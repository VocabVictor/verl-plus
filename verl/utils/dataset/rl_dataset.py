# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import copy
import json
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
from tqdm import tqdm

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


async def _answer_question_async(
    client,
    model: str,
    question_message: str,
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: float = 60.0,
    max_retries: int = 2,
) -> Optional[str]:
    """Call OpenAI API to answer a question (without caption) for caching."""
    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question_message},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=timeout,
            )
            return response.choices[0].message.content.strip()
        except asyncio.TimeoutError:
            logger.warning(f"Timeout answering question (attempt {attempt + 1}/{max_retries + 1})")
        except Exception as e:
            logger.warning(f"Error answering question (attempt {attempt + 1}/{max_retries + 1}): {e}")

        if attempt < max_retries:
            await asyncio.sleep(1.0 * (attempt + 1))

    return None


async def _answer_questions_batch_async(
    question_messages: list[str],
    client,
    model: str,
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: float = 60.0,
    max_retries: int = 2,
    max_concurrency: int = 32,
) -> list[dict]:
    """Answer questions in batch for caching (question-only, no caption)."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def process_one(question_message: str) -> dict:
        async with semaphore:
            if not question_message:
                return {"question": "", "response": None}

            response = await _answer_question_async(
                client=client,
                model=model,
                question_message=question_message,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
            )
            return {"question": question_message, "response": response}

    tasks = [process_one(q) for q in question_messages]
    return await asyncio.gather(*tasks)


def precompute_question_cache(
    dataframe: datasets.Dataset,
    cache_path: str,
    caption_config: dict,
    prompt_key: str = "prompt",
    force_regenerate: bool = False,
) -> str:
    """
    Precompute question cache for CaptionReward after data filtering.

    This caches the OpenAI API responses for QUESTION-ONLY queries (without caption).
    CaptionReward compares:
    - Question-only answer (can be cached, same for all epochs)
    - Question+Caption answer (cannot be cached, depends on model output)

    The cache stores: {question: formatted_question_message, response: openai_answer}

    Args:
        dataframe: Filtered HuggingFace Dataset
        cache_path: Path to save the cache file (JSONL format)
        caption_config: Configuration dict containing API settings
        prompt_key: Key in dataframe containing the prompt/messages
        force_regenerate: If True, regenerate even if cache exists

    Returns:
        Path to the cache file
    """
    # Check if cache already exists and is valid
    print(f"[CACHE CHECK] cache_path={cache_path}, exists={os.path.exists(cache_path)}, force_regenerate={force_regenerate}")
    if os.path.exists(cache_path) and not force_regenerate:
        try:
            existing_count = 0
            valid_count = 0
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    existing_count += 1
                    data = json.loads(line)
                    if data.get("response") is not None:
                        valid_count += 1

            # If cache has enough valid entries, skip regeneration
            # Use 80% threshold to avoid regeneration when dataset filtering varies slightly
            threshold = len(dataframe) * 0.8
            print(f"[CACHE CHECK] valid_count={valid_count}, threshold={threshold} (80% of {len(dataframe)})")
            if valid_count >= threshold:  # 80% threshold
                print(f"[CACHE CHECK] ✅ SKIPPING - Cache has {valid_count} valid entries >= {threshold}")
                logger.info(f"Question cache exists with {valid_count}/{existing_count} valid entries (need {len(dataframe)}), skipping regeneration")
                return cache_path
            else:
                print(f"[CACHE CHECK] ❌ REGENERATING - Cache has {valid_count} valid entries < {threshold}")
                logger.info(f"Question cache has only {valid_count}/{existing_count} valid entries (need 80% of {len(dataframe)}={int(len(dataframe)*0.8)}), regenerating...")
        except Exception as e:
            print(f"[CACHE CHECK] ⚠️ Error reading cache: {e}")
            logger.warning(f"Error reading existing cache: {e}, will regenerate")

    # Get API configuration
    api_key = caption_config.get("api_key")
    if not api_key:
        api_key_env = caption_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)

    if not api_key:
        logger.warning("No API key found for question cache generation, skipping")
        return cache_path

    base_url = caption_config.get("base_url")
    model = caption_config.get("model", "gpt-4o-mini")
    max_concurrency = caption_config.get("max_concurrency", 32)
    temperature = caption_config.get("temperature", 0.0)
    max_tokens = caption_config.get("max_tokens", 256)
    timeout = caption_config.get("timeout", 60.0)
    max_retries = caption_config.get("max_retries", 2)

    # CaptionReward's default templates - must match exactly!
    system_prompt = caption_config.get(
        "system_prompt",
        "You are a careful and precise expert who solves multimodal reasoning tasks."
    )
    question_prompt_template = caption_config.get(
        "question_prompt_template",
        "Question:\n{question}\n\nProvide the best possible answer."
    )

    # Helper to extract question from prompt and remove image/video tags
    # CaptionReward sends PURE TEXT questions to OpenAI (no image tags)
    def extract_question(prompt_messages: list) -> str:
        """Extract the user question from chat messages, removing image/video tags."""
        import re
        for msg in reversed(prompt_messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    # Remove <image> and <video> tags for pure text question
                    text = re.sub(r'<image>\s*', '', content)
                    text = re.sub(r'<video>\s*', '', text)
                    return text.strip()
                elif isinstance(content, list):
                    # Handle multimodal content format - only extract text parts
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    return " ".join(text_parts).strip()
                break
        return ""

    # Extract unique questions from dataframe
    question_messages = []
    seen_questions = set()
    empty_questions = 0

    for i in tqdm(range(len(dataframe)), desc="Extracting questions from prompts", disable=False):
        row = dataframe[i]
        messages = row.get(prompt_key, [])
        question = extract_question(messages)

        if question:
            # Format as CaptionReward does
            question_message = question_prompt_template.format(question=question)
            if question_message not in seen_questions:
                seen_questions.add(question_message)
                question_messages.append(question_message)
        else:
            empty_questions += 1

    logger.info(f"Found {len(question_messages)} unique questions to cache from {len(dataframe)} samples")
    logger.info(f"Samples without questions: {empty_questions}")

    if not question_messages:
        logger.warning("No questions found in dataframe, skipping cache generation")
        return cache_path

    # Create async OpenAI client
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    except ImportError:
        logger.error("openai package not installed, cannot generate question cache")
        return cache_path

    # Generate answers in batches with high concurrency per batch
    # batch_size should match max_concurrency for optimal throughput
    batch_size = max_concurrency
    all_results = []

    async def generate_all():
        nonlocal all_results
        num_batches = (len(question_messages) + batch_size - 1) // batch_size
        logger.info(f"Generating answers for {len(question_messages)} questions in {num_batches} batches (batch_size={batch_size}, concurrency={max_concurrency})")

        for batch_idx in tqdm(range(0, len(question_messages), batch_size),
                              total=num_batches,
                              desc="Generating question answers"):
            batch = question_messages[batch_idx:batch_idx + batch_size]
            results = await _answer_questions_batch_async(
                question_messages=batch,
                client=client,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
                max_concurrency=max_concurrency,
            )
            all_results.extend(results)

    # Run async generation
    asyncio.run(generate_all())

    # Save to cache file
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for result in tqdm(all_results, desc="Saving cache to file", disable=False):
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    valid_count = sum(1 for r in all_results if r.get("response") is not None)
    logger.info(f"Question cache saved to {cache_path} with {valid_count}/{len(all_results)} valid entries")

    return cache_path


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        system_prompt = config.get("system_prompt")
        system_prompt_path = config.get("system_prompt_path")
        self.system_prompt: Optional[str] = None

        if system_prompt_path:
            resolved_path = os.path.expanduser(system_prompt_path)
            if not os.path.isfile(resolved_path):
                raise FileNotFoundError(f"system_prompt_path not found: {resolved_path}")
            with open(resolved_path, "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()

        if system_prompt:
            if isinstance(system_prompt, (list, tuple, ListConfig)):
                system_prompt = "\n".join(str(item) for item in system_prompt)
            self.system_prompt = str(system_prompt).strip()

        if self.system_prompt == "":
            self.system_prompt = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        # Caption cache configuration (for precomputing question cache after filtering)
        self.caption_cache_config = config.get("caption_cache_config", None)
        self.caption_cache_path = config.get("caption_cache_path", None)

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

        # Precompute question cache AFTER filtering (if configured)
        self._maybe_precompute_question_cache()

    def _maybe_precompute_question_cache(self):
        """
        Precompute question cache for caption reward after data filtering.

        This ensures we only generate questions for samples that will actually be used,
        avoiding wasted API calls for filtered-out samples.
        """
        if self.caption_cache_config is None or self.caption_cache_path is None:
            return

        logger.info("Precomputing question cache after data filtering...")

        # Convert DictConfig to dict if needed
        caption_config = dict(self.caption_cache_config) if hasattr(self.caption_cache_config, 'items') else self.caption_cache_config

        precompute_question_cache(
            dataframe=self.dataframe,
            cache_path=self.caption_cache_path,
            caption_config=caption_config,
            prompt_key=self.prompt_key,
            force_regenerate=False,
        )

        logger.info(f"Question cache ready at: {self.caption_cache_path}")

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import MissingImageError, process_image, process_video

                def has_all_images(doc) -> bool:
                    if image_key in doc and doc[image_key]:
                        try:
                            for image in doc[image_key]:
                                process_image(image)
                        except MissingImageError as exc:
                            logger.warning(
                                "Dropping sample because referenced image is missing: %s",
                                getattr(exc, "uri", None) or "<unknown>",
                            )
                            return False
                    return True

                dataframe = dataframe.filter(
                    has_all_images,
                    num_proc=self.num_workers,
                    desc="Dropping samples with missing images",
                )

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    images = None
                    if image_key in doc and doc[image_key]:
                        try:
                            images = [process_image(image) for image in doc[image_key]]
                        except MissingImageError as exc:
                            logger.warning(
                                "Dropping sample during prompt-length filtering because image is missing: %s",
                                getattr(exc, "uri", None) or "<unknown>",
                            )
                            return self.max_prompt_length + 1
                    videos = (
                        [process_video(video) for video in doc[video_key]]
                        if video_key in doc and doc[video_key]
                        else None
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(
                        tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs
                        )
                    )

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.system_prompt is not None:
            system_message = {"role": "system", "content": self.system_prompt}
            if messages:
                first_role = messages[0].get("role")
                if first_role == "system":
                    messages[0]["content"] = self.system_prompt
                else:
                    messages = [system_message] + messages
            else:
                messages = [system_message]

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos = [process_video(video) for video in row_dict_videos]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
