# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
#
# Portions of this file are adapted from ModelScope's ms-swift project
# (https://github.com/modelscope/ms-swift) to ensure reward mechanisms
# remain compatible across projects.

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Type, Union

RewardCallable = Callable[[Sequence[str], Any], List[float]]


logger = logging.getLogger(__name__)


class RewardFunction:
    """Base interface for reward computation on batches of completions."""

    def __call__(self, completions: Sequence[str], **kwargs) -> List[float]:  # pragma: no cover - interface
        raise NotImplementedError


class ReactReward(RewardFunction):
    """ReAct-style reward that compares predicted actions with references."""

    @staticmethod
    def _evaluate_action_reward(
        action_pred: list[str], action_ref: list[str], cand_list: list[str], ref_list: list[str]
    ) -> float:
        f1: list[float] = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0.0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactReward._evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0.0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1.0)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    f1.append(0.0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    f1.append(1.0 if cand_input_json == {} else 0.0)
                else:
                    for key, value in ref_input_json.items():
                        if key in cand_input_json:
                            if cand_input_json[key] == value:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        return 1.0 if f1 and f1[0] == 1.0 else 0.0

    @staticmethod
    def _parse_action(text: str) -> tuple[str, str]:
        if "Action Input:" in text:
            input_idx = text.rindex("Action Input:")
            action_input = text[input_idx + len("Action Input:"):].strip()
        else:
            action_input = "{}"

        if "Action:" in text:
            action_idx = text.rindex("Action:")
            action = text[action_idx + len("Action:"):].strip()
            if "Action Input:" in action:
                input_idx = action.index("Action Input:")
                action = action[:input_idx].strip()
        else:
            action = "none"
        return action, action_input

    @staticmethod
    def _evaluate_rougel(cand_list: list[str], ref_list: list[str]) -> Optional[float]:
        if not ref_list:
            return None
        try:
            from rouge import Rouge

            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            return rouge_score["rouge-l"]["f"]
        except Exception:
            return None

    def __call__(self, completions: Sequence[str], solution: Sequence[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        predictions = list(completions)
        for prediction, reference in zip(predictions, solution, strict=True):
            if prediction.endswith("Observation:"):
                prediction = prediction[: prediction.index("Observation:")].strip()

            prediction = prediction.replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()
            ref_action, ref_input = self._parse_action(reference)
            pred_action, pred_input = self._parse_action(prediction)

            reward = self._evaluate_action_reward(
                [pred_action if pred_action is not None else "none"],
                [ref_action if ref_action is not None else "none"],
                [pred_input if pred_input is not None else "{}"],
                [ref_input if ref_input is not None else "{}"],
            )
            rewards.append(float(reward))
        return rewards


class MathReward(RewardFunction):
    """Symbolic math equivalence reward."""

    def __init__(self) -> None:
        from transformers.utils import strtobool

        self.use_opencompass = strtobool(os.environ.get("USE_OPENCOMPASS_EVALUATOR", "False"))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator

            self.evaluator = MATHEvaluator()

    @staticmethod
    def _extract_boxed_result(text: str) -> str:
        pattern = r"\\boxed{([^}]*)}"
        match = re.search(pattern, text)
        return match.group(1).strip() if match else text

    @staticmethod
    def _clean_latex(latex_str: str) -> str:
        latex_str = re.sub(r"\\\(|\\\)|\\\[|\\]", "", latex_str)
        latex_str = latex_str.replace("}}", "}").replace("{", "").replace("}", "")
        return latex_str.strip()

    @staticmethod
    def _parse_expression(latex_str: str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex

        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @classmethod
    def _compare_consecutive(cls, first: str, second: str) -> bool:
        cleaned = [cls._clean_latex(x) for x in [first, second]]
        parsed = [cls._parse_expression(x) for x in cleaned]
        if hasattr(parsed[0], "equals") and hasattr(parsed[1], "equals"):
            value = parsed[0].equals(parsed[1])
        else:
            value = parsed[0] == parsed[1]
        return bool(value)

    def __call__(self, completions: Sequence[str], solution: Sequence[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        for prediction, ground_truth in zip(completions, solution, strict=True):
            if "# Answer" in prediction:
                prediction = prediction.split("# Answer")[-1]
            if "# Answer" in ground_truth:
                ground_truth = ground_truth.split("# Answer")[-1]
            prediction = self._extract_boxed_result(prediction.strip())
            ground_truth = self._extract_boxed_result(ground_truth.strip())
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = self._compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(RewardFunction):
    """Robust math accuracy reward using math_verify."""

    def __init__(self) -> None:
        import importlib.util

        if importlib.util.find_spec("math_verify") is None:
            raise ImportError(
                "The math_verify package is required but not installed. "
                "Please install it via 'pip install math_verify==0.5.2'."
            )
        # Display name used by loggers/metrics
        self.name = "accuracy"

    def __call__(self, completions: Sequence[str], solution: Sequence[str], **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify

        rewards: List[float] = []
        for content, sol in zip(completions, solution, strict=True):
            gold_parsed = parse(sol, extraction_mode="first_match")
            if gold_parsed:
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed=True,
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                reward = 0.0
            rewards.append(reward)
        return rewards


class FormatReward(RewardFunction):
    """Checks completion format compliance via configurable regex."""

    DEFAULT_PATTERN = r"^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])"
    DEFAULT_FLAGS = re.DOTALL | re.MULTILINE

    def __init__(self, pattern: Optional[str] = None, flags: Optional[Union[int, str, Sequence[str]]] = None) -> None:
        # Display name used by loggers/metrics
        self.name = "format"
        if pattern is None:
            pattern = self.DEFAULT_PATTERN

        flag_value = self.DEFAULT_FLAGS
        if flags is not None:
            flag_value = self._parse_flags(flags)

        self._pattern_str = pattern
        self._flags = flag_value
        self._compiled = re.compile(pattern, flag_value)

    @staticmethod
    def _parse_flags(flags: Union[int, str, Sequence[str]]) -> int:
        if isinstance(flags, int):
            return flags

        if isinstance(flags, str):
            flags = [flag.strip() for flag in flags.split("|") if flag.strip()]

        value = 0
        for name in flags:
            attr = getattr(re, name, None)
            if attr is None:
                raise ValueError(f"Unsupported regex flag '{name}' for FormatReward")
            value |= attr
        return value if value else 0

    def __call__(self, completions: Sequence[str], **kwargs) -> List[float]:
        return [1.0 if self._compiled.match(content) else 0.0 for content in completions]


class ReActFormatReward(RewardFunction):
    """Checks ReAct style format compliance."""

    FORMAT_PATTERN = re.compile(r"^<think>.*?</think>\s*Action:.*?Action Input:.*?$", re.DOTALL | re.MULTILINE)

    def __call__(self, completions: Sequence[str], **kwargs) -> List[float]:
        return [1.0 if self.FORMAT_PATTERN.match(content) else 0.0 for content in completions]


class CosineReward(RewardFunction):
    """Length-aware cosine shaping reward (https://arxiv.org/abs/2502.03373)."""

    def __init__(
        self,
        cosine_min_len_value_wrong: float = -0.5,
        cosine_max_len_value_wrong: float = 0.0,
        cosine_min_len_value_correct: float = 1.0,
        cosine_max_len_value_correct: float = 0.5,
        cosine_max_len: Optional[int] = None,
        accuracy_orm: Optional[RewardFunction] = None,
    ) -> None:
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def _cosine_decay(t: int, T: int, min_value: float, max_value: float) -> float:
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions: Sequence[str], solution: Sequence[str], **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids: Sequence[Sequence[int]] = kwargs.get("response_token_ids", [])
        rewards: List[float] = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards, strict=True):
            is_correct = acc_reward >= 1.0
            if is_correct:
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            length_cap = self.max_len or max(gen_len, 1)
            rewards.append(self._cosine_decay(gen_len, length_cap, min_value, max_value))
        return rewards


class RepetitionPenalty(RewardFunction):
    """Penalizes repeated n-grams (https://arxiv.org/abs/2502.03373)."""

    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0) -> None:
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def _zip_ngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions: Sequence[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        for completion in completions:
            if not completion:
                rewards.append(0.0)
                continue
            tokens = completion.split()
            if len(tokens) < self.ngram_size:
                rewards.append(0.0)
                continue
            ngrams = set()
            total = 0
            for ng in self._zip_ngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1
            scaling = 1 - len(ngrams) / max(total, 1)
            rewards.append(scaling * self.max_penalty)
        return rewards


class SoftOverlongReward(RewardFunction):
    """Soft penalty for overlong completions used in DAPO."""

    def __init__(self, soft_max_length: int, soft_cache_length: int) -> None:
        if soft_cache_length >= soft_max_length:
            raise ValueError("soft_cache_length must be smaller than soft_max_length")
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions: Sequence[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        response_token_ids: Sequence[Sequence[int]] = kwargs.get("response_token_ids", [])
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0.0))
        return rewards


class GSM8KReward(RewardFunction):
    """Reward for GSM8K style math problems."""

    def __init__(self, method: str = "strict", format_score: float = 0.0, score: float = 1.0) -> None:
        if method not in {"strict", "flexible"}:
            raise ValueError("method must be either 'strict' or 'flexible'")
        self.method = method
        self.format_score = format_score
        self.score = score

    @staticmethod
    def _extract_solution(solution_str: str, method: str) -> Optional[str]:
        clip_chars = 300
        if len(solution_str) > clip_chars:
            solution_str = solution_str[-clip_chars:]
        if method == "strict":
            solutions = re.findall(r"#### (\\-?[0-9\\.\\,]+)", solution_str)
            if not solutions:
                return None
            return solutions[-1].replace(",", "").replace("$", "")
        answer = re.findall(r"(\\-?[0-9\\.\\,]+)", solution_str)
        if not answer:
            return None
        for candidate in reversed(answer):
            if candidate not in {"", "."}:
                return candidate
        return None

    def __call__(self, completions: Sequence[str], solution: Sequence[str], **kwargs) -> List[float]:
        rewards: List[float] = []
        for content, gold in zip(completions, solution, strict=True):
            answer = self._extract_solution(content, self.method)
            if answer is None:
                rewards.append(0.0)
            elif answer == gold:
                rewards.append(self.score)
            else:
                rewards.append(self.format_score)
        return rewards


class Geo3KReward(RewardFunction):
    """Reward for GEO3K benchmark using MathRuler grading."""

    def __init__(self, use_boxed: bool = True, format_score: float = 0.1) -> None:
        try:
            from mathruler.grader import extract_boxed_content, grade_answer  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "mathruler is required for Geo3KReward. Install via 'pip install mathruler'."
            ) from exc
        self.use_boxed = use_boxed
        self.format_score = format_score

    @staticmethod
    def _format_reward(predict_str: str) -> float:
        pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
        return 1.0 if pattern.fullmatch(predict_str) else 0.0

    def __call__(self, completions: Sequence[str], solution: Sequence[str], **kwargs) -> List[float]:
        from mathruler.grader import extract_boxed_content, grade_answer

        rewards: List[float] = []
        for content, gold in zip(completions, solution, strict=True):
            answer = extract_boxed_content(content) if self.use_boxed else content
            acc = 1.0 if grade_answer(answer, gold) else 0.0
            fmt = self._format_reward(content)
            rewards.append((1.0 - self.format_score) * acc + self.format_score * fmt)
        return rewards


class GainReward(RewardFunction):
    """Gain-based reward that compares caption-aided vs direct answers."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: str | None = "bfloat16",
        trust_remote_code: bool = True,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        generation_kwargs: Mapping[str, Any] | None = None,
        caption_prefix: str = "Here is an auxiliary caption describing the image:",
        caption_template: str | None = None,
        semantic_match_case_sensitive: bool = False,
        **_unused_kwargs,
    ) -> None:
        if not model_path:
            raise ValueError("GainReward requires 'model_path' to be provided.")

        self.model_path = model_path
        self.device = self._resolve_device(device)
        self._target_dtype = self._resolve_dtype(torch_dtype)
        self.trust_remote_code = trust_remote_code
        self.max_new_tokens = max_new_tokens
        self.caption_prefix = caption_prefix.strip()
        self.caption_template = caption_template
        self.semantic_match_case_sensitive = semantic_match_case_sensitive

        base_generation = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
        }
        self._generation_kwargs = {k: v for k, v in base_generation.items() if v is not None}
        if generation_kwargs:
            self._generation_kwargs.update(dict(generation_kwargs))

        self._model = None
        self._processor = None
        self._tokenizer = None
        self._pad_token_id: Optional[int] = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        if device in {"auto", ""}:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _resolve_dtype(dtype: str | None):
        if dtype is None or str(dtype).lower() == "auto":
            return None

        import torch

        normalized = str(dtype).lower()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported torch_dtype '{dtype}' for GainReward.")
        return mapping[normalized]

    def _ensure_model(self) -> None:
        if self._model is not None and self._processor is not None and self._tokenizer is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "GainReward requires the 'transformers' package. Install via 'pip install transformers'."
            ) from exc

        import torch

        processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
        model_kwargs: dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if self._target_dtype is not None:
            model_kwargs["torch_dtype"] = self._target_dtype

        model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        model.eval()
        if self.device.startswith("cuda"):
            model.to(self.device)

        self._processor = processor
        self._tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self._model = model
        self._pad_token_id = getattr(self._tokenizer, "pad_token_id", None)
        if self._pad_token_id is None:
            self._pad_token_id = getattr(self._tokenizer, "eos_token_id", None)

        if self._pad_token_id is None:
            raise ValueError("GainReward could not determine pad/eos token id for generation.")

        if "pad_token_id" not in self._generation_kwargs:
            self._generation_kwargs["pad_token_id"] = self._pad_token_id

    def _prepare_inputs(self, prompt: str, images: Optional[Sequence[Any]]):
        self._ensure_model()

        processor_kwargs: dict[str, Any] = {"text": [prompt], "return_tensors": "pt"}
        if images:
            processor_kwargs["images"] = list(images)

        inputs = self._processor(**processor_kwargs)

        import torch

        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(self.device)
            elif value is None:
                inputs.pop(key)
        return inputs

    def _generate(self, prompt: str, images: Optional[Sequence[Any]]) -> str:
        inputs = self._prepare_inputs(prompt, images)

        import torch

        input_length = inputs["input_ids"].shape[-1]
        generation_kwargs = dict(self._generation_kwargs)
        if "max_new_tokens" not in generation_kwargs:
            generation_kwargs["max_new_tokens"] = self.max_new_tokens

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation_kwargs)

        generated_ids = output_ids[0, input_length:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=False)

    @staticmethod
    def _extract_last_answer(text: str) -> str:
        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
        matches = list(answer_pattern.finditer(text))
        if matches:
            return matches[-1].group(1).strip()
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped
        return ""

    def _call_openai_batch(self, messages: Sequence[str]) -> Dict[str, str]:
        unique_messages = list(dict.fromkeys(messages))
        if not unique_messages:
            return {}

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                tasks = [loop.run_in_executor(executor, self._call_openai_sync, msg) for msg in unique_messages]
                outputs = loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            try:
                asyncio.set_event_loop(None)
            finally:
                loop.close()

        return dict(zip(unique_messages, outputs))

    def _semantic_match(self, prediction: str, ground_truth: str) -> bool:
        from verl.utils.reward_score.search_r1_like_qa_em import normalize_answer

        pred_answer = self._extract_last_answer(prediction)
        gold_answer = self._extract_last_answer(ground_truth)

        if not pred_answer:
            return False

        if not self.semantic_match_case_sensitive:
            pred_norm = normalize_answer(pred_answer)
            gold_norm = normalize_answer(gold_answer)
        else:
            pred_norm = unicodedata.normalize("NFKC", pred_answer)
            gold_norm = unicodedata.normalize("NFKC", gold_answer)

        if not gold_norm:
            gold_norm = normalize_answer(ground_truth) if not self.semantic_match_case_sensitive else ground_truth

        return pred_norm == gold_norm or gold_norm in pred_norm or pred_norm in gold_norm

    def _inject_caption(self, prompt: str, caption: str) -> str:
        caption = caption.strip()
        if not caption:
            return prompt

        caption_message = (
            self.caption_template.format(caption=caption)
            if self.caption_template
            else f"\n\n{self.caption_prefix}\n{caption}\n"
        )

        user_end_token = "<|im_end|>"
        last_user_end = prompt.rfind(user_end_token)
        if last_user_end == -1:
            return prompt + caption_message
        return prompt[:last_user_end] + caption_message + prompt[last_user_end:]

    def __call__(
        self,
        completions: Sequence[str],
        *,
        prompts: Sequence[str],
        solution: Sequence[str],
        multi_modal_data: Sequence[Optional[Mapping[str, Any]]] | None = None,
        **_kwargs,
    ) -> List[float]:
        if multi_modal_data is None:
            raise ValueError("GainReward requires 'multi_modal_data' with image references.")

        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            prompt = prompts[idx]
            ground_truth = solution[idx]
            mm_data = multi_modal_data[idx] or {}
            images = mm_data.get("image") if isinstance(mm_data, Mapping) else None
            caption_only = self._extract_caption(completion)
            caption_payload = caption_only if caption_only else completion

            try:
                direct_answer = self._generate(prompt, images)
            except Exception as exc:  # pragma: no cover - runtime safeguard
                logger.error("GainReward failed during direct inference: %s", exc)
                direct_answer = ""

            try:
                caption_prompt = self._inject_caption(prompt, caption_payload)
                caption_answer = self._generate(caption_prompt, images)
            except Exception as exc:  # pragma: no cover - runtime safeguard
                logger.error("GainReward failed during caption inference: %s", exc)
                caption_answer = ""

            caption_correct = self._semantic_match(caption_answer, ground_truth)
            direct_correct = self._semantic_match(direct_answer, ground_truth)

            if caption_correct and not direct_correct:
                reward = 1.0
            elif caption_correct and direct_correct:
                reward = 0.7
            elif not caption_correct and not direct_correct:
                reward = 0.2
            else:
                reward = 0.0

            rewards.append(float(reward))

        return rewards

    @staticmethod
    def _extract_caption(completion: str) -> str:
        pattern = re.compile(r"<caption>(.*?)</caption>", re.DOTALL | re.IGNORECASE)
        match = pattern.search(completion)
        if match:
            return match.group(1).strip()
        return ""


class CaptionReward(RewardFunction):
    """Evaluates caption utility by comparing OpenAI answers with/without the caption."""

    DEFAULT_SYSTEM_PROMPT = "You are a careful and precise expert who solves multimodal reasoning tasks."
    DEFAULT_QUESTION_PROMPT = "Question:\n{question}\n\nProvide the best possible answer."
    DEFAULT_CAPTION_PROMPT = (
        "You are given a caption produced by another assistant that may help answer the question.\n"
        "Caption:\n{caption}\n\nQuestion:\n{question}"
    )

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 128,
        max_retries: int = 2,
        system_prompt: Optional[str] = None,
        question_prompt_template: Optional[str] = None,
        caption_prompt_template: Optional[str] = None,
        timeout: Optional[float] = None,
        semantic_case_sensitive: bool = False,
        max_concurrency: int = 8,
        question_cache_path: Optional[str] = None,
        precompute_cache: bool = False,
        train_files: Optional[List[str]] = None,
    ) -> None:
        if not model:
            raise ValueError("CaptionReward requires an OpenAI model name via 'model'.")

        resolved_key = api_key or os.getenv(api_key_env or "")
        if not resolved_key:
            raise ValueError(
                "CaptionReward requires an OpenAI API key. Provide via 'api_key' or environment variable."
            )

        self.model = model
        self.api_key = resolved_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.question_prompt_template = question_prompt_template or self.DEFAULT_QUESTION_PROMPT
        self.caption_prompt_template = caption_prompt_template or self.DEFAULT_CAPTION_PROMPT
        self.timeout = timeout
        self.semantic_case_sensitive = semantic_case_sensitive
        self.max_concurrency = max(1, int(max_concurrency))

        self._client = None
        self._question_cache: Dict[str, str] = {}
        self._caption_cache: Dict[str, str] = {}
        self.precompute_cache = precompute_cache
        self.train_files = train_files

        # Load precomputed question cache if provided
        if question_cache_path and os.path.exists(question_cache_path):
            self._load_question_cache(question_cache_path)

        # Precompute cache if enabled and cache doesn't exist
        # Check environment variable for precompute cache
        precompute_env = os.getenv("PRECOMPUTE_QUESTION_CACHE", "false").lower() == "true"

        if (precompute_env or precompute_cache) and question_cache_path and train_files:
            if not os.path.exists(question_cache_path):
                logger.info("Precomputing question cache from training data...")
                self._precompute_question_cache(question_cache_path, train_files)

    def _load_question_cache(self, cache_path: str) -> None:
        """Load precomputed question cache from JSONL file."""
        logger.info(f"Loading precomputed question cache from {cache_path}")
        try:
            loaded_count = 0
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if "question" in data and "response" in data and data["response"] is not None:
                            self._question_cache[data["question"]] = data["response"]
                            loaded_count += 1
                    except json.JSONDecodeError:
                        continue
            logger.info(f"Loaded {loaded_count} precomputed question responses")
        except Exception as e:
            logger.warning(f"Failed to load question cache from {cache_path}: {e}")

    def _precompute_question_cache(self, cache_path: str, train_files: List[str]) -> None:
        """Precompute question cache from training data files."""
        import asyncio
        import aiohttp
        from tqdm import tqdm

        self._ensure_client()

        # Extract all unique questions from training files
        logger.info("Extracting unique questions from training data...")
        unique_questions = self._extract_unique_questions(train_files)
        logger.info(f"Found {len(unique_questions)} unique questions")

        # Filter out questions already in cache
        uncached_questions = [q for q in unique_questions if q not in self._question_cache]
        logger.info(f"Need to generate responses for {len(uncached_questions)} questions")

        if not uncached_questions:
            logger.info("All questions are already cached!")
            return

        # Generate cache using async API calls
        asyncio.run(self._generate_cache_async(uncached_questions, cache_path))

    def _extract_unique_questions(self, train_files: List[str]) -> List[str]:
        """Extract unique questions from training parquet files."""
        unique_questions = set()

        for file_path in train_files:
            try:
                import pandas as pd
                df = pd.read_parquet(file_path)

                for _, row in df.iterrows():
                    # Extract prompt from numpy array
                    prompt_list = row['prompt']
                    if not isinstance(prompt_list, list):
                        continue

                    # Find the user message (question)
                    for msg in prompt_list:
                        if msg.get('role') == 'user':
                            # Extract question after 'Question:\n'
                            content = msg.get('content', '')
                            if 'Question:\n' in content:
                                question = content.split('Question:\n')[1].strip()
                                unique_questions.add(question)
                            break
            except Exception as e:
                logger.warning(f"Failed to extract questions from {file_path}: {e}")

        return list(unique_questions)

    async def _generate_cache_async(self, questions: List[str], cache_path: str) -> None:
        """Generate cache using async API calls."""
        import aiohttp
        from tqdm import tqdm
        import time

        # Create output directory if needed
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        connector = aiohttp.TCPConnector(limit=self.max_concurrency * 2)
        timeout = aiohttp.ClientTimeout(total=60)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        # Rate limiting
        requests_per_second = 50
        request_times = []

        async def call_api(session: aiohttp.ClientSession, question: str) -> str:
            async with semaphore:
                # Rate limiting
                now = time.time()
                request_times[:] = [t for t in request_times if now - t < 1.0]
                if len(request_times) >= requests_per_second:
                    sleep_time = 1.0 - (now - request_times[0])
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                request_times.append(now)

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                data = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Please answer the question concisely and accurately."
                        },
                        {
                            "role": "user",
                            "content": question
                        }
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }

                try:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=timeout
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return result["choices"][0]["message"]["content"]
                        else:
                            error_text = await response.text()
                            logger.error(f"API error. Status: {response.status}, Error: {error_text}")
                            return None
                except Exception as e:
                    logger.error(f"Exception when calling API: {str(e)}")
                    return None

        # Save progress periodically
        def save_progress():
            temp_path = cache_path + ".tmp"
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    for question, response in self._question_cache.items():
                        f.write(json.dumps({"question": question, "response": response}) + "\n")
                os.rename(temp_path, cache_path)
            except Exception as e:
                logger.error(f"Failed to save progress: {e}")

        # Process all questions
        success_count = 0
        error_count = 0
        save_interval = 100

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for question in questions:
                task = asyncio.create_task(call_api(session, question))
                tasks.append((question, task))

            with tqdm(total=len(questions), desc="Generating cache") as pbar:
                for question, task in tasks:
                    try:
                        response = await task
                        if response:
                            self._question_cache[question] = response
                            success_count += 1
                        else:
                            error_count += 1

                        pbar.update(1)

                        # Save progress periodically
                        if (success_count + error_count) % save_interval == 0:
                            save_progress()
                            logger.info(f"Progress: {success_count + error_count}/{len(questions)} "
                                      f"(Success: {success_count}, Errors: {error_count})")
                    except Exception as e:
                        logger.error(f"Error processing question: {str(e)}")
                        error_count += 1
                        pbar.update(1)

        # Final save
        save_progress()
        logger.info(f"Cache generation complete! Success: {success_count}, Errors: {error_count}")

    def _ensure_client(self) -> None:
        if self._client is not None:
            return

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install the openai package to use CaptionReward.") from exc

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = OpenAI(**client_kwargs)

    @staticmethod
    def _extract_caption(completion: str) -> str:
        match = re.search(r"<caption>(.*?)</caption>", completion, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_question(prompt: str) -> str:
        """Extract pure text question from prompt, removing image/video tags."""
        match = re.findall(r"<\|im_start\|>user\s*(.*?)<\|im_end\|>", prompt, re.DOTALL | re.IGNORECASE)
        if match:
            text = match[-1].strip()
        else:
            text = prompt.strip()
        # Remove <image> and <video> tags for pure text question to OpenAI
        text = re.sub(r'<image>\s*', '', text)
        text = re.sub(r'<video>\s*', '', text)
        return text.strip()

    @staticmethod
    def _extract_generated_answer(text: str) -> str:
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if stripped:
                return stripped
        return text.strip()

    def _call_openai_sync(self, user_message: str) -> str:
        self._ensure_client()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.timeout is not None:
            request_kwargs["timeout"] = self.timeout

        for attempt in range(max(1, self.max_retries)):
            try:
                try:
                    response = self._client.chat.completions.create(**request_kwargs)
                except TypeError as type_err:
                    if request_kwargs.pop("timeout", None) is not None:
                        response = self._client.chat.completions.create(**request_kwargs)
                    else:
                        raise type_err
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # pragma: no cover - runtime safeguard
                logger.warning(
                    "CaptionReward OpenAI call failed (attempt %s/%s): %s", attempt + 1, self.max_retries, exc
                )
        return ""

    def _call_openai_batch(self, messages: Sequence[str]) -> Dict[str, str]:
        """Issue OpenAI chat completions concurrently for a list of prompts."""

        unique_messages = list(dict.fromkeys(messages))
        if not unique_messages:
            return {}

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                tasks = [loop.run_in_executor(executor, self._call_openai_sync, msg) for msg in unique_messages]
                outputs = loop.run_until_complete(asyncio.gather(*tasks))
        finally:
            try:
                asyncio.set_event_loop(None)
            finally:
                loop.close()

        result = dict(zip(unique_messages, outputs))

        return result

    def _semantic_match(self, prediction: str, ground_truth: str) -> bool:
        from verl.utils.reward_score.search_r1_like_qa_em import normalize_answer

        if not prediction:
            return False

        pred_answer = self._extract_generated_answer(prediction)
        gold_answer = self._extract_generated_answer(ground_truth)

        if not pred_answer:
            return False

        if self.semantic_case_sensitive:
            pred_norm = unicodedata.normalize("NFKC", pred_answer)
            gold_norm = unicodedata.normalize("NFKC", gold_answer)
            return pred_norm == gold_norm or gold_norm in pred_norm or pred_norm in gold_norm

        pred_norm = normalize_answer(pred_answer)
        gold_norm = normalize_answer(gold_answer)
        return pred_norm == gold_norm or gold_norm in pred_norm or pred_norm in gold_norm

    def __call__(
        self,
        completions: Sequence[str],
        *,
        prompts: Sequence[str],
        solution: Sequence[str],
        **_kwargs,
    ) -> List[float]:
        pending_question_messages: list[str] = []
        pending_caption_messages: list[str] = []
        pending_question_set: set[str] = set()
        pending_caption_set: set[str] = set()

        samples: list[tuple[str, str, Optional[str]]] = []
        for prompt, completion, ground_truth in zip(prompts, completions, solution, strict=True):
            question = self._extract_question(prompt)
            caption = self._extract_caption(completion)

            question_message = self.question_prompt_template.format(question=question)
            if question_message not in self._question_cache and question_message not in pending_question_set:
                pending_question_set.add(question_message)
                pending_question_messages.append(question_message)

            if caption:
                caption_message = self.caption_prompt_template.format(question=question, caption=caption)
                if caption_message not in self._caption_cache and caption_message not in pending_caption_set:
                    pending_caption_set.add(caption_message)
                    pending_caption_messages.append(caption_message)
            else:
                caption_message = None

            samples.append((ground_truth, question_message, caption_message))

        if pending_question_messages:
            self._question_cache.update(self._call_openai_batch(pending_question_messages))
        if pending_caption_messages:
            self._caption_cache.update(self._call_openai_batch(pending_caption_messages))

        rewards: List[float] = []
        for ground_truth, question_message, caption_message in samples:
            direct_answer = self._question_cache.get(question_message, "")
            direct_correct = self._semantic_match(direct_answer, ground_truth)

            if caption_message is not None:
                caption_answer = self._caption_cache.get(caption_message, "")
                caption_correct = self._semantic_match(caption_answer, ground_truth)
            else:
                caption_correct = False

            if caption_correct and not direct_correct:
                reward = 1.0
            elif caption_correct and direct_correct:
                reward = 0.7
            elif not caption_correct and not direct_correct:
                reward = 0.2
            else:
                reward = 0.0

            rewards.append(float(reward))

        return rewards

@dataclass
class RewardSpec:
    """Container describing an instantiated reward function."""

    name: str
    fn: RewardFunction


_REWARD_REGISTRY: Dict[str, Type[RewardFunction]] = {
    "toolbench": ReactReward,
    "react": ReactReward,
    "math": MathReward,
    "accuracy": MathAccuracy,
    "format": FormatReward,
    "react_format": ReActFormatReward,
    "cosine": CosineReward,
    "repetition": RepetitionPenalty,
    "soft_overlong": SoftOverlongReward,
    "gsm8k": GSM8KReward,
    "geo3k": Geo3KReward,
    "gain": GainReward,
    "caption": CaptionReward,
}


def register_reward(name: str) -> Callable[[Type[RewardFunction]], Type[RewardFunction]]:
    """Decorator to register a reward class under ``name``."""

    lowered = name.lower()

    def decorator(cls: Type[RewardFunction]) -> Type[RewardFunction]:
        if lowered in _REWARD_REGISTRY and _REWARD_REGISTRY[lowered] is not cls:
            raise ValueError(f"Reward '{name}' already registered")
        _REWARD_REGISTRY[lowered] = cls
        return cls

    return decorator


def list_registered_rewards() -> List[str]:
    return sorted(_REWARD_REGISTRY.keys())


def resolve_reward_cls(name: str) -> Type[RewardFunction]:
    lowered = name.lower()
    if lowered not in _REWARD_REGISTRY:
        raise KeyError(f"Reward '{name}' is not registered. Available: {list_registered_rewards()}")
    return _REWARD_REGISTRY[lowered]


def instantiate_reward(
    name: str,
    config: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> RewardFunction:
    """Instantiate a reward function using config/default parameters."""

    cls = resolve_reward_cls(name)
    overrides = dict(overrides or {})

    reward_kwargs: Mapping[str, Any] = {}
    if config is not None:
        reward_kwargs = config.get("reward_kwargs", {}) if isinstance(config, Mapping) else {}
        per_reward_cfg = reward_kwargs.get(name, {}) if isinstance(reward_kwargs, Mapping) else {}
    else:
        per_reward_cfg = {}

    combined: Dict[str, Any] = {}
    if isinstance(per_reward_cfg, Mapping):
        combined.update(per_reward_cfg)
    combined.update(overrides)

    signature = inspect.signature(cls.__init__)
    init_kwargs: Dict[str, Any] = {}

    for param_name, parameter in signature.parameters.items():
        if param_name == "self" or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if param_name in combined:
            init_kwargs[param_name] = combined[param_name]
            continue
        if config is not None and isinstance(config, Mapping) and param_name in config:
            value = config[param_name]
            if value is not None:
                init_kwargs[param_name] = value
                continue
        if parameter.default is inspect._empty:
            raise ValueError(
                f"Missing required parameter '{param_name}' for reward '{name}'. "
                "Provide it under reward_model.{param_name} or "
                f"reward_model.reward_kwargs.{name}.{param_name}."
            )

    return cls(**init_kwargs)


def build_reward_specs(
    reward_entries: Iterable[str | Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> List[RewardSpec]:
    """Instantiate rewards from config entries."""

    specs: List[RewardSpec] = []
    for entry in reward_entries:
        if isinstance(entry, Mapping):
            if "name" not in entry:
                raise ValueError(f"Reward mapping requires a 'name' field: {entry}")
            name = str(entry["name"])
            overrides = entry.get("params", {})
        else:
            name = str(entry)
            overrides = {}
        fn = instantiate_reward(name, config=config, overrides=overrides)
        display_name = getattr(fn, "name", name)
        specs.append(RewardSpec(name=display_name, fn=fn))
    return specs


def reward_specs_to_dict(specs: Sequence[RewardSpec]) -> Dict[str, RewardFunction]:
    return {spec.name: spec.fn for spec in specs}


__all__ = [
    "RewardFunction",
    "RewardSpec",
    "register_reward",
    "list_registered_rewards",
    "resolve_reward_cls",
    "instantiate_reward",
    "build_reward_specs",
    "reward_specs_to_dict",
    "GainReward",
    "CaptionReward",
]
