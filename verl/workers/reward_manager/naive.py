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

from collections import defaultdict
from typing import Iterable, Mapping, Optional, Sequence

import torch

from verl import DataProto
from verl.reward import RewardSpec, build_reward_specs
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("rule")
class RuleRewardEvaluator(AbstractRewardManager):
    """Rule-based reward evaluator compatible with ms-swift style reward functions."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key: Optional[str] = None,
        reward_funcs: Optional[Iterable[str | Mapping[str, object]]] = None,
        reward_specs: Optional[Sequence[RewardSpec]] = None,
        reward_weights: Optional[Sequence[float]] = None,
        reward_config: Optional[Mapping[str, object]] = None,
        **_ignored_kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.custom_compute = compute_score
        self.reward_fn_key = reward_fn_key or "data_source"

        if reward_specs is None and reward_funcs:
            reward_specs = build_reward_specs(reward_funcs, config=reward_config)
        self.reward_specs = list(reward_specs or [])

        if reward_weights is None:
            self.reward_weights = [1.0] * len(self.reward_specs)
        else:
            self.reward_weights = list(reward_weights)

        if self.reward_specs and len(self.reward_specs) != len(self.reward_weights):
            raise ValueError(
                "Number of reward_weights must match number of reward functions. "
                f"Got {len(self.reward_weights)} weights for {len(self.reward_specs)} functions."
            )

        if not self.reward_specs and self.custom_compute is None:
            raise ValueError(
                "RuleRewardEvaluator requires reward_funcs or a custom compute_score callable."
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        # If RM scores already exist, bypass rule-based computation
        if "rm_scores" in data.batch:
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        if self.reward_specs:
            result = self._compute_batch_rewards(data)
        else:
            result = self._compute_custom_rewards(data)

        if return_dict:
            return result
        return result["reward_tensor"]

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _compute_batch_rewards(self, data: DataProto) -> dict[str, object]:
        device = data.batch["responses"].device

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list[float]] = defaultdict(list)

        prompts: list[str] = []
        completions: list[str] = []
        ground_truths: list[str] = []
        extra_infos: list[Mapping[str, object]] = []
        multi_modal_batches: list[Mapping[str, object] | None] = []
        response_token_ids: list[list[int]] = []
        valid_response_lengths: list[int] = []

        # Collect decoded data
        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            attention_mask = item.batch["attention_mask"]

            valid_prompt_length = int(attention_mask[:prompt_length].sum().item())
            prompt_tokens = prompt_ids[-valid_prompt_length:]
            prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)

            response_ids = item.batch["responses"]
            valid_response_length = int(attention_mask[prompt_length:].sum().item())
            if valid_response_length > 0:
                response_tokens = response_ids[:valid_response_length]
            else:
                response_tokens = response_ids[:0]
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

            prompts.append(prompt_text)
            completions.append(response_text)
            response_token_ids.append(response_tokens.tolist())
            ground_truths.append(item.non_tensor_batch["reward_model"]["ground_truth"])
            extra_infos.append(item.non_tensor_batch.get("extra_info", {}) or {})
            multi_modal_batches.append(item.non_tensor_batch.get("multi_modal_data"))
            valid_response_lengths.append(valid_response_length)

        weights_tensor = torch.tensor(self.reward_weights, dtype=torch.float32, device=device)
        rewards_per_func = torch.zeros((len(completions), len(self.reward_specs)), dtype=torch.float32, device=device)

        common_kwargs = {
            "solution": ground_truths,
            "prompts": prompts,
            "response_token_ids": response_token_ids,
            "extra_info": extra_infos,
            "multi_modal_data": multi_modal_batches,
        }

        def _normalize_name(name: str) -> str:
            key = name.strip().lower().replace("_", "-")
            if key in {"format", "accuracy"}:
                return name  # keep original casing from spec
            if key in {"caption-format", "captionformat", "caption-format-reward"}:
                return "format"
            return name

        for col, spec in enumerate(self.reward_specs):
            raw_outputs = spec.fn(
                completions,
                **common_kwargs,
            )
            if len(raw_outputs) != len(completions):
                raise ValueError(
                    f"Reward '{spec.name}' returned {len(raw_outputs)} scores for "
                    f"batch size {len(completions)}."
                )
            # Use a normalized display name for the metric key.
            # - keep "format" and "accuracy" as-is
            # - alias any "caption-format" variants to "format"
            converted = [0.0 if value is None else float(value) for value in raw_outputs]
            rewards_per_func[:, col] = torch.tensor(converted, dtype=torch.float32, device=device)
            display_name = _normalize_name(spec.name)
            reward_extra_info[display_name].extend(converted)

        total_rewards = (rewards_per_func * weights_tensor.unsqueeze(0)).sum(dim=1)
        total_rewards_list = total_rewards.tolist()
        # Keep the aggregated reward under a concise key as well.
        reward_extra_info["total"].extend(total_rewards_list)

        already_printed = 0
        for idx, (prompt, completion, ground_truth, reward_value) in enumerate(
            zip(prompts, completions, ground_truths, total_rewards_list, strict=True)
        ):
            valid_len = valid_response_lengths[idx]
            if valid_len > 0:
                reward_tensor[idx, valid_len - 1] = reward_value

            if already_printed < self.num_examine:
                already_printed += 1
                print("[prompt]", prompt)
                print("[response]", completion)
                print("[ground_truth]", ground_truth)
                for col, spec in enumerate(self.reward_specs):
                    print(f"[{_normalize_name(spec.name)}]", rewards_per_func[idx, col].item())
                print("[score]", reward_value)

        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info,
        }

    def _compute_custom_rewards(self, data: DataProto) -> dict[str, object]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list[float]] = defaultdict(list)
        already_printed = 0

        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            attention_mask = item.batch["attention_mask"]

            valid_prompt_length = int(attention_mask[:prompt_length].sum().item())
            prompt_tokens = prompt_ids[-valid_prompt_length:]

            response_ids = item.batch["responses"]
            valid_response_length = int(attention_mask[prompt_length:].sum().item())
            response_tokens = response_ids[:valid_response_length]

            prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
            response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            extra_info = item.non_tensor_batch.get("extra_info", {})
            score = self.custom_compute(
                solution_str=response_text,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward_value = float(score.get("score", 0.0))
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward_value = float(score)
                reward_extra_info["score"].append(reward_value)

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = reward_value

            if already_printed < self.num_examine:
                already_printed += 1
                print("[prompt]", prompt_text)
                print("[response]", response_text)
                print("[ground_truth]", ground_truth)
                print("[score]", reward_value)

        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": reward_extra_info,
        }


NaiveRewardManager = RuleRewardEvaluator
register("naive")(RuleRewardEvaluator)
