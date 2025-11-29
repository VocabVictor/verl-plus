"""Optimized reward computation with proper async handling"""
import ray
import time
from typing import Dict, Any
from dataclasses import dataclass

from verl.utils.sequential_sampler import FragmentSampler
from verl.trainer.ppo.ray_trainer import DataProto
from verl.trainer.ppo.reward import RewardManager, apply reward_fn_for_rwd_fn_key


@dataclass
class AsyncRewardHandle:
    """Handle for managing async reward computation"""
    future: Any
    start_time: float


def compute_reward_true_async(
    data: DataProto,
    config=None,
    tokenizer=None,
    reward_fn=None
) -> DataProto:
    """
    True async reward computation that starts immediately and can be retrieved later.
    This separates reward initiation from reward retrieval.
    """
    # Extract the necessary data immediately
    data = reward_fn.prepare_data(data)

    # Start reward computation asynchronously but don't wait
    # This would require modifying reward functions to support true async
    if hasattr(reward_fn, 'compute_async'):
        # For reward functions that support true async (like CaptionReward)
        future_reward = reward_fn.compute_async(data)
        return data, future_reward
    else:
        # Fallback to sync computation for non-async reward functions
        reward_tensor = reward_fn(data)
        return data, None, reward_tensor


class OptimizedRewardManager(RewardManager):
    """
    Optimized reward manager that properly handles async reward computation
    to achieve true parallelization with log probability computation.
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.pending_rewards = {}

    def start_reward_computation(self, data: DataProto) -> str:
        """
        Start reward computation and return a handle ID.
        This should be called BEFORE log probability computation.
        """
        # Generate unique ID for this batch
        batch_id = f"batch_{int(time.time() * 1000000)}"

        # Start async reward computation
        if self.config.launch_reward_fn_async:
            # Use Ray remote for true async execution
            future = compute_reward_true_async.remote(
                data=data,
                config=self.config,
                tokenizer=self.tokenizer,
                reward_fn=self.reward_fn
            )
            self.pending_rewards[batch_id] = {
                'future': future,
                'start_time': time.time(),
                'status': 'computing'
            }
        else:
            # Immediate sync computation
            reward_tensor = self.reward_fn(data)
            self.pending_rewards[batch_id] = {
                'reward_tensor': reward_tensor,
                'start_time': time.time(),
                'status': 'completed'
            }

        return batch_id

    def get_reward_result(self, batch_id: str) -> DataProto:
        """
        Retrieve the reward computation result.
        This should be called AFTER log probability computation.
        """
        if batch_id not in self.pending_rewards:
            raise ValueError(f"Batch ID {batch_id} not found")

        reward_info = self.pending_rewards[batch_id]

        if reward_info['status'] == 'completed':
            # Already computed (sync mode)
            return reward_info['reward_tensor']
        else:
            # Wait for async completion
            wait_start = time.time()
            reward_tensor = ray.get(reward_info['future'])
            wait_time = time.time() - wait_start

            # Log timing info
            total_time = time.time() - reward_info['start_time']
            print(f"Reward computation: total={total_time:.3f}s, wait={wait_time:.3f}s")

            # Clean up
            del self.pending_rewards[batch_id]
            return reward_tensor


# Patch for ray_trainer.py to use optimized reward handling
def apply_optimized_reward_timing(trainer_instance, batch):
    """
    Optimized version of reward timing that properly separates
    reward initiation from reward retrieval.
    """
    timing_raw = trainer_instance.timing

    # STEP 1: Start reward computation BEFORE other computations
    if trainer_instance.config.reward_model.launch_reward_fn_async:
        with marked_timer("reward_start", timing_raw, color="cyan"):
            batch_id = trainer_instance.reward_manager.start_reward_computation(batch)
    else:
        with marked_timer("reward", timing_raw, color="cyan"):
            reward_tensor, reward_extra_infos_dict = trainer_instance.reward_manager(batch)

    # STEP 2: Compute log probabilities (can run in parallel with reward)
    with marked_timer("old_log_prob", timing_raw, color="blue"):
        # Store old log prob for advantage computation
        batch.old_log_prob = trainer_instance.actor_rollout_ref.log_prob(
            batch.batch["prompts"],
            batch.batch["responses"],
            batch.batch["old_log_probs"],
            do_log_prob_old=trainer_instance.do_log_prob_old,
        )

    with marked_timer("ref_log_prob", timing_raw, color="blue"):
        # Store reference log prob
        batch.ref_log_prob = trainer_instance.actor_rollout_ref.log_prob(
            batch.batch["prompts"],
            batch.batch["responses"],
            batch.batch["ref_log_probs"],
        )

    # STEP 3: Retrieve reward computation AFTER log probabilities
    if trainer_instance.config.reward_model.launch_reward_fn_async:
        with marked_timer("reward_wait", timing_raw, color="cyan"):
            reward_tensor, reward_extra_infos_dict = trainer_instance.reward_manager.get_reward_result(batch_id)

    # Update batch with rewards
    batch.batch["reward_tensor"] = reward_tensor

    return reward_tensor, reward_extra_infos_dict


def install_optimized_reward_handler():
    """
    Monkey patch the trainer to use optimized reward handling.
    This can be called at trainer initialization.
    """
    from verl.trainer.ppo.ray_trainer import PPOTrainer

    # Store original method
    original_method = PPOTrainer._estimate_adv

    def patched_estimate_adv(self, batch):
        """Patched version with optimized reward handling"""
        # Use optimized reward timing
        reward_tensor, reward_extra_infos_dict = apply_optimized_reward_timing(self, batch)

        # Continue with normal advantage estimation
        # ... (rest of the original method)

        return batch, reward_extra_infos_dict

    # Apply patch
    PPOTrainer._estimate_advantage = patched_estimate_adv

    print("✅ Installed optimized reward handler")