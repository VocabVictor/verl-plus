#!/usr/bin/env bash
set -Eeuo pipefail

# Micromamba env (you told me to use /root/wzh/micromamba/verl)
MICROMAMBA_ROOT="/root/wzh/micromamba"
MICROMAMBA_BIN="$MICROMAMBA_ROOT/bin/micromamba"
if [[ ! -x "$MICROMAMBA_BIN" ]]; then
  echo "micromamba binary not found at $MICROMAMBA_BIN" >&2
  exit 1
fi
export MAMBA_ROOT_PREFIX="$MICROMAMBA_ROOT"
# shellcheck disable=SC1090
eval "$("$MICROMAMBA_BIN" shell hook -s bash)"
micromamba activate verl

# Common envs
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="true"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
unset PYTORCH_CUDA_ALLOC_CONF

# Paths (override by exporting the envs before running this script)
WORKDIR="${WORKDIR:-/root/wzh/rl/verl-plus}"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
PROMPT_SV="${PROMPT_SV:-/root/wzh/rl/verl-plus/prompt_self_verification.txt}"

# Data (override TRAIN_PARQUET/VAL_PARQUET if needed)
DATA_ROOT="${DATA_ROOT:-/data/wzh/wzh/datasets/MM-EUREKA/verl}"
TRAIN_PARQUET="${TRAIN_PARQUET:-$DATA_ROOT/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-$DATA_ROOT/val.parquet}"
if [[ ! -f "$TRAIN_PARQUET" ]]; then
  echo "Missing train parquet: $TRAIN_PARQUET" >&2
  exit 1
fi
if [[ ! -f "$VAL_PARQUET" ]]; then
  echo "Missing val parquet: $VAL_PARQUET" >&2
  exit 1
fi

# Model used for RL training (VL)
MODEL_PATH="${MODEL_PATH:-/data/wzh/wzh/models/qwen/Qwen2.5-VL-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/outputs/qwen25vl3b_selfverify}"
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Local model directory not found: $MODEL_PATH" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
cd "$WORKDIR"

# Offline filter can optionally use a separate (often text-only) model for speed/stability.
# Default to a local Qwen2.5-7B-Instruct text model when available.
OFFLINE_FILTER_MODEL="${OFFLINE_FILTER_MODEL:-/data/wzh/wzh/models/qwen/Qwen2.5-7B-Instruct}"

FILTERED_DATA_DIR="${FILTERED_DATA_DIR:-$WORKDIR/filtered_datasets}"
mkdir -p "$FILTERED_DATA_DIR"
FILTERED_TRAIN_PARQUET="${FILTERED_TRAIN_PARQUET:-$FILTERED_DATA_DIR/train_filtered.parquet}"
# Use vLLM by default for faster offline filtering; requires Ray, which we now
# auto-init in verl.tools.offline_filter when engine==vllm.
OFFLINE_FILTER_ENGINE="${OFFLINE_FILTER_ENGINE:-vllm}"
# Use a relatively large batch size by default; can be overridden via env.
OFFLINE_FILTER_BATCH_SIZE="${OFFLINE_FILTER_BATCH_SIZE:-32}"
OFFLINE_FILTER_MAX_NEW="${OFFLINE_FILTER_MAX_NEW:-512}"
OFFLINE_FILTER_REPEAT="${OFFLINE_FILTER_REPEAT:-8}"
OFFLINE_FILTER_TEMP_MIN="${OFFLINE_FILTER_TEMP_MIN:-0.1}"
OFFLINE_FILTER_TEMP_MAX="${OFFLINE_FILTER_TEMP_MAX:-0.9}"
OFFLINE_FILTER_TP="${OFFLINE_FILTER_TP:-8}"

maybe_run_offline_filter() {
  if [[ -f "$FILTERED_TRAIN_PARQUET" ]] && [[ "$FILTERED_TRAIN_PARQUET" -nt "$TRAIN_PARQUET" ]]; then
    echo "Offline filtered dataset already up to date: $FILTERED_TRAIN_PARQUET"
    return
  fi

  echo "[offline-filter] Generating filtered dataset at $FILTERED_TRAIN_PARQUET"
  python -m verl.tools.offline_filter \
    --data-files "$TRAIN_PARQUET" \
    --model "$OFFLINE_FILTER_MODEL" \
    --tokenizer "$OFFLINE_FILTER_MODEL" \
    --output "$FILTERED_TRAIN_PARQUET" \
    --max-prompt-length 8192 \
    --batch-size "$OFFLINE_FILTER_BATCH_SIZE" \
    --max-new-tokens "$OFFLINE_FILTER_MAX_NEW" \
    --repeat "$OFFLINE_FILTER_REPEAT" \
    --temperature-range "$OFFLINE_FILTER_TEMP_MIN" "$OFFLINE_FILTER_TEMP_MAX" \
    --metric-type reward \
    --metric-name accuracy \
    --solution-key answer \
    --keep-min 0.9 \
    --invert \
    --text-only \
    --engine "$OFFLINE_FILTER_ENGINE" \
    --tensor-model-parallel-size "$OFFLINE_FILTER_TP" \
    --gpu-memory-utilization 0.6
}

maybe_run_offline_filter
TRAIN_PARQUET="$FILTERED_TRAIN_PARQUET"

# Optional WandB
export WANDB_API_KEY="aa3765e540d2c141804b236d76c1aa7a7b4ba04e"
export WANDB_MODE="online"

TRAIN_FILES="['$TRAIN_PARQUET']"
VAL_FILES="['$VAL_PARQUET']"

OVERRIDES=(
  # Algorithm
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"

  # Data and prompts
  "data.train_files=${TRAIN_FILES}"
  "data.val_files=${VAL_FILES}"
  "data.train_batch_size=256"
  "data.max_prompt_length=8192"
  "data.max_response_length=512"
  "data.filter_overlong_prompts=True"
  "data.truncation=right"
  "data.return_multi_modal_inputs=True"
  "data.system_prompt_path=${PROMPT_SV}"
  "data.filter_overlong_prompts_workers=8"

  "data.offline_filter.enable=False"

  # Actor/rollout/reference
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.trust_remote_code=True"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.ppo_mini_batch_size=64"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2"
  "actor_rollout_ref.actor.use_remove_padding=True"
  "actor_rollout_ref.actor.freeze_vision_tower=True"
  "actor_rollout_ref.actor.use_dynamic_bsz=True"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=0.001"
  "actor_rollout_ref.actor.entropy_coeff=0.0"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.load_format=safetensors"
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.n=8"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=6"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=6"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"

  # Reward: self-verify (answer correctness + verdict consistency)
  "data.reward_fn_key=reward_model"
  "reward_model.reward_manager=rule"
  "reward_model.reward_funcs=[\"self_verify\"]"
  "reward_model.reward_weights=[1.0]"
  # If your task has verifiable answers, enable base accuracy judge
  "reward_model.reward_kwargs.self_verify.base_reward=accuracy"
  "reward_model.reward_kwargs.self_verify.lambda_verdict=0.3"
  "reward_model.reward_kwargs.self_verify.gate_verdict_on_correct=True"
  "reward_model.reward_kwargs.self_verify.case_insensitive_match=True"
  "reward_model.reward_kwargs.self_verify.strip_punct=True"

  # Optional: add a light format gate to enforce 4-tag structure (<think>/<answer>/<verify>/<verdict>)
  # "reward_model.reward_funcs=[\"self_verify\",\"format\"]"
  # "reward_model.reward_weights=[1.0,0.2]"
  # "reward_model.reward_kwargs.format.pattern='^\\s*<think>[\\s\\S]*?</think>\\s*<answer>[\\s\\S]*?</answer>\\s*<verify>[\\s\\S]*?</verify>\\s*<verdict>\\s*(CORRECT|INCORRECT)\\s*</verdict>\\s*$'"
  # "reward_model.reward_kwargs.format.flags='DOTALL|MULTILINE'"

  # Disable built-in online group filtering unless you want it
  "algorithm.filter_groups.enable=False"

  # Trainer
  "trainer.project_name=cvpr"
  "trainer.experiment_name=qwen25vl_3b_selfverify_grpo"
  "trainer.logger=[\"console\",\"wandb\"]"
  "trainer.n_gpus_per_node=8"
  "trainer.nnodes=1"
  "trainer.save_freq=200"
  "trainer.test_freq=0"
  "trainer.val_before_train=False"
  "trainer.early_stop.enable=True"
  "trainer.early_stop.metric=reward/accuracy/mean"
  "trainer.early_stop.min_steps=0"
  "trainer.early_stop.window_size=10"
  "trainer.early_stop.threshold=0.0001"
  "trainer.early_stop.patience=10"
  "trainer.total_epochs=1"
  "trainer.default_local_dir=${OUTPUT_DIR}"
)

python -m verl.trainer.main_ppo "${OVERRIDES[@]}" "$@"
