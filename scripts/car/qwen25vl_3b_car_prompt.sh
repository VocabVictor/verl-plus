#!/usr/bin/env bash
set -Eeuo pipefail

MICROMAMBA_ROOT="/data/wzh/wzh/micromamba"
MICROMAMBA_BIN="$MICROMAMBA_ROOT/bin/micromamba"

if [[ ! -x "$MICROMAMBA_BIN" ]]; then
  echo "micromamba binary not found at $MICROMAMBA_BIN" >&2
  exit 1
fi

export MAMBA_ROOT_PREFIX="$MICROMAMBA_ROOT"
# shellcheck disable=SC1090
eval "$("$MICROMAMBA_BIN" shell hook -s bash)"
micromamba activate verl

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM="true"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
unset PYTORCH_CUDA_ALLOC_CONF
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"

WORKDIR="/data/wzh/wzh/rl/verl-plus"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
DATA_ROOT="/data/wzh/wzh/datasets/MM-EUREKA/verl"
TRAIN_PARQUET="$DATA_ROOT/train.parquet"
VAL_PARQUET="$DATA_ROOT/val.parquet"
if [[ ! -f "$TRAIN_PARQUET" ]]; then
  echo "Missing train parquet: $TRAIN_PARQUET" >&2
  exit 1
fi
if [[ ! -f "$VAL_PARQUET" ]]; then
  echo "Missing val parquet: $VAL_PARQUET" >&2
  exit 1
fi

# Force using local model path only; fail fast if missing (no HF fallback)
MODEL_PATH="${MODEL_PATH:-/data/wzh/wzh/models/qwen/Qwen2.5-VL-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/outputs/qwen25vl3b_cap_prompt}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Local model directory not found: $MODEL_PATH" >&2
  exit 1
fi
echo "Using local model directory: $MODEL_PATH" >&2

mkdir -p "$OUTPUT_DIR"
cd "$WORKDIR"

if [[ -f "$WORKDIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$WORKDIR/.env"
  set +a
fi

TRAIN_FILES="['$TRAIN_PARQUET']"
VAL_FILES="['$VAL_PARQUET']"

export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY before running}"
export WANDB_MODE="online"

OVERRIDES=(
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "data.train_files=${TRAIN_FILES}"
  "data.val_files=${VAL_FILES}"
  "data.train_batch_size=256"
  "data.max_prompt_length=8192"
  "data.max_response_length=512"
  "data.filter_overlong_prompts=True"
  "data.truncation=right"
  "data.return_multi_modal_inputs=True"
  "data.system_prompt_path=${WORKDIR}/prompt_caption.txt"
  "data.filter_overlong_prompts_workers=8"
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
  "data.reward_fn_key=reward_model"
  "reward_model.reward_manager=rule"
  "reward_model.reward_funcs=[\"format\",\"accuracy\"]"
  "reward_model.reward_weights=[1.0,1.0]"
  # Require caption + think + answer in the completion format (quote to satisfy Hydra grammar)
  "reward_model.reward_kwargs.format.pattern='^<caption>.*?</caption>\\s*<think>.*?</think>\\s*<answer>.*?</answer>(?![\\s\\S])'"
  "reward_model.reward_kwargs.format.flags=DOTALL|MULTILINE"
  "algorithm.filter_groups.enable=False"
  "algorithm.filter_groups.metric=accuracy"
  "algorithm.filter_groups.mode=std"
  "algorithm.filter_groups.mean_min=0.1"
  "algorithm.filter_groups.mean_max=null"
  "algorithm.filter_groups.warmup_steps=10"
  "algorithm.filter_groups.max_num_gen_batches=10"
  "trainer.project_name=cvpr"
  "trainer.experiment_name=mm_eruka_qwen25vl_3b_cap_prompt"
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
