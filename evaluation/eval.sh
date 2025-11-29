#!/usr/bin/env bash
set -Eeuo pipefail

MICROMAMBA_BIN="/data/wzh/wzh/micromamba/bin/micromamba"
export MAMBA_ROOT_PREFIX="/data/wzh/wzh/micromamba"

eval "$($MICROMAMBA_BIN shell hook -s bash)"
micromamba activate vlmeval

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    export OPENAI_API_KEY
fi

if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
    export OPENAI_BASE_URL
fi

# Ensure vLLM workers start with spawn to avoid CUDA fork issues
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}

TEMP_CONFIG=""
cleanup() {
    if [[ -n "$TEMP_CONFIG" && -f "$TEMP_CONFIG" ]]; then
        rm -f "$TEMP_CONFIG"
    fi
}
trap cleanup EXIT

MODEL_NAME=""
MODEL_CLASS=""
MODEL_PATH=""
MODEL_MAX_NEW_TOKENS=""
MODEL_SYSTEM_PROMPT=""
MODEL_SYSTEM_PROMPT_FILE=""
DATA_NAME=""
DATA_CLASS=""
DATASET_ID=""
CONFIG_FILE=""
DATA_CACHE_ROOT=""
PASSTHROUGH_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash ./eval.sh [wrapper-options] --mode all --work-dir ... [other run.py args]

Wrapper options (optional, ignored when --config is provided):
  --model-name NAME            Optional key used in config (defaults to sanitized basename of model path).
  --model-class CLASS          Class under vlmeval.vlm or vlmeval.api (e.g., Qwen2VLChat).
  --model-path PATH            Local/remote model path passed to the class.
  --model-max-new-tokens NUM   Optional max_new_tokens override.
  --system-prompt TEXT         Inline system prompt string (quote it).
  --system-prompt-file FILE    Read system prompt content from FILE.
  --dataset-name NAME          Dataset key inside the config.
  --dataset-class CLASS        Dataset builder class (e.g., MathVista).
  --dataset-id ID              Value for the dataset field (defaults to dataset-name).
  --data-root PATH             Override LMUData cache directory for downloaded datasets.
  --help-wrapper               Show this message.

Example:
  bash ./eval.sh \
    --model-name QWEN25VL3B_LOCAL \
    --model-class Qwen2VLChat \
    --model-path /root/wzh/rl/verl-plus/outputs/qwen25vl3b_car/hf_merged \
    --system-prompt-file /root/wzh/rl/verl-plus/prompt.txt \
    --dataset-name MathVista_MINI \
    --dataset-class MathVista \
    --mode all \
    --work-dir /root/wzh/VLMEvalKit/outputs/qwen25vl3b_local \
    --judge gpt-4o-mini
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-class)
            [[ $# -lt 2 ]] && { echo "Error: --model-class needs a value." >&2; exit 1; }
            MODEL_CLASS="$2"
            shift 2
            ;;
        --model-path)
            [[ $# -lt 2 ]] && { echo "Error: --model-path needs a value." >&2; exit 1; }
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-max-new-tokens)
            [[ $# -lt 2 ]] && { echo "Error: --model-max-new-tokens needs a value." >&2; exit 1; }
            MODEL_MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --system-prompt)
            [[ $# -lt 2 ]] && { echo "Error: --system-prompt needs a value." >&2; exit 1; }
            MODEL_SYSTEM_PROMPT="$2"
            shift 2
            ;;
        --system-prompt-file)
            [[ $# -lt 2 ]] && { echo "Error: --system-prompt-file needs a value." >&2; exit 1; }
            MODEL_SYSTEM_PROMPT_FILE="$2"
            shift 2
            ;;
        --dataset-name)
            [[ $# -lt 2 ]] && { echo "Error: --dataset-name needs a value." >&2; exit 1; }
            DATA_NAME="$2"
            shift 2
            ;;
        --dataset-class)
            [[ $# -lt 2 ]] && { echo "Error: --dataset-class needs a value." >&2; exit 1; }
            DATA_CLASS="$2"
            shift 2
            ;;
        --dataset-id)
            [[ $# -lt 2 ]] && { echo "Error: --dataset-id needs a value." >&2; exit 1; }
            DATASET_ID="$2"
            shift 2
            ;;
        --data-root)
            [[ $# -lt 2 ]] && { echo "Error: --data-root needs a value." >&2; exit 1; }
            DATA_CACHE_ROOT="$2"
            shift 2
            ;;
        --help-wrapper)
            usage
            exit 0
            ;;
        --config)
            [[ $# -lt 2 ]] && { echo "Error: --config needs a value." >&2; exit 1; }
            CONFIG_FILE="$2"
            PASSTHROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
        --)
            shift
            while [[ $# -gt 0 ]]; do
                PASSTHROUGH_ARGS+=("$1")
                shift
            done
            break
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$CONFIG_FILE" && ( -n "$MODEL_NAME" || -n "$DATA_NAME" || -n "$MODEL_CLASS" || -n "$DATA_CLASS" ) ]]; then
    echo "Wrapper notice: --config supplied, ignoring auto-config options." >&2
fi

if [[ -n "$DATA_CACHE_ROOT" ]]; then
    mkdir -p "$DATA_CACHE_ROOT"
    export LMUData="$DATA_CACHE_ROOT"
fi

if [[ -z "$CONFIG_FILE" ]]; then
    if [[ -z "$MODEL_NAME" && -z "$DATA_NAME" ]]; then
        echo "Error: provide --config or set the wrapper model/dataset options." >&2
        exit 1
    fi

    missing=()
    [[ -z "$MODEL_CLASS" ]] && missing+=("--model-class")
    [[ -z "$MODEL_PATH" ]] && missing+=("--model-path")
    [[ -z "$DATA_NAME" ]] && missing+=("--dataset-name")
    [[ -z "$DATA_CLASS" ]] && missing+=("--dataset-class")

    if (( ${#missing[@]} > 0 )); then
        printf 'Error: missing required options for auto config: %s\n' "${missing[*]}" >&2
        exit 1
    fi

    if [[ -z "$DATASET_ID" ]]; then
        DATASET_ID="$DATA_NAME"
    fi

    if [[ -z "$MODEL_NAME" ]]; then
        trimmed="${MODEL_PATH%/}"
        trimmed="${trimmed##*/}"
        if [[ -z "$trimmed" ]]; then
            trimmed="model"
        fi
        MODEL_NAME="${trimmed//[^A-Za-z0-9._-]/_}"
    fi

    if [[ -n "$MODEL_SYSTEM_PROMPT_FILE" ]]; then
        if [[ ! -f "$MODEL_SYSTEM_PROMPT_FILE" ]]; then
            echo "Error: system prompt file '$MODEL_SYSTEM_PROMPT_FILE' not found." >&2
            exit 1
        fi
        MODEL_SYSTEM_PROMPT="$(cat "$MODEL_SYSTEM_PROMPT_FILE")"
    fi

    TEMP_CONFIG="$(mktemp "${TMPDIR:-/tmp}/eval_config.XXXXXX.json")"
    export EVAL_MODEL_NAME="$MODEL_NAME"
    export EVAL_MODEL_CLASS="$MODEL_CLASS"
    export EVAL_MODEL_PATH="$MODEL_PATH"
    export EVAL_MODEL_MAX_NEW_TOKENS="$MODEL_MAX_NEW_TOKENS"
    export EVAL_MODEL_SYSTEM_PROMPT="$MODEL_SYSTEM_PROMPT"
    export EVAL_DATA_NAME="$DATA_NAME"
    export EVAL_DATA_CLASS="$DATA_CLASS"
    export EVAL_DATASET_ID="$DATASET_ID"

    python - <<'PY' "$TEMP_CONFIG"
import json
import os
import sys

config_path = sys.argv[1]

model_entry = {
    "class": os.environ["EVAL_MODEL_CLASS"],
    "model_path": os.environ["EVAL_MODEL_PATH"],
}

max_new = os.environ.get("EVAL_MODEL_MAX_NEW_TOKENS")
if max_new:
    try:
        model_entry["max_new_tokens"] = int(max_new)
    except ValueError:
        model_entry["max_new_tokens"] = max_new

system_prompt = os.environ.get("EVAL_MODEL_SYSTEM_PROMPT")
if system_prompt:
    model_entry["system_prompt"] = system_prompt

dataset_entry = {
    "class": os.environ["EVAL_DATA_CLASS"],
    "dataset": os.environ["EVAL_DATASET_ID"],
}

config = {
    "model": {os.environ["EVAL_MODEL_NAME"]: model_entry},
    "data": {os.environ["EVAL_DATA_NAME"]: dataset_entry},
}

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
PY

    echo "Wrapper: generated temporary config at ${TEMP_CONFIG} for ${MODEL_NAME}/${DATA_NAME}" >&2
    PASSTHROUGH_ARGS+=("--config" "$TEMP_CONFIG")
fi

python /root/wzh/VLMEvalKit/run.py "${PASSTHROUGH_ARGS[@]}"
