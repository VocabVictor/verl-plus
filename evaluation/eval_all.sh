  bash /data/wzh/wzh/rl/verl-plus/evaluation/eval.sh \
    --config /data/wzh//root/wzh/rl/verl-plus/evaluation/configs/qwen_local.json \
    --mode all \
    --data-root /data/wzh/wzh/datasets/vlmeval_cache \
    --work-dir /data/wzh/wzh/VLMEvalKit/outputs/qwen_local_$(date +%Y%m%d_%H%M%S) \
    --judge gpt-4o-mini \
    --dataset-workers 4 \
    --verbose