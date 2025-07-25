# lora_llm_full_vit: 23GiB
# lora: 21.6GiB
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
verl sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset 'tany0699/garbage265#20000' \
    --split_dataset_ratio 0.01 \
    --train_type custom \
    --external_plugins 'examples/train/multimodal/lora_llm_full_vit/custom_plugin.py' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --vit_lr 1e-5 \
    --aligner_lr 1e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --num_labels 265 \
    --task_type seq_cls \
    --save_only_model true
