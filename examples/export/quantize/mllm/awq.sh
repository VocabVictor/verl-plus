CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
verl export \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'modelscope/coco_2014_caption:validation#500' \
              'swift/VideoChatGPT:Generic#500' \
    --quant_n_samples 256 \
    --quant_batch_size -1 \
    --max_length 2048 \
    --quant_method awq \
    --quant_bits 4 \
    --output_dir Qwen2.5-VL-3B-Instruct-AWQ
