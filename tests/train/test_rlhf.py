import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

kwargs = {
    'per_device_train_batch_size': 2,
    'save_steps': 5,
    'gradient_accumulation_steps': 4,
    'num_train_epochs': 1,
}


def test_llm():
    from verl.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='dpo',
            model='Qwen/Qwen2-7B-Instruct',
            dataset=['hjh0119/shareAI-Llama3-DPO-zh-en-emoji#100'],
            split_dataset_ratio=0.01,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm():
    from verl.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    os.environ['MAX_PIXLES'] = f'{1280 * 28 * 28}'
    result = rlhf_main(
        RLHFArguments(
            rlhf_type='dpo',
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['verl/RLAIF-V-Dataset#100'],
            split_dataset_ratio=0.01,
            dataset_num_proc=8,
            max_pixels=512 * 512,
            **kwargs))
    last_model_checkpoint = result['last_model_checkpoint']
    infer_main(InferArguments(adapters=last_model_checkpoint, load_data_args=True, merge_lora=True))


def test_mllm_zero3():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    os.environ['MAX_PIXLES'] = f'{1280 * 28 * 28}'
    from verl.llm import rlhf_main, RLHFArguments, infer_main, InferArguments
    rlhf_main(
        RLHFArguments(
            rlhf_type='dpo',
            model='Qwen/Qwen2-VL-7B-Instruct',
            dataset=['verl/RLAIF-V-Dataset#100'],
            split_dataset_ratio=0.01,
            dataset_num_proc=8,
            max_pixels=512 * 512,
            deepspeed='zero3',
            **kwargs))


if __name__ == '__main__':
    # test_llm()
    test_mllm()
    # test_mllm_zero3()
