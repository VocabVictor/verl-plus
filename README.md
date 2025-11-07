# verl-plus (custom fork)

This repository is a lightweight fork of [verl](https://github.com/volcengine/verl) that I use for experimenting with multimodal RLHF on **MM-EUREKA** and **Qwen2.5-VL** models. All upstream marketing/CI docs were removed so the README only documents what I actually run.

## Layout

- `examples/` – launch scripts (e.g. `qwen25vl_3b_car.sh`, `qwen25vl_3b 1k.sh`). Adjust these to point at your dataset/model paths before running.
- `examples/reward/process_fn_gain.py` – helper functions for the gain reward worker.
- `verl/` – the core library; only the pieces I touched for gain reward + resource management differ from upstream.
- `prompt*.txt` – system prompts used in current experiments.

## Requirements

- CUDA 12.x, Python 3.10, and the `verl` conda/micromamba environment from `/data/wzh/wzh/micromamba`.
- Local checkpoints:
  - `/data/wzh/wzh/models/qwen/Qwen2.5-VL-3B-Instruct`
  - `/data/wzh/wzh/models/qwen/Qwen2.5-7B-Instruct`
- Dataset parquet files in `/data/wzh/wzh/datasets/MM-EUREKA/verl/{train,val}.parquet`.

## Running the experiments

```bash
cd /root/wzh/cvpr/car
nohup env PYTHONUNBUFFERED=1 ./qwen25vl_3b_car.sh > qwen25vl3b_nohup.log 2>&1 &
```

Variants:

- `qwen25vl_3b 1k.sh` – same training but with `max_response_length=1024`.
- `qwen25vl_3b_car_prompt.sh` – prompt tuning variant.

Watch logs via `tail -f qwen25vl3b_nohup.log`. If Ray/SGlang workers fail because GPUs are busy, clear the old processes (`nvidia-smi`, kill PIDs) before relaunching.

## Gain reward setup

The gain reward worker shares code with SGLang. For now it simply runs as a Ray worker with its own GPU pool. Make sure `sglang` is installed in the `verl` environment:

```bash
/data/wzh/wzh/micromamba/bin/micromamba run -n verl python -m pip install sglang
```

Relevant overrides live in `examples/qwen25vl_3b_car.sh` under `reward_model.*`. Adjust `reward_model.model.path` if you swap the evaluation checkpoint.

## CI & branches

All upstream GitHub Actions/Dependabot automation is disabled. The only active branch is `feature/gain-reward`; push there for personal usage. Feel free to trim more docs if they get stale.
