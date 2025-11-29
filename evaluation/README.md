# VERL模型VLMEvalKit评测工具

这个目录包含了用于评测VERL训练模型的VLMEvalKit统一评测脚本。

## 目录结构

```
evaluation/
├── eval.sh            # 统一评测脚本
├── results/           # 评测结果存放目录
└── README.md          # 使用说明
```

## 可用模型别名

- **qwen25vl3b_car**: Qwen2.5-VL-3B 汽车数据集训练版本
- **qwen25vl3b_1k**: Qwen2.5-VL-3B 1k数据集版本
- **qwen25vl3b_2k**: Qwen2.5-VL-3B 2k数据集版本
- **qwen25vl7b_car**: Qwen2.5-VL-7B 汽车数据集训练版本
- **mm_eruka_qwen25vl_3b_cap**: MM-EUREKA Qwen2.5-VL-3B 字幕版本
- **qwen25vl3b_base**: Qwen2.5-VL-3B 基础模型（对照组）

*注：模型路径在 `eval.sh` 脚本中维护，如需添加新模型请直接编辑脚本中的 `MODEL_PATHS` 数组。*

## 快速开始

```bash
# 进入评测目录
cd /data/wzh/wzh/rl/verl-plus/evaluation

# 1. 基础用法：指定模型和数据集
bash eval.sh --models qwen25vl3b_car --datasets MMBench_DEV_EN,POPE

# 2. 多模型对比
bash eval.sh --models qwen25vl3b_car,qwen25vl3b_base --datasets MathVista_MINI,DynaMath

# 3. 查看帮助
bash eval.sh --help
```

## 常用评测数据集

### 核心数据集
- **MMBench_DEV_EN**: 多模态基准测试
- **MME**: 多模态评估基准
- **POPE**: 物体识别评估

### 扩展数据集
- **LLaVA_Bench**: 对话理解
- **AI2D_TEST**: 图表理解
- **MathVista_MINI**: 数学推理
- **MMMU_DEV_VAL**: 学科理解
- **ScienceQA_VAL**: 科学推理
- **OCRBench**: 文字识别

## 结果查看

评测结果将保存在 `results/eval_YYYYMMDD_HHMMSS/` 目录下。

```bash
# 查看最新结果
ls -la results/
```

## 故障排除

如果遇到 `VLMEvalKit` 目录不存在的错误，请确保该工具包已安装在正确位置，或在 `eval.sh` 中修改 `VLMEVAL_ROOT` 变量。