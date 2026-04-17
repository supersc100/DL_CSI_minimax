# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

大规模MIMO系统中基于深度学习的信道状态信息(CSI)反馈研究。项目核心思想是利用预训练大模型(DeepSeek-LLM-7B)强大的表示能力，通过迁移学习实现下行CSI到上行CSI的预测。

## 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    CSI Feedback Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│  Sionna 2.0 (PyTorch)   PyTorch                DeepSeek (Frozen)│
│  ──────────────────    ────────                ────────────────│
│  生成下行CSI  ──►  自定义Embedding  ──►   Transformer Block   │
│  信道数据         (2feat→hidden_dim)         (LoRA微调)       │
│                        + 位置编码                               │
│                                                    │             │
│                                                    ▼             │
│                                           ┌──────────────┐      │
│                                           │  Regression   │      │
│                                           │  Head         │      │
│                                           │  (seq_len×2)  │      │
│                                           └──────────────┘      │
│                                                    │             │
│                                                    ▼             │
│                                           预测上行CSI (MSE Loss) │
└─────────────────────────────────────────────────────────────────┘
```

## 研究阶段与实施计划

### Phase 1: 下行→上行CSI预测
- 加载离线DeepSeek模型 + tokenizer
- 冻结全部参数
- 自定义Embedding层：输入`[seq_len, 2]` (实部/虚部) → 投影到`hidden_dim` + 位置编码
- 替换`embed_tokens`和`lm_head`
- LoRA微调

### Phase 2+: (待定)

## 关键技术点

### CSI数据表示
- 每个时间步2个特征：复数的实部(real)和虚部(imag)
- 下行CSI: `[batch, seq_len, 2]` → 展平后送入embedding
- 上行CSI预测: 每个位置输出2个值（实部/虚部）

### Embedding层设计
```
Input CSI: [batch, seq_len, 2]
    ↓
Linear(2 → hidden_dim) + 原有RoPE位置编码
    ↓
Output: [batch, seq_len, hidden_dim]
```

### 损失函数
- MSE Loss：预测的上行CSI vs 真实上行CSI

### Sionna数据生成
- 基于Sionna 2.0 (PyTorch原生支持)
- 信道特性：多径效应、衰落、MIMO传输
- 支持TDD/FDD两种系统模式：
  - **TDD模式**: 基于信道互易性，上下行共享同一信道（添加微调扰动）
  - **FDD模式**: 上下行使用不同载波频率，分别独立生成，共享路径延迟以保持多径相关性

### CSI数据维度
- `cir_to_ofdm_channel`输出: `[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]`
  - `num_rx/num_tx`: 空间流数量（本项目均为1）
  - `num_rx_ant/num_tx_ant`: 天线元素数量（本项目分别为16和64）
  - `num_ofdm_symbols`: OFDM符号数（本项目为14）
- 经squeeze和OFDM符号平均后: `[batch, num_rx_ant, num_tx_ant, num_subcarriers]`
- 经`_freq_to_csi_features`转换为: `[batch, num_subcarriers, num_rx_ant * num_tx_ant * 2]`
- 最终输出: `[batch, seq_len, 2]`（通过`_reshape_to_seq`调整）

## 实际文件结构

```
DL_CSI_minimax/
├── config/
│   └── csi_config.yaml           # 超参数配置（模型、LoRA、训练）
├── models/
│   ├── __init__.py
│   ├── deepseek_csi_model.py      # 核心模型：embedding + frozen deepseek + regression head
│   └── lora_utils.py              # LoRA配置工具
├── data/
│   ├── __init__.py
│   ├── sionna_csi_generator.py    # Sionna信道数据生成 (PyTorch/Sionna 2.0)
│   ├── data_converter.py          # TF → PyTorch 格式转换
│   └── csi_dataset.py             # PyTorch Dataset
├── training/
│   ├── __init__.py
│   └── trainer.py                 # 训练器
├── scripts/
│   ├── generate_data.py           # 数据生成入口脚本
│   ├── download_model.py          # DeepSeek模型下载脚本
│   └── train.py                   # 训练入口脚本
├── tests/
│   └── test_deepseek_csi.py       # 模型测试
├── requirements.txt
└── CLAUDE.md
```

## 环境依赖

- torch>=2.0
- transformers>=4.36.0 (DeepSeek)
- peft>=0.7.0 (LoRA)
- sionna>=2.0 (CSI数据生成，PyTorch原生)
- pyyaml, numpy, h5py

## 常用命令

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载DeepSeek模型（离线部署）
python scripts/download_model.py \
    --model_name deepseek-ai/deepseek-llm-7b-base \
    --output_dir ./models/deepseek-7b

# 3. 生成CSI数据 (TDD模式，默认)
python scripts/generate_data.py --num_samples 10000 --output_dir ./data

# 3b. 生成CSI数据 (FDD模式)
python scripts/generate_data.py --num_samples 10000 --system_type FDD \
    --dl_frequency 3.5e9 --ul_frequency 2.1e9 --output_dir ./data

# 4. 训练模型
python scripts/train.py --config config/csi_config.yaml

# 5. 单步训练测试
python scripts/train.py --config config/csi_config.yaml --max_steps 10

# 6. 模型验证
python -m pytest tests/test_deepseek_csi.py -v
```

## 模型核心类

### `DeepSeekCSIModel`
主模型类，位于 [models/deepseek_csi_model.py](models/deepseek_csi_model.py)
- `__init__()`: 加载DeepSeek、冻结参数、替换embedding/lm_head
- `forward(csi_down)`: 前向传播，输入`[batch, seq_len, 2]`，输出`[batch, seq_len, 2]`

### `CSITrainer`
训练器类，位于 [training/trainer.py](training/trainer.py)
- `fit()`: 完整训练循环
- `train_epoch()`: 单轮训练
- `evaluate()`: 评估模型

### `SionnaCSIGenerator`
数据生成器，位于 [data/sionna_csi_generator.py](data/sionna_csi_generator.py)
- `generate_channel_batch()`: 生成下行/上行CSI对（支持TDD/FDD）
- `generate_dataset()`: 保存为HDF5格式
- 支持TDD（基于信道互易性）和FDD（独立频率信道）两种模式
