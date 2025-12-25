# Minimal Transformer

从零开始实现 Transformer，逐步构建 **"Attention Is All You Need"** 论文中的所有核心组件。

本项目专为教学设计，每个模块都包含详细的中文注释和可运行的示例。

## 📖 项目结构

```
minimal-transformer/
│
├── attention/                          # 🎯 注意力机制
│   ├── scaled_dot_product_attention.py # 缩放点积注意力（基础）
│   ├── self_attention.py               # 自注意力
│   ├── cross_attention.py              # 交叉注意力
│   ├── masked_self_attention.py        # 带掩码的自注意力
│   ├── multi_head_attention_core.py    # 多头注意力核心实现
│   ├── multi_head_self_attention.py    # 多头自注意力
│   ├── multi_head_cross_attention.py   # 多头交叉注意力
│   └── multi_head_masked_self_attention.py # 多头带掩码自注意力
│
├── layers/                             # 🧱 网络层组件
│   ├── feed_forward.py                 # 位置前馈网络 (FFN)
│   ├── residual_layer_norm.py          # 残差连接 & 层归一化
│   └── positional_encoding.py          # 位置编码
│
├── models/                             # 🏗️ 完整模型
│   ├── encoder.py                      # Transformer Encoder
│   ├── decoder.py                      # Transformer Decoder
│   └── transformer.py                  # 完整 Transformer
│
└── utils.py                            # 🔧 工具函数
```

## 🚀 快速开始

### 环境要求

```bash
pip install torch matplotlib numpy
```

### 运行示例

每个模块都可以独立运行，包含详细的演示：

```bash
# 1. 理解缩放点积注意力
python attention/scaled_dot_product_attention.py

# 2. 理解三种注意力机制
python attention/self_attention.py
python attention/cross_attention.py
python attention/masked_self_attention.py

# 3. 理解多头注意力
python attention/multi_head_self_attention.py

# 4. 理解其他组件
python layers/feed_forward.py
python layers/positional_encoding.py
python layers/residual_layer_norm.py

# 5. 运行完整 Transformer
python models/transformer.py
```

## 🎓 学习路径

建议按以下顺序学习：

### 第一阶段：注意力机制基础

```
scaled_dot_product_attention.py
         ↓
    ┌────┴────┐
    ↓         ↓
self_attention  cross_attention
    ↓
masked_self_attention
```

**核心公式**：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### 第二阶段：多头注意力

```
multi_head_attention_core.py  ← 公共实现
         ↓
    ┌────┼────┐
    ↓    ↓    ↓
  自注意力  交叉注意力  带掩码自注意力
```

**核心思想**：多个注意力头从不同角度理解序列关系

### 第三阶段：辅助组件

| 组件 | 作用 |
|------|------|
| 位置编码 | 注入位置信息（注意力本身是置换不变的） |
| 前馈网络 | 非线性变换，增强表达能力 |
| 残差连接 | 缓解梯度消失，帮助训练深层网络 |
| 层归一化 | 稳定训练，加速收敛 |

### 第四阶段：完整模型

```
┌─────────────────────────────────────────────────────────────┐
│                       Transformer                           │
│                                                             │
│  源序列 ──→ [Embedding + PE] ──→ Encoder ──┐               │
│                                            ↓               │
│  目标序列 ──→ [Embedding + PE] ──→ Decoder ──→ Output      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📚 核心概念

### 三种注意力机制

| 类型 | Q 来源 | K, V 来源 | 掩码 | 用途 |
|------|--------|-----------|------|------|
| 自注意力 | 输入序列 | 输入序列 | 可选 | Encoder |
| 交叉注意力 | Decoder | Encoder | 可选 | Decoder 关注 Encoder |
| 带掩码自注意力 | 输入序列 | 输入序列 | 因果掩码 | Decoder 自回归生成 |

### Transformer 结构 (原论文)

```
Encoder Layer:                    Decoder Layer:
┌──────────────────┐              ┌──────────────────┐
│ Self-Attention   │              │ Masked           │
│ + Add & Norm     │              │ Self-Attention   │
├──────────────────┤              │ + Add & Norm     │
│ Feed Forward     │              ├──────────────────┤
│ + Add & Norm     │              │ Cross-Attention  │
└──────────────────┘              │ + Add & Norm     │
                                  ├──────────────────┤
                                  │ Feed Forward     │
                                  │ + Add & Norm     │
                                  └──────────────────┘
```

## 🔬 代码示例

### 使用多头自注意力

```python
from attention import MultiHeadSelfAttention

# 创建模块
mhsa = MultiHeadSelfAttention(embed_size=512, num_heads=8)

# 输入: (batch_size, seq_len, embed_size)
x = torch.randn(2, 10, 512)

# 前向传播
output, attention_weights = mhsa(x)
# output: (2, 10, 512)
# attention_weights: (2, 8, 10, 10)
```

### 使用完整 Transformer

```python
from models import Transformer

# 创建模型
model = Transformer(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048
)

# 输入
src = torch.randint(0, 10000, (2, 20))  # 源序列
tgt = torch.randint(0, 10000, (2, 15))  # 目标序列

# 前向传播
logits = model(src, tgt)
# logits: (2, 15, 10000)
```

## 📖 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 原始论文
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 可视化讲解
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) - 代码注解

## 📄 License

MIT License
