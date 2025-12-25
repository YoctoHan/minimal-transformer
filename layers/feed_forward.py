"""
前馈神经网络模块 (Position-wise Feed-Forward Network)

来自论文 "Attention Is All You Need"
公式: FFN(x) = max(0, x @ W₁ + b₁) @ W₂ + b₂
     或: FFN(x) = ReLU(x @ W₁ + b₁) @ W₂ + b₂

特点:
    - 对每个位置独立应用 (Position-wise)
    - 不同位置之间不交互（与注意力机制不同）
    - 同一层内，不同位置共享参数
"""

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """
    逐位置前馈网络 (Position-wise Feed-Forward Network)
    
    结构: Linear → ReLU → Dropout → Linear → Dropout
    
    作用:
        - 在注意力层之后进一步处理特征
        - 引入非线性变换
        - 扩展再压缩 (d_model → d_ff → d_model)
    
    参数说明:
        - d_model: 输入/输出维度 (论文中为 512)
        - d_ff: 中间层维度 (论文中为 2048 = 4 × d_model)
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        参数:
            d_model: 输入和输出的维度
            d_ff: 中间隐藏层的维度 (通常是 d_model 的 4 倍)
            dropout: Dropout 概率
        """
        super(PositionwiseFeedForward, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 两层线性变换
        self.linear1 = nn.Linear(d_model, d_ff)    # 扩展
        self.linear2 = nn.Linear(d_ff, d_model)    # 压缩
        
        # 激活函数和 Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播。
        
        参数:
            x: 输入张量 (batch_size, seq_len, d_model)
            
        返回:
            输出张量 (batch_size, seq_len, d_model)
        """
        # x: (batch, seq_len, d_model)
        
        # 第一层: 线性变换 + ReLU
        x = self.linear1(x)      # (batch, seq_len, d_ff)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 第二层: 线性变换
        x = self.linear2(x)      # (batch, seq_len, d_model)
        x = self.dropout(x)
        
        return x


class PositionwiseFeedForwardGELU(nn.Module):
    """
    使用 GELU 激活函数的前馈网络 (GPT/BERT 风格)
    
    GELU 相比 ReLU:
        - 更平滑，没有"硬"截断
        - 在现代 Transformer 中更常用
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForwardGELU, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数 (LLaMA 风格)
    
    公式: SwiGLU(x) = Swish(x @ W₁) ⊙ (x @ W₃) @ W₂
    
    特点:
        - 使用门控机制
        - 在 LLaMA, PaLM 等模型中使用
        - 参数量略多，但效果更好
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(SwiGLU, self).__init__()
        
        # SwiGLU 需要三个线性层
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # 门控
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Swish(x @ W₁) ⊙ (x @ W₃)
        swish = nn.functional.silu(self.w1(x))  # Swish = SiLU
        gate = self.w3(x)
        x = swish * gate
        
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        
        return x


if __name__ == "__main__":
    print("=" * 70)
    print("       前馈神经网络 (Position-wise Feed-Forward Network)")
    print("=" * 70)
    
    print("\n来自论文: 'Attention Is All You Need'")
    print("\n公式: FFN(x) = ReLU(x @ W₁ + b₁) @ W₂ + b₂")
    
    print("""
核心特点:

1. 【逐位置独立】Position-wise
   - 对序列中每个位置独立应用
   - 位置之间不交互（与注意力不同）
   - 同一层内，所有位置共享参数
   
2. 【扩展-压缩】结构
   - 输入: d_model = 512
   - 中间: d_ff = 2048 (4倍扩展)
   - 输出: d_model = 512
   
3. 【非线性变换】
   - 注意力是线性的加权求和
   - FFN 引入非线性 (ReLU/GELU)
""")
    
    print("=" * 70)
    print("结构图")
    print("=" * 70)
    print("""
输入 x: (batch, seq_len, d_model)
          │
          ▼
    ┌──────────────┐
    │   Linear 1   │   d_model → d_ff (扩展)
    │  (512 → 2048)│
    └──────────────┘
          │
          ▼
    ┌──────────────┐
    │     ReLU     │   激活函数
    └──────────────┘
          │
          ▼
    ┌──────────────┐
    │   Dropout    │
    └──────────────┘
          │
          ▼
    ┌──────────────┐
    │   Linear 2   │   d_ff → d_model (压缩)
    │  (2048 → 512)│
    └──────────────┘
          │
          ▼
    ┌──────────────┐
    │   Dropout    │
    └──────────────┘
          │
          ▼
输出: (batch, seq_len, d_model)
""")
    
    # 设置参数
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    d_model = 8
    d_ff = 32  # 4 × d_model
    
    print("=" * 70)
    print("示例演示")
    print("=" * 70)
    print(f"\n参数:")
    print(f"  d_model = {d_model}")
    print(f"  d_ff = {d_ff} (= {d_ff // d_model} × d_model)")
    print(f"  batch_size = {batch_size}")
    print(f"  seq_len = {seq_len}")
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"\n输入形状: {x.shape}")
    
    # 创建 FFN 层
    ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.0)
    
    # 设置为评估模式（禁用 dropout）
    ffn.eval()
    
    with torch.no_grad():
        output = ffn(x)
    
    print(f"输出形状: {output.shape}")
    print("  → 与输入形状相同")
    
    # 展示参数量
    print("\n" + "=" * 70)
    print("参数量分析")
    print("=" * 70)
    
    params_linear1 = d_model * d_ff + d_ff  # 权重 + 偏置
    params_linear2 = d_ff * d_model + d_model
    total_params = params_linear1 + params_linear2
    
    print(f"\nLinear 1: {d_model} × {d_ff} + {d_ff} = {params_linear1}")
    print(f"Linear 2: {d_ff} × {d_model} + {d_model} = {params_linear2}")
    print(f"总参数量: {total_params}")
    
    # 实际模型参数
    actual_params = sum(p.numel() for p in ffn.parameters())
    print(f"\n验证 (实际参数量): {actual_params}")
    
    # 逐位置独立性演示
    print("\n" + "=" * 70)
    print("关键概念: 逐位置独立 (Position-wise)")
    print("=" * 70)
    
    print("""
【注意力 vs FFN 的区别】

注意力机制:
  - 位置之间有交互
  - 每个位置的输出依赖于所有位置的输入
  
  位置1 ←→ 位置2 ←→ 位置3 ←→ 位置4
  
FFN (逐位置):
  - 位置之间无交互
  - 每个位置的输出只依赖于该位置的输入
  
  位置1    位置2    位置3    位置4
    ↓        ↓        ↓        ↓
   FFN      FFN      FFN      FFN   (共享参数)
    ↓        ↓        ↓        ↓
  输出1    输出2    输出3    输出4
""")
    
    # 验证逐位置独立性
    print("验证: 修改一个位置，不影响其他位置")
    print("-" * 50)
    
    x_modified = x.clone()
    x_modified[0, 0, :] = 0  # 修改第一个样本的第一个位置
    
    with torch.no_grad():
        output_modified = ffn(x_modified)
    
    # 检查其他位置是否改变
    diff_pos0 = (output[0, 0] - output_modified[0, 0]).abs().sum().item()
    diff_pos1 = (output[0, 1] - output_modified[0, 1]).abs().sum().item()
    diff_pos2 = (output[0, 2] - output_modified[0, 2]).abs().sum().item()
    
    print(f"位置 0 的输出差异: {diff_pos0:.4f} (已修改，应该有差异)")
    print(f"位置 1 的输出差异: {diff_pos1:.4f} (未修改，应该为 0)")
    print(f"位置 2 的输出差异: {diff_pos2:.4f} (未修改，应该为 0)")
    print("\n  → 证明 FFN 对每个位置独立计算 ✓")
    
    # 不同激活函数对比
    print("\n" + "=" * 70)
    print("激活函数变体")
    print("=" * 70)
    print("""
┌─────────────────┬────────────────────────────────────────────────┐
│     变体        │                 说明                           │
├─────────────────┼────────────────────────────────────────────────┤
│ ReLU            │ 原始 Transformer，max(0, x)                    │
├─────────────────┼────────────────────────────────────────────────┤
│ GELU            │ GPT, BERT 使用，更平滑                         │
│                 │ ≈ x × Φ(x)，其中 Φ 是标准正态分布 CDF          │
├─────────────────┼────────────────────────────────────────────────┤
│ SwiGLU          │ LLaMA, PaLM 使用，门控机制                     │
│                 │ Swish(xW₁) ⊙ xW₃                               │
└─────────────────┴────────────────────────────────────────────────┘
""")
    
    # 创建不同变体
    ffn_gelu = PositionwiseFeedForwardGELU(d_model, d_ff, dropout=0.0)
    ffn_swiglu = SwiGLU(d_model, d_ff, dropout=0.0)
    
    ffn_gelu.eval()
    ffn_swiglu.eval()
    
    with torch.no_grad():
        out_relu = ffn(x)
        out_gelu = ffn_gelu(x)
        out_swiglu = ffn_swiglu(x)
    
    print("三种变体的输出形状都相同:")
    print(f"  ReLU FFN:   {out_relu.shape}")
    print(f"  GELU FFN:   {out_gelu.shape}")
    print(f"  SwiGLU FFN: {out_swiglu.shape}")
    
    # FFN 在 Transformer 中的位置
    print("\n" + "=" * 70)
    print("FFN 在 Transformer 中的位置")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                    Transformer Block                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                                                           │  │
│  │          ┌─────────────────────────┐                      │  │
│  │          │   Multi-Head Attention  │                      │  │
│  │          └─────────────────────────┘                      │  │
│  │                      ↓                                    │  │
│  │          ┌─────────────────────────┐                      │  │
│  │          │   Add & Layer Norm      │                      │  │
│  │          └─────────────────────────┘                      │  │
│  │                      ↓                                    │  │
│  │          ┌─────────────────────────┐                      │  │
│  │          │        FFN              │  ← 这里！            │  │
│  │          │  (Position-wise)        │                      │  │
│  │          └─────────────────────────┘                      │  │
│  │                      ↓                                    │  │
│  │          ┌─────────────────────────┐                      │  │
│  │          │   Add & Layer Norm      │                      │  │
│  │          └─────────────────────────┘                      │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

作用:
  1. 注意力层: 捕捉位置之间的关系 (建模依赖)
  2. FFN 层: 对每个位置独立变换 (特征提取)
  
两者配合，实现强大的序列建模能力。
""")
    
    # 论文中的超参数
    print("=" * 70)
    print("论文中的超参数设置")
    print("=" * 70)
    print("""
┌──────────────────┬──────────────┬──────────────┐
│      模型        │   d_model    │     d_ff     │
├──────────────────┼──────────────┼──────────────┤
│ Transformer-base │     512      │    2048      │
│ Transformer-big  │    1024      │    4096      │
├──────────────────┼──────────────┼──────────────┤
│ BERT-base        │     768      │    3072      │
│ BERT-large       │    1024      │    4096      │
├──────────────────┼──────────────┼──────────────┤
│ GPT-2 (small)    │     768      │    3072      │
│ GPT-2 (xl)       │    1600      │    6400      │
├──────────────────┼──────────────┼──────────────┤
│ LLaMA-7B         │    4096      │   11008      │
│ LLaMA-65B        │    8192      │   22016      │
└──────────────────┴──────────────┴──────────────┘

通常 d_ff = 4 × d_model (或约为 2.7 × d_model for LLaMA)
""")

