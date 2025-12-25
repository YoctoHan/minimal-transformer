"""
残差连接与层归一化模块

包含:
1. 残差连接 (Residual Connection)
2. 层归一化 (Layer Normalization)
3. Add & Norm (残差连接 + 层归一化)

这些是 Transformer 训练稳定的关键组件。
"""

import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    """
    残差连接 (Residual Connection / Skip Connection)
    
    公式: output = x + sublayer(x)
    
    作用:
        - 缓解梯度消失问题
        - 允许信息直接跨层传递
        - 使深层网络更容易训练
    
    来源: "Deep Residual Learning for Image Recognition" (ResNet)
    """
    
    def __init__(self, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer_output):
        """
        参数:
            x: 原始输入 (残差)
            sublayer_output: 子层的输出 (如注意力层或 FFN 的输出)
            
        返回:
            x + dropout(sublayer_output)
        """
        return x + self.dropout(sublayer_output)


class LayerNorm(nn.Module):
    """
    层归一化 (Layer Normalization)
    
    公式: y = (x - μ) / √(σ² + ε) × γ + β
    
    其中:
        - μ: 沿最后一个维度的均值
        - σ²: 沿最后一个维度的方差
        - γ (gamma): 可学习的缩放参数
        - β (beta): 可学习的偏移参数
        - ε: 防止除零的小常数
    
    与 Batch Normalization 的区别:
        - BN: 沿 batch 维度归一化 (每个特征独立)
        - LN: 沿 feature 维度归一化 (每个样本独立)
    
    为什么用 LN 而不是 BN:
        - 序列长度可变，BN 难以处理
        - LN 不依赖 batch size
        - LN 对单个样本也能工作 (推理时)
    """
    
    def __init__(self, d_model, eps=1e-6):
        """
        参数:
            d_model: 特征维度
            eps: 防止除零的小常数
        """
        super(LayerNorm, self).__init__()
        
        self.gamma = nn.Parameter(torch.ones(d_model))   # 缩放参数
        self.beta = nn.Parameter(torch.zeros(d_model))   # 偏移参数
        self.eps = eps
    
    def forward(self, x):
        """
        参数:
            x: 输入张量 (..., d_model)
            
        返回:
            归一化后的张量，形状与输入相同
        """
        # 沿最后一个维度计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # 归一化
        x_norm = (x - mean) / (std + self.eps)
        
        # 缩放和偏移
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    """
    RMS Normalization (Root Mean Square Layer Normalization)
    
    公式: y = x / √(mean(x²) + ε) × γ
    
    特点:
        - 比 LayerNorm 更简单，没有中心化 (减均值) 步骤
        - 计算更高效
        - 在 LLaMA, T5 等模型中使用
    
    来自论文: "Root Mean Square Layer Normalization"
    """
    
    def __init__(self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x):
        # 计算 RMS (均方根)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # 归一化并缩放
        return x / rms * self.gamma


class AddNorm(nn.Module):
    """
    Add & Norm: 残差连接 + 层归一化
    
    Post-LN (原始 Transformer):
        output = LayerNorm(x + Sublayer(x))
        
    Pre-LN (更稳定，现代 Transformer 常用):
        output = x + Sublayer(LayerNorm(x))
    
    本实现默认使用 Post-LN (原始论文风格)
    """
    
    def __init__(self, d_model, dropout=0.1, pre_norm=False):
        """
        参数:
            d_model: 特征维度
            dropout: Dropout 概率
            pre_norm: 是否使用 Pre-LN (默认 False，使用 Post-LN)
        """
        super(AddNorm, self).__init__()
        
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm
    
    def forward(self, x, sublayer_fn):
        """
        参数:
            x: 输入张量
            sublayer_fn: 子层函数 (如 attention 或 ffn)
            
        返回:
            Add & Norm 的输出
        """
        if self.pre_norm:
            # Pre-LN: x + Sublayer(LayerNorm(x))
            return x + self.dropout(sublayer_fn(self.layer_norm(x)))
        else:
            # Post-LN: LayerNorm(x + Sublayer(x))
            return self.layer_norm(x + self.dropout(sublayer_fn(x)))


class AddNormSimple(nn.Module):
    """
    简化版 Add & Norm (接收子层输出而非子层函数)
    
    更直观的接口，适合教学演示
    """
    
    def __init__(self, d_model, dropout=0.1):
        super(AddNormSimple, self).__init__()
        
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer_output):
        """
        参数:
            x: 原始输入 (残差)
            sublayer_output: 子层的输出
            
        返回:
            LayerNorm(x + Dropout(sublayer_output))
        """
        return self.layer_norm(x + self.dropout(sublayer_output))


if __name__ == "__main__":
    print("=" * 70)
    print("           残差连接与层归一化 (Residual & Layer Normalization)")
    print("=" * 70)
    
    # ==================== 残差连接 ====================
    print("\n" + "=" * 70)
    print("一、残差连接 (Residual Connection)")
    print("=" * 70)
    
    print("""
公式: output = x + sublayer(x)

作用:
  1. 【缓解梯度消失】梯度可以直接通过 skip connection 传递
  2. 【恒等映射】最坏情况下，网络可以学习 sublayer(x) = 0
  3. 【更容易优化】深层网络变得可训练

示意图:
                    ┌────────────────────┐
                    │                    │
        x ─────────┬┴───→ Sublayer ─────┬┴───→ x + Sublayer(x)
                   │                    │
                   └────────────────────┘
                      (skip connection)
""")
    
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 3
    d_model = 4
    
    x = torch.randn(batch_size, seq_len, d_model)
    sublayer_out = torch.randn(batch_size, seq_len, d_model)
    
    residual = ResidualConnection(dropout=0.0)
    output = residual(x, sublayer_out)
    
    print(f"输入 x 形状: {x.shape}")
    print(f"子层输出形状: {sublayer_out.shape}")
    print(f"残差连接输出形状: {output.shape}")
    
    print("\n验证: output = x + sublayer_out")
    print(f"  手动计算与模块输出一致: {torch.allclose(output, x + sublayer_out)}")
    
    # ==================== 层归一化 ====================
    print("\n" + "=" * 70)
    print("二、层归一化 (Layer Normalization)")
    print("=" * 70)
    
    print("""
公式: y = (x - μ) / √(σ² + ε) × γ + β

归一化维度:
  - 沿最后一个维度 (d_model) 计算均值和方差
  - 每个位置独立归一化

Layer Norm vs Batch Norm:

┌──────────────────┬─────────────────────┬─────────────────────┐
│                  │    Batch Norm       │     Layer Norm      │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 归一化维度        │ 沿 batch 维度       │ 沿 feature 维度     │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 依赖 batch size  │ 是                  │ 否                  │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 适合可变长序列    │ 困难                │ 适合                │
├──────────────────┼─────────────────────┼─────────────────────┤
│ 推理时           │ 需要 running stats  │ 即时计算            │
└──────────────────┴─────────────────────┴─────────────────────┘
""")
    
    print("示例:")
    print("-" * 50)
    
    # 创建一个简单的例子
    x_demo = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                            [2.0, 4.0, 6.0, 8.0]]])  # (1, 2, 4)
    
    print(f"输入 x:\n{x_demo[0]}")
    
    layer_norm = LayerNorm(d_model=4)
    layer_norm.eval()  # 确保参数是初始值
    
    with torch.no_grad():
        # 手动重置参数以便演示
        layer_norm.gamma.fill_(1.0)
        layer_norm.beta.fill_(0.0)
        
        x_normed = layer_norm(x_demo)
    
    print(f"\n归一化后:")
    print(x_normed[0])
    
    # 验证
    print("\n验证 (第一个位置):")
    row = x_demo[0, 0]
    mean = row.mean()
    std = row.std()
    print(f"  原始值: {row.tolist()}")
    print(f"  均值 μ = {mean:.4f}")
    print(f"  标准差 σ = {std:.4f}")
    print(f"  归一化后均值 ≈ 0: {x_normed[0, 0].mean().item():.6f}")
    print(f"  归一化后标准差 ≈ 1: {x_normed[0, 0].std().item():.6f}")
    
    # ==================== RMS Norm ====================
    print("\n" + "=" * 70)
    print("三、RMS Normalization (LLaMA 风格)")
    print("=" * 70)
    
    print("""
公式: y = x / RMS(x) × γ
     其中 RMS(x) = √(mean(x²))

与 Layer Norm 的区别:
  - 没有减均值 (中心化) 步骤
  - 没有 β 参数
  - 计算更简单，速度更快
  - 在 LLaMA, T5 等现代模型中使用
""")
    
    rms_norm = RMSNorm(d_model=4)
    with torch.no_grad():
        rms_norm.gamma.fill_(1.0)
        x_rms = rms_norm(x_demo)
    
    print(f"输入 x:\n{x_demo[0]}")
    print(f"\nRMS Norm 后:\n{x_rms[0]}")
    
    # ==================== Add & Norm ====================
    print("\n" + "=" * 70)
    print("四、Add & Norm (残差 + 归一化)")
    print("=" * 70)
    
    print("""
Transformer 中的两种用法:

【Post-LN】(原始论文)
  output = LayerNorm(x + Sublayer(x))
  
  流程:  x ──┬──→ Sublayer ──┬──→ Add ──→ LayerNorm ──→ output
             │               │
             └───────────────┘

【Pre-LN】(现代实践，更稳定)
  output = x + Sublayer(LayerNorm(x))
  
  流程:  x ──┬──→ LayerNorm ──→ Sublayer ──┬──→ Add ──→ output
             │                             │
             └─────────────────────────────┘
""")
    
    print("Post-LN vs Pre-LN 对比:")
    print("-" * 50)
    print("""
┌─────────────┬───────────────────────┬───────────────────────┐
│             │       Post-LN         │        Pre-LN         │
├─────────────┼───────────────────────┼───────────────────────┤
│ 训练稳定性  │ 较差，需要 warmup     │ 更好，可省略 warmup   │
├─────────────┼───────────────────────┼───────────────────────┤
│ 最终性能    │ 略好 (充分训练时)     │ 略差                  │
├─────────────┼───────────────────────┼───────────────────────┤
│ 使用模型    │ 原始 Transformer      │ GPT-2, GPT-3 等       │
└─────────────┴───────────────────────┴───────────────────────┘
""")
    
    # 演示 Add & Norm
    print("Add & Norm 演示:")
    print("-" * 50)
    
    x = torch.randn(batch_size, seq_len, d_model)
    sublayer_out = torch.randn(batch_size, seq_len, d_model)
    
    add_norm = AddNormSimple(d_model, dropout=0.0)
    
    with torch.no_grad():
        output = add_norm(x, sublayer_out)
    
    print(f"输入 x 形状: {x.shape}")
    print(f"子层输出形状: {sublayer_out.shape}")
    print(f"Add & Norm 输出形状: {output.shape}")
    
    # 验证输出是归一化的
    print(f"\n验证输出是否归一化 (第一个样本，第一个位置):")
    print(f"  均值 ≈ 0: {output[0, 0].mean().item():.6f}")
    print(f"  标准差 ≈ 1: {output[0, 0].std().item():.6f}")
    
    # ==================== 在 Transformer 中的位置 ====================
    print("\n" + "=" * 70)
    print("五、在 Transformer 中的位置")
    print("=" * 70)
    
    print("""
Encoder Block (Post-LN 风格):

    输入 x
       │
       ▼
  ┌─────────────────────────────┐
  │   Multi-Head Attention      │
  └─────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────┐
  │   Add & Norm                │  ← x + Attention(x), 然后 LayerNorm
  └─────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────┐
  │   Feed Forward              │
  └─────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────┐
  │   Add & Norm                │  ← x + FFN(x), 然后 LayerNorm
  └─────────────────────────────┘
       │
       ▼
     输出


Decoder Block (Post-LN 风格):

    输入 x                     Encoder 输出
       │                            │
       ▼                            │
  ┌────────────────────────┐        │
  │ Masked Self-Attention  │        │
  └────────────────────────┘        │
       │                            │
       ▼                            │
  ┌────────────────────────┐        │
  │     Add & Norm         │        │
  └────────────────────────┘        │
       │                            │
       ▼                            ▼
  ┌─────────────────────────────────────┐
  │        Cross-Attention              │
  └─────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │           Add & Norm                │
  └─────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │          Feed Forward               │
  └─────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │           Add & Norm                │
  └─────────────────────────────────────┘
                    │
                    ▼
                  输出
""")
    
    # ==================== 为什么重要 ====================
    print("=" * 70)
    print("六、为什么这些组件很重要")
    print("=" * 70)
    
    print("""
【残差连接】
  ✓ 使梯度能够直接回传，避免梯度消失
  ✓ 允许堆叠更多层 (Transformer 通常 6-96 层)
  ✓ 网络可以学习增量变化，而非完全重建

【层归一化】
  ✓ 稳定每层的输入分布
  ✓ 加速训练收敛
  ✓ 减少对初始化的敏感性
  ✓ 有轻微的正则化效果

【两者结合】
  ✓ 残差连接保留原始信息
  ✓ 层归一化稳定数值范围
  ✓ 共同作用使深层 Transformer 可训练

没有这两个组件，Transformer 几乎无法训练！
""")
    
    # ==================== 参数统计 ====================
    print("=" * 70)
    print("七、参数统计")
    print("=" * 70)
    
    d = 512  # 典型的 d_model
    
    print(f"\n以 d_model = {d} 为例:")
    print(f"\nLayerNorm 参数:")
    print(f"  γ (gamma): {d} 参数")
    print(f"  β (beta): {d} 参数")
    print(f"  总计: {2 * d} = {2 * d} 参数")
    
    print(f"\nRMSNorm 参数:")
    print(f"  γ (gamma): {d} 参数")
    print(f"  总计: {d} 参数")
    
    print(f"\n残差连接: 0 参数 (只是加法)")
    
    print(f"\n对比 Transformer 中的其他层:")
    print(f"  Multi-Head Attention: ~4 × {d}² = {4 * d * d:,} 参数")
    print(f"  FFN: ~8 × {d}² = {8 * d * d:,} 参数")
    print(f"  LayerNorm: 2 × {d} = {2 * d} 参数")
    print(f"\n  → LayerNorm 参数量很小，但作用很大！")

