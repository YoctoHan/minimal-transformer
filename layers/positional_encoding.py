"""
位置编码模块 (Positional Encoding)

为什么需要位置编码？
  - 注意力机制是置换不变的 (permutation invariant)
  - "我爱你" 和 "你爱我" 在纯注意力看来是一样的
  - 需要显式注入位置信息

实现:
1. 正弦位置编码 (Sinusoidal) - 原始 Transformer
2. 可学习位置嵌入 (Learned) - BERT, GPT
3. 旋转位置编码 (RoPE) - LLaMA, GPT-NeoX
"""

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np


class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码 (Sinusoidal Positional Encoding)
    
    来自论文 "Attention Is All You Need"
    
    公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    特点:
        - 不需要学习，固定计算
        - 可以推广到比训练时更长的序列
        - 相对位置可以通过线性变换表示
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        参数:
            d_model: 嵌入维度
            max_len: 支持的最大序列长度
            dropout: Dropout 概率
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 位置索引: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 维度索引的缩放因子: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 0, 2, 4, ...
        pe[:, 1::2] = torch.cos(position * div_term)  # 1, 3, 5, ...
        
        # 增加 batch 维度: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer (不是参数，但会随模型保存)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        参数:
            x: 输入张量 (batch_size, seq_len, d_model)
            
        返回:
            添加位置编码后的张量
        """
        seq_len = x.size(1)
        # 截取需要的长度，加到输入上
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEmbedding(nn.Module):
    """
    可学习的位置嵌入 (Learned Positional Embedding)
    
    用于: BERT, GPT, RoBERTa 等
    
    特点:
        - 位置编码是可学习的参数
        - 更灵活，可以学习任务特定的位置模式
        - 缺点: 不能推广到训练时未见过的位置
    """
    
    def __init__(self, d_model, max_len=512, dropout=0.1):
        """
        参数:
            d_model: 嵌入维度
            max_len: 支持的最大序列长度
            dropout: Dropout 概率
        """
        super(LearnedPositionalEmbedding, self).__init__()
        
        # 可学习的位置嵌入矩阵
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 位置索引
        self.register_buffer('positions', torch.arange(max_len))
    
    def forward(self, x):
        """
        参数:
            x: 输入张量 (batch_size, seq_len, d_model)
            
        返回:
            添加位置编码后的张量
        """
        seq_len = x.size(1)
        positions = self.positions[:seq_len]
        
        # 获取位置嵌入并加到输入上
        pos_emb = self.embedding(positions)  # (seq_len, d_model)
        x = x + pos_emb.unsqueeze(0)  # 广播到 batch
        
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding, RoPE)
    
    来自论文: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    用于: LLaMA, GPT-NeoX, PaLM 等
    
    核心思想:
        - 不是加法，而是乘法 (旋转)
        - 对 Q 和 K 应用旋转矩阵
        - 相对位置信息自然编码在 Q·K 的点积中
    
    优点:
        - 可以推广到任意长度
        - 相对位置信息更自然
        - 效果更好
    """
    
    def __init__(self, d_model, max_len=2048, base=10000):
        """
        参数:
            d_model: 嵌入维度 (必须是偶数)
            max_len: 最大序列长度
            base: 频率基数
        """
        super(RotaryPositionalEmbedding, self).__init__()
        
        assert d_model % 2 == 0, "d_model 必须是偶数"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # 计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算 cos 和 sin
        self._build_cache(max_len)
    
    def _build_cache(self, seq_len):
        """预计算 cos 和 sin 缓存"""
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, d_model/2)
        
        # 复制一份，用于完整维度
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, d_model)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x, seq_len=None):
        """
        应用旋转位置编码
        
        参数:
            x: 输入张量 (batch, seq_len, d_model) 或 (batch, heads, seq_len, head_dim)
            seq_len: 序列长度 (如果 x 的维度不是 3)
            
        返回:
            旋转后的张量
        """
        if seq_len is None:
            seq_len = x.size(-2)
        
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        return self._apply_rotary(x, cos, sin)
    
    def _apply_rotary(self, x, cos, sin):
        """应用旋转变换"""
        # 将 x 分成两半
        x1 = x[..., : self.d_model // 2]
        x2 = x[..., self.d_model // 2 :]
        
        # 旋转: [x1, x2] → [x1*cos - x2*sin, x1*sin + x2*cos]
        cos = cos[:x.size(-2)]
        sin = sin[:x.size(-2)]
        
        # 调整维度以支持广播
        if x.dim() == 4:  # (batch, heads, seq, dim)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:  # (batch, seq, dim)
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        cos1, cos2 = cos[..., :self.d_model//2], cos[..., self.d_model//2:]
        sin1, sin2 = sin[..., :self.d_model//2], sin[..., self.d_model//2:]
        
        rotated = torch.cat([
            x1 * cos1 - x2 * sin1,
            x1 * sin2 + x2 * cos2
        ], dim=-1)
        
        return rotated


def visualize_sinusoidal_encoding(d_model=128, max_len=100):
    """可视化正弦位置编码"""
    pe = SinusoidalPositionalEncoding(d_model, max_len, dropout=0.0)
    encoding = pe.pe[0].numpy()  # (max_len, d_model)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(encoding, aspect='auto', cmap='RdBu')
    plt.colorbar(label='编码值')
    plt.xlabel('维度 (d_model)')
    plt.ylabel('位置')
    plt.title('正弦位置编码可视化')
    plt.savefig('positional_encoding_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("已保存热力图: positional_encoding_heatmap.png")


if __name__ == "__main__":
    print("=" * 70)
    print("              位置编码 (Positional Encoding)")
    print("=" * 70)
    
    print("""
为什么需要位置编码？

注意力机制是【置换不变】的 (Permutation Invariant):
  - 输入 ["我", "爱", "你"] 和 ["你", "爱", "我"]
  - 如果只看注意力，模型无法区分这两个句子
  - 位置编码告诉模型每个 token 在序列中的位置
""")
    
    # ==================== 正弦位置编码 ====================
    print("=" * 70)
    print("一、正弦位置编码 (Sinusoidal Positional Encoding)")
    print("=" * 70)
    
    print("""
公式:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中:
    - pos: 位置索引 (0, 1, 2, ...)
    - i: 维度索引 (0, 1, ..., d_model/2 - 1)
    - 偶数维度用 sin，奇数维度用 cos
""")
    
    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    d_model = 8
    
    print(f"\n参数: batch={batch_size}, seq_len={seq_len}, d_model={d_model}")
    
    # 创建输入 (假设是词嵌入)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 正弦位置编码
    sinusoidal_pe = SinusoidalPositionalEncoding(d_model, max_len=100, dropout=0.0)
    
    print("\n位置编码矩阵 (前4个位置):")
    pe_matrix = sinusoidal_pe.pe[0, :seq_len, :].numpy()
    print(f"形状: {pe_matrix.shape}")
    
    # 打印前几个位置的编码
    for pos in range(seq_len):
        print(f"  位置 {pos}: [{', '.join([f'{v:.3f}' for v in pe_matrix[pos, :4]])} ...]")
    
    # 应用位置编码
    x_with_pe = sinusoidal_pe(x)
    print(f"\n输入形状: {x.shape}")
    print(f"添加位置编码后形状: {x_with_pe.shape}")
    print("  → 形状不变，只是加上了位置信息")
    
    # ==================== 相对位置的线性关系 ====================
    print("\n" + "-" * 50)
    print("正弦编码的特殊性质: 相对位置可以线性表示")
    print("-" * 50)
    
    print("""
PE(pos + k) 可以表示为 PE(pos) 的线性变换！

这意味着模型可以学习到相对位置关系:
  - PE(位置5) - PE(位置3) ≈ PE(位置7) - PE(位置5)
  - 相距相同的位置，它们的位置编码差是相似的
""")
    
    # 验证
    pe_vals = sinusoidal_pe.pe[0, :10, :4].numpy()
    diff_1_0 = pe_vals[1] - pe_vals[0]
    diff_5_4 = pe_vals[5] - pe_vals[4]
    print(f"\nPE(1) - PE(0) = [{', '.join([f'{v:.3f}' for v in diff_1_0])}]")
    print(f"PE(5) - PE(4) = [{', '.join([f'{v:.3f}' for v in diff_5_4])}]")
    print("  → 相邻位置的差异是相似的 ✓")
    
    # ==================== 可学习位置嵌入 ====================
    print("\n" + "=" * 70)
    print("二、可学习位置嵌入 (Learned Positional Embedding)")
    print("=" * 70)
    
    print("""
特点:
  - 位置编码是可学习的 nn.Embedding
  - BERT, GPT, RoBERTa 等使用
  - 更灵活，可以学习任务特定的位置模式

缺点:
  - 不能推广到训练时未见过的长度
  - 例如: 训练时 max_len=512，推理时不能处理 1000 个 token
""")
    
    learned_pe = LearnedPositionalEmbedding(d_model, max_len=100, dropout=0.0)
    
    print(f"\n可学习参数量: {learned_pe.embedding.weight.numel()}")
    print(f"  = max_len × d_model = 100 × {d_model} = {100 * d_model}")
    
    x_with_learned = learned_pe(x)
    print(f"\n输入形状: {x.shape}")
    print(f"添加位置编码后形状: {x_with_learned.shape}")
    
    # ==================== 旋转位置编码 ====================
    print("\n" + "=" * 70)
    print("三、旋转位置编码 (RoPE)")
    print("=" * 70)
    
    print("""
核心思想:
  - 不是"加法"，而是"旋转"
  - 对 Q 和 K 应用旋转矩阵
  - Q·K 的点积自然包含相对位置信息

公式 (2D 简化):
    [q₁, q₂] × R(θ) = [q₁·cos(θ) - q₂·sin(θ), q₁·sin(θ) + q₂·cos(θ)]
    
    其中 θ = pos × base_frequency

用于:
  - LLaMA, LLaMA-2
  - GPT-NeoX
  - PaLM
  - Mistral
""")
    
    rope = RotaryPositionalEmbedding(d_model, max_len=100)
    
    # 应用 RoPE
    x_with_rope = rope(x)
    print(f"\n输入形状: {x.shape}")
    print(f"应用 RoPE 后形状: {x_with_rope.shape}")
    
    print("\nRoPE 的优势:")
    print("  ✓ 可以推广到任意长度")
    print("  ✓ 相对位置信息自然编码")
    print("  ✓ 计算效率高")
    print("  ✓ 效果通常更好")
    
    # ==================== 三种方法对比 ====================
    print("\n" + "=" * 70)
    print("四、三种方法对比")
    print("=" * 70)
    
    print("""
┌─────────────────┬───────────────┬───────────────┬───────────────┐
│                 │   Sinusoidal  │    Learned    │     RoPE      │
├─────────────────┼───────────────┼───────────────┼───────────────┤
│ 是否可学习       │      否       │      是       │      否       │
├─────────────────┼───────────────┼───────────────┼───────────────┤
│ 推广到更长序列   │      是       │      否       │      是       │
├─────────────────┼───────────────┼───────────────┼───────────────┤
│ 相对位置信息     │    隐式       │    隐式       │    显式       │
├─────────────────┼───────────────┼───────────────┼───────────────┤
│ 应用方式         │    加法       │    加法       │    乘法       │
├─────────────────┼───────────────┼───────────────┼───────────────┤
│ 使用模型         │  Transformer  │  BERT, GPT-2  │  LLaMA, PaLM  │
│                 │    原始       │               │               │
└─────────────────┴───────────────┴───────────────┴───────────────┘
""")
    
    # ==================== 在 Transformer 中的位置 ====================
    print("=" * 70)
    print("五、在 Transformer 中的位置")
    print("=" * 70)
    
    print("""
Embedding 层之后，第一个 Encoder/Decoder 层之前:

    输入 token IDs: [101, 2769, 4263, 872, 102]
              │
              ▼
    ┌─────────────────────────────┐
    │    Token Embedding          │   词嵌入表
    └─────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────┐
    │  + Positional Encoding      │   ← 加上位置编码
    └─────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────┐
    │       Dropout               │
    └─────────────────────────────┘
              │
              ▼
         Encoder/Decoder 层

注意:
  - RoPE 不是加到 Embedding 上
  - RoPE 是在 Attention 计算中对 Q, K 应用旋转
""")
    
    # ==================== 频率的含义 ====================
    print("=" * 70)
    print("六、理解正弦编码的频率")
    print("=" * 70)
    
    print("""
不同维度有不同的"波长":

  - 低维度 (i 小): 高频，波长短，变化快
    → 编码细粒度的位置差异
    → 例: 区分位置 0 和位置 1
    
  - 高维度 (i 大): 低频，波长长，变化慢
    → 编码粗粒度的位置信息
    → 例: 区分位置 0 和位置 100

类比: 二进制表示
  - 最低位: 0101010101... (变化最快)
  - 最高位: 0000000011... (变化最慢)
  
正弦编码类似，但是连续的:
  - 低维度: ∿∿∿∿∿∿∿∿ (高频波)
  - 高维度: ∿      ∿ (低频波)
""")
    
    # 生成可视化
    print("\n生成位置编码可视化...")
    try:
        visualize_sinusoidal_encoding(d_model=64, max_len=50)
    except Exception as e:
        print(f"  (可视化需要 matplotlib: {e})")
    
    # ==================== 代码使用示例 ====================
    print("\n" + "=" * 70)
    print("七、使用示例")
    print("=" * 70)
    
    print("""
# 原始 Transformer 风格
pe = SinusoidalPositionalEncoding(d_model=512, max_len=5000)
x = token_embedding(input_ids)  # (batch, seq_len, 512)
x = pe(x)                       # 加上位置编码

# BERT/GPT 风格
pe = LearnedPositionalEmbedding(d_model=768, max_len=512)
x = token_embedding(input_ids)  # (batch, seq_len, 768)
x = pe(x)                       # 加上位置编码

# LLaMA 风格 (RoPE)
rope = RotaryPositionalEmbedding(head_dim=64, max_len=4096)
# 在 Attention 中:
Q = rope(Q)  # 对 Query 应用旋转
K = rope(K)  # 对 Key 应用旋转
# V 不需要位置编码
scores = Q @ K.T / sqrt(d_k)
""")
    
    # ==================== 参数量对比 ====================
    print("=" * 70)
    print("八、参数量对比")
    print("=" * 70)
    
    d = 512
    max_l = 512
    
    print(f"\n以 d_model={d}, max_len={max_l} 为例:")
    print(f"\n  Sinusoidal: 0 个参数 (固定计算)")
    print(f"  Learned:    {d * max_l:,} 个参数 ({d} × {max_l})")
    print(f"  RoPE:       0 个参数 (固定计算)")
    
    print("\n相比之下:")
    print(f"  Token Embedding (vocab=30000): {30000 * d:,} 个参数")
    print("  → 位置编码的参数量相对较小")

