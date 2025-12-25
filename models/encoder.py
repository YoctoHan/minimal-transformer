"""
Transformer Encoder 模块

结构 (来自 "Attention Is All You Need"):
    
    EncoderLayer:
        - Multi-Head Self-Attention
        - Add & Norm
        - Position-wise Feed-Forward
        - Add & Norm
    
    Encoder:
        - Input Embedding
        - Positional Encoding
        - N × EncoderLayer
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention.multi_head_self_attention import MultiHeadSelfAttention
from layers.feed_forward import PositionwiseFeedForward
from layers.residual_layer_norm import LayerNorm
from layers.positional_encoding import SinusoidalPositionalEncoding


class EncoderLayer(nn.Module):
    """
    Transformer Encoder 的单层
    
    结构:
        x → Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm → output
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: FFN 中间层维度
            dropout: Dropout 概率
        """
        super(EncoderLayer, self).__init__()
        
        # 多头自注意力
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads)
        
        # 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 两个 Layer Normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        参数:
            x: 输入 (batch_size, seq_len, d_model)
            mask: 注意力掩码 (可选，用于 padding)
            
        返回:
            output: (batch_size, seq_len, d_model)
        """
        # 1. Multi-Head Self-Attention + Add & Norm
        attn_output, attn_weights = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Feed-Forward + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


class Encoder(nn.Module):
    """
    Transformer Encoder
    
    结构:
        Input → Embedding → Positional Encoding → N × EncoderLayer → Output
    """
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, 
                 max_len=5000, dropout=0.1):
        """
        参数:
            vocab_size: 词表大小
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: FFN 中间层维度
            num_layers: Encoder 层数
            max_len: 最大序列长度
            dropout: Dropout 概率
        """
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        
        # N 个 Encoder 层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最后的 Layer Norm (有些实现会加)
        self.norm = LayerNorm(d_model)
    
    def forward(self, src, src_mask=None):
        """
        参数:
            src: 源序列 token IDs (batch_size, src_len)
            src_mask: 源序列掩码 (可选)
            
        返回:
            output: Encoder 输出 (batch_size, src_len, d_model)
            attention_weights: 每层的注意力权重
        """
        # 词嵌入 + 缩放 (论文中乘以 sqrt(d_model))
        x = self.embedding(src) * (self.d_model ** 0.5)
        
        # 位置编码
        x = self.positional_encoding(x)
        
        # 通过每一层
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            attention_weights.append(attn_weights)
        
        # 最后的归一化
        x = self.norm(x)
        
        return x, attention_weights


if __name__ == "__main__":
    print("=" * 70)
    print("                   Transformer Encoder")
    print("=" * 70)
    
    print("""
Encoder 结构 (来自 "Attention Is All You Need"):

    ┌─────────────────────────────────────────────────────┐
    │                    Encoder                          │
    │  ┌───────────────────────────────────────────────┐  │
    │  │            Input Embedding                    │  │
    │  └───────────────────────────────────────────────┘  │
    │                        │                            │
    │                        ▼                            │
    │  ┌───────────────────────────────────────────────┐  │
    │  │        + Positional Encoding                  │  │
    │  └───────────────────────────────────────────────┘  │
    │                        │                            │
    │           ┌────────────┴────────────┐               │
    │           │                         │               │
    │           ▼                         │               │
    │  ┌─────────────────────────┐        │               │
    │  │ Multi-Head              │        │               │
    │  │ Self-Attention          │        │               │
    │  └─────────────────────────┘        │               │
    │           │                         │               │
    │           ▼                         │               │
    │  ┌─────────────────────────┐        │               │
    │  │    Add & Norm           │◄───────┘               │
    │  └─────────────────────────┘                        │
    │           │                                         │
    │           ├────────────────────────┐                │
    │           ▼                        │                │
    │  ┌─────────────────────────┐       │                │
    │  │   Feed Forward          │       │                │
    │  └─────────────────────────┘       │                │
    │           │                        │                │
    │           ▼                        │                │
    │  ┌─────────────────────────┐       │                │
    │  │    Add & Norm           │◄──────┘                │
    │  └─────────────────────────┘                        │
    │           │                                         │
    │           ▼                                         │
    │        (重复 N 次)                                   │
    │           │                                         │
    │           ▼                                         │
    │  ┌───────────────────────────────────────────────┐  │
    │  │              Output                           │  │
    │  └───────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────┘
""")
    
    # 设置参数 (论文中的 base 配置)
    torch.manual_seed(42)
    
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    max_len = 100
    dropout = 0.1
    
    batch_size = 2
    src_len = 10
    
    print("参数配置 (Transformer-base):")
    print(f"  vocab_size = {vocab_size}")
    print(f"  d_model = {d_model}")
    print(f"  num_heads = {num_heads}")
    print(f"  d_ff = {d_ff}")
    print(f"  num_layers = {num_layers}")
    print(f"  dropout = {dropout}")
    
    print("\n" + "=" * 70)
    print("创建 Encoder")
    print("=" * 70)
    
    # 创建 Encoder
    encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
    
    # 计算参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 创建输入
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    
    print(f"\n输入形状: {src.shape}")
    print(f"  → (batch_size={batch_size}, src_len={src_len})")
    
    # 前向传播
    encoder.eval()
    with torch.no_grad():
        output, attn_weights = encoder(src)
    
    print(f"\n输出形状: {output.shape}")
    print(f"  → (batch_size={batch_size}, src_len={src_len}, d_model={d_model})")
    
    print(f"\n注意力权重数量: {len(attn_weights)} 层")
    print(f"每层权重形状: {attn_weights[0].shape}")
    print(f"  → (batch_size, num_heads, src_len, src_len)")
    
    # 使用 Padding Mask
    print("\n" + "=" * 70)
    print("使用 Padding Mask")
    print("=" * 70)
    
    # 假设第一个样本长度为 8，第二个样本长度为 5
    # 创建 padding mask (需要 unsqueeze 以匹配注意力计算的维度)
    src_mask = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # 样本1: 8个有效token
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 样本2: 5个有效token
    ]).unsqueeze(1)  # (batch, 1, seq_len) 格式
    
    print(f"Padding Mask 形状: {src_mask.shape}")
    print(f"样本1 有效长度: {src_mask[0].sum().item()}")
    print(f"样本2 有效长度: {src_mask[1].sum().item()}")
    print(f"  → 形状为 (batch, 1, seq_len) 以便广播")
    
    with torch.no_grad():
        output_masked, _ = encoder(src, src_mask)
    
    print(f"\n带 Mask 的输出形状: {output_masked.shape}")
    print("  → Padding 位置的输出不应该被使用")


