"""
Transformer Decoder 模块

结构 (来自 "Attention Is All You Need"):
    
    DecoderLayer:
        - Masked Multi-Head Self-Attention
        - Add & Norm
        - Multi-Head Cross-Attention (with Encoder output)
        - Add & Norm
        - Position-wise Feed-Forward
        - Add & Norm
    
    Decoder:
        - Output Embedding
        - Positional Encoding
        - N × DecoderLayer
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention.multi_head_masked_self_attention import MultiHeadMaskedSelfAttention
from attention.multi_head_cross_attention import MultiHeadCrossAttention
from layers.feed_forward import PositionwiseFeedForward
from layers.residual_layer_norm import LayerNorm
from layers.positional_encoding import SinusoidalPositionalEncoding


class DecoderLayer(nn.Module):
    """
    Transformer Decoder 的单层
    
    结构:
        x → Masked Self-Attention → Add & Norm 
          → Cross-Attention (with encoder output) → Add & Norm 
          → FFN → Add & Norm → output
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # 带掩码的多头自注意力
        self.masked_self_attention = MultiHeadMaskedSelfAttention(d_model, num_heads)
        
        # 多头交叉注意力
        self.cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        
        # 前馈网络
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 三个 Layer Normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. Masked Multi-Head Self-Attention + Add & Norm
        self_attn_output, self_attn_weights = self.masked_self_attention(x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 2. Multi-Head Cross-Attention + Add & Norm
        cross_attn_output, cross_attn_weights = self.cross_attention(x, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. Feed-Forward + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attn_weights, cross_attn_weights


class Decoder(nn.Module):
    """
    Transformer Decoder
    """
    
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers,
                 max_len=5000, dropout=0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = LayerNorm(d_model)
    
    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt) * (self.d_model ** 0.5)
        x = self.positional_encoding(x)
        
        self_attention_weights = []
        cross_attention_weights = []
        
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)
        
        x = self.norm(x)
        
        return x, self_attention_weights, cross_attention_weights


if __name__ == "__main__":
    print("=" * 70)
    print("                   Transformer Decoder")
    print("=" * 70)
    
    print("""
Decoder Layer 结构:

    ┌────────────────────────────────────────────────┐
    │                DecoderLayer                    │
    │                                                │
    │   输入 x ─────────────┬───────────────────┐    │
    │         │             │                   │    │
    │         ▼             │                   │    │
    │   ┌──────────────┐    │                   │    │
    │   │ Masked       │    │                   │    │
    │   │ Self-Attn    │    │                   │    │
    │   └──────────────┘    │                   │    │
    │         │             │                   │    │
    │         ▼             │                   │    │
    │   ┌──────────────┐    │                   │    │
    │   │  Add & Norm  │◄───┘                   │    │
    │   └──────────────┘                        │    │
    │         │ ──────────────┬─────────────────│    │
    │         ▼               │    Encoder      │    │
    │   ┌──────────────┐      │    Output       │    │
    │   │ Cross-Attn   │◄─────┼────────────     │    │
    │   └──────────────┘      │                 │    │
    │         │               │                 │    │
    │         ▼               │                 │    │
    │   ┌──────────────┐      │                 │    │
    │   │  Add & Norm  │◄─────┘                 │    │
    │   └──────────────┘                        │    │
    │         │ ────────────────────────────┐   │    │
    │         ▼                             │   │    │
    │   ┌──────────────┐                    │   │    │
    │   │ Feed Forward │                    │   │    │
    │   └──────────────┘                    │   │    │
    │         │                             │   │    │
    │         ▼                             │   │    │
    │   ┌──────────────┐                    │   │    │
    │   │  Add & Norm  │◄───────────────────┘   │    │
    │   └──────────────┘                        │    │
    │         │                                 │    │
    │         ▼                                 │    │
    │      输出                                  │    │
    └────────────────────────────────────────────────┘
""")
    
    torch.manual_seed(42)
    
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    print(f"参数: d_model={d_model}, heads={num_heads}, layers={num_layers}")
    
    decoder = Decoder(vocab_size, d_model, num_heads, d_ff, num_layers)
    
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"总参数量: {total_params:,}")
    
    encoder_output = torch.randn(batch_size, src_len, d_model)
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    
    print(f"\nEncoder 输出: {encoder_output.shape}")
    print(f"目标序列: {tgt.shape}")
    
    decoder.eval()
    with torch.no_grad():
        output, self_attn, cross_attn = decoder(tgt, encoder_output)
    
    print(f"\nDecoder 输出: {output.shape}")
    print(f"自注意力权重: {len(self_attn)} 层, 每层 {self_attn[0].shape}")
    print(f"交叉注意力权重: {len(cross_attn)} 层, 每层 {cross_attn[0].shape}")

