"""
完整的 Transformer 模型

来自论文 "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    """
    完整的 Transformer 模型
    """
    
    def __init__(self, 
                 src_vocab_size,
                 tgt_vocab_size,
                 d_model=512,
                 num_heads=8,
                 d_ff=2048,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 max_len=5000,
                 dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, d_ff,
            num_encoder_layers, max_len, dropout
        )
        
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads, d_ff,
            num_decoder_layers, max_len, dropout
        )
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output, _ = self.encoder(src, src_mask)
        decoder_output, _, _ = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.output_projection(decoder_output)
        return logits
    
    def encode(self, src, src_mask=None):
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        decoder_output, _, _ = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)


def create_masks(src, tgt, pad_idx=0):
    """创建源序列和目标序列的掩码"""
    src_mask = (src != pad_idx).unsqueeze(1)
    
    tgt_len = tgt.size(1)
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device))
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1)
    tgt_mask = causal_mask.unsqueeze(0) * tgt_padding_mask
    
    return src_mask, tgt_mask


if __name__ == "__main__":
    print("=" * 70)
    print("                      Transformer")
    print("             'Attention Is All You Need'")
    print("=" * 70)
    
    print("""
完整 Transformer 结构:

┌─────────────────────────────────────────────────────────────────────┐
│                         Transformer                                 │
│                                                                     │
│   源序列                                    目标序列 (右移)           │
│    │                                          │                     │
│    ▼                                          ▼                     │
│ ┌──────────────────┐                    ┌──────────────────┐        │
│ │ Input Embedding  │                    │Output Embedding  │        │
│ │       +          │                    │       +          │        │
│ │ Pos Encoding     │                    │ Pos Encoding     │        │
│ └──────────────────┘                    └──────────────────┘        │
│         │                                       │                   │
│         ▼                                       ▼                   │
│ ┌──────────────────┐                    ┌──────────────────┐        │
│ │                  │                    │ Masked           │        │
│ │  Self-Attention  │                    │ Self-Attention   │        │
│ │  + Add & Norm    │                    │ + Add & Norm     │        │
│ │                  │                    │                  │        │
│ │  Feed Forward    │    ┌───────────────│ Cross-Attention  │        │
│ │  + Add & Norm    │    │               │ + Add & Norm     │        │
│ │                  │────┘               │                  │        │
│ │    × N layers    │                    │ Feed Forward     │        │
│ │                  │                    │ + Add & Norm     │        │
│ └──────────────────┘                    │                  │        │
│      ENCODER                            │    × N layers    │        │
│                                         └──────────────────┘        │
│                                               DECODER               │
│                                                 │                   │
│                                                 ▼                   │
│                                         ┌──────────────────┐        │
│                                         │  Linear + Softmax│        │
│                                         └──────────────────┘        │
│                                                 │                   │
│                                                 ▼                   │
│                                           输出概率分布               │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    torch.manual_seed(42)
    
    # 论文 base 配置
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    print("=" * 70)
    print("模型配置 (Transformer-base)")
    print("=" * 70)
    print(f"  d_model = {d_model}")
    print(f"  num_heads = {num_heads}")
    print(f"  d_ff = {d_ff}")
    print(f"  num_layers = {num_layers}")
    print(f"  src_vocab = {src_vocab_size}, tgt_vocab = {tgt_vocab_size}")
    
    print("\n创建模型...")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers
    )
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\n参数量统计:")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")
    print(f"  Output Projection: {d_model * tgt_vocab_size:,}")
    print(f"  总计: {total_params:,}")
    
    print("\n" + "=" * 70)
    print("前向传播测试")
    print("=" * 70)
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    
    src_mask, tgt_mask = create_masks(src, tgt, pad_idx=0)
    
    print(f"\n源序列: {src.shape}")
    print(f"目标序列: {tgt.shape}")
    
    model.eval()
    with torch.no_grad():
        logits = model(src, tgt, src_mask, tgt_mask)
    
    print(f"\n输出 logits: {logits.shape}")
    print(f"  → (batch={batch_size}, tgt_len={tgt_len}, vocab={tgt_vocab_size})")
    
    predictions = logits.argmax(dim=-1)
    print(f"\n预测 tokens: {predictions.shape}")
    print(f"第一个样本: {predictions[0].tolist()}")
    
    print("\n" + "=" * 70)
    print("自回归生成 (贪心解码)")
    print("=" * 70)
    
    def greedy_decode(model, src, src_mask, max_len, start_symbol):
        model.eval()
        encoder_output, _ = model.encode(src, src_mask)
        tgt = torch.full((src.size(0), 1), start_symbol, dtype=torch.long)
        
        for _ in range(max_len - 1):
            tgt_mask = torch.tril(torch.ones(tgt.size(1), tgt.size(1)))
            tgt_mask = tgt_mask.unsqueeze(0)
            
            logits = model.decode(tgt, encoder_output, src_mask, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, next_token], dim=1)
        
        return tgt
    
    with torch.no_grad():
        generated = greedy_decode(model, src[:1], src_mask[:1], max_len=12, start_symbol=1)
    
    print(f"生成序列: {generated[0].tolist()}")
    print(f"生成长度: {generated.size(1)}")
    
    print("\n" + "=" * 70)
    print("论文配置参考")
    print("=" * 70)
    print("""
┌─────────────────┬──────────────┬──────────────┐
│     参数        │ Base Model   │  Big Model   │
├─────────────────┼──────────────┼──────────────┤
│ d_model         │     512      │    1024      │
│ d_ff            │    2048      │    4096      │
│ num_heads       │      8       │     16       │
│ num_layers      │      6       │      6       │
│ dropout         │     0.1      │     0.3      │
├─────────────────┼──────────────┼──────────────┤
│ 参数量          │    ~65M      │    ~213M     │
└─────────────────┴──────────────┴──────────────┘
""")
    
    print("=" * 70)
    print("项目模块结构")
    print("=" * 70)
    print("""
transformer.py (本文件)
    │
    ├── encoder.py
    │       └── EncoderLayer
    │               ├── multi_head_self_attention.py
    │               ├── feed_forward.py
    │               └── residual_layer_norm.py
    │
    ├── decoder.py
    │       └── DecoderLayer
    │               ├── multi_head_masked_self_attention.py
    │               ├── multi_head_cross_attention.py
    │               ├── feed_forward.py
    │               └── residual_layer_norm.py
    │
    └── positional_encoding.py
    
底层依赖:
    ├── multi_head_attention_core.py
    ├── scaled_dot_product_attention.py
    └── utils.py
""")

