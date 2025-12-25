import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_head_attention_core import multi_head_scaled_dot_product_attention
from utils import print_attention_matrix


class MultiHeadCrossAttention(nn.Module):
    """
    多头交叉注意力机制 (Multi-Head Cross-Attention)
    
    核心思想:
        - Q 来自 Decoder，K/V 来自 Encoder（交叉注意力）
        - 分成多个头，让模型从不同角度关注源序列（多头）
    
    用途: Transformer Decoder 中连接 Encoder 和 Decoder
    
    例子: 机器翻译 "The cat sat on the mat" → "猫坐在垫子上"
        - 头1 可能关注: 主要名词对齐 (猫 → cat, 垫子 → mat)
        - 头2 可能关注: 动作对齐 (坐 → sat)
        - 头3 可能关注: 位置词对齐 (在...上 → on)
    """
    
    def __init__(self, embed_size, num_heads):
        """
        参数:
            embed_size: 嵌入维度 (d_model)
            num_heads: 注意力头的数量
        """
        super(MultiHeadCrossAttention, self).__init__()
        
        assert embed_size % num_heads == 0, \
            f"embed_size ({embed_size}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # Q 来自 Decoder
        self.w_q = nn.Linear(embed_size, embed_size)
        # K, V 来自 Encoder
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)
        
        # 输出线性变换
        self.w_o = nn.Linear(embed_size, embed_size)
    
    def forward(self, decoder_input, encoder_output, mask=None):
        """
        前向传播。
        
        参数:
            decoder_input: Decoder 的输入 (batch_size, tgt_len, embed_size)
            encoder_output: Encoder 的输出 (batch_size, src_len, embed_size)
            mask: 可选的掩码 (屏蔽 Encoder 中的 padding)

        返回:
            out: 注意力输出 (batch_size, tgt_len, embed_size)
            attention_weights: 每个头的注意力权重 (batch_size, num_heads, tgt_len, src_len)
        """
        # 1. 线性变换
        Q = self.w_q(decoder_input)    # (batch, tgt_len, embed_size)
        K = self.w_k(encoder_output)   # (batch, src_len, embed_size)
        V = self.w_v(encoder_output)   # (batch, src_len, embed_size)
        
        # 2. 多头缩放点积注意力 (使用公共实现)
        context, attention_weights = multi_head_scaled_dot_product_attention(
            Q, K, V, self.num_heads, mask
        )
        
        # 3. 输出线性变换
        out = self.w_o(context)
        
        return out, attention_weights


if __name__ == "__main__":
    print("=" * 70)
    print("         多头交叉注意力机制 (Multi-Head Cross-Attention)")
    print("=" * 70)
    
    print("\n核心特点:")
    print("  → 交叉: Q 来自 Decoder，K/V 来自 Encoder")
    print("  → 多头: 从多个角度关注源序列的不同方面")
    
    print("""
示意图:

   Decoder 输入                    Encoder 输出
        │                              │
        ↓                         ┌────┴────┐
       W_q                       W_k       W_v
        │                         │         │
        ↓                         ↓         ↓
  ┌─────────────────────────────────────────────┐
  │                  分头                        │
  └─────────────────────────────────────────────┘
        │                         │         │
        ↓                         ↓         ↓
      ┌───┐                     ┌───┐     ┌───┐
  头1 │ Q │ ──────────────────→ │ K │     │ V │ ─→ 头1 输出
      └───┘                     └───┘     └───┘
      ┌───┐                     ┌───┐     ┌───┐
  头2 │ Q │ ──────────────────→ │ K │     │ V │ ─→ 头2 输出
      └───┘                     └───┘     └───┘
        ⋮                          ⋮         ⋮
        │                          │         │
        └──────────────┬───────────┘
                       ↓
              ┌────────────────┐
              │  拼接 + W_o    │
              └────────────────┘
                       │
                       ↓
                     输出
""")
    
    # 设置参数
    torch.manual_seed(42)
    batch_size = 1
    embed_size = 8
    num_heads = 2
    
    # 模拟翻译任务
    src_tokens = ["I", "love", "you", "very", "much"]    # 源语言 (5个词)
    tgt_tokens = ["我", "非常", "爱", "你"]               # 目标语言 (4个词)
    
    src_len = len(src_tokens)
    tgt_len = len(tgt_tokens)
    
    print("-" * 70)
    print("示例: 机器翻译")
    print("-" * 70)
    print(f"  源语言 (Encoder): {src_tokens} (长度={src_len})")
    print(f"  目标语言 (Decoder): {tgt_tokens} (长度={tgt_len})")
    print(f"  num_heads = {num_heads}, head_dim = {embed_size // num_heads}")
    
    encoder_output = torch.randn(batch_size, src_len, embed_size)
    decoder_input = torch.randn(batch_size, tgt_len, embed_size)
    
    print(f"\nEncoder 输出形状: {encoder_output.shape}")
    print(f"Decoder 输入形状: {decoder_input.shape}")
    
    # 创建多头交叉注意力层
    mhca = MultiHeadCrossAttention(embed_size, num_heads)
    
    print("\n" + "=" * 70)
    print("计算多头交叉注意力")
    print("=" * 70)
    
    with torch.no_grad():
        output, attn_weights = mhca(decoder_input, encoder_output)
    
    print(f"\n注意力权重形状: {attn_weights.shape}")
    print(f"  → (batch={batch_size}, heads={num_heads}, tgt_len={tgt_len}, src_len={src_len})")
    
    # 展示每个头的注意力权重
    for head_idx in range(num_heads):
        weights = attn_weights[0, head_idx].numpy()
        print_attention_matrix(tgt_tokens, src_tokens, weights,
                               f"头 {head_idx + 1} 的交叉注意力权重", "目标词", "源词")
    
    print(f"\n输出形状: {output.shape}")
    print("  → 与 Decoder 输入形状相同")
    
    # 分析不同头的对齐模式
    print("\n" + "=" * 70)
    print("分析: 不同头学习不同的对齐模式")
    print("=" * 70)
    
    for head_idx in range(num_heads):
        print(f"\n【头 {head_idx + 1}】目标词对齐到源词:")
        weights = attn_weights[0, head_idx].numpy()
        for i, tgt_token in enumerate(tgt_tokens):
            max_idx = weights[i].argmax()
            max_weight = weights[i][max_idx]
            print(f"  「{tgt_token}」→「{src_tokens[max_idx]}」(权重={max_weight:.2f})")
    
    # 对比单头和多头
    print("\n" + "=" * 70)
    print("多头交叉注意力的优势")
    print("=" * 70)
    print("""
在翻译任务中的作用:

1. 【词汇对齐】
   某些头专门学习词汇层面的对应关系
   例: "love" → "爱", "you" → "你"

2. 【短语对齐】
   某些头学习短语级别的对应
   例: "very much" → "非常"

3. 【结构对齐】
   某些头关注句法结构的对应
   例: 英语的 SVO 顺序 → 中文的顺序调整

4. 【长距离依赖】
   某些头可能关注远距离的修饰关系

每个头的关注点不同，合并后模型能获得更丰富的源语言信息。
""")
    
    # 维度变换详解
    print("=" * 70)
    print("维度变换详解")
    print("=" * 70)
    print(f"""
Decoder 输入:     ({batch_size}, {tgt_len}, {embed_size})  ← Q 的来源
Encoder 输出:     ({batch_size}, {src_len}, {embed_size})  ← K, V 的来源

线性变换后:
  Q: ({batch_size}, {tgt_len}, {embed_size})
  K: ({batch_size}, {src_len}, {embed_size})
  V: ({batch_size}, {src_len}, {embed_size})

分头后:
  Q: ({batch_size}, {num_heads}, {tgt_len}, {embed_size // num_heads})
  K: ({batch_size}, {num_heads}, {src_len}, {embed_size // num_heads})
  V: ({batch_size}, {num_heads}, {src_len}, {embed_size // num_heads})

注意力权重:
  scores: ({batch_size}, {num_heads}, {tgt_len}, {src_len})
  
  注意: 矩阵形状是 (tgt_len × src_len)，不是方阵！
  因为 Q 和 K 来自不同长度的序列。

合并后输出:
  ({batch_size}, {tgt_len}, {embed_size})
""")
    
    # Padding Mask 示例
    print("=" * 70)
    print("处理 Encoder 侧的 Padding")
    print("=" * 70)
    
    src_padded = ["I", "love", "<PAD>", "<PAD>", "<PAD>"]
    print(f"\n源序列 (带 padding): {src_padded} (实际长度=2)")
    
    # Encoder padding mask
    encoder_mask = torch.tensor([[1, 1, 0, 0, 0]])  # (batch, src_len)
    
    with torch.no_grad():
        _, attn_weights_masked = mhca(decoder_input, encoder_output, mask=encoder_mask)
    
    for head_idx in range(num_heads):
        weights = attn_weights_masked[0, head_idx].numpy()
        print_attention_matrix(tgt_tokens, src_padded, weights,
                               f"头 {head_idx + 1} (带 Encoder Padding Mask)", "目标词", "源词")
    
    print("\n  → 所有头的后三列 (<PAD> 位置) 权重都为 0")
    print("  → Decoder 不会从 Encoder 的填充位置获取信息")
    
    # 总结
    print("\n" + "=" * 70)
    print("总结: 多头交叉注意力在 Transformer 中的位置")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                         Transformer                             │
├─────────────────────────────┬───────────────────────────────────┤
│         Encoder             │            Decoder                │
├─────────────────────────────┼───────────────────────────────────┤
│                             │                                   │
│  ┌───────────────────────┐  │  ┌───────────────────────────┐   │
│  │ Multi-Head            │  │  │ Masked Multi-Head         │   │
│  │ Self-Attention        │  │  │ Self-Attention            │   │
│  └───────────────────────┘  │  └───────────────────────────┘   │
│           ↓                 │              ↓                    │
│  ┌───────────────────────┐  │  ┌───────────────────────────┐   │
│  │ Feed Forward          │  │  │ Multi-Head                │   │
│  └───────────────────────┘  │  │ Cross-Attention ←─────────┼───┤
│           ↓                 │  └───────────────────────────┘   │
│      Encoder Output ────────┼──────────→ K, V                  │
│                             │              ↓                    │
│                             │  ┌───────────────────────────┐   │
│                             │  │ Feed Forward              │   │
│                             │  └───────────────────────────┘   │
└─────────────────────────────┴───────────────────────────────────┘
""")

