import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaled_dot_product_attention import scaled_dot_product_attention
from utils import print_attention_matrix


class CrossAttention(nn.Module):
    """
    交叉注意力机制 (Cross-Attention)
    
    特点: Q 来自一个序列，K 和 V 来自另一个序列
    用途: Decoder 关注 Encoder 的输出
    
    例子: 机器翻译 "I love you" → "我爱你"
          → 生成 "爱" 时，需要关注英文的 "love"
    """
    
    def __init__(self, embed_size):
        """
        参数:
            embed_size: 嵌入维度 (d_model)
        """
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size

        # Q 来自 Decoder，K/V 来自 Encoder
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)

    def forward(self, decoder_input, encoder_output, mask=None):
        """
        前向传播。
        
        参数:
            decoder_input: Decoder 的输入 (batch_size, seq_len_dec, embed_size)
            encoder_output: Encoder 的输出 (batch_size, seq_len_enc, embed_size)
            mask: 可选的掩码 (屏蔽 Encoder 中的 padding)

        返回:
            out: 注意力输出 (batch_size, seq_len_dec, embed_size)
            attention_weights: 注意力权重 (batch_size, seq_len_dec, seq_len_enc)
        """
        # Q 来自 Decoder 输入
        Q = self.w_q(decoder_input)
        # K, V 来自 Encoder 输出
        K = self.w_k(encoder_output)
        V = self.w_v(encoder_output)

        out, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        return out, attention_weights


if __name__ == "__main__":
    print("=" * 60)
    print("             交叉注意力机制 (Cross-Attention)")
    print("=" * 60)
    
    print("\n核心特点:")
    print("  → Q 来自【Decoder】，K 和 V 来自【Encoder】")
    print("  → 让 Decoder 的每个位置都能关注 Encoder 的所有位置")
    
    print("\n应用场景:")
    print("  → Transformer Decoder 中的 Encoder-Decoder Attention")
    print("  → 机器翻译、文本摘要等 Seq2Seq 任务")
    
    print("""
示意图:
                    ┌─────────────────────────────────┐
                    │         Cross-Attention         │
                    └─────────────────────────────────┘
                                   ▲
                    ┌──────────────┼──────────────┐
                    │              │              │
                  ┌─┴─┐          ┌─┴─┐          ┌─┴─┐
                  │W_q│          │W_k│          │W_v│
                  └─┬─┘          └─┬─┘          └─┬─┘
                    │              │              │
              ┌─────┴─────┐  ┌─────┴──────────────┴─────┐
              │  Decoder  │  │        Encoder           │
              │   输入    │  │         输出             │
              └───────────┘  └──────────────────────────┘
                   ↑                     ↑
                 目标语言              源语言
""")
    
    # 设置参数
    torch.manual_seed(42)
    batch_size = 1
    embed_size = 8
    
    # 模拟翻译任务: "I love you" → "我爱你"
    src_tokens = ["I", "love", "you"]      # 源语言 (英文)
    tgt_tokens = ["我", "爱", "你"]         # 目标语言 (中文)
    
    src_len = len(src_tokens)
    tgt_len = len(tgt_tokens)
    
    print("-" * 60)
    print("示例: 机器翻译")
    print("-" * 60)
    print(f"  源语言 (Encoder 输入): {src_tokens}")
    print(f"  目标语言 (Decoder 输入): {tgt_tokens}")
    
    # 模拟 Encoder 输出和 Decoder 输入
    encoder_output = torch.randn(batch_size, src_len, embed_size)
    decoder_input = torch.randn(batch_size, tgt_len, embed_size)
    
    print(f"\nEncoder 输出形状: {encoder_output.shape}")
    print(f"Decoder 输入形状: {decoder_input.shape}")
    
    # 创建交叉注意力层
    cross_attn = CrossAttention(embed_size)
    
    # 计算交叉注意力
    print("\n" + "=" * 60)
    print("计算交叉注意力")
    print("=" * 60)
    
    with torch.no_grad():
        output, attn_weights = cross_attn(decoder_input, encoder_output)
    
    weights = attn_weights[0].numpy()
    print_attention_matrix(tgt_tokens, src_tokens, weights, 
                           "交叉注意力权重矩阵", "目标词", "源词")
    
    print("\n解读 (哪个中文词关注哪个英文词最多):")
    for i, tgt_token in enumerate(tgt_tokens):
        max_idx = weights[i].argmax()
        max_weight = weights[i][max_idx]
        print(f"  '{tgt_token}' 最关注 '{src_tokens[max_idx]}' (权重={max_weight:.2f})")
    
    print(f"\n输出形状: {output.shape}")
    print("  → 与 Decoder 输入形状相同")
    print("  → 每个 Decoder 位置融合了 Encoder 的信息")
    
    # 不同长度示例
    print("\n" + "=" * 60)
    print("关键点: 源序列和目标序列长度可以不同")
    print("=" * 60)
    
    # 更长的源序列
    long_src = ["The", "quick", "brown", "fox", "jumps"]
    short_tgt = ["敏捷", "的", "狐狸"]
    
    print(f"\n源序列 (长度={len(long_src)}): {long_src}")
    print(f"目标序列 (长度={len(short_tgt)}): {short_tgt}")
    
    encoder_out_long = torch.randn(batch_size, len(long_src), embed_size)
    decoder_in_short = torch.randn(batch_size, len(short_tgt), embed_size)
    
    with torch.no_grad():
        out_diff, weights_diff = cross_attn(decoder_in_short, encoder_out_long)
    
    print(f"\n注意力权重形状: {weights_diff.shape}")
    print(f"  → ({batch_size}, {len(short_tgt)}, {len(long_src)})")
    print(f"  → 每个目标词对所有源词的注意力")
    
    w = weights_diff[0].numpy()
    print_attention_matrix(short_tgt, long_src, w, 
                           "注意力权重矩阵", "目标词", "源词")
    
    # Padding Mask 示例
    print("\n" + "=" * 60)
    print("处理 Padding: 屏蔽 Encoder 侧的填充位置")
    print("=" * 60)
    
    print("\n假设 Encoder 输入: ['I', 'love', '<PAD>'] (实际长度=2)")
    
    # 创建 encoder padding mask
    encoder_mask = torch.tensor([[[1, 1, 0]]])  # 1=有效, 0=填充
    
    encoder_padded = torch.randn(batch_size, 3, embed_size)
    decoder_in = torch.randn(batch_size, 2, embed_size)
    
    with torch.no_grad():
        _, weights_masked = cross_attn(decoder_in, encoder_padded, mask=encoder_mask)
    
    w_m = weights_masked[0].numpy()
    tgt_labels = ["目标1", "目标2"]
    src_labels = ["I", "love", "<PAD>"]
    print_attention_matrix(tgt_labels, src_labels, w_m,
                           "带 Mask 的交叉注意力权重", "目标词", "源词")
    print("\n  → <PAD> 列权重为 0，不会影响翻译结果")
    
    # 总结
    print("\n" + "=" * 60)
    print("总结: 交叉注意力 vs 自注意力")
    print("=" * 60)
    print("""
┌──────────────────┬───────────────────┬───────────────────┐
│                  │   Self-Attention  │  Cross-Attention  │
├──────────────────┼───────────────────┼───────────────────┤
│  Q 的来源        │   输入序列 x      │   Decoder 输入    │
├──────────────────┼───────────────────┼───────────────────┤
│  K, V 的来源     │   输入序列 x      │   Encoder 输出    │
├──────────────────┼───────────────────┼───────────────────┤
│  注意力矩阵形状  │  (seq_len, seq_len) │ (tgt_len, src_len) │
├──────────────────┼───────────────────┼───────────────────┤
│  用途            │  捕捉序列内部关系  │  连接两个序列     │
└──────────────────┴───────────────────┴───────────────────┘
""")

