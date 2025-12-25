"""
多头注意力核心模块 (Multi-Head Attention Core)

提供多头注意力的公共实现，被以下模块复用：
- multi_head_self_attention.py
- multi_head_cross_attention.py  
- multi_head_masked_self_attention.py
"""

import torch
import torch.nn.functional as F
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import print_self_attention_matrix, print_attention_matrix


def split_heads(x, num_heads):
    """
    将张量分成多个头。
    
    参数:
        x: 输入张量 (batch_size, seq_len, embed_size)
        num_heads: 头的数量
        
    返回:
        分头后的张量 (batch_size, num_heads, seq_len, head_dim)
    """
    batch_size, seq_len, embed_size = x.size()
    head_dim = embed_size // num_heads
    
    # (batch, seq_len, embed_size) → (batch, seq_len, num_heads, head_dim)
    x = x.view(batch_size, seq_len, num_heads, head_dim)
    
    # (batch, seq_len, num_heads, head_dim) → (batch, num_heads, seq_len, head_dim)
    return x.transpose(1, 2)


def merge_heads(x):
    """
    合并多个头。
    
    参数:
        x: 分头后的张量 (batch_size, num_heads, seq_len, head_dim)
        
    返回:
        合并后的张量 (batch_size, seq_len, embed_size)
    """
    batch_size, num_heads, seq_len, head_dim = x.size()
    embed_size = num_heads * head_dim
    
    # (batch, num_heads, seq_len, head_dim) → (batch, seq_len, num_heads, head_dim)
    x = x.transpose(1, 2).contiguous()
    
    # (batch, seq_len, num_heads, head_dim) → (batch, seq_len, embed_size)
    return x.view(batch_size, seq_len, embed_size)


def multi_head_scaled_dot_product_attention(Q, K, V, num_heads, mask=None):
    """
    多头缩放点积注意力计算。
    
    公式: MultiHead(Q, K, V) = Concat(head_1, ..., head_h)
          其中 head_i = softmax(Q_i @ K_i^T / √d_k) @ V_i
    
    参数:
        Q: 查询矩阵 (batch_size, seq_len_q, embed_size)
        K: 键矩阵 (batch_size, seq_len_k, embed_size)
        V: 值矩阵 (batch_size, seq_len_v, embed_size)
        num_heads: 注意力头的数量
        mask: 掩码矩阵 (可选)

    返回:
        output: 注意力输出 (batch_size, seq_len_q, embed_size)
        attention_weights: 每个头的注意力权重 (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    embed_size = Q.size(-1)
    head_dim = embed_size // num_heads
    
    # 1. 分头
    Q = split_heads(Q, num_heads)  # (batch, num_heads, seq_len_q, head_dim)
    K = split_heads(K, num_heads)  # (batch, num_heads, seq_len_k, head_dim)
    V = split_heads(V, num_heads)  # (batch, num_heads, seq_len_v, head_dim)
    
    # 2. 计算缩放点积注意力
    # scores: (batch, num_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    
    # 3. 应用掩码
    if mask is not None:
        # 扩展 mask 维度以匹配 scores
        if mask.dim() == 2:
            # (seq_len, seq_len) → (1, 1, seq_len, seq_len)
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            # (batch, 1, seq_len) 或 (batch, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 4. Softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # 5. 加权求和
    context = torch.matmul(attention_weights, V)
    # context: (batch, num_heads, seq_len_q, head_dim)
    
    # 6. 合并多头
    output = merge_heads(context)
    # output: (batch, seq_len_q, embed_size)
    
    return output, attention_weights


if __name__ == "__main__":
    print("=" * 70)
    print("      多头缩放点积注意力 (Multi-Head Scaled Dot-Product Attention)")
    print("=" * 70)
    
    print("\n这是多头注意力的核心计算模块，被以下脚本复用:")
    print("  → multi_head_self_attention.py")
    print("  → multi_head_cross_attention.py")
    print("  → multi_head_masked_self_attention.py")
    
    print("""
公式:
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)
    
    其中 head_i = softmax(Q_i @ K_i^T / √d_k) @ V_i

核心函数:
    1. split_heads(x, num_heads)     - 分头
    2. merge_heads(x)                 - 合并多头
    3. multi_head_scaled_dot_product_attention(Q, K, V, num_heads, mask)
""")
    
    print("=" * 70)
    print("演示: 分头与合并")
    print("=" * 70)
    
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 4
    embed_size = 8
    num_heads = 2
    head_dim = embed_size // num_heads
    
    print(f"\n参数: embed_size={embed_size}, num_heads={num_heads}, head_dim={head_dim}")
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, embed_size)
    print(f"\n输入 x 形状: {x.shape}")
    print(f"  → (batch={batch_size}, seq_len={seq_len}, embed_size={embed_size})")
    
    # 分头
    x_split = split_heads(x, num_heads)
    print(f"\n分头后形状: {x_split.shape}")
    print(f"  → (batch={batch_size}, num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim})")
    
    # 合并
    x_merged = merge_heads(x_split)
    print(f"\n合并后形状: {x_merged.shape}")
    print(f"  → 恢复为原始形状")
    
    # 验证
    print(f"\n验证: 分头后合并是否还原? {torch.allclose(x, x_merged)}")
    
    # 演示多头注意力计算
    print("\n" + "=" * 70)
    print("演示: 多头自注意力计算")
    print("=" * 70)
    
    tokens = ["我", "喜欢", "机器", "学习"]
    Q = K = V = torch.randn(batch_size, seq_len, embed_size)
    
    output, attn_weights = multi_head_scaled_dot_product_attention(Q, K, V, num_heads)
    
    print(f"\n输入 Q, K, V 形状: {Q.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attn_weights.shape}")
    
    for head_idx in range(num_heads):
        weights = attn_weights[0, head_idx].numpy()
        print_self_attention_matrix(tokens, weights, f"头 {head_idx + 1}")
    
    # 演示带掩码的计算
    print("\n" + "=" * 70)
    print("演示: 带因果掩码的多头注意力")
    print("=" * 70)
    
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    print(f"\n因果掩码形状: {causal_mask.shape}")
    
    output_masked, attn_weights_masked = multi_head_scaled_dot_product_attention(
        Q, K, V, num_heads, mask=causal_mask
    )
    
    for head_idx in range(num_heads):
        weights = attn_weights_masked[0, head_idx].numpy()
        print_self_attention_matrix(tokens, weights, f"头 {head_idx + 1} (带因果掩码)")
    
    print("\n  → 所有头的右上角都是 0")
    
    # 演示交叉注意力
    print("\n" + "=" * 70)
    print("演示: 多头交叉注意力 (Q 和 K/V 长度不同)")
    print("=" * 70)
    
    src_tokens = ["I", "love", "you"]
    tgt_tokens = ["我", "爱"]
    
    Q_cross = torch.randn(batch_size, 2, embed_size)  # 目标: 2个词
    K_cross = torch.randn(batch_size, 3, embed_size)  # 源: 3个词
    V_cross = K_cross
    
    output_cross, attn_weights_cross = multi_head_scaled_dot_product_attention(
        Q_cross, K_cross, V_cross, num_heads
    )
    
    print(f"\nQ 形状: {Q_cross.shape} (目标序列)")
    print(f"K, V 形状: {K_cross.shape} (源序列)")
    print(f"输出形状: {output_cross.shape}")
    print(f"注意力权重形状: {attn_weights_cross.shape}")
    
    for head_idx in range(num_heads):
        weights = attn_weights_cross[0, head_idx].numpy()
        print_attention_matrix(tgt_tokens, src_tokens, weights, 
                               f"头 {head_idx + 1} (交叉注意力)", "目标", "源")
    
    # 总结
    print("\n" + "=" * 70)
    print("模块结构总结")
    print("=" * 70)
    print("""
                    ┌─────────────────────────────────────────┐
                    │   multi_head_attention_core.py          │
                    │                                         │
                    │  • split_heads()                        │
                    │  • merge_heads()                        │
                    │  • multi_head_scaled_dot_product_attention() │
                    └───────────────┬─────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│ multi_head_       │   │ multi_head_       │   │ multi_head_masked │
│ self_attention.py │   │ cross_attention.py│   │ _self_attention.py│
└───────────────────┘   └───────────────────┘   └───────────────────┘
""")

