import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaled_dot_product_attention import scaled_dot_product_attention
from utils import print_self_attention_matrix


class SelfAttention(nn.Module):
    """
    自注意力机制 (Self-Attention)
    
    特点: Q, K, V 都来自同一个输入序列
    用途: Encoder 中捕捉序列内部的依赖关系
    
    例子: 理解句子 "小明喜欢小红，因为她很善良"
          → "她" 需要关注 "小红" 来理解指代关系
    """
    
    def __init__(self, embed_size):
        """
        参数:
            embed_size: 嵌入维度 (d_model)
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size

        # Q, K, V 的线性变换层
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        """
        前向传播。
        
        参数:
            x: 输入序列 (batch_size, seq_len, embed_size)
            mask: 可选的掩码矩阵 (用于屏蔽 padding)

        返回:
            out: 注意力输出 (batch_size, seq_len, embed_size)
            attention_weights: 注意力权重 (batch_size, seq_len, seq_len)
        """
        # 自注意力: Q, K, V 都来自同一个输入 x
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        out, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        return out, attention_weights


if __name__ == "__main__":
    print("=" * 60)
    print("              自注意力机制 (Self-Attention)")
    print("=" * 60)
    
    print("\n核心特点:")
    print("  → Q, K, V 都来自【同一个】输入序列")
    print("  → 让序列中的每个位置都能关注其他所有位置")
    
    print("\n应用场景:")
    print("  → Transformer Encoder 的自注意力层")
    print("  → BERT 等双向语言模型")
    
    print("""
示意图:
                    ┌─────────────────────────────────┐
                    │         Self-Attention          │
                    └─────────────────────────────────┘
                                   ▲
                    ┌──────────────┼──────────────┐
                    │              │              │
                  ┌─┴─┐          ┌─┴─┐          ┌─┴─┐
                  │W_q│          │W_k│          │W_v│
                  └─┬─┘          └─┬─┘          └─┬─┘
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                              ┌────┴────┐
                              │  输入 x  │  ← 同一个输入
                              └─────────┘
""")
    
    # 设置参数
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 4
    embed_size = 8
    
    print("-" * 60)
    print(f"示例参数: batch={batch_size}, seq_len={seq_len}, embed_size={embed_size}")
    print("-" * 60)
    
    # 模拟输入: 一个句子有4个词，每个词是8维向量
    x = torch.randn(batch_size, seq_len, embed_size)
    
    # 给每个位置一个标签便于理解
    tokens = ["我", "喜欢", "机器", "学习"]
    
    print(f"\n输入序列 x: {tokens}")
    print(f"  → 形状: {x.shape}")
    
    # 创建自注意力层
    self_attn = SelfAttention(embed_size)
    
    # 计算自注意力
    print("\n" + "=" * 60)
    print("计算自注意力")
    print("=" * 60)
    
    with torch.no_grad():
        output, attn_weights = self_attn(x)
    
    weights = attn_weights[0].numpy()
    print_self_attention_matrix(tokens, weights)
    
    print("\n解读示例:")
    for i, token in enumerate(tokens):
        max_idx = weights[i].argmax()
        max_weight = weights[i][max_idx]
        print(f"  「{token}」最关注「{tokens[max_idx]}」(权重={max_weight:.2f})")
    
    print(f"\n输出形状: {output.shape}")
    print("  → 与输入形状相同，但每个位置融合了全局信息")
    
    # 与普通注意力的对比
    print("\n" + "=" * 60)
    print("自注意力 vs 普通注意力")
    print("=" * 60)
    print("""
┌─────────────────┬───────────────────────────────────────┐
│                 │              输入来源                 │
│      类型       ├───────────┬───────────┬───────────────┤
│                 │     Q     │     K     │       V       │
├─────────────────┼───────────┼───────────┼───────────────┤
│ Self-Attention  │     x     │     x     │       x       │
│   (自注意力)    │  (同一个) │  (同一个) │    (同一个)   │
├─────────────────┼───────────┼───────────┼───────────────┤
│ Cross-Attention │  decoder  │  encoder  │    encoder    │
│  (交叉注意力)   │   输出    │   输出    │      输出     │
└─────────────────┴───────────┴───────────┴───────────────┘
""")
    
    # Padding Mask 示例
    print("=" * 60)
    print("处理 Padding: 使用掩码屏蔽填充位置")
    print("=" * 60)
    
    tokens_padded = ["我", "喜欢", "<PAD>", "<PAD>"]
    print(f"\n假设序列: {tokens_padded} (实际长度=2)")
    
    # 创建 padding mask
    padding_mask = torch.tensor([[[1, 1, 0, 0]]])  # 1=有效, 0=填充
    
    with torch.no_grad():
        _, attn_weights_masked = self_attn(x, mask=padding_mask)
    
    weights_m = attn_weights_masked[0].numpy()
    print_self_attention_matrix(tokens_padded, weights_m, "带 Padding Mask 的注意力权重")
    
    print("\n  → 最后两列 (<PAD> 位置) 权重为 0")