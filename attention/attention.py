import torch
import torch.nn as nn
from scaled_dot_product_attention import scaled_dot_product_attention

class Attention(nn.Module):
    def __init__(self, embed_size):
        """
        单头注意力机制。
        
        参数:
            embed_size: 输入序列（Inputs）的嵌入（Input Embedding）维度，也是论文中所提到的d_model。
        """
        super(Attention, self).__init__()
        self.embed_size = embed_size

        # 定义线性层，用于生成查询、键和值矩阵
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, mask=None):
        """
        前向传播函数。
        
        参数:
            q: 查询矩阵 (batch_size, seq_len_q, embed_size)
            k: 键矩阵 (batch_size, seq_len_k, embed_size)
            v: 值矩阵 (batch_size, seq_len_v, embed_size)
            mask: 掩码矩阵，用于屏蔽不应关注的位置 (batch_size, seq_len_q, seq_len_k)

        返回:
            out: 注意力加权后的输出
            attention_weights: 注意力权重矩阵
        """
        # 将输入序列通过线性变换生成 Q, K, V
        Q = self.w_q(q)  # (batch_size, seq_len_q, embed_size)
        K = self.w_k(k)  # (batch_size, seq_len_k, embed_size)
        V = self.w_v(v)  # (batch_size, seq_len_v, embed_size)

        # 使用缩放点积注意力函数计算输出和权重
        out, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        return out, attention_weights


if __name__ == "__main__":
    print("=" * 60)
    print("              单头注意力机制 (Single-Head Attention)")
    print("=" * 60)
    
    print("\n与直接使用 Q/K/V 的区别:")
    print("  → 输入 x 通过三个可学习的线性变换 (W_q, W_k, W_v) 生成 Q, K, V")
    print("  → 这让模型能够学习'如何提问'、'如何被检索'、'如何提供信息'")
    print()
    print("  输入 x ──┬── W_q ──→ Q (查询)")
    print("           ├── W_k ──→ K (键)")
    print("           └── W_v ──→ V (值)")
    print()
    print("  然后: Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V")
    
    # 设置参数
    torch.manual_seed(42)
    batch_size = 1   # 简化为单样本
    seq_len = 3      # 3个词
    embed_size = 4   # 嵌入维度 (d_model)
    
    print("\n" + "-" * 60)
    print(f"示例参数: batch_size={batch_size}, seq_len={seq_len}, embed_size={embed_size}")
    print("-" * 60)
    
    # 创建输入序列 (假设已经过 embedding 层)
    x = torch.randn(batch_size, seq_len, embed_size)
    print(f"\n输入 x (已嵌入的序列):")
    print(f"  → 形状: ({batch_size}, {seq_len}, {embed_size}) = (batch, 词数, 嵌入维度)")
    print(x[0])  # 只显示第一个样本
    
    # 创建注意力层
    attention = Attention(embed_size)
    
    # 查看线性变换的权重
    print("\n" + "=" * 60)
    print("线性变换权重 (可学习参数)")
    print("=" * 60)
    print(f"\nW_q 权重形状: {attention.w_q.weight.shape}")
    print(f"W_k 权重形状: {attention.w_k.weight.shape}")
    print(f"W_v 权重形状: {attention.w_v.weight.shape}")
    print("  → 每个权重矩阵: (embed_size, embed_size)")
    print("  → 训练过程中这些权重会被优化")
    
    # 展示线性变换过程
    print("\n" + "=" * 60)
    print("第1步: 线性变换生成 Q, K, V")
    print("=" * 60)
    
    with torch.no_grad():
        Q = attention.w_q(x)
        K = attention.w_k(x)
        V = attention.w_v(x)
    
    print("\nQ = x @ W_q^T + b_q")
    print(f"Q 形状: {Q.shape}")
    print(Q[0])
    
    print("\nK = x @ W_k^T + b_k")
    print(f"K 形状: {K.shape}")
    print(K[0])
    
    print("\nV = x @ W_v^T + b_v")
    print(f"V 形状: {V.shape}")
    print(V[0])
    
    # 计算注意力
    print("\n" + "=" * 60)
    print("第2步: 缩放点积注意力")
    print("=" * 60)
    print("使用 scaled_dot_product_attention(Q, K, V)")
    
    with torch.no_grad():
        output, attention_weights = attention(x, x, x)  # 自注意力: q=k=v=x
    
    print(f"\n注意力权重:")
    print(f"  → 形状: {attention_weights.shape}")
    print(attention_weights[0])
    
    print(f"\n输出:")
    print(f"  → 形状: {output.shape} (与输入相同)")
    print(output[0])
    
    # 自注意力 vs 交叉注意力
    print("\n" + "=" * 60)
    print("自注意力 (Self-Attention) vs 交叉注意力 (Cross-Attention)")
    print("=" * 60)
    
    print("\n【自注意力】q = k = v = 同一序列")
    print("  → 用途: Encoder, Decoder 的自注意力层")
    print("  → 示例: attention(x, x, x)")
    print("  → 每个词关注同一序列中的其他词")
    
    print("\n【交叉注意力】q ≠ k = v")
    print("  → 用途: Decoder 中关注 Encoder 输出")
    print("  → 示例: attention(decoder_x, encoder_out, encoder_out)")
    print("  → Decoder 的每个词关注 Encoder 的所有词")
    
    # 交叉注意力示例
    print("\n" + "-" * 60)
    print("交叉注意力示例:")
    print("-" * 60)
    
    encoder_out = torch.randn(batch_size, 5, embed_size)  # Encoder输出: 5个词
    decoder_x = torch.randn(batch_size, 3, embed_size)    # Decoder输入: 3个词
    
    print(f"Encoder 输出形状: {encoder_out.shape} (5个词)")
    print(f"Decoder 输入形状: {decoder_x.shape} (3个词)")
    
    with torch.no_grad():
        cross_out, cross_weights = attention(decoder_x, encoder_out, encoder_out)
    
    print(f"\n交叉注意力权重形状: {cross_weights.shape}")
    print("  → (1, 3, 5): Decoder的每个词对Encoder的5个词的注意力")
    print(cross_weights[0])
    
    print(f"\n交叉注意力输出形状: {cross_out.shape}")
    print("  → 与 Decoder 输入形状相同")