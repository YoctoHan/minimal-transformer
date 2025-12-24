import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力计算。
    
    参数:
        Q: 查询矩阵 (batch_size, seq_len_q, embed_size)
        K: 键矩阵 (batch_size, seq_len_k, embed_size)
        V: 值矩阵 (batch_size, seq_len_v, embed_size)
        mask: 掩码矩阵，用于屏蔽不应该关注的位置 (可选)

    返回:
        output: 注意力加权后的输出矩阵
        attention_weights: 注意力权重矩阵
    """
    embed_size = Q.size(-1)  # embed_size
    
    # 计算点积并进行缩放
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(embed_size)

    # 如果提供了掩码矩阵，则将掩码对应位置的分数设为 -inf
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 对缩放后的分数应用 Softmax 函数，得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)

    # 加权求和，计算输出
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 定义参数
    batch_size = 2
    seq_len = 4
    embed_size = 8
    
    # 创建随机的 Q, K, V 矩阵
    Q = torch.randn(batch_size, seq_len, embed_size)
    K = torch.randn(batch_size, seq_len, embed_size)
    V = torch.randn(batch_size, seq_len, embed_size)
    
    # 计算注意力
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print("Q shape:", Q.shape)
    print("K shape:", K.shape)
    print("V shape:", V.shape)
    print("\n输出 shape:", output.shape)
    print("注意力权重 shape:", attention_weights.shape)
    print("\n注意力权重 (第一个样本):")
    print(attention_weights[0])
    
    # 使用掩码的示例 (因果掩码，用于自回归模型)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    breakpoint()
    output_masked, attention_weights_masked = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    
    print("\n使用因果掩码后的注意力权重 (第一个样本):")
    print(attention_weights_masked[0])