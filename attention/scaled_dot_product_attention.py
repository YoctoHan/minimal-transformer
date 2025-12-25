import torch
import torch.nn.functional as F
import math
from utils import print_matrix


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力计算。
    
    公式: Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
    
    参数:
        Q: 查询矩阵 (seq_len_q, d_k)
        K: 键矩阵 (seq_len_k, d_k)
        V: 值矩阵 (seq_len_v, d_v)
        mask: 掩码矩阵，用于屏蔽不应该关注的位置 (可选)

    返回:
        output: 注意力加权后的输出矩阵
        attention_weights: 注意力权重矩阵
    """
    d_k = Q.size(-1)  # 键的维度
    
    # 第1步: 计算 Q @ K^T (点积)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 第2步: 缩放 (除以 √d_k)
    scores = scores / math.sqrt(d_k)

    # 如果提供了掩码矩阵，则将掩码对应位置的分数设为 -inf
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 第3步: Softmax (将分数转换为概率分布)
    attention_weights = F.softmax(scores, dim=-1)

    # 第4步: 加权求和
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights


if __name__ == "__main__":
    print("=" * 60)
    print("        缩放点积注意力 (Scaled Dot-Product Attention)")
    print("=" * 60)
    print("\n公式: Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V")
    print("\n想象一个场景：你在图书馆找书")
    print("  - Q (Query): 你想找的书的描述 '我想要一本关于Python的书'")
    print("  - K (Key):   每本书的标签 '编程'、'烹饪'、'历史'...")  
    print("  - V (Value): 每本书的实际内容")
    print("  - 注意力权重: 每本书与你需求的匹配程度")
    
    print("\n" + "-" * 60)
    print("示例：3个词，每个词用4维向量表示")
    print("-" * 60)
    
    # 使用简单的维度便于理解
    seq_len = 3  # 3个词/token
    d_k = 4      # 每个词用4维向量表示
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 创建 Q, K, V 矩阵 (实际中这些是通过线性变换从输入得到的)
    Q = torch.randn(seq_len, d_k)
    K = torch.randn(seq_len, d_k)
    V = torch.randn(seq_len, d_k)
    
    print_matrix("Q (查询矩阵)", Q, f"形状: {seq_len}个词 × {d_k}维")
    print_matrix("K (键矩阵)", K, f"形状: {seq_len}个词 × {d_k}维")
    print_matrix("V (值矩阵)", V, f"形状: {seq_len}个词 × {d_k}维")
    
    # ========== 第1步: 计算点积 ==========
    print("\n" + "=" * 60)
    print("第1步: 计算 Q @ K^T (点积)")
    print("=" * 60)
    print("每个位置 [i,j] 表示: 第i个查询与第j个键的相似度")
    
    scores_raw = torch.matmul(Q, K.transpose(-2, -1))
    print_matrix("Q @ K^T", scores_raw, f"形状: {seq_len} × {seq_len}")
    
    # ========== 第2步: 缩放 ==========
    print("\n" + "=" * 60)
    print(f"第2步: 缩放 (除以 √d_k = √{d_k} = {math.sqrt(d_k):.2f})")
    print("=" * 60)
    print("为什么要缩放？")
    print("  → 当 d_k 很大时，点积值会很大，导致 softmax 梯度很小")
    print("  → 缩放可以让训练更稳定")
    
    scores_scaled = scores_raw / math.sqrt(d_k)
    print_matrix("缩放后的分数", scores_scaled)
    
    # ========== 第3步: Softmax ==========
    print("\n" + "=" * 60)
    print("第3步: Softmax (将分数转换为概率分布)")
    print("=" * 60)
    print("每一行经过 softmax 后，所有值加起来 = 1")
    print("  → 表示当前词对其他词的'注意力分配'")
    
    attention_weights = F.softmax(scores_scaled, dim=-1)
    print_matrix("注意力权重", attention_weights)
    
    # 验证每行和为1
    row_sums = attention_weights.sum(dim=-1)
    print(f"\n验证每行之和: {row_sums}")
    print("  → 每行确实约等于 1.0 ✓")
    
    # ========== 第4步: 加权求和 ==========
    print("\n" + "=" * 60)
    print("第4步: 加权求和 (attention_weights @ V)")
    print("=" * 60)
    print("用注意力权重对 V 进行加权平均")
    print("  → 权重大的词贡献更多，权重小的词贡献更少")
    
    output = torch.matmul(attention_weights, V)
    print_matrix("最终输出", output, f"形状: {seq_len} × {d_k}")
    
    # ========== 使用封装好的函数验证 ==========
    print("\n" + "=" * 60)
    print("使用封装函数验证结果")
    print("=" * 60)
    
    output_func, weights_func = scaled_dot_product_attention(Q, K, V)
    print(f"\n结果一致: {torch.allclose(output, output_func)} ✓")
    
    # ========== 可选：掩码示例 ==========
    print("\n" + "=" * 60)
    print("[进阶] 因果掩码 (Causal Mask)")
    print("=" * 60)
    print("在语言模型中，生成第i个词时只能看到前面的词")
    print("通过掩码实现：把'未来'位置的分数设为 -∞，softmax 后变成 0")
    
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    print_matrix("因果掩码 (下三角矩阵)", causal_mask, "1=可见, 0=屏蔽")
    
    _, weights_masked = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    print_matrix("带掩码的注意力权重", weights_masked, "注意右上角都是0")
