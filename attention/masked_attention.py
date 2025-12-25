import torch
import torch.nn as nn
from scaled_dot_product_attention import scaled_dot_product_attention
from utils import print_mask_matrix, get_display_width, pad_to_width


def create_causal_mask(seq_len):
    """
    创建因果掩码 (Causal Mask)，用于自回归模型。
    
    作用: 防止位置 i 关注位置 j (j > i)，即不能"看到未来"
    
    参数:
        seq_len: 序列长度
        
    返回:
        mask: 下三角矩阵 (seq_len, seq_len)，1=可见，0=屏蔽
    """
    return torch.tril(torch.ones(seq_len, seq_len))


def create_padding_mask(seq_lens, max_len):
    """
    创建填充掩码 (Padding Mask)，用于屏蔽填充位置。
    
    作用: 批次中不同序列长度不同，短序列需要填充，填充位置不应被关注
    
    参数:
        seq_lens: 每个序列的实际长度，形状 (batch_size,)
        max_len: 填充后的最大长度
        
    返回:
        mask: 形状 (batch_size, 1, max_len)，1=有效，0=填充
    """
    batch_size = len(seq_lens)
    mask = torch.zeros(batch_size, max_len)
    for i, length in enumerate(seq_lens):
        mask[i, :length] = 1
    return mask.unsqueeze(1)  # (batch_size, 1, max_len)


def create_combined_mask(seq_lens, max_len):
    """
    创建组合掩码，同时应用因果掩码和填充掩码。
    
    用于: Decoder 的自注意力层（既要防止看到未来，又要屏蔽填充）
    
    参数:
        seq_lens: 每个序列的实际长度
        max_len: 填充后的最大长度
        
    返回:
        mask: 形状 (batch_size, max_len, max_len)
    """
    causal = create_causal_mask(max_len)  # (max_len, max_len)
    padding = create_padding_mask(seq_lens, max_len)  # (batch_size, 1, max_len)
    # 广播相乘: causal 会扩展到每个 batch
    return causal.unsqueeze(0) * padding  # (batch_size, max_len, max_len)


class MaskedAttention(nn.Module):
    def __init__(self, embed_size):
        """
        带掩码的单头注意力机制。
        
        参数:
            embed_size: 嵌入维度 (d_model)
        """
        super(MaskedAttention, self).__init__()
        self.embed_size = embed_size

        # 定义线性层，用于生成 Q, K, V
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, mask=None):
        """
        前向传播函数。
        
        参数:
            q: 查询输入 (batch_size, seq_len_q, embed_size)
            k: 键输入 (batch_size, seq_len_k, embed_size)
            v: 值输入 (batch_size, seq_len_v, embed_size)
            mask: 掩码矩阵，1=可见，0=屏蔽

        返回:
            out: 注意力加权后的输出
            attention_weights: 注意力权重矩阵
        """
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        out, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        return out, attention_weights


if __name__ == "__main__":
    print("=" * 60)
    print("           带掩码的注意力机制 (Masked Attention)")
    print("=" * 60)
    
    print("\n为什么需要掩码？")
    print("  1. 【因果掩码】生成文本时，不能偷看后面的词")
    print("  2. 【填充掩码】批处理时，不应关注填充的 <PAD> 位置")
    
    torch.manual_seed(42)
    
    # ==================== 因果掩码 ====================
    print("\n" + "=" * 60)
    print("一、因果掩码 (Causal Mask)")
    print("=" * 60)
    print("\n场景: GPT 等自回归模型生成文本")
    print("  → 生成第 i 个词时，只能看到前 i-1 个词")
    print("  → 防止'作弊'偷看答案")
    
    seq_len = 4
    causal_mask = create_causal_mask(seq_len)
    
    print(f"\n因果掩码 (seq_len={seq_len}):")
    print("  行=查询位置, 列=可关注的键位置")
    print("  1=可见, 0=屏蔽")
    print()
    
    # 可视化
    tokens = ["词1", "词2", "词3", "词4"]
    print_mask_matrix(tokens, causal_mask.numpy(), "因果掩码可视化")
    
    print("\n解读:")
    print("  → 词1 只能看自己")
    print("  → 词2 能看 词1, 词2")
    print("  → 词3 能看 词1, 词2, 词3")
    print("  → 词4 能看所有词")
    
    # 因果掩码下的注意力
    print("\n" + "-" * 60)
    print("因果掩码的注意力计算示例:")
    print("-" * 60)
    
    batch_size = 1
    embed_size = 4
    x = torch.randn(batch_size, seq_len, embed_size)
    
    attention = MaskedAttention(embed_size)
    
    with torch.no_grad():
        # 无掩码
        _, weights_no_mask = attention(x, x, x, mask=None)
        # 有因果掩码
        _, weights_causal = attention(x, x, x, mask=causal_mask)
    
    print("\n无掩码时的注意力权重:")
    print(weights_no_mask[0].numpy().round(3))
    
    print("\n有因果掩码时的注意力权重:")
    print(weights_causal[0].numpy().round(3))
    print("  → 右上角全为 0 (未来位置被屏蔽)")
    
    # ==================== 填充掩码 ====================
    print("\n" + "=" * 60)
    print("二、填充掩码 (Padding Mask)")
    print("=" * 60)
    print("\n场景: 批处理不同长度的句子")
    print("  → 短句子用 <PAD> 填充到相同长度")
    print("  → 计算注意力时不应关注 <PAD>")
    
    print("\n示例: 两个句子")
    print("  句子1: ['我', '爱', '学习', '<PAD>']  实际长度=3")
    print("  句子2: ['你', '好', '<PAD>', '<PAD>']  实际长度=2")
    
    seq_lens = [3, 2]  # 实际长度
    max_len = 4        # 填充后长度
    
    padding_mask = create_padding_mask(seq_lens, max_len)
    
    print(f"\n填充掩码 (batch_size=2, max_len={max_len}):")
    print("  1=有效词, 0=填充位置")
    for i, mask in enumerate(padding_mask):
        print(f"  句子{i+1}: {mask[0].tolist()}")
    
    # 填充掩码下的注意力
    print("\n" + "-" * 60)
    print("填充掩码的注意力计算:")
    print("-" * 60)
    
    batch_x = torch.randn(2, max_len, embed_size)
    
    with torch.no_grad():
        _, weights_padding = attention(batch_x, batch_x, batch_x, mask=padding_mask)
    
    print("\n句子1 的注意力权重 (只有最后1列是0):")
    print(weights_padding[0].numpy().round(3))
    
    print("\n句子2 的注意力权重 (最后2列是0):")
    print(weights_padding[1].numpy().round(3))
    print("  → <PAD> 位置的注意力权重为 0")
    
    # ==================== 组合掩码 ====================
    print("\n" + "=" * 60)
    print("三、组合掩码 (Causal + Padding)")
    print("=" * 60)
    print("\n场景: Decoder 的自注意力层")
    print("  → 既要防止看到未来 (因果)")
    print("  → 又要屏蔽填充位置 (填充)")
    
    combined_mask = create_combined_mask(seq_lens, max_len)
    
    print("\n组合掩码:")
    print("\n句子1 (实际长度=3):")
    print(combined_mask[0].numpy().astype(int))
    
    print("\n句子2 (实际长度=2):")
    print(combined_mask[1].numpy().astype(int))
    print("  → 下三角 (因果) + 最后几列为0 (填充)")
    
    # 组合掩码下的注意力
    print("\n" + "-" * 60)
    print("组合掩码的注意力计算:")
    print("-" * 60)
    
    with torch.no_grad():
        _, weights_combined = attention(batch_x, batch_x, batch_x, mask=combined_mask)
    
    print("\n句子1 的注意力权重:")
    print(weights_combined[0].numpy().round(3))
    
    print("\n句子2 的注意力权重:")
    print(weights_combined[1].numpy().round(3))
    
    # ==================== 总结 ====================
    print("\n" + "=" * 60)
    print("总结: 掩码在 Transformer 中的应用")
    print("=" * 60)
    
    print("""
┌─────────────────┬──────────────────┬─────────────────────────┐
│     组件        │    掩码类型      │          作用           │
├─────────────────┼──────────────────┼─────────────────────────┤
│ Encoder         │    Padding       │ 屏蔽 <PAD> 位置         │
│ Self-Attention  │                  │                         │
├─────────────────┼──────────────────┼─────────────────────────┤
│ Decoder         │ Causal + Padding │ 防止看未来 + 屏蔽填充   │
│ Self-Attention  │                  │                         │
├─────────────────┼──────────────────┼─────────────────────────┤
│ Decoder         │    Padding       │ 屏蔽 Encoder 的填充位置 │
│ Cross-Attention │  (Encoder侧)     │                         │
└─────────────────┴──────────────────┴─────────────────────────┘
""")

