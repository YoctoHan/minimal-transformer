import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaled_dot_product_attention import scaled_dot_product_attention
from utils import print_self_attention_matrix, print_mask_matrix


def create_causal_mask(seq_len):
    """
    创建因果掩码 (下三角矩阵)
    
    作用: 位置 i 只能关注位置 0, 1, ..., i (不能看到未来)
    """
    return torch.tril(torch.ones(seq_len, seq_len))


class MaskedSelfAttention(nn.Module):
    """
    带掩码的自注意力机制 (Masked Self-Attention)
    
    特点: 
        - Q, K, V 来自同一个输入 (自注意力)
        - 使用因果掩码，防止看到未来位置 (掩码)
    
    用途: Decoder 的自注意力层，用于自回归生成
    
    例子: 生成 "我爱学习"
          → 生成 "学" 时，只能看到 "我"、"爱"，不能偷看 "习"
    """
    
    def __init__(self, embed_size):
        """
        参数:
            embed_size: 嵌入维度 (d_model)
        """
        super(MaskedSelfAttention, self).__init__()
        self.embed_size = embed_size

        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        """
        前向传播。
        
        参数:
            x: 输入序列 (batch_size, seq_len, embed_size)
            mask: 掩码矩阵，如果为 None 则自动生成因果掩码

        返回:
            out: 注意力输出
            attention_weights: 注意力权重
        """
        seq_len = x.size(1)
        
        # 如果没有提供掩码，自动生成因果掩码
        if mask is None:
            mask = create_causal_mask(seq_len).to(x.device)
        
        # 自注意力: Q, K, V 都来自 x
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        out, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        return out, attention_weights


if __name__ == "__main__":
    print("=" * 60)
    print("       带掩码的自注意力 (Masked Self-Attention)")
    print("=" * 60)
    
    print("\n核心特点:")
    print("  → 自注意力: Q, K, V 来自同一序列")
    print("  → 因果掩码: 每个位置只能看到自己和之前的位置")
    
    print("\n为什么需要掩码?")
    print("  → 自回归生成时，不能让模型'作弊'偷看答案")
    print("  → 训练时一次性处理整个序列，但要模拟逐个生成的过程")
    
    print("\n应用场景:")
    print("  → GPT 系列模型")
    print("  → Transformer Decoder 的自注意力层")
    
    print("""
对比示意图:

【Self-Attention (无掩码)】          【Masked Self-Attention (有掩码)】
  
  每个词可以看到所有词:                每个词只能看到之前的词:
  
     词1 词2 词3 词4                      词1 词2 词3 词4
词1   ✓   ✓   ✓   ✓                  词1   ✓   ✗   ✗   ✗
词2   ✓   ✓   ✓   ✓                  词2   ✓   ✓   ✗   ✗
词3   ✓   ✓   ✓   ✓                  词3   ✓   ✓   ✓   ✗
词4   ✓   ✓   ✓   ✓                  词4   ✓   ✓   ✓   ✓
""")
    
    # 设置参数
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 5
    embed_size = 8
    
    # 模拟生成任务
    tokens = ["<BOS>", "今", "天", "天气", "好"]
    
    print("-" * 60)
    print("示例: 文本生成任务")
    print("-" * 60)
    print(f"目标序列: {tokens}")
    print("生成过程 (自回归):")
    print("  第1步: 输入 <BOS> → 预测 '今'")
    print("  第2步: 输入 <BOS>今 → 预测 '天'")
    print("  第3步: 输入 <BOS>今天 → 预测 '天气'")
    print("  第4步: 输入 <BOS>今天天气 → 预测 '好'")
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, embed_size)
    
    print(f"\n输入形状: {x.shape}")
    
    # 创建带掩码的自注意力层
    masked_self_attn = MaskedSelfAttention(embed_size)
    
    # 展示因果掩码
    print("\n" + "=" * 60)
    print("因果掩码 (Causal Mask)")
    print("=" * 60)
    
    causal_mask = create_causal_mask(seq_len)
    
    print_mask_matrix(tokens, causal_mask.numpy(), "下三角矩阵 (因果掩码)")
    
    # 计算注意力
    print("\n" + "=" * 60)
    print("计算带掩码的自注意力")
    print("=" * 60)
    
    with torch.no_grad():
        output, attn_weights = masked_self_attn(x)
    
    weights = attn_weights[0].numpy()
    print_self_attention_matrix(tokens, weights, "注意力权重 (应用掩码后)")
    
    print("\n验证:")
    print("  → 右上角全为 0 (未来位置被屏蔽)")
    print("  → 每行的有效权重之和 = 1")
    for i, token in enumerate(tokens):
        valid_sum = weights[i, :i+1].sum()
        print(f"     {token}: 有效权重和 = {valid_sum:.3f}")
    
    # 对比有无掩码
    print("\n" + "=" * 60)
    print("对比: 有掩码 vs 无掩码")
    print("=" * 60)
    
    # 无掩码的普通自注意力
    no_mask = torch.ones(seq_len, seq_len)
    with torch.no_grad():
        _, weights_no_mask = masked_self_attn(x, mask=no_mask)
    
    w_no = weights_no_mask[0].numpy()
    print_self_attention_matrix(tokens, w_no, "【无掩码】注意力权重")
    print_self_attention_matrix(tokens, weights, "【有掩码】注意力权重")
    
    # 逐步生成模拟
    print("\n" + "=" * 60)
    print("模拟自回归生成过程")
    print("=" * 60)
    print("\n在推理时，序列长度逐渐增加:")
    
    for step in range(1, seq_len + 1):
        current_tokens = tokens[:step]
        x_step = x[:, :step, :]
        
        with torch.no_grad():
            _, weights_step = masked_self_attn(x_step)
        
        print(f"\n第{step}步 - 输入: {current_tokens}")
        print(f"  注意力权重形状: {weights_step.shape}")
        print(f"  最后一个位置 '{tokens[step-1]}' 的注意力分布:")
        w = weights_step[0, -1, :].numpy()
        for j, weight in enumerate(w):
            bar = "█" * int(weight * 20)
            print(f"    {current_tokens[j]:>4}: {weight:.2f} {bar}")
    
    # 总结
    print("\n" + "=" * 60)
    print("总结: Transformer 中的三种注意力")
    print("=" * 60)
    print("""
┌─────────────────────┬────────────┬────────────┬────────────┐
│                     │     Q      │    K, V    │    掩码    │
├─────────────────────┼────────────┼────────────┼────────────┤
│  Self-Attention     │     x      │     x      │    无/Pad  │
│  (Encoder)          │            │            │            │
├─────────────────────┼────────────┼────────────┼────────────┤
│  Masked Self-Attn   │     x      │     x      │ Causal+Pad │
│  (Decoder)          │            │            │            │
├─────────────────────┼────────────┼────────────┼────────────┤
│  Cross-Attention    │  decoder   │  encoder   │  Pad(enc)  │
│  (Decoder)          │   输入     │   输出     │            │
└─────────────────────┴────────────┴────────────┴────────────┘

在 Transformer 中的位置:

Encoder:
  └── Self-Attention (可选 Padding Mask)

Decoder:
  ├── Masked Self-Attention (Causal Mask + Padding Mask)
  └── Cross-Attention (Encoder 的 Padding Mask)
""")

