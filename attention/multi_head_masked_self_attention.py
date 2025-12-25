import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_head_attention_core import multi_head_scaled_dot_product_attention
from utils import print_self_attention_matrix, print_mask_matrix


def create_causal_mask(seq_len):
    """
    创建因果掩码 (下三角矩阵)
    
    作用: 位置 i 只能关注位置 0, 1, ..., i
    """
    return torch.tril(torch.ones(seq_len, seq_len))


class MultiHeadMaskedSelfAttention(nn.Module):
    """
    多头带掩码的自注意力机制 (Multi-Head Masked Self-Attention)
    
    核心特点:
        - 自注意力: Q, K, V 来自同一输入
        - 多头: 从多个角度理解序列
        - 因果掩码: 防止看到未来位置（用于自回归生成）
    
    用途: Transformer Decoder 的第一个注意力层，GPT 系列模型
    
    例子: 生成句子 "今天天气真好"
        - 生成 "天" 时，只能看到 "今"
        - 生成 "气" 时，只能看到 "今天天"
        - 多个头可以从不同角度学习这种受限的依赖关系
    """
    
    def __init__(self, embed_size, num_heads):
        """
        参数:
            embed_size: 嵌入维度 (d_model)
            num_heads: 注意力头的数量
        """
        super(MultiHeadMaskedSelfAttention, self).__init__()
        
        assert embed_size % num_heads == 0, \
            f"embed_size ({embed_size}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)
        self.w_o = nn.Linear(embed_size, embed_size)
    
    def forward(self, x, mask=None):
        """
        前向传播。
        
        参数:
            x: 输入序列 (batch_size, seq_len, embed_size)
            mask: 掩码矩阵，如果为 None 则自动生成因果掩码

        返回:
            out: 注意力输出 (batch_size, seq_len, embed_size)
            attention_weights: 每个头的注意力权重 (batch_size, num_heads, seq_len, seq_len)
        """
        seq_len = x.size(1)
        
        # 如果没有提供掩码，自动生成因果掩码
        if mask is None:
            mask = create_causal_mask(seq_len).to(x.device)
        
        # 1. 线性变换
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        # 2. 多头缩放点积注意力 (使用公共实现)
        context, attention_weights = multi_head_scaled_dot_product_attention(
            Q, K, V, self.num_heads, mask
        )
        
        # 3. 输出线性变换
        out = self.w_o(context)
        
        return out, attention_weights


if __name__ == "__main__":
    print("=" * 70)
    print("   多头带掩码的自注意力 (Multi-Head Masked Self-Attention)")
    print("=" * 70)
    
    print("\n核心特点:")
    print("  → 自注意力: Q, K, V 来自同一序列")
    print("  → 多头: 从多个角度学习依赖关系")
    print("  → 因果掩码: 每个位置只能看到自己和之前的位置")
    
    print("\n应用场景:")
    print("  → GPT 系列模型的核心组件")
    print("  → Transformer Decoder 的第一个注意力层")
    
    print("""
示意图:

             输入 x
                │
       ┌────────┼────────┐
       ↓        ↓        ↓
      W_q      W_k      W_v
       │        │        │
       ↓        ↓        ↓
  ┌─────────────────────────┐
  │         分头            │
  └─────────────────────────┘
       │        │        │
  ┌────┴────────┴────────┴────┐
  │   头1    头2    头3  ...  │  ← 每个头独立计算
  │    ↓      ↓      ↓        │
  │ ┌─────────────────────┐   │
  │ │    因果掩码 Mask    │   │  ← 所有头共享同一个掩码
  │ │  ┌───────────────┐  │   │
  │ │  │ 1  0  0  0    │  │   │
  │ │  │ 1  1  0  0    │  │   │
  │ │  │ 1  1  1  0    │  │   │
  │ │  │ 1  1  1  1    │  │   │
  │ │  └───────────────┘  │   │
  │ └─────────────────────┘   │
  └───────────────────────────┘
                │
       ┌────────┴────────┐
       │   拼接 Concat   │
       └────────┬────────┘
                ↓
               W_o
                │
                ↓
              输出
""")
    
    # 设置参数
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 5
    embed_size = 8
    num_heads = 2
    
    tokens = ["<BOS>", "今", "天", "天气", "好"]
    
    print("-" * 70)
    print("示例: 文本生成任务")
    print("-" * 70)
    print(f"目标序列: {tokens}")
    print(f"参数: embed_size={embed_size}, num_heads={num_heads}, head_dim={embed_size // num_heads}")
    
    x = torch.randn(batch_size, seq_len, embed_size)
    print(f"\n输入形状: {x.shape}")
    
    # 创建多头带掩码的自注意力层
    mhmsa = MultiHeadMaskedSelfAttention(embed_size, num_heads)
    
    # 展示因果掩码
    print("\n" + "=" * 70)
    print("因果掩码 (所有头共享)")
    print("=" * 70)
    
    causal_mask = create_causal_mask(seq_len)
    print_mask_matrix(tokens, causal_mask.numpy(), "因果掩码 (下三角矩阵)")
    
    # 计算注意力
    print("\n" + "=" * 70)
    print("计算多头带掩码的自注意力")
    print("=" * 70)
    
    with torch.no_grad():
        output, attn_weights = mhmsa(x)
    
    print(f"\n注意力权重形状: {attn_weights.shape}")
    print(f"  → (batch={batch_size}, heads={num_heads}, seq={seq_len}, seq={seq_len})")
    
    # 展示每个头的注意力权重
    for head_idx in range(num_heads):
        weights = attn_weights[0, head_idx].numpy()
        print_self_attention_matrix(tokens, weights, 
                                    f"头 {head_idx + 1} 的注意力权重 (带因果掩码)")
    
    print("\n验证: 每个头的右上角都是 0（未来位置被屏蔽）")
    
    # 对比不同头的关注模式
    print("\n" + "=" * 70)
    print("分析: 不同头的关注模式")
    print("=" * 70)
    
    for head_idx in range(num_heads):
        print(f"\n【头 {head_idx + 1}】各位置最关注的词:")
        weights = attn_weights[0, head_idx].numpy()
        for i, token in enumerate(tokens):
            # 只考虑可见的位置 (0 到 i)
            visible_weights = weights[i, :i+1]
            max_idx = visible_weights.argmax()
            max_weight = visible_weights[max_idx]
            print(f"  「{token}」→「{tokens[max_idx]}」(权重={max_weight:.2f})")
    
    # 无掩码 vs 有掩码
    print("\n" + "=" * 70)
    print("对比: 无掩码 vs 有掩码")
    print("=" * 70)
    
    # 无掩码
    no_mask = torch.ones(seq_len, seq_len)
    with torch.no_grad():
        _, weights_no_mask = mhmsa(x, mask=no_mask)
    
    print("\n【头 1 - 无掩码】")
    w_no = weights_no_mask[0, 0].numpy()
    print_self_attention_matrix(tokens, w_no, "无掩码时的注意力权重")
    
    print("\n【头 1 - 有因果掩码】")
    w_causal = attn_weights[0, 0].numpy()
    print_self_attention_matrix(tokens, w_causal, "有因果掩码时的注意力权重")
    
    print("\n关键区别:")
    print("  → 无掩码: 每个位置可以看到所有位置")
    print("  → 有掩码: 每个位置只能看到自己和之前的位置")
    
    # 自回归生成模拟
    print("\n" + "=" * 70)
    print("模拟自回归生成过程")
    print("=" * 70)
    
    print("\n训练时:")
    print("  → 一次性处理整个序列 (并行)")
    print("  → 使用因果掩码模拟逐个生成")
    
    print("\n推理时:")
    print("  → 序列长度逐渐增加")
    print("  → 每一步只预测一个 token")
    
    print("\n" + "-" * 70)
    for step in range(1, min(4, seq_len) + 1):
        current_tokens = tokens[:step]
        x_step = x[:, :step, :]
        
        with torch.no_grad():
            _, weights_step = mhmsa(x_step)
        
        print(f"\n第 {step} 步 - 已生成: {current_tokens}")
        print(f"  注意力权重形状: {weights_step.shape}")
        print(f"  最后一个位置「{tokens[step-1]}」的注意力分布 (头1):")
        w = weights_step[0, 0, -1, :].numpy()
        for j, weight in enumerate(w):
            bar = "█" * int(weight * 15)
            print(f"    {current_tokens[j]:>5}: {weight:.2f} {bar}")
    
    # 与 GPT 的关系
    print("\n" + "=" * 70)
    print("与 GPT 模型的关系")
    print("=" * 70)
    print("""
GPT 模型结构:

┌─────────────────────────────────────────────────────────────────┐
│                          GPT Block × N                          │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │    Multi-Head Masked Self-Attention                 │  │  │
│  │  │    (本脚本实现的模块)                                 │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                          ↓                                │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │    Layer Normalization + Residual Connection        │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                          ↓                                │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │    Feed Forward Network                             │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                          ↓                                │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │    Layer Normalization + Residual Connection        │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

GPT-1:  12 层, 768 维, 12 头
GPT-2:  48 层, 1600 维, 25 头 (最大版本)
GPT-3:  96 层, 12288 维, 96 头
""")
    
    # 总结表格
    print("=" * 70)
    print("总结: Transformer 中的多头注意力")
    print("=" * 70)
    print("""
┌───────────────────────┬───────────┬───────────┬────────────────┐
│        类型           │  Q 来源   │  K,V 来源 │     掩码       │
├───────────────────────┼───────────┼───────────┼────────────────┤
│ Multi-Head            │           │           │                │
│ Self-Attention        │     x     │     x     │   Padding      │
│ (Encoder)             │           │           │                │
├───────────────────────┼───────────┼───────────┼────────────────┤
│ Multi-Head Masked     │           │           │                │
│ Self-Attention        │     x     │     x     │ Causal+Padding │
│ (Decoder / GPT)       │           │           │                │
├───────────────────────┼───────────┼───────────┼────────────────┤
│ Multi-Head            │  decoder  │  encoder  │                │
│ Cross-Attention       │   输入    │   输出    │ Padding(enc)   │
│ (Decoder)             │           │           │                │
└───────────────────────┴───────────┴───────────┴────────────────┘
""")

