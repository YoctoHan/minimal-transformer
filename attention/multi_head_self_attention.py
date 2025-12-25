import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_head_attention_core import multi_head_scaled_dot_product_attention
from utils import print_self_attention_matrix


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention)
    
    核心思想:
        - 将注意力分成多个"头"，每个头关注不同的特征子空间
        - 让模型能同时从不同角度理解序列关系
    
    例子: 理解句子 "小猫在沙发上睡觉"
        - 头1 可能关注: 主语-谓语关系 (小猫 → 睡觉)
        - 头2 可能关注: 位置关系 (睡觉 → 沙发上)
        - 头3 可能关注: 修饰关系 (小 → 猫)
    
    公式:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
        其中 head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)
    """
    
    def __init__(self, embed_size, num_heads):
        """
        参数:
            embed_size: 嵌入维度 (d_model)，必须能被 num_heads 整除
            num_heads: 注意力头的数量
        """
        super(MultiHeadSelfAttention, self).__init__()
        
        assert embed_size % num_heads == 0, \
            f"embed_size ({embed_size}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads  # 每个头的维度
        
        # Q, K, V 的线性变换 (一次性计算所有头)
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)
        
        # 输出线性变换
        self.w_o = nn.Linear(embed_size, embed_size)
    
    def forward(self, x, mask=None):
        """
        前向传播。
        
        参数:
            x: 输入序列 (batch_size, seq_len, embed_size)
            mask: 可选的掩码矩阵

        返回:
            out: 注意力输出 (batch_size, seq_len, embed_size)
            attention_weights: 每个头的注意力权重 (batch_size, num_heads, seq_len, seq_len)
        """
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
    print("          多头自注意力机制 (Multi-Head Self-Attention)")
    print("=" * 70)
    
    print("\n为什么需要多头?")
    print("  → 单头注意力只能学习一种关注模式")
    print("  → 多头让模型同时从不同角度理解序列关系")
    print("  → 不同的头可以关注不同类型的依赖关系")
    
    print("""
示意图:

          输入 x
             │
    ┌────────┼────────┐
    ↓        ↓        ↓
   W_q      W_k      W_v
    │        │        │
    ↓        ↓        ↓
  ┌─────────────────────┐
  │       分头          │  ← 将 embed_size 分成 num_heads 份
  │  (split into heads) │
  └─────────────────────┘
    │        │        │
    ↓        ↓        ↓
  ┌───┐    ┌───┐    ┌───┐
  │头1│    │头2│    │...│    ← 每个头独立计算注意力
  └───┘    └───┘    └───┘
    │        │        │
    └────────┼────────┘
             ↓
  ┌─────────────────────┐
  │      拼接 Concat    │
  └─────────────────────┘
             │
             ↓
            W_o         ← 输出线性变换
             │
             ↓
          输出
""")
    
    # 设置参数
    torch.manual_seed(42)
    batch_size = 1
    seq_len = 4
    embed_size = 8
    num_heads = 2  # 2个头，每个头维度为 8/2=4
    
    print("-" * 70)
    print(f"示例参数:")
    print(f"  embed_size = {embed_size}")
    print(f"  num_heads = {num_heads}")
    print(f"  head_dim = embed_size / num_heads = {embed_size // num_heads}")
    print("-" * 70)
    
    tokens = ["我", "喜欢", "机器", "学习"]
    x = torch.randn(batch_size, seq_len, embed_size)
    
    print(f"\n输入序列: {tokens}")
    print(f"输入形状: {x.shape}")
    
    # 创建多头自注意力层
    mha = MultiHeadSelfAttention(embed_size, num_heads)
    
    print("\n" + "=" * 70)
    print("计算多头自注意力")
    print("=" * 70)
    
    with torch.no_grad():
        output, attn_weights = mha(x)
    
    print(f"\n注意力权重形状: {attn_weights.shape}")
    print(f"  → (batch={batch_size}, heads={num_heads}, seq={seq_len}, seq={seq_len})")
    
    # 展示每个头的注意力权重
    for head_idx in range(num_heads):
        weights = attn_weights[0, head_idx].numpy()
        print_self_attention_matrix(tokens, weights, f"头 {head_idx + 1} 的注意力权重")
    
    print(f"\n输出形状: {output.shape}")
    print("  → 与输入形状相同")
    
    # 对比不同头的关注点
    print("\n" + "=" * 70)
    print("分析: 不同头关注不同内容")
    print("=" * 70)
    
    for head_idx in range(num_heads):
        print(f"\n【头 {head_idx + 1}】各词最关注的位置:")
        weights = attn_weights[0, head_idx].numpy()
        for i, token in enumerate(tokens):
            max_idx = weights[i].argmax()
            max_weight = weights[i][max_idx]
            print(f"  「{token}」→「{tokens[max_idx]}」(权重={max_weight:.2f})")
    
    # 单头 vs 多头对比
    print("\n" + "=" * 70)
    print("单头 vs 多头 对比")
    print("=" * 70)
    print("""
┌─────────────────┬─────────────────────┬─────────────────────────────┐
│                 │      单头注意力     │        多头注意力           │
├─────────────────┼─────────────────────┼─────────────────────────────┤
│ 注意力模式      │ 只能学习一种模式    │ 同时学习多种模式            │
├─────────────────┼─────────────────────┼─────────────────────────────┤
│ 参数量          │ 3 × d² + d²        │ 相同 (只是拆分方式不同)     │
│ (W_q,W_k,W_v,W_o)│                    │                             │
├─────────────────┼─────────────────────┼─────────────────────────────┤
│ 每头维度        │ d_model             │ d_model / num_heads         │
├─────────────────┼─────────────────────┼─────────────────────────────┤
│ 计算复杂度      │ O(n² × d)           │ O(n² × d) (总计相同)        │
├─────────────────┼─────────────────────┼─────────────────────────────┤
│ 表达能力        │ 较弱                │ 更强，可捕捉多种关系        │
└─────────────────┴─────────────────────┴─────────────────────────────┘
""")
    
    # 维度变换详解
    print("=" * 70)
    print("维度变换详解 (以本例为例)")
    print("=" * 70)
    print(f"""
输入 x:        ({batch_size}, {seq_len}, {embed_size})
               (batch, seq_len, embed_size)
                              ↓
线性变换后 Q:  ({batch_size}, {seq_len}, {embed_size})
               (batch, seq_len, embed_size)
                              ↓
分头 reshape:  ({batch_size}, {seq_len}, {num_heads}, {embed_size // num_heads})
               (batch, seq_len, num_heads, head_dim)
                              ↓
转置 transpose: ({batch_size}, {num_heads}, {seq_len}, {embed_size // num_heads})
               (batch, num_heads, seq_len, head_dim)
                              ↓
注意力计算后:  ({batch_size}, {num_heads}, {seq_len}, {embed_size // num_heads})
               (batch, num_heads, seq_len, head_dim)
                              ↓
转置回来:      ({batch_size}, {seq_len}, {num_heads}, {embed_size // num_heads})
               (batch, seq_len, num_heads, head_dim)
                              ↓
合并 reshape:  ({batch_size}, {seq_len}, {embed_size})
               (batch, seq_len, embed_size)
                              ↓
输出变换 W_o:  ({batch_size}, {seq_len}, {embed_size})
               (batch, seq_len, embed_size)
""")
    
    # Padding Mask 示例
    print("=" * 70)
    print("使用 Padding Mask")
    print("=" * 70)
    
    tokens_padded = ["我", "喜欢", "<PAD>", "<PAD>"]
    print(f"\n序列: {tokens_padded} (实际长度=2)")
    
    padding_mask = torch.tensor([[1, 1, 0, 0]])  # (batch, seq_len)
    
    with torch.no_grad():
        _, attn_weights_masked = mha(x, mask=padding_mask)
    
    for head_idx in range(num_heads):
        weights = attn_weights_masked[0, head_idx].numpy()
        print_self_attention_matrix(tokens_padded, weights, 
                                    f"头 {head_idx + 1} (带 Padding Mask)")
    
    print("\n  → 每个头的最后两列 (<PAD> 位置) 权重都为 0")

