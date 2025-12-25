"""
工具函数模块

提供中英文混合显示对齐、注意力矩阵打印等通用功能。
"""


def get_display_width(s):
    """
    计算字符串的显示宽度（中文字符算2，英文算1）
    
    参数:
        s: 字符串
        
    返回:
        width: 显示宽度
    """
    width = 0
    for char in s:
        if '\u4e00' <= char <= '\u9fff' or char in '，。！？：；""''（）':
            width += 2
        else:
            width += 1
    return width


def pad_to_width(s, target_width, align='left'):
    """
    将字符串填充到指定显示宽度
    
    参数:
        s: 字符串
        target_width: 目标宽度
        align: 对齐方式 ('left', 'right', 'center')
        
    返回:
        填充后的字符串
    """
    current_width = get_display_width(s)
    padding = target_width - current_width
    if padding <= 0:
        return s
    if align == 'left':
        return s + ' ' * padding
    elif align == 'right':
        return ' ' * padding + s
    else:  # center
        left_pad = padding // 2
        right_pad = padding - left_pad
        return ' ' * left_pad + s + ' ' * right_pad


def print_attention_matrix(row_tokens, col_tokens, weights, title="注意力权重矩阵",
                           row_label="行", col_label="列"):
    """
    美观地打印注意力权重矩阵（支持不同行列标签，适用于交叉注意力）
    
    参数:
        row_tokens: 行标签列表 (查询词)
        col_tokens: 列标签列表 (键词)
        weights: 注意力权重矩阵 (numpy array)
        title: 标题
        row_label: 行的说明
        col_label: 列的说明
    """
    print(f"\n{title}:")
    print(f"  {row_label} = 查询词, {col_label} = 键词")
    print()
    
    # 计算每列需要的宽度
    col_width = 6  # 每个数值的宽度
    row_label_width = max(get_display_width(t) for t in row_tokens) + 2
    
    # 打印表头
    header = " " * row_label_width + "│"
    for token in col_tokens:
        header += pad_to_width(token, col_width, 'center') + "│"
    
    separator = "─" * row_label_width + "┼" + ("─" * col_width + "┼") * len(col_tokens)
    
    print(header)
    print(separator)
    
    # 打印每行
    for i, token in enumerate(row_tokens):
        row = pad_to_width(token, row_label_width, 'right') + "│"
        for j in range(len(col_tokens)):
            val = f"{weights[i][j]:.2f}"
            row += pad_to_width(val, col_width, 'center') + "│"
        print(row)


def print_self_attention_matrix(tokens, weights, title="注意力权重矩阵"):
    """
    打印自注意力矩阵的便捷函数（行列标签相同）
    
    参数:
        tokens: token 列表
        weights: 注意力权重矩阵
        title: 标题
    """
    print_attention_matrix(tokens, tokens, weights, title, "行", "列")


def print_matrix(name, matrix, explanation=""):
    """
    美观地打印矩阵（用于展示中间计算结果）
    
    参数:
        name: 矩阵名称
        matrix: 要打印的矩阵
        explanation: 可选的说明文字
    """
    print(f"\n{name}:")
    if explanation:
        print(f"  → {explanation}")
    print(matrix)


def print_mask_matrix(tokens, mask, title="掩码矩阵"):
    """
    打印掩码矩阵（用 ✓/✗ 可视化）
    
    参数:
        tokens: token 列表
        mask: 掩码矩阵 (1=可见, 0=屏蔽)
        title: 标题
    """
    print(f"\n{title}:")
    print("  1=可见 (✓), 0=屏蔽 (✗)")
    print()
    
    col_width = max(get_display_width(t) for t in tokens) + 1
    row_label_width = col_width + 2
    
    # 表头
    header = " " * row_label_width + "│"
    for token in tokens:
        header += pad_to_width(token, col_width, 'center') + "│"
    
    separator = "─" * row_label_width + "┼" + ("─" * col_width + "┼") * len(tokens)
    
    print(header)
    print(separator)
    
    # 每行
    for i, token in enumerate(tokens):
        row = pad_to_width(token, row_label_width, 'right') + "│"
        for j in range(len(tokens)):
            val = "✓" if mask[i][j] == 1 else "✗"
            row += pad_to_width(val, col_width, 'center') + "│"
        print(row)


if __name__ == "__main__":
    print("=" * 60)
    print("                  工具函数演示")
    print("=" * 60)
    
    # 演示 get_display_width
    print("\n1. get_display_width() - 计算显示宽度")
    print("-" * 40)
    test_strings = ["Hello", "你好", "Hello世界", "<PAD>"]
    for s in test_strings:
        print(f"  '{s}' → 显示宽度 = {get_display_width(s)}")
    
    # 演示 pad_to_width
    print("\n2. pad_to_width() - 填充到指定宽度")
    print("-" * 40)
    print(f"  左对齐: '{pad_to_width('你好', 10, 'left')}'")
    print(f"  右对齐: '{pad_to_width('你好', 10, 'right')}'")
    print(f"  居中:   '{pad_to_width('你好', 10, 'center')}'")
    
    # 演示 print_self_attention_matrix
    print("\n3. print_self_attention_matrix() - 打印自注意力矩阵")
    print("-" * 40)
    import numpy as np
    tokens = ["我", "喜欢", "学习"]
    weights = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.5, 0.3],
        [0.3, 0.3, 0.4]
    ])
    print_self_attention_matrix(tokens, weights)
    
    # 演示 print_attention_matrix (交叉注意力)
    print("\n4. print_attention_matrix() - 打印交叉注意力矩阵")
    print("-" * 40)
    src_tokens = ["I", "love", "you"]
    tgt_tokens = ["我", "爱", "你"]
    cross_weights = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7]
    ])
    print_attention_matrix(tgt_tokens, src_tokens, cross_weights, 
                           "交叉注意力权重", "目标词", "源词")
    
    # 演示 print_mask_matrix
    print("\n5. print_mask_matrix() - 打印掩码矩阵")
    print("-" * 40)
    mask_tokens = ["词1", "词2", "词3"]
    mask = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]
    ])
    print_mask_matrix(mask_tokens, mask, "因果掩码")

