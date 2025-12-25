"""
Transformer 层模块

包含:
- feed_forward: 位置前馈网络 (FFN)
- residual_layer_norm: 残差连接和层归一化
- positional_encoding: 位置编码
"""

from .feed_forward import (
    PositionwiseFeedForward,
    PositionwiseFeedForwardGELU,
    SwiGLU
)
from .residual_layer_norm import (
    ResidualConnection,
    LayerNorm,
    RMSNorm,
    AddNorm,
    AddNormSimple
)
from .positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEmbedding,
    RotaryPositionalEmbedding
)

__all__ = [
    'PositionwiseFeedForward',
    'PositionwiseFeedForwardGELU',
    'SwiGLU',
    'ResidualConnection',
    'LayerNorm',
    'RMSNorm',
    'AddNorm',
    'AddNormSimple',
    'SinusoidalPositionalEncoding',
    'LearnedPositionalEmbedding',
    'RotaryPositionalEmbedding',
]

