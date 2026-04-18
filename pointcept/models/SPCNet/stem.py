# stem.py
import spconv.pytorch as spconv
import torch.nn as nn

"""这个stem模块是oacnns的embedding部分"""
def create_stem(in_channels, embed_channels, norm_fn):
    return spconv.SparseSequential(
        spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            padding=1,
            indice_key="stem",
            bias=False,
        ),
        norm_fn(embed_channels),
        nn.ReLU(),
        spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            padding=1,
            indice_key="stem",
            bias=False,
        ),
        norm_fn(embed_channels),
        nn.ReLU(),
        spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            padding=1,
            indice_key="stem",
            bias=False,
        ),
        norm_fn(embed_channels),
        nn.ReLU(),
    )
