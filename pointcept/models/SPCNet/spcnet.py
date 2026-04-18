"""
SPCNet (Superpoint Context Network)
-----------------------------------
This file contains the complete definition of SPCNet and its components.
It is logically divided into 6 modules:
    1. Utilities & Basic Layers
    2. Graph & Superpoint Core Operations
    3. Attention Mechanisms
    4. Point Processing Modules (Down/Up-sampling)
    5. Intermediate Structural Blocks
    6. Main Network Architecture (SPCNet)
"""

import math
from functools import partial
import numpy as np
from addict import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_scatter import scatter_mean, scatter_softmax, scatter_add
from torch_geometric.nn import GCNConv, knn_graph
from torch_geometric.data import Data
from torch_geometric.utils import scatter, add_self_loops

import spconv.pytorch as spconv
from timm.models.layers import DropPath

# Pointcept & Specific imports
from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential
from pointcept.models.SPCNet.mamba_layer import MambaBlock
from pointcept.models.SPCNet.transformer import TransformerBlock

try:
    import flash_attn
except ImportError:
    flash_attn = None


# ============================================================================== #
# MODULE 1: UTILITIES & BASIC LAYERS                                             #
# ============================================================================== #

def kmeans_plus_plus_init(x, k, device=None):
    """
    Description:
        Selects k initial centers using the k-means++ algorithm (PyTorch native).
    Inputs:
        x (Tensor): Data points of shape [N, D] (float32).
        k (int): Number of initial centers.
        device (torch.device, optional): Target device.
    Outputs:
        centers (Tensor): Initialized centers of shape [k, D].
    """
    if device is None:
        device = x.device
    N, D = x.shape
    x = x.detach()

    idx = torch.randint(0, N, (1,), device=device)
    centers = [x[idx]]
    dist_sq = torch.cdist(x, centers[0], p=2).squeeze().pow(2)

    for _ in range(1, k):
        dist_sq_sum = dist_sq.sum()
        r = torch.rand(1, device=device) * dist_sq_sum
        candidate_idx = torch.searchsorted(dist_sq.cumsum(dim=0), r).item()

        new_center = x[candidate_idx:candidate_idx + 1]
        centers.append(new_center)

        dist_to_new_center_sq = torch.cdist(x, new_center, p=2).squeeze().pow(2)
        dist_sq = torch.minimum(dist_sq, dist_to_new_center_sq)

    centers = torch.cat(centers, dim=0)
    return centers


class RPE(nn.Module):
    """
    Description:
        Relative Position Encoding (RPE) module. Generates structural positional encodings.
    Inputs:
        coord (Tensor): Coordinate tensor for calculating positional bounds.
    Outputs:
        out (Tensor): RPE features of shape (N, H, K, K).
    """

    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
                coord.clamp(-self.pos_bnd, self.pos_bnd)
                + self.pos_bnd
                + torch.arange(3, device=coord.device) * self.rpe_num
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)
        return out


class MLP(nn.Module):
    """
    Description:
        Standard Multi-Layer Perceptron with activation and dropout.
    Inputs:
        x (Tensor): Input tensor of shape (..., in_channels).
    Outputs:
        x (Tensor): Transformed tensor of shape (..., out_channels).
    """

    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================================== #
# MODULE 2: GRAPH & SUPERPOINT CORE OPERATIONS                                   #
# ============================================================================== #

class SuperpointGCN(nn.Module):
    """
    Description:
        Graph Convolutional Network applied over superpoints to aggregate local structural data.
    Inputs:
        data (torch_geometric.data.Data): Contains node features (x), edge_index, and edge_attr.
    Outputs:
        x (Tensor): Updated node features with structural context.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, improved=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, improved=improved))
        self.norms.append(nn.LayerNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, improved=improved))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        if edge_attr is not None:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_attr=edge_attr, fill_value=1.0, num_nodes=x.size(0)
            )
        else:
            edge_index, edge_weight = add_self_loops(
                edge_index, fill_value=1.0, num_nodes=x.size(0)
            )

        edge_weight = torch.ones((edge_index.size(1),), device=x.device)
        residual = x

        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            if i < len(self.norms):
                x = self.norms[i](x)
            if i < len(self.convs) - 1:
                x = self.relu(x)

            if i == len(self.convs) - 1:
                x = x + residual
            residual = x

        return x


# ============================================================================== #
# MODULE 3: ATTENTION MECHANISMS                                                 #
# ============================================================================== #

class CrossMultiAttention_updateSP(nn.Module):
    """
    Description:
        Cross attention mechanism to update superpoint features using raw point features.
    Inputs:
        sp_feat (Tensor): Superpoint queries [num_superpoints, in_channels_q].
        rawPoint_feat (Tensor): Raw point keys/values [N, in_channels_kv].
        point_assignments (Tensor): Mapping of points to superpoints [N].
    Outputs:
        updated_sp_feat (Tensor): Updated superpoint features.
        attn_scores (Tensor): Mean attention scores [N].
    """

    def __init__(self, in_channels_q, in_channels_kv, emb_dim, num_heads, att_dropout=0.0, dropout=0.0,
                 rpe_dim=3, k_rpe=False, q_rpe=False, v_rpe=False, k_delta_rpe=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.depth = emb_dim // num_heads
        self.scale = self.depth ** -0.5

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.Wq = nn.Linear(in_channels_q, emb_dim, bias=False)
        self.Wkv = nn.Linear(in_channels_kv * 2, emb_dim * 2, bias=False)

        self.k_rpe = nn.Linear(rpe_dim, emb_dim, bias=False) if k_rpe else None
        self.q_rpe = nn.Linear(rpe_dim, emb_dim, bias=False) if q_rpe else None
        self.v_rpe = nn.Linear(rpe_dim, emb_dim, bias=False) if v_rpe else None
        self.k_delta_rpe = nn.Linear(in_channels_kv, emb_dim, bias=False) if k_delta_rpe else None

        self.proj_out = nn.Linear(emb_dim, in_channels_q)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(att_dropout)
        self.layer_norm = nn.LayerNorm(in_channels_q)

    def forward(self, sp_feat, rawPoint_feat, point_assignments, sp_rpe=None, rawPoint_rpe=None):
        H, C = self.num_heads, self.emb_dim
        num_superpoints, N = sp_feat.size(0), rawPoint_feat.size(0)
        device = rawPoint_feat.device

        Q = self.Wq(sp_feat).reshape(num_superpoints, H, C // H)
        k, v = (self.Wkv(torch.cat([rawPoint_feat, rawPoint_feat], -1))
                .reshape(N, 2, H, C // H)
                .permute(1, 0, 2, 3)
                .unbind(0))

        if self.q_rpe is not None and sp_rpe is not None:
            Q = Q + self.norm1(self.q_rpe(sp_rpe)).reshape(num_superpoints, H, C // H)
        if self.k_rpe is not None and rawPoint_rpe is not None:
            k = k + self.norm2(self.k_rpe(rawPoint_rpe)).reshape(N, H, C // H)
        if self.v_rpe is not None and rawPoint_rpe is not None:
            v = v + self.v_rpe(rawPoint_rpe).reshape(N, H, C // H)
        if self.k_delta_rpe is not None:
            k = k + self.k_delta_rpe(rawPoint_feat - sp_feat[point_assignments]).reshape(N, H, C // H)

        Q_flat, K_flat, V_flat = map(lambda x: x.reshape(-1, C // H), (Q, k, v))

        point_assignments_expanded = point_assignments.unsqueeze(1).expand(-1, H)
        head_indices = torch.arange(H, device=device).unsqueeze(0).expand(N, H)
        indices_flat = (point_assignments_expanded * H + head_indices).reshape(-1)

        Q_assigned = Q_flat[indices_flat]
        attn_scores_flat = torch.sum(Q_assigned * K_flat, dim=-1) * self.scale
        del Q_assigned

        attn_weights_flat = self.attn_dropout(scatter_softmax(attn_scores_flat, indices_flat))
        context_flat = V_flat * attn_weights_flat.unsqueeze(-1)
        context = scatter_add(context_flat, indices_flat, dim=0, dim_size=num_superpoints * H)
        context = context.reshape(num_superpoints, -1)

        updated_sp_feat = self.dropout(context)
        updated_sp_feat = self.layer_norm(sp_feat + updated_sp_feat)
        attn_scores = attn_scores_flat.reshape(N, H).mean(dim=1)

        return updated_sp_feat, attn_scores


class SerializedAttention(PointModule):
    """
    Description:
        Attention mechanism operating on serialized point sequences.
    Inputs:
        point (Point): Pointcept structure containing serialized point features.
    Outputs:
        point (Point): Updated point structure.
    """

    def __init__(self, channels, num_heads, patch_size, norm_layer=nn.LayerNorm, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, order_index=0, enable_rpe=False, enable_flash=True,
                 upcast_attention=True, upcast_softmax=True, embedding_is_mamba=False):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index

        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        self.embedding_is_mamba = embedding_is_mamba

        if enable_flash:
            assert not enable_rpe, "Set enable_rpe to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = nn.Dropout(attn_drop)

        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

        self.norm_fn1 = partial(nn.LayerNorm, eps=1e-3)
        self.norm_fn2 = partial(nn.LayerNorm, eps=1e-3)
        self.norm_fn3 = partial(nn.LayerNorm, eps=1e-3)

        self.rpe_mlp = nn.Sequential(
            nn.Linear(3, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels)
        )

        # TODO: Add normalization and ReLU between Mamba layers
        self.mamba_blocks = PointSequential(
            self.norm_fn1(channels),
            MambaBlock(
                dim=channels, layer_idx=1, norm_cls=nn.LayerNorm,
                fused_add_norm=True, residual_in_fp32=True, drop_path=0.2,
            ),
        )

        self.SparseConv1 = PointSequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, padding=1, bias=False, indice_key="Local"),
            self.norm_fn2(channels),
            nn.GELU(),
        )
        self.SparseConv2 = PointSequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, padding=1, bias=False, indice_key="Local"),
            self.norm_fn3(channels),
            nn.GELU(),
        )

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"

        if pad_key not in point.keys() or unpad_key not in point.keys() or cu_seqlens_key not in point.keys():
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                    torch.div(bincount + self.patch_size - 1, self.patch_size, rounding_mode="trunc") * self.patch_size
            )

            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))

            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []

            for i in range(len(offset)):
                unpad[_offset[i]: _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[_offset_pad[i + 1] - self.patch_size + (bincount[i] % self.patch_size): _offset_pad[i + 1]] = \
                        pad[_offset_pad[i + 1] - 2 * self.patch_size + (bincount[i] % self.patch_size): _offset_pad[
                                                                                                            i + 1] - self.patch_size]
                pad[_offset_pad[i]: _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]

                cu_seqlens.append(
                    torch.arange(_offset_pad[i], _offset_pad[i + 1], step=self.patch_size, dtype=torch.int32,
                                 device=offset.device)
                )

            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1])

        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point: Point):
        if not self.enable_flash:
            self.patch_size = min(offset2bincount(point.offset).min().tolist(), self.patch_size_max)

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        point.pad = pad
        point.unpad = unpad
        point.cu_seqlens = cu_seqlens

        order = point.serialized_order[self.order_index]
        inverse = point.serialized_inverse[self.order_index]
        residual = point.feat

        point = self.SparseConv1(point)
        point = self.SparseConv2(point)
        point.feat = point.feat + residual

        if self.channels == 512:
            residual = point.feat
            point = self.mamba_blocks(point, residual=None, order=order, inverse=inverse)
            point.feat = point.feat + residual

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SelfAttentionSPBlock(PointModule):
    """
    Description:
        Self Attention operating explicitly over Superpoint features and structure.
    Inputs:
        sp_center_feat (Tensor): Superpoint features.
        edge_index (Tensor): Connectivity matrix.
        edge_attr (Tensor): Edge features.
        norm_index (Tensor): Indices for layer normalization.
    Outputs:
        sp_center_feat (Tensor): Refined superpoint features.
    """

    def __init__(self, channels, k=8, mlp_ratio=4.0, proj_drop=0.0, drop_path=0.0,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_norm=True,
                 num_transformer_layers=1, num_heads=16, edge_feat_dim=3, emb_dim=256):
        super().__init__()
        self.k = k
        self.channels = channels
        self.pre_norm = pre_norm

        self.edge_attr_mlp = MLP(in_channels=edge_feat_dim, hidden_channels=32, out_channels=channels,
                                 act_layer=act_layer, drop=proj_drop)

        self.transformer_for_spinteractive = nn.ModuleList([
            nn.ModuleDict({
                'transformer': TransformerBlock(
                    dim=channels, num_heads=num_heads, qk_dim=emb_dim, k_rpe=True, q_rpe=True,
                    k_delta_rpe=True, q_delta_rpe=True, qk_share_rpe=True, q_on_minus_rpe=True, heads_share_rpe=True,
                ),
                'LayerNorm': nn.LayerNorm(channels)
            })
            for _ in range(num_transformer_layers)
        ])

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm1 = norm_layer(channels)
        self.norm2 = norm_layer(channels)
        self.mlp = MLP(in_channels=channels, hidden_channels=int(channels * mlp_ratio), out_channels=channels,
                       act_layer=act_layer, drop=proj_drop)

    def forward(self, sp_center_feat, edge_index, edge_attr, norm_index):
        sp_center_feat = sp_center_feat.to(sp_center_feat.device)
        for module in self.transformer_for_spinteractive:
            sp_center_feat, _, _ = module['transformer'](
                sp_center_feat, norm_index=norm_index, edge_index=edge_index, edge_attr=edge_attr,
                device=sp_center_feat.device
            )
            sp_center_feat = module['LayerNorm'](sp_center_feat)
        return sp_center_feat


# ============================================================================== #
# MODULE 4: POINT PROCESSING MODULES                                             #
# ============================================================================== #

class Embedding(PointModule):
    """
    Description:
        Initial stem embedding layer for point cloud inputs using Sparse Convolution.
    Inputs:
        point (Point): Initial point structure.
    Outputs:
        point (Point): Point structure with initialized embeddings.
    """

    def __init__(self, in_channels, embed_channels, norm_layer=None, norm_fn=None, act_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.norm_fn = norm_fn

        self.stem = PointSequential(
            spconv.SubMConv3d(in_channels, embed_channels, kernel_size=3, padding=1, bias=False, indice_key="stem"),
            self.norm_fn(embed_channels),
            nn.ReLU(),
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        point.embedding = point.feat
        return point


class SerializedPooling(PointModule):
    """
    Description:
        Downsamples the point cloud by grouping according to serialized codes.
    Inputs:
        point (Point): Full resolution point structure.
    Outputs:
        point (Point): Downsampled point structure with aggregated features.
    """

    def __init__(self, in_channels, out_channels, stride=2, norm_layer=None, act_layer=None,
                 reduce="max", shuffle_orders=True, traceable=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable
        self.proj = nn.Linear(in_channels, out_channels)

        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True, return_counts=True)
        _, indices = torch.sort(cluster)

        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        code = code[:, head_indices]

        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1, index=order, src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1)
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        point_dict = Dict(
            feat=torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce),
            coord=torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point

        point = Point(point_dict)
        if hasattr(self, 'norm') and self.norm is not None:
            point = self.norm(point)
        if hasattr(self, 'act') and self.act is not None:
            point = self.act(point)

        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    """
    Description:
        Upsamples the point cloud using recorded parent clustering data.
    Inputs:
        point (Point): Downsampled point structure.
    Outputs:
        parent (Point): Reconstructed, higher-resolution point structure.
    """

    def __init__(self, in_channels, skip_channels, out_channels, norm_layer=None, act_layer=None, traceable=False):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))
        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")

        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


# ============================================================================== #
# MODULE 5: INTERMEDIATE STRUCTURAL BLOCKS                                       #
# ============================================================================== #

class Block(PointModule):
    """
    Description:
        Core network block wrapping Attention, SpConv, Mamba, and MLP blocks.
    Inputs:
        point (Point): Point data.
    Outputs:
        point (Point): Transformed point data.
    """

    def __init__(self, channels, num_heads, patch_size=48, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, norm_fn=None,
                 act_layer=nn.GELU, pre_norm=True, order_index=0, cpe_indice_key=None, enable_rpe=False,
                 enable_flash=True, upcast_attention=True, upcast_softmax=True, embedding_is_mamba=False):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.cpe = PointSequential(
            spconv.SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=cpe_indice_key),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )
        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels, patch_size=patch_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop, order_index=order_index, enable_rpe=enable_rpe,
            enable_flash=enable_flash, upcast_attention=upcast_attention, upcast_softmax=upcast_softmax,
            embedding_is_mamba=embedding_is_mamba,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(in_channels=channels, hidden_channels=int(channels * mlp_ratio), out_channels=channels,
                act_layer=act_layer, drop=proj_drop)
        )
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

        self.mamba_blocks = PointSequential(
            MambaBlock(dim=channels, layer_idx=1, norm_cls=nn.LayerNorm, fused_add_norm=True, residual_in_fp32=True,
                       drop_path=0.2),
        )
        self.norm3 = PointSequential(norm_layer(channels))

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)

        point = self.drop_path(self.attn(point))
        point.feat = shortcut + point.feat

        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.mlp(point))
        point.feat = shortcut + point.feat

        if not self.pre_norm:
            point = self.norm2(point)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class UpdateSPBlock(PointModule):
    """
    Description:
        Block to update superpoint representations through cross attention & MLP.
    Inputs:
        sp_center_feat (Tensor): Superpoint features.
        rawPoint_feat (Tensor): Raw point features.
        point_assignments (Tensor): Assignments.
        sp_rpe (Tensor): Superpoint relative encodings.
        rawPoint_rpe (Tensor): Point relative encodings.
    Outputs:
        Tuple: (sp_center_feat (Tensor), attn_scores (Tensor))
    """

    def __init__(self, channels, mlp_ratio=4.0, proj_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU, pre_norm=True):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.norm1 = norm_layer(channels)
        self.norm2 = norm_layer(channels)
        self.mlp = MLP(in_channels=channels, hidden_channels=int(channels * mlp_ratio), out_channels=channels,
                       act_layer=act_layer, drop=proj_drop)
        self.cross_attention_update = CrossMultiAttention_updateSP(
            in_channels_q=64, in_channels_kv=64, num_heads=4, att_dropout=0.1, dropout=0.1,
            rpe_dim=3, k_rpe=False, q_rpe=False, v_rpe=False, k_delta_rpe=False
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, sp_center_feat, rawPoint_feat, point_assignments, sp_rpe, rawPoint_rpe):
        shortcut = sp_center_feat
        if self.pre_norm:
            sp_center_feat = self.norm1(sp_center_feat)

        updateSP, attn_scores = self.drop_path(
            self.cross_attention_update(sp_center_feat, rawPoint_feat, point_assignments, sp_rpe, rawPoint_rpe)
        )
        sp_center_feat = shortcut + updateSP

        shortcut = sp_center_feat
        if self.pre_norm:
            sp_center_feat = self.norm2(sp_center_feat)
        sp_center_feat = self.drop_path(self.mlp(sp_center_feat))
        sp_center_feat = shortcut + sp_center_feat

        return sp_center_feat, attn_scores


class UpdateSuperpointsModule(nn.Module):
    """
    Description:
        High-level module managing KNN finding and superpoint updates iteratively.
    """

    def __init__(self, cross_attention_update_sp, norm, K=5, is_momentum=False):
        super().__init__()
        self.cross_attention_update_sp = cross_attention_update_sp
        self.norm = norm
        self.K = K
        self.is_momentum = is_momentum
        self.norm_fn_rpe1 = nn.LayerNorm(64)
        self.norm_fn_rpe2 = nn.LayerNorm(64)
        self.is_cossim = False

    @torch.no_grad()
    def update_point_assignments(self, point_assignments, attn_scores, num_superpoints, last_iteration,
                                 retain_ratio=0.5):
        N = point_assignments.shape[0]
        device = point_assignments.device
        if last_iteration:
            return None

        counts = torch.bincount(point_assignments, minlength=num_superpoints)
        retain_mask = torch.zeros(N, dtype=torch.bool, device=device)
        attn_scores = attn_scores.squeeze()

        sorted_indices = torch.argsort(point_assignments * N - attn_scores, descending=False)
        sorted_point_assignments = point_assignments[sorted_indices]

        retain_counts = torch.ceil(counts.float() * retain_ratio).long()
        point_offsets = torch.cumsum(counts, dim=0)
        point_offsets = torch.cat([torch.tensor([0], device=device), point_offsets[:-1]])

        relative_indices = torch.arange(N, device=device) - point_offsets[sorted_point_assignments]
        mask_per_superpoint = relative_indices < retain_counts[sorted_point_assignments]
        retain_mask[sorted_indices] = mask_per_superpoint

        return retain_mask

    def forward(self, point, rawPoint_feat, rawPoint_coord, hilbert_feat_coord, num_segments0, points_per_segment0,
                points_feat, points_coord, num_segments1, points_per_segment1, segments_per_level0,
                level0_to_level1_indices, point_assignments, retain_mask, last_iteration):

        device = points_feat.device
        neighbor_range = 1
        neighbor_offsets = torch.arange(-neighbor_range, neighbor_range + 1, device=device)

        neighbor_indices = torch.arange(num_segments0, device=device).unsqueeze(1) + neighbor_offsets.unsqueeze(0)
        neighbor_indices = neighbor_indices.clamp(0, num_segments0 - 1)
        sp_indices = level0_to_level1_indices[neighbor_indices.view(-1)].view(num_segments0, -1)

        sp_center_feat_neighbors = point.sp_center_feat[sp_indices.view(-1)].view(num_segments0, -1,
                                                                                  point.sp_center_feat.size(-1))
        sp_center_coord_neighbors = point.sp_center_coord[sp_indices.view(-1)].view(num_segments0, -1,
                                                                                    point.sp_center_coord.size(-1))

        distances = torch.cdist(points_coord, sp_center_coord_neighbors, p=2)
        _, knn_indices = torch.topk(distances, self.K, largest=False, dim=-1)
        del distances

        batch_indices = torch.arange(num_segments0, device=device).view(-1, 1, 1).expand(-1, points_per_segment0,
                                                                                         self.K)
        knn_sp_feats = sp_center_feat_neighbors[batch_indices, knn_indices]

        if self.is_cossim:
            similarities = F.cosine_similarity(points_feat.unsqueeze(2), knn_sp_feats, dim=-1)
        else:
            similarities = torch.sum(points_feat.unsqueeze(2) * knn_sp_feats, dim=-1)

        similarities = F.softmax(similarities, dim=-1)
        max_similarities, max_indices = similarities.max(dim=-1)
        del similarities

        sp_indices_expanded = sp_indices.unsqueeze(1).expand(-1, points_per_segment0, -1)
        assigned_sp_indices = torch.gather(sp_indices_expanded, 2, knn_indices).gather(2, max_indices.unsqueeze(
            -1)).squeeze(-1)
        del sp_indices_expanded, knn_indices

        if self.is_momentum:
            point_assignments_iter = assigned_sp_indices.reshape(-1)
            if point_assignments is None:
                point_assignments = point_assignments_iter.clone()
            else:
                if retain_mask is not None:
                    point_assignments_iter[retain_mask] = point_assignments[retain_mask]
                point_assignments = point_assignments_iter
        else:
            point_assignments = assigned_sp_indices.reshape(-1)

        point_assignments = point_assignments.to(device)
        attn_scores = None
        sp_rpe = point.sp_center_coord
        rawPoint_rpe = hilbert_feat_coord - point.sp_center_coord[point_assignments]

        for module in self.cross_attention_update_sp:
            point.sp_center_feat, attn_scores = module(point.sp_center_feat, rawPoint_feat, point_assignments, sp_rpe,
                                                       rawPoint_rpe)
            point.sp_center_feat = self.norm(point.sp_center_feat)

        point.sp_center_coord = scatter(hilbert_feat_coord, point_assignments, dim=0, reduce="mean",
                                        dim_size=point.sp_center_feat.size(0))

        hilbert_feat_level1 = rawPoint_feat.reshape(num_segments1, points_per_segment1, -1)
        hilbert_feat_level0 = hilbert_feat_level1.reshape(num_segments0, points_per_segment0, -1)
        points_feat = hilbert_feat_level0

        if self.is_momentum:
            retain_mask = self.update_point_assignments(point_assignments, attn_scores, num_superpoints=num_segments1,
                                                        last_iteration=last_iteration)
        else:
            retain_mask = None

        return point_assignments, retain_mask, point.sp_center_feat, point.sp_center_coord, points_feat, hilbert_feat_level1


class SuperpointGraphModule(nn.Module):
    """
    Description:
        Connects superpoints as a graph and applies SuperpointGCN.
    """

    def __init__(self, SPgrah, norm, SuperpointGCN):
        super().__init__()
        self.SPgrah = SPgrah
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)
        self.SuperpointGCN = SuperpointGCN

    def forward(self, sp_center_feat, edge_index_tran, edge_attr_rpe, norm_index, sp_crood):
        if sp_center_feat.size(0) == 512:
            norm_index = norm_index[0]
        if sp_center_feat.size(0) == 2048:
            norm_index = norm_index[1]

        sp_center_feat = self.norm1(sp_center_feat)
        shortcut = sp_center_feat
        sp_center_feat = self.SPgrah(sp_center_feat, edge_index_tran, edge_attr_rpe, norm_index)
        sp_center_feat = sp_center_feat + shortcut
        sp_center_feat = self.norm2(sp_center_feat)
        shortcut = sp_center_feat

        edge_index_gcn = knn_graph(sp_crood, k=3, batch=None, loop=False)
        similarity = F.cosine_similarity(sp_center_feat[edge_index_gcn[0]], sp_center_feat[edge_index_gcn[1]], dim=-1)
        similarity = torch.sigmoid(similarity)
        edge_attr = similarity.unsqueeze(-1)

        data = Data(x=sp_center_feat, edge_index=edge_index_gcn, edge_attr=edge_attr)
        sp_center_feat_graph = self.SuperpointGCN(data)
        sp_center_feat_graph = shortcut + sp_center_feat_graph

        return sp_center_feat_graph


# ============================================================================== #
# MODULE 6: MAIN NETWORK ARCHITECTURE (SPCNet)                                   #
# ============================================================================== #

@MODELS.register_module("SPC-Net")
class SPCNet(PointModule):
    """
    Description:
        The main architecture connecting the Encoder/Decoder, Attention, and Graph Superpoint operations.
    Inputs:
        data_dict (dict): Raw input dict mapping to Point structure.
    Outputs:
        point (Point): Final processed point representation.
    """

    def __init__(self, in_channels=6, order=("hilbert", "hilbert-trans"), stride=(2, 2, 2, 2),
                 enc_depths=(2, 2, 2, 6, 2), enc_channels=(32, 64, 128, 256, 512), enc_num_head=(2, 4, 8, 16, 32),
                 enc_patch_size=(48, 48, 48, 48, 48), dec_depths=(2, 2, 2, 2), dec_channels=(64, 64, 128, 256),
                 dec_num_head=(4, 4, 8, 16), dec_patch_size=(48, 48, 48, 48), mlp_ratio=4, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, drop_path=0.3, pre_norm=True, norm_layer=nn.LayerNorm,
                 shuffle_orders=True,
                 enable_rpe=False, enable_flash=True, upcast_attention=False, upcast_softmax=False, cls_mode=False,
                 pdnorm_bn=False, pdnorm_ln=False, norm_fn=None, pdnorm_decouple=True, pdnorm_adaptive=False,
                 pdnorm_affine=True, pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D")):
        super().__init__()

        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        self.num_levels = 2
        self.is_concat = False
        self.embedding_is_mamba = True
        self.is_cossim = False

        if pdnorm_bn:
            bn_layer = partial(PDNorm,
                               norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine),
                               conditions=pdnorm_conditions, decouple=pdnorm_decouple, adaptive=pdnorm_adaptive)
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        if pdnorm_ln:
            ln_layer = partial(PDNorm, norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                               conditions=pdnorm_conditions, decouple=pdnorm_decouple, adaptive=pdnorm_adaptive)
        else:
            ln_layer = nn.LayerNorm

        act_layer = nn.GELU

        self.embedding = Embedding(in_channels=in_channels, embed_channels=enc_channels[0], norm_layer=ln_layer,
                                   norm_fn=ln_layer, act_layer=act_layer)

        self.serialized_pooling = PointSequential()
        self.serialized_pooling.add(
            SerializedPooling(in_channels=64, out_channels=64, stride=8, norm_layer=bn_layer, act_layer=act_layer),
            name="down")

        self.attn = SerializedAttention(
            channels=64, patch_size=1024, num_heads=4, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=proj_drop, order_index=0, enable_rpe=enable_rpe, enable_flash=enable_flash,
            upcast_attention=upcast_attention, upcast_softmax=upcast_softmax,
        )

        self._init_update_blocks()
        self._init_mlps_and_graphs(act_layer, proj_drop, norm_layer)
        self._init_encoder(enc_depths, enc_channels, enc_num_head, enc_patch_size, mlp_ratio, qkv_bias, qk_scale,
                           attn_drop, proj_drop, drop_path, ln_layer, bn_layer, act_layer, pre_norm, enable_rpe,
                           enable_flash, upcast_attention, upcast_softmax, stride)
        self._init_decoder(dec_depths, dec_channels, enc_channels, dec_num_head, dec_patch_size, mlp_ratio, qkv_bias,
                           qk_scale, attn_drop, proj_drop, drop_path, ln_layer, bn_layer, act_layer, pre_norm,
                           enable_rpe, enable_flash, upcast_attention, upcast_softmax)

    def _init_update_blocks(self):
        block_kwargs = dict(channels=64, mlp_ratio=4.0, proj_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
                            act_layer=nn.GELU, pre_norm=True)
        self.cross_attention_update_sp = nn.ModuleList([UpdateSPBlock(**block_kwargs)])
        self.cross_attention_update_sp_level0 = nn.ModuleList([UpdateSPBlock(**block_kwargs)])
        self.cross_attention_update_sp_rawPoint = nn.ModuleList([UpdateSPBlock(**block_kwargs)])

    def _init_mlps_and_graphs(self, act_layer, proj_drop, norm_layer):
        self.norm1 = norm_layer(64)
        self.norm2 = norm_layer(64)
        self.norm3 = norm_layer(64)
        self.norm4 = norm_layer(64)

        self.SPgrah = SelfAttentionSPBlock(
            channels=64, k=6, mlp_ratio=4.0, proj_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
            pre_norm=True, num_transformer_layers=1, num_heads=16, edge_feat_dim=6, emb_dim=256
        )
        self.SuperpointGCN = SuperpointGCN(64, 64, 64, num_layers=1, improved=True)
        self.edge_attr_mlp = MLP(in_channels=3, hidden_channels=32, out_channels=64, act_layer=act_layer,
                                 drop=proj_drop)
        self.edge_attr_liner = nn.Linear(3, 64, bias=False)
        self.Level_fuse_mlp = MLP(in_channels=192, hidden_channels=128, out_channels=64, act_layer=act_layer,
                                  drop=proj_drop)
        self.Point_proj_mlp = MLP(in_channels=64, hidden_channels=128, out_channels=64, act_layer=act_layer,
                                  drop=proj_drop)
        self.Super_proj_mlp = MLP(in_channels=64, hidden_channels=64, out_channels=64, act_layer=act_layer,
                                  drop=proj_drop)
        self.Object_proj_mlp = MLP(in_channels=64, hidden_channels=64, out_channels=64, act_layer=act_layer,
                                   drop=proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.num_iterations = 1
        self.update_superpoints_modules = nn.ModuleList(
            [UpdateSuperpointsModule(self.cross_attention_update_sp, self.norm1, K=5) for _ in
             range(self.num_iterations)])

        self.num_sp_graph_iterations = 4
        self.sp_graph_modules = nn.ModuleList(
            [SuperpointGraphModule(self.SPgrah, self.norm2, self.SuperpointGCN) for _ in
             range(self.num_sp_graph_iterations)])

    def _init_encoder(self, enc_depths, enc_channels, enc_num_head, enc_patch_size, mlp_ratio, qkv_bias, qk_scale,
                      attn_drop, proj_drop, drop_path, ln_layer, bn_layer, act_layer, pre_norm, enable_rpe,
                      enable_flash, upcast_attention, upcast_softmax, stride):
        enc_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]): sum(enc_depths[: s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(SerializedPooling(in_channels=enc_channels[s - 1], out_channels=enc_channels[s],
                                          stride=stride[s - 1], norm_layer=bn_layer, act_layer=act_layer), name="down")
            for i in range(enc_depths[s]):
                enc.add(
                    Block(channels=enc_channels[s], num_heads=enc_num_head[s], patch_size=enc_patch_size[s],
                          mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          attn_drop=attn_drop, proj_drop=proj_drop, drop_path=enc_drop_path_[i], norm_layer=ln_layer,
                          norm_fn=bn_layer, act_layer=act_layer,
                          pre_norm=pre_norm, order_index=i % len(self.order), cpe_indice_key=f"stage{s}",
                          enable_rpe=enable_rpe, enable_flash=enable_flash,
                          upcast_attention=upcast_attention, upcast_softmax=upcast_softmax,
                          embedding_is_mamba=self.embedding_is_mamba), name=f"block{i}"
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

    def _init_decoder(self, dec_depths, dec_channels, enc_channels, dec_num_head, dec_patch_size, mlp_ratio, qkv_bias,
                      qk_scale, attn_drop, proj_drop, drop_path, ln_layer, bn_layer, act_layer, pre_norm, enable_rpe,
                      enable_flash, upcast_attention, upcast_softmax):
        if not self.cls_mode:
            dec_drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[sum(dec_depths[:s]): sum(dec_depths[: s + 1])]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(SerializedUnpooling(in_channels=dec_channels[s + 1], skip_channels=enc_channels[s],
                                            out_channels=dec_channels[s], norm_layer=bn_layer, act_layer=act_layer),
                        name="up")
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(channels=dec_channels[s], num_heads=dec_num_head[s], patch_size=dec_patch_size[s],
                              mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=proj_drop, drop_path=dec_drop_path_[i],
                              norm_layer=ln_layer, norm_fn=bn_layer, act_layer=act_layer, pre_norm=pre_norm,
                              order_index=i % len(self.order), cpe_indice_key=f"stage{s}", enable_rpe=enable_rpe,
                              enable_flash=enable_flash, upcast_attention=upcast_attention,
                              upcast_softmax=upcast_softmax), name=f"block{i}"
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    @torch.no_grad()
    def norm_index(self, mode='graph', node_num=1024):
        """Index to be used for LayerNorm per graph/node/segment."""
        if getattr(self, 'batch', None) is not None:
            batch = self.batch
        if node_num == 2048:
            batch = torch.zeros(2048, dtype=torch.long)
        if node_num == 512:
            batch = torch.zeros(512, dtype=torch.long)
        if mode == 'graph':
            return batch
        elif mode == 'node':
            return torch.arange(self.num_nodes, device=self.device)
        else:
            raise NotImplementedError(f"Unknown mode='{mode}'")

    def prepare_initial_data(self, point, num_levels=2, num_segments_list=None):
        if num_segments_list is None:
            if num_levels == 1:
                num_segments_list = [1024]
            elif num_levels == 2:
                num_segments_list = [512, 2048]
            elif num_levels == 3:
                num_segments_list = [128, 256, 1024]
            else:
                raise ValueError(f"Unsupported num_levels: {num_levels}")

        device = point.feat.device
        pad, unpad, cu_seqlens = self.attn.get_padding_and_inverse(point)
        inverse = unpad[point.serialized_inverse[0]]

        hilbert_order = point.serialized_order[0][pad]
        point.hilbert_order = hilbert_order
        hilbert_feat = point.feat[hilbert_order]
        hilbert_feat_coord = point.coord[hilbert_order]
        total_points = hilbert_feat.size(0)

        num_levels = len(num_segments_list)
        hilbert_feat_levels = []
        hilbert_feat_coord_levels = []
        num_segments_levels = []
        points_per_segment_levels = []

        for num_segments in num_segments_list:
            points_per_segment = total_points // num_segments
            hilbert_feat_level = hilbert_feat[:total_points].reshape(num_segments, points_per_segment, -1)
            hilbert_feat_coord_level = hilbert_feat_coord[:total_points].reshape(num_segments, points_per_segment, -1)

            hilbert_feat_levels.append(hilbert_feat_level.to(device))
            hilbert_feat_coord_levels.append(hilbert_feat_coord_level.to(device))
            num_segments_levels.append(num_segments)
            points_per_segment_levels.append(points_per_segment)

        initial_point_assignments = torch.repeat_interleave(
            torch.arange(num_segments_levels[-1], device=device), points_per_segment_levels[-1]
        )

        level_indices = {}
        level_inverse_indices = {}

        for i in range(num_levels - 1):
            num_segments_higher = num_segments_levels[i]
            num_segments_lower = num_segments_levels[i + 1]
            segments_per_higher = num_segments_lower // num_segments_higher

            higher_to_lower_indices = torch.arange(num_segments_higher, device=device).unsqueeze(
                1) * segments_per_higher
            higher_to_lower_indices = higher_to_lower_indices + torch.arange(segments_per_higher, device=device)
            level_indices[(i, i + 1)] = higher_to_lower_indices

            lower_to_higher_indices = torch.repeat_interleave(torch.arange(num_segments_higher, device=device),
                                                              segments_per_higher)
            level_inverse_indices[(i + 1, i)] = lower_to_higher_indices[:num_segments_lower]

        point.hilbert_feat = hilbert_feat_levels[-1]
        point.sp_center_feat = torch.mean(hilbert_feat_levels[-1], dim=1)
        point.sp_center_coord = torch.mean(hilbert_feat_coord_levels[-1], dim=1)

        rawPoint_feat = hilbert_feat_levels[-1].reshape(-1, hilbert_feat_levels[-1].size(-1))
        rawPoint_coord = hilbert_feat_coord_levels[-1].reshape(-1, hilbert_feat_coord_levels[-1].size(-1))

        return {
            'hilbert_feat': hilbert_feat, 'hilbert_feat_coord': hilbert_feat_coord, 'total_points': total_points,
            'hilbert_feat_levels': hilbert_feat_levels, 'hilbert_feat_coord_levels': hilbert_feat_coord_levels,
            'num_segments_levels': num_segments_levels, 'points_per_segment_levels': points_per_segment_levels,
            'level_indices': level_indices, 'level_inverse_indices': level_inverse_indices,
            'rawPoint_feat': rawPoint_feat, 'rawPoint_coord': rawPoint_coord, 'inverse': inverse,
            'segments_per_higher': segments_per_higher, 'initial_point_assignments': initial_point_assignments[inverse]
        }

    def fuse_features(self, point, point_assignments, inverse, sp_center_feats, sp_center_coords, level_indices,
                      num_segments_levels):
        device = point.feat.device
        k_gcn_list = []
        shortcut = sp_center_feats[-1]

        for num_segments in num_segments_levels:
            if num_segments >= 1024:
                k_gcn_list.append(6)
            elif num_segments >= 256 or num_segments >= 128:
                k_gcn_list.append(3)
            else:
                k_gcn_list.append(3)

        num_levels = len(sp_center_feats)
        assert len(k_gcn_list) == num_levels

        for index in range(num_levels):
            sp_feat = sp_center_feats[index]
            sp_coord = sp_center_coords[index]

            edge_index_tran = knn_graph(sp_feat, k=6, batch=None, loop=False)
            src_coords = sp_coord[edge_index_tran[0]]
            dst_coords = sp_coord[edge_index_tran[1]]
            relative_position = dst_coords - src_coords
            edge_attr_rpe = self.edge_attr_mlp(relative_position)

            norm_index = []
            norm_index1 = self.norm_index(mode='graph', node_num=512).to(sp_feat.device)
            norm_index2 = self.norm_index(mode='graph', node_num=2048).to(sp_feat.device)
            norm_index.append(norm_index1)
            norm_index.append(norm_index2)

            if index == 0:
                sp_center_feats[index] = sp_feat
            if index == 1:
                for module in self.sp_graph_modules:
                    sp_feat = module(sp_feat, edge_index_tran, edge_attr_rpe, norm_index, sp_coord)
                sp_center_feats[index] = sp_feat + shortcut

        for level in range(num_levels - 1):
            level_index = level_indices
            sp_feat_higher = sp_center_feats[level][level_index]
            sp_center_feats[level] = sp_feat_higher

        return sp_center_feats

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)

        data = self.prepare_initial_data(point, num_levels=self.num_levels)
        device = point.feat.device  # Inherit dynamic device naturally

        hilbert_feat = data['hilbert_feat']
        hilbert_feat_coord = data['hilbert_feat_coord']
        hilbert_feat_levels = data['hilbert_feat_levels']
        hilbert_feat_coord_levels = data['hilbert_feat_coord_levels']
        num_segments_levels = data['num_segments_levels']
        points_per_segment_levels = data['points_per_segment_levels']
        level_indices = data['level_indices']
        level_inverse_indices = data['level_inverse_indices']
        rawPoint_feat = data['rawPoint_feat']
        rawPoint_coord = data['rawPoint_coord']
        inverse = data['inverse']
        segments_per_level0 = data['segments_per_higher']
        initial_point_assignments = data['initial_point_assignments'].to(device)

        point_assignments = None
        retain_mask = None
        point.sp_center_feat = point.sp_center_feat.to(device)
        point.sp_center_coord = point.sp_center_coord.to(device)
        point.initial_point_assignments = initial_point_assignments

        for i, module in enumerate(self.update_superpoints_modules):
            last_iteration = (i == self.num_iterations - 1)
            point_assignments, retain_mask, point.sp_center_feat, point.sp_center_coord, points_feat, hilbert_feat_level1 = module(
                point, rawPoint_feat, rawPoint_coord, hilbert_feat_coord, num_segments_levels[-2],
                points_per_segment_levels[-2],
                hilbert_feat_levels[-2], hilbert_feat_coord_levels[-2], num_segments_levels[-1],
                points_per_segment_levels[-1],
                segments_per_level0, level_indices[(0, 1)], point_assignments, retain_mask, last_iteration,
            )

        level0_point_assignments = level_inverse_indices[(1, 0)][point_assignments]
        point.l2_initial_point_assignments = level0_point_assignments[inverse].to(device)

        sp_center_feats = [point.sp_center_feat]
        sp_center_coords = [point.sp_center_coord]

        for l in range(len(num_segments_levels) - 1, 0, -1):
            num_segments = num_segments_levels[l - 1]
            sp_center_feat = scatter_mean(hilbert_feat, level0_point_assignments, dim=0, dim_size=num_segments)
            sp_center_coord = scatter_mean(hilbert_feat_coord, level0_point_assignments, dim=0, dim_size=num_segments)
            sp_center_feats.append(sp_center_feat)
            sp_center_coords.append(sp_center_coord)

        sp_center_feats = sp_center_feats[::-1]
        sp_center_coords = sp_center_coords[::-1]

        level0_feats = sp_center_feats[0]
        level1_feats = sp_center_feats[1]
        shortcut = level0_feats

        level0_point_assignments_updated = None
        assignments = None

        for iteration in range(3):
            if self.is_cossim:
                S = F.cosine_similarity(level1_feats.unsqueeze(1), level0_feats.unsqueeze(0), dim=2)
            else:
                S = torch.matmul(level1_feats, level0_feats.T)

            S = F.softmax(S, dim=-1)
            assignments = torch.argmax(S, dim=1)
            level0_point_assignments_updated = assignments[point_assignments]

            for module in self.cross_attention_update_sp_level0:
                level0_feats, _ = module(level0_feats, level1_feats, assignments, sp_rpe=None, rawPoint_rpe=None)

            level0_feats = level0_feats + shortcut
            level_inverse_indices[(1, 0)] = assignments
            level0_point_assignments_updated = level0_point_assignments_updated

        sp_center_feats[0] = self.norm3(level0_feats)
        sp_center_coords[0] = scatter(sp_center_coords[1], level_inverse_indices[(1, 0)], dim=0, reduce="mean",
                                      dim_size=sp_center_feats[0].size(0))

        sp_center_feats = self.fuse_features(point, point_assignments, inverse, sp_center_feats, sp_center_coords,
                                             level_inverse_indices[(1, 0)], num_segments_levels)
        sp_center_feat0 = sp_center_feats[0]

        point.sp_center_feat = sp_center_feats[-1]
        point.raw_to_super_index = point_assignments[inverse].to(device)
        point.feat = point.feat.to(device)

        super_feat = point.sp_center_feat[point.raw_to_super_index]
        object_feat = sp_center_feat0[point.raw_to_super_index]

        if self.is_concat:
            super_feat_proj = self.Super_proj_mlp(super_feat)
            object_feat_proj = self.Object_proj_mlp(object_feat)
            point_feat_combined = torch.cat([point.feat, super_feat_proj, object_feat_proj], dim=-1)
        else:
            point_feat_combined = super_feat + point.feat + object_feat

        point.feat = point_feat_combined
        point.hilbert_feat = hilbert_feat_levels[-1]
        point.rawPoint_feat = rawPoint_feat
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        point.point_assignment = point_assignments

        level0_point_assignments_updated = level0_point_assignments_updated[inverse].to(device)
        point.objectLevel_raw_to_point_index = level0_point_assignments_updated

        return point