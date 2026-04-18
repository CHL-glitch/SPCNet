from torch import nn
from pointcept.models.SPCNet.nn import SelfAttentionBlock, FFN, DropPath, LayerNorm, \
    INDEX_BASED_NORMS

__all__ = ['TransformerBlock']


# TODO: Careful with how we define the index for LayerNorm:
#  cluster-wise or cloud-wise ? Maybe cloud-wise, seems more stable...


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=True, qk_dim=8, qk_scale=None, in_rpe_dim=18, ffn_ratio=4,
                 residual_drop=None, attn_drop=None, drop_path=None, activation=nn.LeakyReLU(), norm=LayerNorm,
                 pre_norm=True, no_sa=False, no_ffn=False, k_rpe=False, q_rpe=False, v_rpe=False, k_delta_rpe=False,
                 q_delta_rpe=False, qk_share_rpe=False, q_on_minus_rpe=False, heads_share_rpe=False):
        super().__init__()

        self.dim = dim
        self.pre_norm = pre_norm

        # Self-Attention residual branch
        self.no_sa = no_sa
        if not no_sa:
            self.sa_norm = norm(dim)
            self.sa = SelfAttentionBlock(
                dim,
                num_heads=num_heads,
                in_dim=None,
                out_dim=dim,
                qkv_bias=qkv_bias,
                qk_dim=qk_dim,
                qk_scale=qk_scale,
                in_rpe_dim=in_rpe_dim,
                attn_drop=attn_drop,
                drop=residual_drop,
                k_rpe=k_rpe,
                q_rpe=q_rpe,
                v_rpe=v_rpe,
                k_delta_rpe=k_delta_rpe,
                q_delta_rpe=q_delta_rpe,
                qk_share_rpe=qk_share_rpe,
                q_on_minus_rpe=q_on_minus_rpe,
                heads_share_rpe=heads_share_rpe
            )

        # Feed-Forward Network residual branch
        self.no_ffn = no_ffn
        if not no_ffn:
            self.ffn_norm = norm(dim)
            self.ffn_ratio = ffn_ratio
            self.ffn = FFN(
                dim,
                hidden_dim=int(dim * ffn_ratio),
                activation=activation,
                drop=residual_drop
            )

        # Optional DropPath module for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path is not None and drop_path > 0 else nn.Identity()

    def forward(self, x, norm_index, edge_index=None, edge_attr=None, device=None):
        # Move all tensors and modules to the specified device
        self.to(device)
        x = x.to(device)
        norm_index = norm_index.to(device)
        if edge_index is not None:
            edge_index = edge_index.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)

        # Keep track of x for the residual connection
        shortcut = x

        # Self-Attention residual branch. Skip the SA block if no edges are provided
        if not self.no_sa:
            if self.pre_norm:
                x = self._forward_norm(self.sa_norm, x, norm_index).to(device)
                x = self.sa(x, edge_index, edge_attr=edge_attr, device=device)
                x = shortcut + self.drop_path(x).to(device)
            else:
                x = self.sa(x, edge_index, edge_attr=edge_attr, device=device)
                x = self.drop_path(x).to(device)
                x = self._forward_norm(self.sa_norm, shortcut + x, norm_index).to(device)

        # Feed-Forward Network residual branch
        if not self.no_ffn:
            if self.pre_norm:
                x = self._forward_norm(self.ffn_norm, x, norm_index).to(device)
                x = self.ffn(x).to(device)
                x = shortcut + self.drop_path(x).to(device)
            else:
                x = self.ffn(x).to(device)
                x = self.drop_path(x).to(device)
                x = self._forward_norm(self.ffn_norm, shortcut + x, norm_index).to(device)

        return x, norm_index, edge_index

    @staticmethod
    def _forward_norm(norm, x, norm_index):
        # Ensure norm is on the same device as x
        norm = norm.to(x.device)
        norm_index = norm_index.to(x.device)

        if isinstance(norm, INDEX_BASED_NORMS):
            return norm(x, batch=norm_index).to(x.device)
        return norm(x)

