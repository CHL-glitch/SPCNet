import torch
import torch.nn as nn
from functools import partial

from pointcept.models.mamba.mamba_ssm.modules.mamba_simple import Mamba
from pointcept.models.mamba.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from timm.models.layers import DropPath


class MambaBlock(nn.Module):
    def __init__(
            self, dim, layer_idx,
            norm_cls=nn.LayerNorm, fused_add_norm=False,
            residual_in_fp32=False, drop_path=0., ssm_cfg={}, factory_kwargs={}
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states, residual=None, inverse=None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        hidden_states = hidden_states.unsqueeze(0)  # ptv3的输入为（N，c），而mamba要求的输入为（B，N, C）
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual.unsqueeze(0)
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        hidden_states = hidden_states.squeeze(0)
        residual = residual.squeeze(0)
        return hidden_states[inverse], residual[inverse]

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

# import torch
# import torch.nn as nn
# from functools import partial
#
# from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
# from timm.models.layers import DropPath
# from pointcept.models.utils.structure import Point
#
#
#
# class MambaBlock(nn.Module):
#     def __init__(
#             self, dim, layer_idx,
#             norm_cls=nn.LayerNorm, fused_add_norm=False,
#             residual_in_fp32=False, drop_path=0., ssm_cfg={}, factory_kwargs={}
#     ):
#         """
#         Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"
#
#         This Block has a slightly different structure compared to a regular
#         prenorm Transformer block.
#         The standard block is: LN -> MHA/MLP -> Add.
#         [Ref: https://arxiv.org/abs/2002.04745]
#         Here we have: Add -> LN -> Mixer, returning both
#         the hidden_states (output of the mixer) and the residual.
#         This is purely for performance reasons, as we can fuse add and LayerNorm.
#         The residual needs to be provided (except for the very first block).
#         """
#         super().__init__()
#         mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
#         self.residual_in_fp32 = residual_in_fp32
#         self.fused_add_norm = fused_add_norm
#         self.mixer = mixer_cls(dim)
#         self.norm = norm_cls(dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         if self.fused_add_norm:
#             assert RMSNorm is not None, "RMSNorm import fails"
#             assert isinstance(
#                 self.norm, (nn.LayerNorm, RMSNorm)
#             ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"
#
#     def forward(
#             self, hidden_states, residual=None, inference_params=None
#     ):
#         r"""Pass the input through the encoder layer.
#
#         Args:
#             hidden_states: the sequence to the encoder layer (required).
#             residual: hidden_states = Mixer(LN(residual))
#         """
#         print(type(hidden_states))
#         if isinstance(hidden_states, Point):
#             hidden_states, residual = self._forward_point(hidden_states, residual, inference_params)
#         else:
#             hidden_states, residual = self._forward_tensor(hidden_states, residual, inference_params)
#         return hidden_states, residual
#
#     def _forward_tensor(
#             self, hidden_states, residual=None, inference_params=None
#     ):
#         if not self.fused_add_norm:
#             if residual is None:
#                 residual = hidden_states
#             else:
#                 residual = residual + self.drop_path(hidden_states)
#
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#             if self.residual_in_fp32:
#                 residual = residual.to(torch.float32)
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             if residual is None:
#                 hidden_states, residual = fused_add_norm_fn(
#                     hidden_states,
#                     self.norm.weight,
#                     self.norm.bias,
#                     residual=residual,
#                     prenorm=True,
#                     residual_in_fp32=self.residual_in_fp32,
#                     eps=self.norm.eps,
#                 )
#             else:
#                 hidden_states, residual = fused_add_norm_fn(
#                     self.drop_path(hidden_states),
#                     self.norm.weight,
#                     self.norm.bias,
#                     residual=residual,
#                     prenorm=True,
#                     residual_in_fp32=self.residual_in_fp32,
#                     eps=self.norm.eps,
#                 )
#         hidden_states = self.mixer(hidden_states, inference_params=inference_params)
#         return hidden_states, residual
#
#     # def _forward_point(
#     #         self, point, residual=None, inference_params=None
#     # ):
#     #     hidden_states = point['feat']
#     #     if residual is None:
#     #         residual = hidden_states
#     #     else:
#     #         residual = residual + self.drop_path(hidden_states)
#     #
#     #     hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#     #     if self.residual_in_fp32:
#     #         residual = residual.to(torch.float32)
#     #
#     #     hidden_states = self.mixer(hidden_states, inference_params=inference_params)
#     #     point['feat'] = hidden_states
#     #     return point, residual
#     def _forward_point(
#             self, point: Point, residual=None, inference_params=None
#     ):
#         print(point.feat.shape)
#         hidden_states = point.feat.unsqueeze(0)
#         if not self.fused_add_norm:
#             if residual is None:
#                 residual = hidden_states
#             else:
#                 residual = residual + self.drop_path(hidden_states)
#
#             hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
#             if self.residual_in_fp32:
#                 residual = residual.to(torch.float32)
#         else:
#             fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
#             if residual is None:
#                 hidden_states, residual = fused_add_norm_fn(
#                     hidden_states,
#                     self.norm.weight,
#                     self.norm.bias,
#                     residual=residual,
#                     prenorm=True,
#                     residual_in_fp32=self.residual_in_fp32,
#                     eps=self.norm.eps,
#                 )
#             else:
#                 hidden_states, residual = fused_add_norm_fn(
#                     self.drop_path(hidden_states),
#                     self.norm.weight,
#                     self.norm.bias,
#                     residual=residual,
#                     prenorm=True,
#                     residual_in_fp32=self.residual_in_fp32,
#                     eps=self.norm.eps,
#                 )
#
#         hidden_states = self.mixer(hidden_states, inference_params=inference_params)
#         point['feat'] = hidden_states.squeeze(0)
#         return point, residual
#
#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
#
