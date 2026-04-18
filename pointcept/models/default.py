import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model
from pointcept.models.losses.misc import CosineSimilarityLoss
from pointcept.models.losses.misc import BalancingLoss
from pointcept.models.losses.misc import LMNNLoss_SP_segment_OPT
from pointcept.models.losses.misc import ContrastiveLoss_SP_segment_OPT, SupervisedContrastiveLoss, \
    LMNNLoss_SP_segment_OPT_Improved, SuperPointContrastiveLoss, ImprovedSupervisedContrastiveLoss, \
    ImprovedSuperPointContrastiveLoss3, ImprovedSuperPointContrastiveLoss2, ModifiedSuperPointContrastiveLoss, \
    OptimizedSuperPointContrastiveLoss, SuperpointDiscriminativeLossopt
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_mean
import torch
from pointcept.models.SPCNet.utils.superpointMatrix import compute_graph_nn_2, \
    relax_edge_binary, compute_boundary_recall, compute_boundary_precision, compute_graph_nn_optimized, \
    compute_boundary_precision_optimized, \
    compute_boundary_recall_optimized, relax_edge_binary_optimized


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
            self,
            num_classes,
            backbone_out_channels,
            backbone=None,
            criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.sp_seg_head = (
            nn.Linear(64, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.cosine_loss = CosineSimilarityLoss(loss_weight=0.5)
        self.LMNN_loss = LMNNLoss_SP_segment_OPT()
        self.ContrastiveLoss = ContrastiveLoss_SP_segment_OPT()
        self.SPContrastiveLoss_for_raw = SupervisedContrastiveLoss(loss_weight=0.1)
        self.LMNN_loss_opt = LMNNLoss_SP_segment_OPT_Improved(loss_weight=0.001)
        self.SuperPointPushPullloss = SuperPointContrastiveLoss()
        self.SPContrastiveLoss_for_sp = ImprovedSuperPointContrastiveLoss3(total_weight=0.2)
        self.ModifiedSuperPointContrastiveLoss = ModifiedSuperPointContrastiveLoss()
        self.OptimizedSuperPointContrastiveLoss = OptimizedSuperPointContrastiveLoss(loss_weight=0.1)
        self.SuperpointDiscriminativeLoss = SuperpointDiscriminativeLossopt(loss_weight=0.4)

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        if "segment" in input_dict:
            point.segment = input_dict["segment"]
        # subpoint, supoint = point.subpoint, point.supoint
        # 取出聚合后的超点特征，形状为（m，64）,以及原始点的特征，形状为（n，64）
        superPoint_feat = point.sp_center_feat
        rawPoint_feat = point.rawPoint_feat
        raw_to_super_index = point.raw_to_super_index
        # label_inds = point.hilbert_point_label
        # hilbert_feat = point.hilbert_feat
        # num_segments0 = hilbert_feat.size(0)
        # points_per_segment0 = hilbert_feat.size(1)
        point_assignment = point.point_assignment
        # obeject_raw_to_super_index = point.objectLevel_raw_to_point_index
        ini_segment = point.initial_point_assignments
        # l2_ini_segment = point.l2_initial_point_assignments
        # hilbert_point_label1 = point.segment[point.hilbert_order].reshape(num_segments0, points_per_segment0)
        # label_inds_1d = input_dict["segment"][point.hilbert_order]
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2

        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        sp_seg_logits = self.sp_seg_head(superPoint_feat)  # sp_feat        # train
        if self.training:

            # compute superpoint label
            valid_mask = input_dict["segment"][point.hilbert_order] >= 0
            valid_label_inds = input_dict["segment"][point.hilbert_order][valid_mask]
            valid_raw_to_super_index = point_assignment[valid_mask]
            num_classes = input_dict["segment"][point.hilbert_order].max().item() + 1
            label_one_hot = F.one_hot(valid_label_inds, num_classes=num_classes).float()
            superpoint_label_counts = scatter_add(label_one_hot, valid_raw_to_super_index, dim=0,
                                                  dim_size=superPoint_feat.size(0))
            # superpoint_label_sums = superpoint_label_counts.sum(dim=1, keepdim=True) + 1e-8
            # label_probs = superpoint_label_counts / superpoint_label_sums
            superpoint_labels = torch.argmax(superpoint_label_counts, dim=1)

            loss = self.criteria(seg_logits, input_dict["segment"])
            sp_loss = self.criteria(sp_seg_logits, superpoint_labels)
            # input_dict["segment"] = input_dict["segment"]
            loss = loss + \
                   self.SuperpointDiscriminativeLoss(superPoint_feat, rawPoint_feat, point_assignment,
                                                     input_dict["segment"][point.hilbert_order]) + 0.1 * sp_loss

            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():

            # compute superpoint label
            valid_mask = input_dict["segment"][point.hilbert_order] >= 0
            valid_label_inds = input_dict["segment"][point.hilbert_order][valid_mask]
            valid_raw_to_super_index = point_assignment[valid_mask]
            num_classes = input_dict["segment"][point.hilbert_order].max().item() + 1
            label_one_hot = F.one_hot(valid_label_inds, num_classes=num_classes).float()
            superpoint_label_counts = scatter_add(label_one_hot, valid_raw_to_super_index, dim=0,
                                                  dim_size=superPoint_feat.size(0))
            # superpoint_label_sums = superpoint_label_counts.sum(dim=1, keepdim=True) + 1e-8
            # label_probs = superpoint_label_counts / superpoint_label_sums
            superpoint_labels = torch.argmax(superpoint_label_counts, dim=1)

            loss = self.criteria(seg_logits, input_dict["segment"])
            sp_loss = self.criteria(sp_seg_logits, superpoint_labels)
            # input_dict["segment"] = input_dict["segment"]
            loss = loss + \
                   self.SuperpointDiscriminativeLoss(superPoint_feat, rawPoint_feat, point_assignment,
                                                     input_dict["segment"][point.hilbert_order]) + 0.1 * sp_loss

            return dict(loss=loss, seg_logits=seg_logits, pred_in_component=raw_to_super_index,
                        initial_segment=ini_segment,
                        )
        # test
        else:
            return dict(seg_logits=seg_logits, pred_in_component=raw_to_super_index)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
            self,
            backbone=None,
            criteria=None,
            num_classes=40,
            backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
