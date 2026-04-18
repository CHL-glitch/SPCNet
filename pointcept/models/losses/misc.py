

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES
from torch_scatter import scatter_add, scatter_max, scatter_mean


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
            self,
            weight=None,
            size_average=None,
            reduce=None,
            reduction="mean",
            label_smoothing=0.0,
            loss_weight=1.0,
            ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
            self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
                F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                        torch.sum(
                            pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                        )
                        + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss


"""余弦相识度loss"""


class CosineSimilarityLoss(nn.Module):
    def __init__(self, loss_weight=1.0, use_sqrt_weight=False):
        """
        Args:
            loss_weight (float): 损失的权重系数。默认值为 1.0。
            use_sqrt_weight (bool): 是否使用平方根权重以平衡超点大小的影响。默认值为 False。
        """
        super(CosineSimilarityLoss, self).__init__()
        self.loss_weight = loss_weight
        self.use_sqrt_weight = use_sqrt_weight

    def forward(self, superPoint_feat, rawPoint_feat, point_assignment):
        """
        Args:
            superPoint_feat (torch.Tensor): 形状为 (num_superpoints, feature_dim)，超点中心特征。
            rawPoint_feat (torch.Tensor): 形状为 (n_points, feature_dim)，原始点特征。
            point_assignment (torch.Tensor): 形状为 (n_points,) ，每个原始点对应的超点索引。

        Returns:
            torch.Tensor: 标量损失值。
        """
        device = superPoint_feat.device
        rawPoint_feat = rawPoint_feat.to(device)
        point_assignment = point_assignment.to(device)

        num_superpoints = superPoint_feat.size(0)
        n_points = rawPoint_feat.size(0)

        # 归一化特征
        superPoint_feat_norm = F.normalize(superPoint_feat, dim=1)  # (num_superpoints, feature_dim)
        rawPoint_feat_norm = F.normalize(rawPoint_feat, dim=1)  # (n_points, feature_dim)

        # 计算每个超点包含的点数
        counts = torch.bincount(point_assignment, minlength=num_superpoints).float()  # (num_superpoints,)

        # 计算每个超点内原始点特征的均值
        # 使用 scatter 操作
        sp_mean_feat = torch.zeros_like(superPoint_feat).to(device)  # (num_superpoints, feature_dim)
        sp_mean_feat = sp_mean_feat.index_add(0, point_assignment, rawPoint_feat_norm)

        # 避免除以零
        counts_expand = counts.unsqueeze(1).clamp(min=1.0)  # (num_superpoints, 1)
        sp_mean_feat = sp_mean_feat / counts_expand  # (num_superpoints, feature_dim)

        # 计算超点中心特征与其对应的原始点特征均值之间的余弦相似度
        cos_sim = F.cosine_similarity(superPoint_feat_norm, sp_mean_feat, dim=1)  # (num_superpoints,)
        # 计算余弦差异
        cos_diff = 1.0 - cos_sim  # (num_superpoints,)
        # 计算权重
        if self.use_sqrt_weight:
            weights = torch.sqrt(counts) / n_points  # 平方根权重
        else:
            weights = counts / n_points  # 原始权重

        # 计算加权的余弦差异
        weighted_cos_diff = cos_diff * weights  # (num_superpoints,)

        # 计算总的损失值
        total_loss = weighted_cos_diff.sum()

        return total_loss * self.loss_weight


@LOSSES.register_module()
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, loss_weight=1.0):
        """监督式对比损失函数，使用批内对比学习
        Args:
            temperature (float): 温度参数，控制分布的平滑度。默认值为 0.07。
            loss_weight (float): 损失的权重系数。默认值为 1.0。
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, superPoint_feat, rawPoint_feat, raw_to_super_index, label_inds):
        """
        Args:
            superPoint_feat (torch.Tensor): 形状为 (num_superpoints, feature_dim)，超点特征。
            rawPoint_feat (torch.Tensor): 形状为 (n, feature_dim)，原始点特征。
            raw_to_super_index (torch.Tensor): 形状为 (n,)，每个原始点对应的超点索引。
            label_inds (torch.Tensor): 形状为 (n,)，每个原始点的类别标签。

        Returns:
            torch.Tensor: 标量损失值。
        """
        device = rawPoint_feat.device
        rawPoint_feat = rawPoint_feat.to(device)
        superPoint_feat = superPoint_feat.to(device)
        raw_to_super_index = raw_to_super_index.to(device)
        label_inds = label_inds.to(device)

        # 1. 计算每个超点的类别标签（使用多数投票法）
        num_classes = label_inds.max() + 1  # 假设标签是从0开始的整数
        num_superpoints = superPoint_feat.size(0)

        # 计算每个超点中每个类别的出现次数
        # 使用线性索引：superpoint_index * num_classes + label
        linear_indices = raw_to_super_index * num_classes + label_inds  # (n,)
        counts = torch.bincount(linear_indices, minlength=num_superpoints * num_classes)
        counts = counts.view(num_superpoints, num_classes)  # (num_superpoints, num_classes)

        # 对每个超点，找到计数最多的类别，作为该超点的标签
        superPoint_labels = counts.argmax(dim=1)  # (num_superpoints,)

        # 2. 将原始点特征和超点特征拼接
        all_features = torch.cat([rawPoint_feat, superPoint_feat], dim=0)  # (n + num_superpoints, feature_dim)

        # 3. 将原始点标签和超点标签拼接
        all_labels = torch.cat([label_inds, superPoint_labels], dim=0)  # (n + num_superpoints,)

        # 4. 特征归一化
        all_features = F.normalize(all_features, dim=1)

        # 5. 使用批内对比学习，随机采样一个小批次
        batch_size = 1024  # 根据显存情况调整批次大小
        num_samples = all_features.size(0)
        if num_samples > batch_size:
            # 随机采样一个批次
            indices = torch.randperm(num_samples)[:batch_size]
            features = all_features[indices]
            labels = all_labels[indices]
        else:
            features = all_features
            labels = all_labels

        # 6. 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature  # (batch_size, batch_size)

        # 7. 创建标签掩码
        labels = labels.unsqueeze(1)
        labels_equal = torch.eq(labels, labels.T).float().to(device)

        # 去除自身匹配
        logits_mask = torch.ones_like(labels_equal) - torch.eye(labels_equal.size(0), device=device)
        labels_equal = labels_equal * logits_mask

        # 8. 计算对数概率
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)

        # 9. 计算每个样本的平均正样本对数概率
        mean_log_prob_pos = (labels_equal * log_prob).sum(dim=1) / (labels_equal.sum(dim=1) + 1e-6)

        # 10. 损失取负值并取平均
        loss = -mean_log_prob_pos.mean()
        return loss * self.loss_weight


@LOSSES.register_module()
class ImprovedSupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, loss_weight=1.0):
        """改进的监督式对比损失函数，使用批内对比学习，旨在使同一超点内的原始点特征更接近，同时不同超点间的特征保持分离。

        Args:
            temperature (float): 温度参数，控制分布的平滑度。默认值为 0.07。
            loss_weight (float): 损失的权重系数。默认值为 1.0。
        """
        super(ImprovedSupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, superPoint_feat, rawPoint_feat, raw_to_super_index, label_inds):
        """
        Args:
            superPoint_feat (torch.Tensor): 形状为 (num_superpoints, feature_dim)，超点特征。
            rawPoint_feat (torch.Tensor): 形状为 (n, feature_dim)，原始点特征。
            raw_to_super_index (torch.Tensor): 形状为 (n,)，每个原始点对应的超点索引。
            label_inds (torch.Tensor): 形状为 (n,)，每个原始点的类别标签。

        Returns:
            torch.Tensor: 标量损失值。
        """
        device = rawPoint_feat.device
        rawPoint_feat = rawPoint_feat.to(device)
        superPoint_feat = superPoint_feat.to(device)
        raw_to_super_index = raw_to_super_index.to(device)
        label_inds = label_inds.to(device)

        # 1. 计算每个超点的类别标签（使用多数投票法）
        num_classes = label_inds.max() + 1  # 假设标签是从0开始的整数
        num_superpoints = superPoint_feat.size(0)

        # 使用scatter_方法高效统计每个超点中每个类别的计数
        counts = torch.zeros((num_superpoints, num_classes), device=device)
        counts.index_add_(0, raw_to_super_index, F.one_hot(label_inds, num_classes=num_classes).float())

        # 对每个超点，找到计数最多的类别，作为该超点的标签
        superPoint_labels = counts.argmax(dim=1)  # (num_superpoints,)

        # 2. 将原始点特征和超点特征拼接
        all_features = torch.cat([rawPoint_feat, superPoint_feat], dim=0)  # (n + num_superpoints, feature_dim)

        # 3. 将原始点标签和超点标签拼接
        all_labels = torch.cat([label_inds, superPoint_labels], dim=0)  # (n + num_superpoints,)

        # 4. 特征归一化
        all_features = F.normalize(all_features, dim=1)

        # 5. 构建对比学习的掩码矩阵
        labels = all_labels.unsqueeze(0)  # (1, n + num_superpoints)
        mask = torch.eq(labels, labels.T).float().to(device)  # (n + num_superpoints, n + num_superpoints)

        # 6. 计算相似度矩阵
        anchor_dot_contrast = torch.matmul(all_features,
                                           all_features.T) / self.temperature  # (n + num_superpoints, n + num_superpoints)

        # 7. 去除自身匹配
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask

        # 8. 计算Logits
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask

        # 9. 计算对比损失
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-9)

        # 10. 计算每个样本的损失
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-9)

        # 11. 损失取负值并取平均
        loss = -mean_log_prob_pos.mean()

        return loss * self.loss_weight


@LOSSES.register_module()
class SuperPointContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, loss_weight=1.0):
        """超点对比损失函数，通过使每个超点内的原始点特征更接近超点中心特征，不同超点间的特征保持分离。

        Args:
            temperature (float): 温度参数，控制分布的平滑度。默认值为 0.07。
            loss_weight (float): 损失的权重系数。默认值为 1.0。
        """
        super(SuperPointContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, superPoint_feat, rawPoint_feat, raw_to_super_index):
        """
        Args:
            superPoint_feat (torch.Tensor): 形状为 (m, feature_dim)，超点特征。
            rawPoint_feat (torch.Tensor): 形状为 (n, feature_dim)，原始点特征。
            raw_to_super_index (torch.Tensor): 形状为 (n,)，每个原始点对应的超点索引。

        Returns:
            torch.Tensor: 标量损失值。
        """
        device = rawPoint_feat.device
        rawPoint_feat = rawPoint_feat.to(device)
        superPoint_feat = superPoint_feat.to(device)
        raw_to_super_index = raw_to_super_index.to(device)

        # 特征归一化
        rawPoint_feat = F.normalize(rawPoint_feat, dim=1)  # (n, feature_dim)
        superPoint_feat = F.normalize(superPoint_feat, dim=1)  # (m, feature_dim)

        # 计算相似度矩阵 (n, m)
        similarity_matrix = torch.matmul(rawPoint_feat, superPoint_feat.T) / self.temperature  # (n, m)

        # 每个原始点的目标是其对应的超点索引
        targets = raw_to_super_index.long()  # (n,)

        # 计算交叉熵损失
        loss = F.cross_entropy(similarity_matrix, targets)

        return loss * self.loss_weight


@LOSSES.register_module()
class ImprovedSuperPointContrastiveLoss2(nn.Module):
    def __init__(self, temperature=0.07, num_negatives=10, loss_weight=1.0):
        """改进的超点对比损失函数，通过使每个超点内的原始点特征更接近超点中心特征，同时拉开不同超点间的特征距离。

        Args:
            temperature (float): 温度参数，控制分布的平滑度。默认值为 0.07。
            num_negatives (int): 每个样本的负样本数量。默认值为 5。
            loss_weight (float): 损失的权重系数。默认值为 1.0。
        """
        super(ImprovedSuperPointContrastiveLoss2, self).__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.loss_weight = loss_weight

    def forward(self, superPoint_feat, rawPoint_feat, raw_to_super_index):
        """
        Args:
            superPoint_feat (torch.Tensor): 形状为 (m, feature_dim)，超点特征。
            rawPoint_feat (torch.Tensor): 形状为 (n, feature_dim)，原始点特征。
            raw_to_super_index (torch.Tensor): 形状为 (n,)，每个原始点对应的超点索引。

        Returns:
            torch.Tensor: 标量损失值。
        """
        device = rawPoint_feat.device
        rawPoint_feat = F.normalize(rawPoint_feat.to(device), dim=1)
        superPoint_feat = F.normalize(superPoint_feat.to(device), dim=1)
        raw_to_super_index = raw_to_super_index.to(device)

        n = rawPoint_feat.size(0)
        m = superPoint_feat.size(0)

        # 获取每个原始点对应的超点特征（正样本）
        positive_super_feats = superPoint_feat[raw_to_super_index]  # (n, feature_dim)

        # 计算正样本相似度
        positive_logits = torch.sum(rawPoint_feat * positive_super_feats, dim=1,
                                    keepdim=True) / self.temperature  # (n, 1)

        # 创建掩码以排除正样本索引
        mask = torch.ones((n, m), dtype=torch.bool, device=device)
        mask[torch.arange(n), raw_to_super_index] = False  # 将正样本位置设为False

        # 创建用于采样负样本的概率分布
        probs = mask.float()
        probs_sum = probs.sum(dim=1, keepdim=True)
        # 处理可能出现的除零问题
        probs_sum[probs_sum == 0] = 1.0
        probs = probs / probs_sum

        # 使用torch.multinomial采样负样本索引
        negative_indices = torch.multinomial(probs, num_samples=self.num_negatives,
                                             replacement=True)  # (n, num_negatives)

        # 获取负样本特征
        negative_super_feats = superPoint_feat[negative_indices]  # (n, num_negatives, feature_dim)

        # 计算负样本相似度
        negative_logits = torch.bmm(negative_super_feats, rawPoint_feat.unsqueeze(2)).squeeze(
            2) / self.temperature  # (n, num_negatives)

        # 拼接正样本和负样本相似度
        logits = torch.cat([positive_logits, negative_logits], dim=1)  # (n, 1 + num_negatives)

        # 创建标签：第一个位置是正样本
        labels = torch.zeros(n, dtype=torch.long, device=device)  # (n,)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss * self.loss_weight


@LOSSES.register_module()
class ModifiedSuperPointContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, loss_weight=0.1, diversity_weight=0.001, batch_size=1024):
        """
        修改后的超点对比损失函数，防止特征塌陷，促进模型学习更加区分的特征。

        Args:
            temperature (float): 温度参数，控制分布的平滑度。
            loss_weight (float): 对比损失的权重系数。
            diversity_weight (float): 特征多样性损失的权重系数。
            batch_size (int): 分批处理的原始点数。
        """
        super(ModifiedSuperPointContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.diversity_weight = diversity_weight
        self.batch_size = batch_size

    def forward(self, superPoint_feat, rawPoint_feat, raw_to_super_index, label_inds):
        device = rawPoint_feat.device
        rawPoint_feat = F.normalize(rawPoint_feat, dim=1)
        superPoint_feat = F.normalize(superPoint_feat, dim=1)
        raw_to_super_index = raw_to_super_index.to(device)
        label_inds = label_inds.to(device)

        n = rawPoint_feat.size(0)
        m = superPoint_feat.size(0)

        # 获取每个原始点对应的超点特征
        positive_super_feats = superPoint_feat[raw_to_super_index]  # (n, feature_dim)

        # 计算相似度矩阵
        logits = torch.matmul(rawPoint_feat, superPoint_feat.T) / self.temperature  # (n, m)

        # 构建标签，正样本为对应的超点索引
        labels = raw_to_super_index  # (n,)

        # 使用交叉熵损失
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        # 增加特征多样性损失，防止特征塌陷
        # 计算特征协方差矩阵，鼓励特征的去相关性
        feat_cov = torch.mm(rawPoint_feat.T, rawPoint_feat) / n
        diversity_loss = ((feat_cov - torch.eye(rawPoint_feat.size(1), device=device)) ** 2).sum()

        total_loss = loss * self.loss_weight + diversity_loss * self.diversity_weight
        return total_loss


@LOSSES.register_module()
class OptimizedSuperPointContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, loss_weight=1, batch_size=1024, total_weight=0.1):
        super(OptimizedSuperPointContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.batch_size = batch_size
        self.total_weight = total_weight

    def forward(self, superPoint_feat, rawPoint_feat, raw_to_super_index, label_inds):
        device = rawPoint_feat.device
        rawPoint_feat = F.normalize(rawPoint_feat.to(device), dim=1)
        superPoint_feat = F.normalize(superPoint_feat.to(device), dim=1)
        raw_to_super_index = raw_to_super_index.to(device)
        label_inds = label_inds.to(device)

        num_classes = label_inds.max() + 1
        num_superpoints = superPoint_feat.size(0)

        # 计算每个超点的类别标签（使用多数投票法）
        linear_indices = raw_to_super_index * num_classes + label_inds
        counts = torch.bincount(linear_indices, minlength=num_superpoints * num_classes)
        counts = counts.view(num_superpoints, num_classes)
        superPoint_labels = counts.argmax(dim=1)

        # 获取每个原始点对应的超点特征和标签
        positive_super_feats = superPoint_feat[raw_to_super_index]
        positive_super_labels = superPoint_labels[raw_to_super_index]

        n = rawPoint_feat.size(0)
        total_loss = 0.0
        count = 0

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_raw = rawPoint_feat[start:end]
            batch_positive_super_feats = positive_super_feats[start:end]
            batch_positive_super_labels = positive_super_labels[start:end]

            # 正样本相似度
            positive_logits = torch.sum(batch_raw * batch_positive_super_feats, dim=1, keepdim=True) / self.temperature

            # 所有超点相似度
            all_logits = torch.matmul(batch_raw, superPoint_feat.T) / self.temperature

            # 创建掩码
            mask = torch.zeros_like(all_logits, dtype=torch.bool, device=device)
            mask.scatter_(1, raw_to_super_index[start:end].unsqueeze(1), True)

            # 处理负样本
            label_mask = (batch_positive_super_labels.unsqueeze(1) == superPoint_labels.unsqueeze(0)).to(device)
            combined_mask = mask | label_mask
            negative_logits = all_logits.masked_fill(combined_mask, float('-inf'))

            # 困难负样本挖掘
            k = 10  # 选择前k个最难的负样本
            hardest_negative_logits, _ = torch.topk(negative_logits, k=k, dim=1)

            # 计算InfoNCE损失
            pos_term = torch.exp(positive_logits)
            neg_term = torch.sum(torch.exp(hardest_negative_logits), dim=1, keepdim=True)
            loss = -torch.log(pos_term / (pos_term + neg_term + 1e-8)).mean()

            total_loss += loss
            count += 1

        avg_loss = total_loss / count
        return avg_loss * self.loss_weight * self.total_weight


@LOSSES.register_module()
class ImprovedSuperPointContrastiveLoss3(nn.Module):
    def __init__(self, temperature=0.07, loss_weight=0.7, superpoint_loss_weight=0.3, batch_size=1024,
                 total_weight=0.1):
        """
        改进的超点对比损失函数，通过对比学习使同一超点内相同标签的原始点特征与超点特征更接近，
        不同超点之间的不同标签原始点特征保持分离。同时加入超点之间的对比损失，推远不同标签的超点特征。

        Args:
            temperature (float): 温度参数，控制分布的平滑度。默认值为 0.07。
            loss_weight (float): 损失的权重系数。默认值为 1.0。
            superpoint_loss_weight (float): 超点间对比损失的权重。默认值为 1.0。
            batch_size (int): 用于分批处理的原始点数。默认值为 1024。
        """
        super(ImprovedSuperPointContrastiveLoss3, self).__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight
        self.superpoint_loss_weight = superpoint_loss_weight
        self.batch_size = batch_size
        self.total_weight = total_weight

    def forward(self, superPoint_feat, rawPoint_feat, raw_to_super_index, label_inds):
        device = rawPoint_feat.device
        rawPoint_feat = F.normalize(rawPoint_feat.to(device), dim=1)
        superPoint_feat = F.normalize(superPoint_feat.to(device), dim=1)
        raw_to_super_index = raw_to_super_index.to(device)
        label_inds = label_inds.to(device)

        # 1. 计算每个超点的类别标签（使用多数投票法）
        num_classes = label_inds.max() + 1  # 假设标签是从0开始的整数
        num_superpoints = superPoint_feat.size(0)

        # 计算每个超点中每个类别的出现次数
        linear_indices = raw_to_super_index * num_classes + label_inds  # (n,)
        counts = torch.bincount(linear_indices, minlength=num_superpoints * num_classes)
        counts = counts.view(num_superpoints, num_classes)  # (num_superpoints, num_classes)

        # 对每个超点，找到计数最多的类别，作为该超点的标签
        superPoint_labels = counts.argmax(dim=1)  # (num_superpoints,)

        # 2. 获取每个原始点对应的超点特征和标签
        positive_super_feats = superPoint_feat[raw_to_super_index]  # (n, feature_dim)
        positive_super_labels = superPoint_labels[raw_to_super_index]  # (n,)

        # 3. 计算相似度和损失分块
        n = rawPoint_feat.size(0)
        m = superPoint_feat.size(0)
        total_raw_to_super_loss = 0.0
        total_superpoint_contrastive_loss = 0.0
        count = 0

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_raw = rawPoint_feat[start:end]  # (batch, feature_dim)
            batch_positive_super_feats = positive_super_feats[start:end]  # (batch, feature_dim)
            batch_positive_super_labels = positive_super_labels[start:end]  # (batch,)

            # 正样本相似度
            positive_logits = torch.sum(batch_raw * batch_positive_super_feats, dim=1,
                                        keepdim=True) / self.temperature  # (batch, 1)

            # 所有超点相似度
            all_logits = torch.matmul(batch_raw, superPoint_feat.T) / self.temperature  # (batch, m)

            # 创建掩码
            mask = torch.zeros_like(all_logits, dtype=torch.bool, device=device)
            mask.scatter_(1, raw_to_super_index[start:end].unsqueeze(1), True)

            # 处理负样本
            label_mask = (batch_positive_super_labels.unsqueeze(1) == superPoint_labels.unsqueeze(0)).to(
                device)  # (batch, m)
            combined_mask = mask | label_mask  # 同超点和同标签都不能作为负样本
            negative_logits = all_logits.masked_fill(combined_mask, float('-inf'))  # (batch, m)

            # 分母
            exp_positive = torch.exp(positive_logits)  # (batch, 1)
            exp_negative = torch.exp(negative_logits).sum(dim=1, keepdim=True)  # (batch, 1)

            # 原始点-超点的对比损失
            raw_to_super_loss = -torch.log(exp_positive / (exp_positive + exp_negative + 1e-8)).mean()
            total_raw_to_super_loss += raw_to_super_loss

            count += 1

        raw_to_super_loss_avg = total_raw_to_super_loss / count

        # # 4. 计算超点之间的对比损失，仅推远不同标签的超点
        # # 计算超点相似度矩阵
        # superpoint_sim = torch.matmul(superPoint_feat, superPoint_feat.T) / self.temperature  # (m, m)
        #
        # # 创建掩码，标记不同标签的超点
        # different_label_mask = (superPoint_labels.unsqueeze(1) != superPoint_labels.unsqueeze(0)).to(device)  # (m, m)
        #
        # # 提取不同标签超点的相似度
        # different_super_logits = superpoint_sim.masked_select(different_label_mask)
        #
        # # 定义推远损失，使用 softplus 使得相似度较高的不同标签超点对有更高的损失
        # superpoint_contrastive_loss = F.softplus(different_super_logits).mean()

        # 5. 总损失 = 原始点-超点损失 + 超点-超点对比损失
        # total_loss = raw_to_super_loss_avg * self.loss_weight + superpoint_contrastive_loss * self.superpoint_loss_weight
        total_loss = raw_to_super_loss_avg * self.loss_weight
        return total_loss * self.total_weight


@LOSSES.register_module()
class LMNNLoss(nn.Module):
    """自定义 Large Margin Nearest Neighbor (LMNN) 损失函数。"""

    def __init__(self, n_neighbors):
        super(LMNNLoss, self).__init__()
        self.n_neighbors = n_neighbors

    def _select_targets(self, outputs, label_inds):
        """为每个样本选择目标邻居。"""
        n_samples = outputs.size(0)
        target_neighbors = torch.empty((n_samples, self.n_neighbors), dtype=torch.long, device=outputs.device)
        unique_labels = label_inds.unique()

        for label in unique_labels:
            inds = (label_inds == label).nonzero(as_tuple=True)[0]
            dd = torch.cdist(outputs[inds], outputs[inds], p=2).pow(2)
            dd.fill_diagonal_(float('inf'))
            _, nn = dd.topk(self.n_neighbors, largest=False, dim=1)
            target_neighbors[inds] = inds[nn]
        return target_neighbors

    def _find_impostors(self, furthest_neighbors, outputs, label_inds):
        """查找违反边界条件的 impostors。"""
        margin_radii = 1 + torch.cdist(outputs[furthest_neighbors], outputs).pow(2)
        impostors = []
        unique_labels = label_inds.unique()

        for label in unique_labels[:-1]:
            in_inds = (label_inds == label).nonzero(as_tuple=True)[0]
            out_inds = (label_inds > label).nonzero(as_tuple=True)[0]
            dist = torch.cdist(outputs[out_inds], outputs[in_inds], p=2).pow(2)

            mask = dist < margin_radii[out_inds][:, None]
            i1, j1 = mask.nonzero(as_tuple=True)
            mask = dist < margin_radii[in_inds]
            i2, j2 = mask.nonzero(as_tuple=True)

            i = torch.cat((i1, i2))
            j = torch.cat((j1, j2))

            if i.size(0) > 0:
                pairs = torch.stack((i, j), dim=1).unique(dim=0)
                impostors.append(torch.stack((in_inds[pairs[:, 1]], out_inds[pairs[:, 0]]), dim=0))

        if len(impostors) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=outputs.device)

        return torch.cat(impostors, dim=1)

    def forward(self, outputs, label_inds):
        n = outputs.size(0)

        # 选择目标邻居
        target_neighbors = self._select_targets(outputs, label_inds)

        # 计算成对的欧氏距离平方
        distance = torch.cdist(outputs, outputs, p=2).pow(2)

        # 计算 margin_radii: 1 + 最远目标邻居的最大距离
        furthest_distances, _ = distance.gather(1, target_neighbors).max(dim=1)
        margin_radii = 1 + furthest_distances

        # 查找违反边界条件的 impostors
        furthest_neighbors = target_neighbors.gather(1, furthest_distances.unsqueeze(1).argmax(dim=1,
                                                                                               keepdim=True)).squeeze()
        impostors = self._find_impostors(furthest_neighbors, outputs, label_inds)

        if impostors.size(1) == 0:
            pull_loss = distance.gather(1, target_neighbors).sum()
            return pull_loss / n

        # 计算拉近目标邻居的损失
        pull_loss = distance.gather(1, target_neighbors).sum()

        # 计算推开非目标邻居的损失
        mask = torch.ones_like(distance, dtype=torch.bool)
        mask.scatter_(1, target_neighbors, False)
        mask.fill_diagonal_(False)

        max_dist = margin_radii.unsqueeze(1)
        push_loss = (max_dist - distance.unsqueeze(1)).clamp(min=0)
        push_loss = (push_loss * mask.unsqueeze(1)).sum()

        total_loss = pull_loss + push_loss
        return total_loss / n


@LOSSES.register_module()
class LMNNLoss_SP(nn.Module):
    """自定义 Large Margin Nearest Neighbor (LMNN) 损失函数。"""

    def __init__(self, n_neighbors):
        super(LMNNLoss_SP, self).__init__()
        self.n_neighbors = n_neighbors

    def _select_targets(self, segment_center, segment_points, label_inds):
        """为每个段的超点中心选择目标邻居。"""
        n_segments, points_per_segment, _ = segment_points.size()
        target_neighbors = torch.empty((n_segments, self.n_neighbors), dtype=torch.long, device=segment_points.device)

        for i in range(n_segments):
            same_label_inds = (label_inds[i] == label_inds[i][0]).nonzero(as_tuple=True)[0]
            if same_label_inds.size(0) > 1:
                dd = torch.cdist(segment_center[i].unsqueeze(0), segment_points[i][same_label_inds], p=2).pow(2)
                _, nn = dd.topk(self.n_neighbors, largest=False, dim=1)
                target_neighbors[i] = same_label_inds[nn.squeeze()]
            else:
                target_neighbors[i] = same_label_inds.expand(self.n_neighbors)

        return target_neighbors

    def _find_impostors(self, furthest_neighbors, segment_center, segment_points, label_inds):
        """查找违反边界条件的 impostors。"""
        n_segments, points_per_segment, _ = segment_points.size()
        impostors = []

        for i in range(n_segments):
            different_label_inds = (label_inds[i] != label_inds[i][0]).nonzero(as_tuple=True)[0]
            if different_label_inds.size(0) > 0:
                margin_radius = 1 + torch.cdist(segment_center[i].unsqueeze(0),
                                                segment_points[i][furthest_neighbors[i]].unsqueeze(0), p=2).pow(2)
                dist = torch.cdist(segment_points[i][different_label_inds], segment_center[i].unsqueeze(0), p=2).pow(2)
                mask = dist < margin_radius
                impostors.append(different_label_inds[mask.squeeze()])

        if len(impostors) == 0:
            return torch.empty((n_segments, 0), dtype=torch.long, device=segment_points.device)

        return torch.cat(impostors, dim=0)

    def forward(self, outputs, label_inds):
        # 获取段数量和每段的点数量
        n_segments, points_per_segment, _ = outputs.size()
        n = n_segments * points_per_segment

        # 假设输出的第一个点是每个段的超点中心特征
        segment_center = outputs[:, 0, :]
        segment_points = outputs[:, 1:, :]

        # 选择目标邻居
        target_neighbors = self._select_targets(segment_center, segment_points, label_inds)

        # 计算段内成对的欧氏距离平方
        distance = torch.cdist(segment_points, segment_center.unsqueeze(1), p=2).pow(2)

        # 计算 margin_radii: 1 + 最远目标邻居的最大距离
        furthest_distances, _ = distance.gather(1, target_neighbors.unsqueeze(1)).max(dim=1)
        margin_radii = 1 + furthest_distances

        # 查找违反边界条件的 impostors
        furthest_neighbors = target_neighbors.gather(1, furthest_distances.unsqueeze(1).argmax(dim=1,
                                                                                               keepdim=True)).squeeze()
        impostors = self._find_impostors(furthest_neighbors, segment_center, segment_points, label_inds)

        if impostors.size(1) == 0:
            pull_loss = distance.gather(1, target_neighbors.unsqueeze(1)).sum()
            return pull_loss / n

        # 计算拉近目标邻居的损失
        pull_loss = distance.gather(1, target_neighbors.unsqueeze(1)).sum()

        # 计算推开非目标邻居的损失
        max_dist = margin_radii.unsqueeze(1)
        push_loss = (max_dist - distance).clamp(min=0)
        push_loss = push_loss.gather(1, impostors.unsqueeze(1)).sum()

        total_loss = pull_loss + push_loss
        return total_loss / n


@LOSSES.register_module()
class LMNNLoss_SP_OPT(nn.Module):
    """自定义 Large Margin Nearest Neighbor (LMNN) 损失函数。"""

    def __init__(self):
        super(LMNNLoss_SP_OPT, self).__init__()
        # self.n_neighbors = n_neighbors  # 定义最近邻的数量

    def _select_targets(self, segment_center, outputs, label_inds):
        """为每个段的超点中心选择目标邻居。"""
        # 假设输出的第一个点是每个段的超点中心特征
        segment_points = outputs

        n_segments, points_per_segment, _ = segment_points.size()
        same_label_mask = label_inds.unsqueeze(1) == label_inds.unsqueeze(2)

        # 计算每个段中的同标签点数
        same_label_counts = same_label_mask.sum(dim=2)

        # 确保 k 不超过同标签点数的最小值
        k = 15

        # # 如果 k 值大于同标签点数，将其调整为同标签点数的最大值 if k < self.n_neighbors: print( f"Warning: n_neighbors ({self.n_neighbors})
        # exceeds the number of available points with the same label. " f"Using k={k} instead.")

        # 计算每个 segment_center 和其 34 个点的欧式距离
        dd = torch.cdist(segment_center.unsqueeze(1), segment_points, p=2).squeeze(1).pow(2)
        dd = dd.unsqueeze(-1).expand_as(same_label_mask)
        dd = dd.masked_fill(~same_label_mask, float('inf'))
        _, target_neighbors = dd.topk(k, dim=2, largest=False)

        return target_neighbors

    def _find_impostors(self, furthest_neighbors, segment_center, segment_points, label_inds):
        """查找违反边界条件的 impostors。"""
        n_segments, points_per_segment, _ = segment_points.size()

        # 创建一个掩码，用于标识不同标签的点
        different_label_mask = label_inds.unsqueeze(1) != label_inds.unsqueeze(2)

        # 计算每个段的 margin_radius
        margin_radius = 1 + torch.cdist(
            segment_center.unsqueeze(1),
            segment_points.gather(1, furthest_neighbors.unsqueeze(-1).expand(-1, -1, segment_center.size(-1))),
            p=2
        ).pow(2).max(dim=2, keepdim=True).values

        # 计算段内所有点到段中心的距离平方
        dist = torch.cdist(segment_points, segment_center.unsqueeze(1), p=2).pow(2)

        # 对于同一标签的点，将距离设为无穷大，避免它们被认为是 impostor
        dist_masked = dist.masked_fill(~different_label_mask, float('inf'))

        # 查找距离小于 margin_radius 的不同标签的点，即为 impostor
        impostors = (dist_masked < margin_radius).nonzero(as_tuple=False)
        return impostors  # 返回所有违反条件的 impostors 索引

    def forward(self, segment_center, outputs, label_inds):
        # 获取段数量和每段的点数量
        n_segments, points_per_segment, _ = outputs.size()

        # 将 label_inds reshape 成 (num_segments, points_per_segment) 的形状
        label_inds = label_inds.view(n_segments, points_per_segment)

        segment_points = outputs
        target_neighbors = self._select_targets(segment_center, segment_points, label_inds)

        # 选择目标邻居

        # 计算段内成对的欧氏距离平方
        distance = torch.cdist(segment_points, segment_center.unsqueeze(1), p=2).pow(2)

        # 计算 margin_radii: 1 + 最远目标邻居的最大距离
        furthest_distances = distance.gather(2, target_neighbors).max(dim=2, keepdim=True).values
        margin_radii = 1 + furthest_distances

        # 查找违反边界条件的 impostors
        impostors = self._find_impostors(target_neighbors, segment_center, segment_points, label_inds)

        # 如果没有 impostors，则只计算拉近目标邻居的损失
        if impostors.size(0) == 0:
            pull_loss = distance.gather(2, target_neighbors).sum()
            return pull_loss / (n_segments * points_per_segment)

        # 计算拉近目标邻居的损失
        pull_loss = distance.gather(2, target_neighbors).sum()

        # 计算推开非目标邻居的损失
        impostor_distances = distance.gather(1, impostors[:, 0].unsqueeze(1)).gather(2, impostors[:, 1].unsqueeze(
            1).unsqueeze(2))
        push_loss = (margin_radii - impostor_distances).clamp(min=0).sum()

        # 返回总损失
        total_loss = pull_loss + push_loss
        return total_loss / (n_segments * points_per_segment)


@LOSSES.register_module()
class LMNNLoss_SP_segment(nn.Module):
    """自定义 Large Margin Nearest Neighbor (LMNN) 损失函数。"""

    def __init__(self, alpha=0.5):
        super(LMNNLoss_SP_segment, self).__init__()
        self.alpha = alpha  # 比例系数

    def _compute_mahalanobis_distance(self, center, points):
        """计算 Mahalanobis 距离"""
        diff = points - center.unsqueeze(1)
        cov = torch.matmul(diff.transpose(1, 2), diff) / (points.size(1) - 1)
        inv_cov = torch.inverse(cov + torch.eye(cov.size(-1)).to(cov.device) * 1e-6)
        mdist = torch.sqrt(torch.einsum('bij, bjk, bik -> bi', diff, inv_cov, diff))
        return mdist.sum(dim=1)

    def forward(self, segment_center, rawPoint_feat, label_inds):
        n_segments, points_per_segment, feat_dim = rawPoint_feat.size()

        # 计算每个组的segment_center标签
        label_inds = label_inds.view(n_segments, points_per_segment)
        majority_labels, _ = torch.mode(label_inds, dim=1)  # 通过投票计算majority标签

        # 获取与segment_center标签相同和不同的原始点的索引
        same_label_mask = label_inds == majority_labels.unsqueeze(1)
        different_label_mask = ~same_label_mask

        # 取出与segment_center标签相同的原始点
        same_label_points = rawPoint_feat[same_label_mask].view(n_segments, -1, feat_dim)
        same_label_center = segment_center[same_label_mask.any(dim=1)]

        # 取出与segment_center标签不同的原始点
        different_label_points = rawPoint_feat[different_label_mask].view(n_segments, -1, feat_dim)
        different_label_center = segment_center[different_label_mask.any(dim=1)]

        # 计算第一个部分的损失：拉近目标邻居的损失（pullloss）
        if same_label_points.size(1) > 0:
            pull_loss = self._compute_mahalanobis_distance(same_label_center, same_label_points).mean()
        else:
            pull_loss = torch.tensor(0.0, device=segment_center.device)

        # 计算第二个部分的损失：推开非目标邻居的损失（pushloss）
        if different_label_points.size(1) > 0:
            push_loss = self._compute_mahalanobis_distance(different_label_center, different_label_points).mean()
        else:
            push_loss = torch.tensor(0.0, device=segment_center.device)

        # 计算总损失
        total_loss = self.alpha * pull_loss - (1 - self.alpha) * push_loss
        return total_loss


@LOSSES.register_module()
class LMNNLoss_SP_segment_OPT(nn.Module):
    """自定义 Large Margin Nearest Neighbor (LMNN) 损失函数，包含 margin 控制以防止过度分离。"""

    def __init__(self, alpha=0.5, margin=0.5, normalize=True, reg_lambda=0.01):
        super(LMNNLoss_SP_segment_OPT, self).__init__()
        self.alpha = alpha  # 比例系数
        self.margin = margin  # margin 参数
        self.normalize = normalize  # 是否进行归一化
        self.reg_lambda = reg_lambda  # 正则化项的权重

    def _compute_euclidean_distance(self, centers, points):
        """计算欧式距离"""
        # 计算差异
        diff = points - centers

        # 计算欧式距离
        edist = torch.sqrt(torch.sum(diff ** 2, dim=1))

        # 仅当normalize为True时进行归一化
        if self.normalize:
            edist = edist / (torch.norm(edist, dim=0, keepdim=True) + 1e-6)

        # 返回距离和
        return edist.sum()

    def forward(self, segment_center, rawPoint_feat, label_inds):
        n_segments, points_per_segment, feat_dim = rawPoint_feat.size()

        # 计算每个组的segment_center标签
        majority_labels, _ = torch.mode(label_inds, dim=1)  # 通过投票计算majority标签

        # 获取与segment_center标签相同和不同的原始点的掩码
        same_label_mask = (label_inds == majority_labels.unsqueeze(1))
        different_label_mask = ~same_label_mask

        # 取出与segment_center标签相同的原始点
        same_label_points = rawPoint_feat[same_label_mask].view(-1, feat_dim)
        same_label_centers = segment_center.repeat_interleave(same_label_mask.sum(dim=1), dim=0)

        # 取出与segment_center标签不同的原始点
        different_label_points = rawPoint_feat[different_label_mask].view(-1, feat_dim)
        different_label_centers = segment_center.repeat_interleave(different_label_mask.sum(dim=1), dim=0)

        # 计算拉近目标邻居的损失（pull_loss）
        if same_label_points.size(0) > 0:
            pull_loss = self._compute_euclidean_distance(same_label_centers, same_label_points).mean()
        else:
            pull_loss = torch.tensor(0.0, device=segment_center.device)

        # 计算推开非目标邻居的损失（push_loss）并加入 margin 控制
        if different_label_points.size(0) > 0:
            diff_dist = self._compute_euclidean_distance(different_label_centers, different_label_points)
            # 使用平滑损失函数（hinge loss）替代torch.clamp
            push_loss = torch.relu(self.margin + pull_loss - diff_dist).mean()
        else:
            push_loss = torch.tensor(0.0, device=segment_center.device)

        # 归一化处理
        total_pull_loss = pull_loss / (n_segments + 1e-6)  # 根据 segment 数量归一化
        total_push_loss = push_loss / (n_segments + 1e-6)  # 根据 segment 数量归一化

        # 增加正则化项（例如L2正则化）
        reg_loss = self.reg_lambda * torch.sum(segment_center ** 2) / (2.0 * n_segments)

        # 计算总损失
        total_loss = self.alpha * total_pull_loss + (1 - self.alpha) * total_push_loss + reg_loss
        return total_loss


@LOSSES.register_module()
class LMNNLoss_SP_segment_OPT_Improved(nn.Module):
    """改进后的 Large Margin Nearest Neighbor (LMNN) 损失函数，包含标准化的损失公式、改进的距离度量和其他优化。"""

    def __init__(self, margin=0.5, reg_lambda=0.1, loss_weight=0.1):
        super(LMNNLoss_SP_segment_OPT_Improved, self).__init__()
        self.margin = margin  # margin 参数
        self.reg_lambda = reg_lambda  # 正则化项的权重
        self.loss_weight = loss_weight

    def _compute_cosine_distance(self, x, y):
        """计算余弦距离"""
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        cosine_similarity = torch.sum(x_norm * y_norm, dim=1)
        cosine_distance = 1 - cosine_similarity
        return cosine_distance  # 返回距离

    def forward(self, segment_center, rawPoint_feat, label_inds):
        n_segments, points_per_segment, feat_dim = rawPoint_feat.size()

        # 计算每个超点的标签（多数投票）
        majority_labels, _ = torch.mode(label_inds, dim=1)  # 形状：(n_segments,)

        # 获取与超点标签相同和不同的点的掩码
        same_label_mask = (label_inds == majority_labels.unsqueeze(1))  # 形状：(n_segments, points_per_segment)
        different_label_mask = ~same_label_mask  # 形状：(n_segments, points_per_segment)

        # 展平特征和掩码
        rawPoint_feat_flat = rawPoint_feat.view(n_segments * points_per_segment, feat_dim)  # 形状：(N, feat_dim)
        label_inds_flat = label_inds.view(-1)  # 形状：(N,)
        same_label_mask_flat = same_label_mask.view(-1)  # 形状：(N,)
        different_label_mask_flat = different_label_mask.view(-1)  # 形状：(N,)

        # 生成对应的超点中心索引
        segment_center_expanded = segment_center.unsqueeze(1).expand(-1, points_per_segment, -1)
        segment_center_flat = segment_center_expanded.contiguous().view(-1, feat_dim)  # 形状：(N, feat_dim)

        # 取出与超点标签相同的点及其对应的中心（正样本）
        positive_points = rawPoint_feat_flat[same_label_mask_flat]
        positive_centers = segment_center_flat[same_label_mask_flat]

        # 取出与超点标签不同的点及其对应的中心（负样本）
        negative_points = rawPoint_feat_flat[different_label_mask_flat]
        negative_centers = segment_center_flat[different_label_mask_flat]

        # 如果没有正样本或负样本，返回零损失
        if positive_points.size(0) == 0 or negative_points.size(0) == 0:
            return torch.tensor(0.0, device=segment_center.device)

        # 平衡同类点和异类点的数量
        num_positive = positive_points.size(0)
        num_negative = negative_points.size(0)
        total_count = num_positive + num_negative + 1e-6  # 避免除以零

        positive_weight = total_count / (2.0 * num_positive + 1e-6)
        negative_weight = total_count / (2.0 * num_negative + 1e-6)

        # 构建三元组 (anchor, positive, negative)
        # Anchor：正样本对应的超点中心
        anchor = positive_centers  # 形状：(num_positive, feat_dim)
        positive = positive_points  # 形状：(num_positive, feat_dim)

        # 为每个正样本随机选择一个负样本
        negative_indices = torch.randint(0, num_negative, (num_positive,))
        negative = negative_points[negative_indices]  # 形状：(num_positive, feat_dim)

        # 计算三元组损失（使用余弦距离）
        triplet_loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self._compute_cosine_distance,
            margin=self.margin,
            reduction='mean'
        )

        triplet_loss = triplet_loss_fn(anchor, positive, negative)

        # 加权处理损失
        total_loss = positive_weight * triplet_loss

        # 增加正则化项（L2 正则化）
        reg_loss = self.reg_lambda * torch.sum(segment_center ** 2) / (2.0 * n_segments)

        # 总损失
        total_loss = total_loss + reg_loss
        total_loss = torch.log1p(total_loss)
        return total_loss * self.loss_weight


@LOSSES.register_module()
class ContrastiveLoss_SP_segment_OPT(nn.Module):
    """基于对比学习的损失函数，用于超点聚类任务，利用真实标签信息。"""

    def __init__(self, temperature=0.5):
        super(ContrastiveLoss_SP_segment_OPT, self).__init__()
        self.temperature = temperature  # 温度参数

    def forward(self, segment_center, rawPoint_feat, point_assignments, label_inds):
        """
        Args:
            segment_center: Tensor of shape (num_superpoints, feature_dim), 超点的特征
            rawPoint_feat: Tensor of shape (num_points, feature_dim), 原始点的特征
            point_assignments: Tensor of shape (num_points,), 每个原始点所属的超点索引
            label_inds: Tensor of shape (num_points,), 每个原始点的真实标签
        """
        device = rawPoint_feat.device

        num_points = point_assignments.size(0)
        num_superpoints = segment_center.size(0)
        num_classes = label_inds.max().item() + 1

        # 计算每个超点中每个类别的出现次数
        linear_indices = point_assignments * num_classes + label_inds  # (num_points,)
        counts = torch.bincount(linear_indices, minlength=num_superpoints * num_classes)
        counts = counts.view(num_superpoints, num_classes)  # (num_superpoints, num_classes)

        # 初始化superpoint_labels为-1
        superpoint_labels = torch.full((num_superpoints,), -1, dtype=torch.long, device=device)

        # 找到非空超点的掩码
        superpoint_point_counts = counts.sum(dim=1)  # (num_superpoints,)
        non_empty_superpoints = superpoint_point_counts > 0  # (num_superpoints,)

        # 对于非空超点，找到出现次数最多的标签
        superpoint_labels[non_empty_superpoints] = counts[non_empty_superpoints].argmax(dim=1)

        # 将超点特征和原始点特征拼接在一起
        all_features = torch.cat([segment_center, rawPoint_feat], dim=0)  # (num_superpoints + num_points, feature_dim)

        # 将超点标签和原始点标签拼接在一起
        all_labels = torch.cat([superpoint_labels, label_inds], dim=0)  # (num_superpoints + num_points,)

        # 创建有效的样本掩码，过滤掉标签为-1的超点（即空超点）
        valid_mask = all_labels != -1  # (num_superpoints + num_points,)
        valid_features = all_features[valid_mask]
        valid_labels = all_labels[valid_mask]

        # 计算相似度矩阵
        features_normalized = F.normalize(valid_features, dim=1)
        similarity_matrix = torch.matmul(features_normalized,
                                         features_normalized.T) / self.temperature  # (valid_num, valid_num)

        # 创建标签掩码
        labels = valid_labels.unsqueeze(1)  # (valid_num, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # (valid_num, valid_num)

        # 去除自身对自身的相似度
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask

        # 计算对比损失
        exp_logits = torch.exp(similarity_matrix) * logits_mask  # (valid_num, valid_num)
        log_prob = similarity_matrix - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)  # (valid_num, valid_num)

        # 计算每个样本的平均正样本对数概率
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # 损失取负值并取平均
        loss = -mean_log_prob_pos.mean()

        return loss


@LOSSES.register_module()
class BalancingLoss(nn.Module):
    def __init__(self, n_superpoints=100, loss_weight=0.2, tolerance=0.05):
        super(BalancingLoss, self).__init__()
        self.n_superpoints = n_superpoints
        self.loss_weight = loss_weight
        self.tolerance = tolerance

    def forward(self, superpoint_assignments):
        n_points = superpoint_assignments.size(0)
        ideal_count = n_points / self.n_superpoints
        counts = torch.bincount(superpoint_assignments, minlength=int(self.n_superpoints)).float()

        # 归一化 counts
        normalized_counts = counts / counts.sum()
        ideal_normalized_count = 1.0 / self.n_superpoints

        # 计算归一化后的counts与理想分布之间的均方误差，并使用容差限制敏感性
        ideal_counts = torch.full_like(normalized_counts, ideal_normalized_count)
        # 使用容差创建动态上下限
        lower_bounds = torch.clamp(ideal_counts - self.tolerance, min=0)
        upper_bounds = torch.clamp(ideal_counts + self.tolerance, max=1)
        # 计算超出容忍区间的误差
        excess_loss = torch.where(normalized_counts < lower_bounds,
                                  lower_bounds - normalized_counts,
                                  torch.where(normalized_counts > upper_bounds,
                                              normalized_counts - upper_bounds,
                                              torch.zeros_like(normalized_counts)))

        # 使用 L1 损失代替 MSE，以提高对小偏差的敏感性
        l1_loss = excess_loss.mean()

        return self.loss_weight * l1_loss


@LOSSES.register_module()
class LabelConsistencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(LabelConsistencyLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.loss_weight = loss_weight

    def forward(self, e, tilde_e, w, u):
        """
        Args:
            e (torch.Tensor): Original point labels with shape (n, C) where C = number of classes.
            tilde_e (torch.Tensor): Reconstructed point labels with shape (n, C).
            w (torch.Tensor): Generated superpoint labels with shape (m, C) where m = number of superpoints.
            u (torch.Tensor): Pseudo superpoint labels with shape (m, C).

        Returns:
            torch.Tensor: The total label consistency loss.
        """
        # 计算点标签和重构点标签之间的交叉熵损失
        point_loss = self.cross_entropy_loss(e, tilde_e)

        # 计算生成的超点标签和伪超点标签之间的交叉熵损失
        superpoint_loss = self.cross_entropy_loss(w, u)

        # 将两个损失加权求和
        total_loss = (point_loss + superpoint_loss) * self.loss_weight

        return total_loss


@LOSSES.register_module()
class SuperpointDiscriminativeLossopt(nn.Module):
    def __init__(self, delta_var=0.5, delta_dist=1.2,
                 var_weight=1.0, dist_weight=1.0,
                 reg_weight=0.001, entropy_weight=1.0,
                 loss_weight=0.1, distance_metric='euclidean'):
        super(SuperpointDiscriminativeLossopt, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.var_weight = var_weight
        self.dist_weight = dist_weight
        self.reg_weight = reg_weight
        self.entropy_weight = entropy_weight
        self.loss_weight = loss_weight
        self.distance_metric = distance_metric  # 可选择 'cosine', 'euclidean', or 'dot_product'

    def _compute_distance(self, rawPoint_feat, superPoint_feat, raw_to_super_index):
        """根据选择的距离度量方式计算距离"""
        # L2 归一化操作
        rawPoint_feat_normalized = F.normalize(rawPoint_feat, p=2, dim=1)
        selected_superPoint_feat = superPoint_feat[raw_to_super_index]
        selected_superPoint_feat_normalized = F.normalize(selected_superPoint_feat, p=2, dim=1)
        if self.distance_metric == 'cosine':
            # 计算余弦距离

            cos_sim = torch.sum(rawPoint_feat_normalized * selected_superPoint_feat_normalized, dim=1)
            cos_dist = 1 - cos_sim  # 余弦距离
            return cos_dist ** 2  # (num_points,)

        elif self.distance_metric == 'euclidean':
            # 计算欧氏距离
            euclidean_dist = torch.norm(rawPoint_feat_normalized - selected_superPoint_feat_normalized, p=2, dim=1)
            return euclidean_dist

        # elif self.distance_metric == 'dot_product':
        #     # 计算负的点积作为距离
        #     dot_prod = torch.sum(rawPoint_feat_normalized * selected_superPoint_feat_normalized, dim=1)
        #     return -dot_prod  # 负的点积值，点积越大，距离越小

        else:
            raise ValueError(
                f"Invalid distance metric: {self.distance_metric}. Choose 'cosine', 'euclidean', or 'dot_product'.")

    def forward(self, superPoint_feat, rawPoint_feat, raw_to_super_index, label_inds):
        device = rawPoint_feat.device
        num_superpoints = superPoint_feat.size(0)

        label_inds = label_inds.to(device).long()
        raw_to_super_index = raw_to_super_index.to(device).long()

        assert raw_to_super_index.min() >= 0, "raw_to_super_index contains negative indices."
        assert raw_to_super_index.max() < num_superpoints, "raw_to_super_index contains indices larger than num_superpoints."

        distance = self._compute_distance(rawPoint_feat, superPoint_feat, raw_to_super_index)

        per_superpoint_variance = scatter_mean(distance, raw_to_super_index, dim=0, dim_size=num_superpoints)
        l_var = torch.mean(F.relu(per_superpoint_variance))

        valid_mask = label_inds >= 0
        valid_label_inds = label_inds[valid_mask]
        valid_raw_to_super_index = raw_to_super_index[valid_mask]

        num_classes = label_inds.max().item() + 1
        label_one_hot = F.one_hot(valid_label_inds, num_classes=num_classes).float()

        superpoint_label_counts = scatter_add(label_one_hot, valid_raw_to_super_index, dim=0, dim_size=num_superpoints)
        superpoint_label_sums = superpoint_label_counts.sum(dim=1, keepdim=True) + 1e-8
        label_probs = superpoint_label_counts / superpoint_label_sums
        superpoint_labels = torch.argmax(superpoint_label_counts, dim=1)

        if num_superpoints > 1:
            superPoint_feat_normalized = F.normalize(superPoint_feat, p=2, dim=1)  # 归一化

            if self.distance_metric == 'cosine':
                center_dist = 1 - torch.mm(superPoint_feat_normalized, superPoint_feat_normalized.t())
            elif self.distance_metric == 'euclidean':
                diff = superPoint_feat_normalized.unsqueeze(1) - superPoint_feat_normalized.unsqueeze(0)
                center_dist = torch.norm(diff, p=2, dim=2)
            elif self.distance_metric == 'dot_product':
                center_dist = -torch.mm(superPoint_feat_normalized, superPoint_feat_normalized.t())
            else:
                raise ValueError(f"Invalid distance metric: {self.distance_metric}.")

            valid_superpoint_mask = (superpoint_label_sums.squeeze() > 0)
            valid_superpoint_indices = valid_superpoint_mask.nonzero(as_tuple=False).squeeze()

            if valid_superpoint_indices.numel() > 1:
                valid_superpoint_labels = superpoint_labels[valid_superpoint_indices]
                valid_center_dist = center_dist[valid_superpoint_indices][:, valid_superpoint_indices]

                label_matrix = valid_superpoint_labels.unsqueeze(0) != valid_superpoint_labels.unsqueeze(1)
                mask = label_matrix & (~torch.eye(len(valid_superpoint_indices), dtype=torch.bool, device=device))

                center_dist_different_labels = valid_center_dist[mask]
                l_dist = torch.mean(F.relu(self.delta_dist - center_dist_different_labels) ** 2)
            else:
                l_dist = torch.tensor(0.0, device=device)
        else:
            l_dist = torch.tensor(0.0, device=device)

        l_reg = torch.mean(torch.norm(superPoint_feat, dim=1))

        entropy = -torch.sum(label_probs * torch.log(label_probs + 1e-8), dim=1)
        valid_entropy = entropy[superpoint_label_sums.squeeze() > 0]
        l_entropy = torch.mean(valid_entropy)

        total_loss = (
                self.var_weight * l_var +
                self.dist_weight * l_dist +
                self.reg_weight * l_reg +
                self.entropy_weight * l_entropy
        )

        return total_loss * self.loss_weight

# @LOSSES.register_module()
# class SuperpointDiscriminativeLossopt(nn.Module):
#     def __init__(self, delta_var=0.5, delta_dist=1.2,
#                  var_weight=1.0, dist_weight=1.0,
#                  reg_weight=0.001, entropy_weight=1.0,
#                  loss_weight=0.1, distance_metric='log-euclidean'):
#         super(SuperpointDiscriminativeLossopt, self).__init__()
#         self.delta_var = delta_var
#         self.delta_dist = delta_dist
#         self.var_weight = var_weight
#         self.dist_weight = dist_weight
#         self.reg_weight = reg_weight
#         self.entropy_weight = entropy_weight
#         self.loss_weight = loss_weight
#         self.distance_metric = distance_metric  # 可选择 'cosine', 'euclidean', 'log-euclidean', or 'dot_product'
#
#     def _compute_distance(self, rawPoint_feat, superPoint_feat, raw_to_super_index):
#         """根据选择的距离度量方式计算距离"""
#         # L2 归一化操作
#         rawPoint_feat_normalized = F.normalize(rawPoint_feat, p=2, dim=1)
#         selected_superPoint_feat = superPoint_feat[raw_to_super_index]
#         selected_superPoint_feat_normalized = F.normalize(selected_superPoint_feat, p=2, dim=1)
#
#         if self.distance_metric == 'cosine':
#             # 计算余弦距离
#             cos_sim = torch.sum(rawPoint_feat_normalized * selected_superPoint_feat_normalized, dim=1)
#             cos_dist = 1 - cos_sim  # 余弦距离
#             return cos_dist**2  # (num_points,)
#
#         elif self.distance_metric == 'euclidean':
#             # 计算欧氏距离
#             euclidean_dist = torch.norm(rawPoint_feat_normalized - selected_superPoint_feat_normalized, p=2, dim=1)
#             return euclidean_dist
#
#         elif self.distance_metric == 'log-euclidean':
#             # 计算对数欧氏距离
#             euclidean_dist = torch.norm(rawPoint_feat - selected_superPoint_feat, p=2, dim=1)
#             log_euclidean_dist = torch.log(1 + euclidean_dist)  # 对欧氏距离取对数
#             return log_euclidean_dist
#         # elif self.distance_metric == 'dot_product':
#         #     # 计算负的点积作为距离
#         #     dot_prod = torch.sum(rawPoint_feat_normalized * selected_superPoint_feat_normalized, dim=1)
#         #     return -dot_prod  # 负的点积值，点积越大，距离越小
#
#         else:
#             raise ValueError(
#                 f"Invalid distance metric: {self.distance_metric}. Choose 'cosine', 'euclidean', 'log-euclidean', or 'dot_product'.")
#
#     def forward(self, superPoint_feat, rawPoint_feat, raw_to_super_index, label_inds):
#         device = rawPoint_feat.device
#         num_superpoints = superPoint_feat.size(0)
#
#         label_inds = label_inds.to(device).long()
#         raw_to_super_index = raw_to_super_index.to(device).long()
#
#         assert raw_to_super_index.min() >= 0, "raw_to_super_index contains negative indices."
#         assert raw_to_super_index.max() < num_superpoints, "raw_to_super_index contains indices larger than num_superpoints."
#
#         distance = self._compute_distance(rawPoint_feat, superPoint_feat, raw_to_super_index)
#
#         per_superpoint_variance = scatter_mean(distance, raw_to_super_index, dim=0, dim_size=num_superpoints)
#         l_var = torch.mean(F.relu(per_superpoint_variance))
#
#         valid_mask = label_inds >= 0
#         valid_label_inds = label_inds[valid_mask]
#         valid_raw_to_super_index = raw_to_super_index[valid_mask]
#
#         num_classes = label_inds.max().item() + 1
#         label_one_hot = F.one_hot(valid_label_inds, num_classes=num_classes).float()
#
#         superpoint_label_counts = scatter_add(label_one_hot, valid_raw_to_super_index, dim=0, dim_size=num_superpoints)
#         superpoint_label_sums = superpoint_label_counts.sum(dim=1, keepdim=True) + 1e-8
#         label_probs = superpoint_label_counts / superpoint_label_sums
#         superpoint_labels = torch.argmax(superpoint_label_counts, dim=1)
#
#         if num_superpoints > 1:
#             superPoint_feat_normalized = F.normalize(superPoint_feat, p=2, dim=1)  # 归一化
#
#             if self.distance_metric == 'cosine':
#                 center_dist = 1 - torch.mm(superPoint_feat_normalized, superPoint_feat_normalized.t())
#             elif self.distance_metric == 'euclidean':
#                 diff = superPoint_feat_normalized.unsqueeze(1) - superPoint_feat_normalized.unsqueeze(0)
#                 center_dist = torch.norm(diff, p=2, dim=2)
#             elif self.distance_metric == 'log-euclidean':
#                 diff = superPoint_feat_normalized.unsqueeze(1) - superPoint_feat_normalized.unsqueeze(0)
#                 euclidean_dist = torch.norm(diff, p=2, dim=2)
#                 center_dist = torch.log(1 + euclidean_dist)  # 对中心距离取对数
#             elif self.distance_metric == 'dot_product':
#                 center_dist = -torch.mm(superPoint_feat_normalized, superPoint_feat_normalized.t())
#             else:
#                 raise ValueError(f"Invalid distance metric: {self.distance_metric}.")
#
#             valid_superpoint_mask = (superpoint_label_sums.squeeze() > 0)
#             valid_superpoint_indices = valid_superpoint_mask.nonzero(as_tuple=False).squeeze()
#
#             if valid_superpoint_indices.numel() > 1:
#                 valid_superpoint_labels = superpoint_labels[valid_superpoint_indices]
#                 valid_center_dist = center_dist[valid_superpoint_indices][:, valid_superpoint_indices]
#
#                 label_matrix = valid_superpoint_labels.unsqueeze(0) != valid_superpoint_labels.unsqueeze(1)
#                 mask = label_matrix & (~torch.eye(len(valid_superpoint_indices), dtype=torch.bool, device=device))
#
#                 center_dist_different_labels = valid_center_dist[mask]
#                 l_dist = torch.mean(F.relu(self.delta_dist - center_dist_different_labels) ** 2)
#             else:
#                 l_dist = torch.tensor(0.0, device=device)
#         else:
#             l_dist = torch.tensor(0.0, device=device)
#
#         l_reg = torch.mean(torch.norm(superPoint_feat, dim=1))
#
#         entropy = -torch.sum(label_probs * torch.log(label_probs + 1e-8), dim=1)
#         valid_entropy = entropy[superpoint_label_sums.squeeze() > 0]
#         l_entropy = torch.mean(valid_entropy)
#
#         total_loss = (
#                 self.var_weight * l_var +
#                 self.dist_weight * l_dist +
#                 self.reg_weight * l_reg +
#                 self.entropy_weight * l_entropy
#         )
#
#         return total_loss * self.loss_weight
