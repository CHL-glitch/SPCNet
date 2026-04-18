"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import os
import numpy as np
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4

import pointcept.utils.comm as comm
from pointcept.utils.misc import intersection_and_union_gpu, make_dirs
import torchnet as tnt
from .default import HookBase
from .builder import HOOKS
# from pointcept.models.SPCNet.utils import superpointMatrix
from ...models.SPCNet.utils.superpointMatrix import compute_graph_nn_2
from pointcept.models.SPCNet.utils.superpointMatrix import compute_graph_nn_2, \
    relax_edge_binary, compute_boundary_recall, compute_boundary_precision, compute_graph_nn_optimized, \
    compute_boundary_precision_optimized, \
    compute_boundary_recall_optimized, relax_edge_binary_optimized


@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                label,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "allAcc"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


# @HOOKS.register_module()
# class SemSegEvaluator(HookBase):
#     def after_epoch(self):
#         if self.trainer.cfg.evaluate:
#             self.eval()
#
#     def eval(self):
#         self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
#         self.trainer.model.eval()
#         for i, input_dict in enumerate(self.trainer.val_loader):
#             for key in input_dict.keys():
#                 if isinstance(input_dict[key], torch.Tensor):
#                     input_dict[key] = input_dict[key].cuda(non_blocking=True)
#             with torch.no_grad():
#                 output_dict = self.trainer.model(input_dict)
#             output = output_dict["seg_logits"]
#             loss = output_dict["loss"]
#             pred = output.max(1)[1]
#             segment = input_dict["segment"]
#             # objects = input_dict["instance"]
#
#             if "origin_coord" in input_dict.keys():
#                 idx, _ = pointops.knn_query(
#                     1,
#                     input_dict["coord"].float(),
#                     input_dict["offset"].int(),
#                     input_dict["origin_coord"].float(),
#                     input_dict["origin_offset"].int(),
#                 )
#                 pred = pred[idx.flatten().long()]
#                 segment = input_dict["origin_segment"]
#             intersection, union, target = intersection_and_union_gpu(
#                 pred,
#                 segment,
#                 self.trainer.cfg.data.num_classes,
#                 self.trainer.cfg.data.ignore_index,
#             )
#             if comm.get_world_size() > 1:
#                 dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
#                     target
#                 )
#             intersection, union, target = (
#                 intersection.cpu().numpy(),
#                 union.cpu().numpy(),
#                 target.cpu().numpy(),
#             )
#             # Here there is no need to sync since sync happened in dist.all_reduce
#             self.trainer.storage.put_scalar("val_intersection", intersection)
#             self.trainer.storage.put_scalar("val_union", union)
#             self.trainer.storage.put_scalar("val_target", target)
#             self.trainer.storage.put_scalar("val_loss", loss.item())
#             info = "Test: [{iter}/{max_iter}] ".format(
#                 iter=i + 1, max_iter=len(self.trainer.val_loader)
#             )
#             if "origin_coord" in input_dict.keys():
#                 info = "Interp. " + info
#             self.trainer.logger.info(
#                 info
#                 + "Loss {loss:.4f} ".format(
#                     iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
#                 )
#             )
#         # 获取存储中的历史数据
#         loss_avg = self.trainer.storage.history("val_loss").avg
#         intersection = self.trainer.storage.history("val_intersection").total
#         union = self.trainer.storage.history("val_union").total
#         target = self.trainer.storage.history("val_target").total
#
#         # compute BR and BP
#         # graph_nn, local_neighbors = compute_graph_nn_2(input_dict["coord"], 5, 20)
#         # is_transition = objects[graph_nn["source"]] != objects[graph_nn["target"]]
#
#         # 计算 IoU 和 Accuracy
#         iou_class = intersection / (union + 1e-10)
#         acc_class = intersection / (target + 1e-10)
#         m_iou = np.mean(iou_class)
#         m_acc = np.mean(acc_class)
#         all_acc = sum(intersection) / (sum(target) + 1e-10)
#
#         # 计算 Precision 和 Recall
#         precision = intersection / (union + 1e-10)  # 或者 (predicted_meter.sum + 1e-10) 根据情况
#         recall = intersection / (target + 1e-10)
#
#         # 计算 F1 分数
#         f1 = 2 * precision * recall / (precision + recall + 1e-10)
#
#         # 计算平均 F1 分数 (mF1)
#         m_F1 = np.mean(f1)
#
#         # 计算 Overall Accuracy (OOA)
#         ooA = sum(intersection) / (sum(target) + 1e-10)
#
#         # 计算 Boundary Recall (BR)
#         boundary_recall = np.mean(intersection / (target + 1e-10))
#
#         # 计算 Boundary Precision (BP)
#         boundary_precision = np.mean(intersection / (union + 1e-10))
#
#         # 使用标量而不是数组来格式化日志输出
#         self.trainer.logger.info(
#             "Val result: mIoU/mAcc/allAcc/mF1/OOA/BR/BP/F1 {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}".format(
#                 m_iou, m_acc, all_acc, m_F1, ooA, boundary_recall, boundary_precision
#             )
#         )
#
#         # 逐类别输出 iou 和 accuracy 的结果
#         for i in range(self.trainer.cfg.data.num_classes):
#             self.trainer.logger.info(
#                 "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
#                     idx=i,
#                     name=self.trainer.cfg.data.names[i],
#                     iou=iou_class[i],
#                     accuracy=acc_class[i],
#                 )
#             )
#
#         # 获取当前 epoch
#         current_epoch = self.trainer.epoch + 1
#
#         # 添加到 TensorBoard 中，确保传递的是标量
#         if self.trainer.writer is not None:
#             self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
#             self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
#             self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
#             self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
#             self.trainer.writer.add_scalar("val/mF1", m_F1, current_epoch)
#             # 这里的 f1 是数组，需取平均值
#             self.trainer.writer.add_scalar("val/F1", np.mean(f1), current_epoch)
#             self.trainer.writer.add_scalar("val/ooA", ooA, current_epoch)
#             # 使用边界召回和精度的平均值
#             self.trainer.writer.add_scalar("val/boundary_recall", boundary_recall, current_epoch)
#             self.trainer.writer.add_scalar("val/boundary_precision", boundary_precision, current_epoch)
#         self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
#         self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
#         self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver
#
#     def after_train(self):
#         self.trainer.logger.info(
#             "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
#         )
@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        # # 初始化加权累积变量
        # total_weight_BR = 0.0
        # total_weight_BP = 0.0
        # weighted_sum_BR = 0.0
        # weighted_sum_BP = 0.0
        #
        # total_weight_BR_level0 = 0.0
        # total_weight_BP_level0 = 0.0
        # weighted_sum_BR_level0 = 0.0
        # weighted_sum_BP_level0 = 0.0
        #
        # # 新增用于统计OOA的列表（如果需要对多个batch求平均OOA，可以累积后求平均）
        # OOA_values = []

        # Initialize meters for BR and BP
        BR_meter = tnt.meter.AverageValueMeter()
        BP_meter = tnt.meter.AverageValueMeter()
        BR_meter_level0 = tnt.meter.AverageValueMeter()
        BP_meter_level0 = tnt.meter.AverageValueMeter()
        root_path = "/home/cvmaster/lch/pointcept1/Pointcept/tools/exp"
        save_path = os.path.join(root_path, "train_val_result")  # save train predict result
        base_group_save_path = os.path.join(root_path, "train_val_superpoint_result")  # as same
        base_turth_save_path = os.path.join(root_path, "train_val_original")  # save train predict result
        make_dirs(save_path)
        make_dirs(base_group_save_path)
        make_dirs(base_turth_save_path)
        for i, input_dict in enumerate(self.trainer.val_loader):
            data_name = input_dict["name"][0]
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            # color and gt and coord save at
            segment_val = segment
            coord = input_dict["coord"]
            # color = input_dict["feat"][:, :3]

            # find superpoint index
            pred_in_component = output_dict["pred_in_component"]
            initial_segment = output_dict['initial_segment']
            objects = input_dict["segment"]  # instance or segment. dependant on the tesk
            # save path
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            group_save_path = os.path.join(base_group_save_path, "{}_group.npy".format(data_name))  # superpoint index
            original_save_path = os.path.join(base_turth_save_path, "{}_original.npy".format(data_name))
            # save result
            np.save(pred_save_path, pred.cpu().numpy())
            group_data = {
                "raw_to_super_index": pred_in_component.cpu().numpy(),
                "initial_segment_index": initial_segment.cpu().numpy(),
            }
            np.save(group_save_path, group_data)
            turth_data = {
                "groud_turth": segment_val.cpu().numpy(),
                "coord": coord.cpu().numpy(),
                # "color": color.cpu().numpy(),
            }
            np.save(original_save_path, turth_data)
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),

                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            #############################compute BR and BP for SuperPoint Matrix##############################
            # 计算图
            graph_nn, local_neighbors = compute_graph_nn_optimized(input_dict["coord"], 20, 20)

            # 确保所有数据在同一设备上
            device = graph_nn["source"].device
            objects = objects.to(device)
            pred_in_component = pred_in_component.to(device)
            # pred_in_component_object_level = pred_in_component_object_level.to(device)
            segment = segment.to(device)

            # 计算 is_transition 和 pred_transition
            is_transition = objects[graph_nn["source"]] != objects[graph_nn["target"]]
            pred_transition = pred_in_component[graph_nn["source"]] != pred_in_component[graph_nn["target"]]
            # pred_transition_object_level = (pred_in_component_object_level[graph_nn["source"]] !=
            #                                 pred_in_component_object_level[graph_nn["target"]])

            n_ver = input_dict["coord"].shape[0]

            # 放宽边缘
            relaxed_pred_transition_BR = relax_edge_binary_optimized(
                pred_transition,
                graph_nn["source"],
                graph_nn["target"],
                n_ver,
                tolerance=1
            )
            relaxed_pred_transition_BP = relax_edge_binary_optimized(
                is_transition,
                graph_nn["source"],
                graph_nn["target"],
                n_ver,
                tolerance=1
            )

            # relaxed_pred_transition_object_level_BR = relax_edge_binary_optimized(
            #     pred_transition_object_level,
            #     graph_nn["source"],
            #     graph_nn["target"],
            #     n_ver,
            #     tolerance=1
            # )
            # relaxed_pred_transition_object_level_BP = relax_edge_binary_optimized(
            #     is_transition,
            #     graph_nn["source"],
            #     graph_nn["target"],
            #     n_ver,
            #     tolerance=1
            # )

            # 计算 BR 和 BP (GPU上)
            BR = compute_boundary_recall_optimized(is_transition, relaxed_pred_transition_BR)
            BP = compute_boundary_precision_optimized(relaxed_pred_transition_BP, pred_transition)
            # BR_Level0 = compute_boundary_recall_optimized(is_transition, relaxed_pred_transition_object_level_BR)
            # BP_Level0 = compute_boundary_precision_optimized(relaxed_pred_transition_object_level_BP,
            #                                                  pred_transition_object_level)

            # 在 GPU 上进行加权计算，并在最后一步转 CPU
            is_transition_sum = is_transition.sum()
            pred_transition_sum = pred_transition.sum()
            # pred_transition_object_level_sum = pred_transition_object_level.sum()

            # 将指标累积到 AverageValueMeter 时才转 CPU
            BR_meter.add((BR * is_transition_sum).cpu().item(), n=is_transition_sum.cpu().item())
            BP_meter.add((BP * pred_transition_sum).cpu().item(), n=pred_transition_sum.cpu().item())
            # BR_meter_level0.add((BR_Level0 * is_transition_sum).cpu().item(), n=is_transition_sum.cpu().item())
            # BP_meter_level0.add((BP_Level0 * pred_transition_object_level_sum).cpu().item(),
            #                     n=pred_transition_object_level_sum.cpu().item())

            # 计算 F1 分数（仍在 GPU 上计算，输出时转换）
            F1 = (2 * (BR * BP) / (BR + BP + 1e-10)).cpu().item()
            # F1_level0 = (2 * (BR_Level0 * BP_Level0) / (BR_Level0 + BP_Level0 + 1e-10)).cpu().item()

            num_classes = self.trainer.cfg.data.num_classes

            # 计算 OOA
            sp_unique, sp_inverse = torch.unique(pred_in_component, sorted=True, return_inverse=True)
            M = sp_unique.size(0)
            counts = torch.zeros((M, num_classes), device=segment_val.device, dtype=torch.int64)
            counts.index_put_((sp_inverse, segment_val), torch.ones_like(segment_val, dtype=torch.int64), accumulate=True)
            max_freq, _ = counts.max(dim=1)
            correct_labels = max_freq.sum().item()
            OOA = 100.0 * correct_labels / segment_val.size(0)
            self.trainer.storage.put_scalar("val_OOA", OOA)

            # # 计算 OOA_L0（基于pred_in_component_object_level）
            # sp_unique_l0, sp_inverse_l0 = torch.unique(pred_in_component_object_level, sorted=True, return_inverse=True)
            # M_l0 = sp_unique_l0.size(0)
            # counts_l0 = torch.zeros((M_l0, num_classes), device=segment_val.device, dtype=torch.int64)
            # counts_l0.index_put_((sp_inverse_l0, segment_val), torch.ones_like(segment_val, dtype=torch.int64), accumulate=True)
            # max_freq_l0, _ = counts_l0.max(dim=1)
            # correct_labels_l0 = max_freq_l0.sum().item()
            # OOA_L0 = 100.0 * correct_labels_l0 / segment_val.size(0)
            # self.trainer.storage.put_scalar("val_OOA_L0", OOA_L0)
            # *********************** 优化后的 OOA 计算结束 **********************

            # Store results
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.storage.put_scalar("val_BR", BR)
            self.trainer.storage.put_scalar("val_BP", BP)
            # self.trainer.storage.put_scalar("val_BR_Level0", BR_Level0)
            # self.trainer.storage.put_scalar("val_BP_Level0", BP_Level0)
            self.trainer.storage.put_scalar("val_F1", F1)
            # self.trainer.storage.put_scalar("val_F1_level0", F1_level0)

            # Logging each iteration
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

            # 在所有batch结束后计算全局加权平均BR和BP
        # global_BR = weighted_sum_BR / total_weight_BR if total_weight_BR > 0 else 0.0
        # global_BP = weighted_sum_BP / total_weight_BP if total_weight_BP > 0 else 0.0
        # global_BR_level0 = weighted_sum_BR_level0 / total_weight_BR_level0 if total_weight_BR_level0 > 0 else 0.0
        # global_BP_level0 = weighted_sum_BP_level0 / total_weight_BP_level0 if total_weight_BP_level0 > 0 else 0.0
        # F1 = 2*(global_BR*global_BP)/(global_BR + global_BP)
        # F1_level0 = 2*(global_BR_level0*global_BP_level0)/(global_BR_level0 + global_BP_level0)

        global_BR = BR_meter.value()[0]
        global_BP = BP_meter.value()[0]
        # global_BR_level0 = BR_meter_level0.value()[0]
        # global_BP_level0 = BP_meter_level0.value()[0]

        self.trainer.storage.put_scalar("val_global_BR", global_BR)
        self.trainer.storage.put_scalar("val_global_BP", global_BP)
        # self.trainer.storage.put_scalar("val_global_BR_Level0", global_BR_level0)
        # self.trainer.storage.put_scalar("val_global_BP_Level0", global_BP_level0)

        # global_BR = self.trainer.storage.history("val_BR").avg
        # global_BP = self.trainer.storage.history("val_BP").avg
        # global_BR_level0 = self.trainer.storage.history("val_BR_Level0").avg
        # global_BP_level0 = self.trainer.storage.history("val_BP_Level0").avg
        F1 = 2*(global_BR*global_BP)/(global_BR + global_BP)
        # F1_level0 = 2*(global_BR_level0*global_BP_level0)/(global_BR_level0 + global_BP_level0)
        # 存储全局加权结果
        self.trainer.storage.put_scalar("val_F1", F1)
        # self.trainer.storage.put_scalar("val_F1_level0", F1_level0)

        # Retrieve historical data
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        # br_avg = self.trainer.storage.history("val_BR").avg
        # bp_avg = self.trainer.storage.history("val_BP").avg
        # br_level0_avg = self.trainer.storage.history("val_BR_Level0").avg
        # bp_level0_avg = self.trainer.storage.history("val_BP_Level0").avg

        # Retrieve global BR and BP (Weighted metrics)
        global_br_avg = self.trainer.storage.history("val_global_BR").avg
        global_bp_avg = self.trainer.storage.history("val_global_BP").avg
        # global_br_level0_avg = self.trainer.storage.history("val_global_BR_Level0").avg
        # global_bp_level0_avg = self.trainer.storage.history("val_global_BP_Level0").avg
        F1_avg = self.trainer.storage.history("val_F1").avg
        # F1_level0_avg = self.trainer.storage.history("val_F1_level0").avg
        OOA = self.trainer.storage.history("val_OOA").avg
        # OOA_l0 = self.trainer.storage.history("val_OOA_L0").avg
        # Calculate metrics
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)

        # Log final results
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f} BR/BP {:.4f}/{:.4f} F1/{:.4f} OOA {:.4f}".format(
                m_iou, m_acc, all_acc, global_br_avg, global_bp_avg, F1_avg,
                OOA
            )
        )

        # Log class-wise results
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )

        # Write to TensorBoard
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            self.trainer.writer.add_scalar("val/BR", global_br_avg, current_epoch)
            self.trainer.writer.add_scalar("val/BP", global_bp_avg, current_epoch)
            # self.trainer.writer.add_scalar("val/BR_Level0", global_br_level0_avg, current_epoch)
            # self.trainer.writer.add_scalar("val/BP_Level0", global_bp_level0_avg, current_epoch)
            self.trainer.writer.add_scalar("val/F1", F1_avg, current_epoch)
            # self.trainer.writer.add_scalar("val/F1_Level0", F1_level0_avg, current_epoch)
            self.trainer.writer.add_scalar("val/OOA", OOA, current_epoch)
            # self.trainer.writer.add_scalar("val/OOA_l0", OOA_l0, current_epoch)

        # Save metrics for monitoring
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou
        self.trainer.comm_info["current_metric_name"] = "mIoU"
        # 增加其他指标的记录

        self.trainer.comm_info["current_BR"] = global_BR
        self.trainer.comm_info["current_BP"] = global_BP
        self.trainer.comm_info["current_OOA"] = OOA
        self.trainer.comm_info["current_F1"] = F1_avg  # 增加其他指标的记录

    # def eval(self):
    #     self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    #     self.trainer.model.eval()
    #
    #     # # 初始化加权累积变量
    #     # total_weight_BR = 0.0
    #     # total_weight_BP = 0.0
    #     # weighted_sum_BR = 0.0
    #     # weighted_sum_BP = 0.0
    #     #
    #     # total_weight_BR_level0 = 0.0
    #     # total_weight_BP_level0 = 0.0
    #     # weighted_sum_BR_level0 = 0.0
    #     # weighted_sum_BP_level0 = 0.0
    #     #
    #     # # 新增用于统计OOA的列表（如果需要对多个batch求平均OOA，可以累积后求平均）
    #     # OOA_values = []
    #
    #     # Initialize meters for BR and BP
    #     for i, input_dict in enumerate(self.trainer.val_loader):
    #         for key in input_dict.keys():
    #             if isinstance(input_dict[key], torch.Tensor):
    #                 input_dict[key] = input_dict[key].cuda(non_blocking=True)
    #         with torch.no_grad():
    #             output_dict = self.trainer.model(input_dict)
    #         output = output_dict["seg_logits"]
    #         loss = output_dict["loss"]
    #         pred = output.max(1)[1]
    #         pred_in_component = output_dict["pred_in_component"]
    #         pred_in_component_object_level = output_dict["pred_in_component_object_level"]
    #         segment = input_dict["segment"]
    #         objects = input_dict["segment"]  # instance or segment. dependant on the tesk
    #
    #         if "origin_coord" in input_dict.keys():
    #             idx, _ = pointops.knn_query(
    #                 1,
    #                 input_dict["coord"].float(),
    #
    #                 input_dict["offset"].int(),
    #                 input_dict["origin_coord"].float(),
    #                 input_dict["origin_offset"].int(),
    #             )
    #             pred = pred[idx.flatten().long()]
    #             segment = input_dict["origin_segment"]
    #         intersection, union, target = intersection_and_union_gpu(
    #             pred,
    #             segment,
    #             self.trainer.cfg.data.num_classes,
    #             self.trainer.cfg.data.ignore_index,
    #         )
    #         if comm.get_world_size() > 1:
    #             dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
    #                 target
    #             )
    #         intersection, union, target = (
    #             intersection.cpu().numpy(),
    #             union.cpu().numpy(),
    #             target.cpu().numpy(),
    #         )
    #         #############################compute BR and BP for SuperPoint Matrix##############################
    #         graph_nn, local_neighbors = compute_graph_nn_optimized(input_dict["coord"], 5, 20)
    #
    #         # 确保所有张量都在同一设备上（通常是 GPU）
    #         device = graph_nn["source"].device
    #         objects = objects.to(device)
    #         pred_in_component = pred_in_component.to(device)
    #         pred_in_component_object_level = pred_in_component_object_level.to(device)
    #
    #         # 计算有效的边（排除包含未标注点的边）
    #         # valid_edges = (objects[graph_nn["source"]] != -1) & (objects[graph_nn["target"]] != -1)
    #
    #         # 计算 is_transition，仅在有效边上计算
    #         is_transition = objects[graph_nn["source"]] != objects[graph_nn["target"]]
    #         # is_transition = is_transition & valid_edges  # 排除未标注点
    #
    #         # 计算 pred_transition
    #         pred_transition = pred_in_component[graph_nn["source"]] != pred_in_component[graph_nn["target"]]
    #         # pred_transition = pred_transition & valid_edges  # 确保形状一致
    #
    #         pred_transition_object_level = pred_in_component_object_level[graph_nn["source"]] != \
    #                                        pred_in_component_object_level[graph_nn["target"]]
    #         # pred_transition_object_level = pred_transition_object_level & valid_edges  # 确保形状一致
    #
    #         # 使用优化后的 relax_edge_binary 函数
    #         n_ver = input_dict["coord"].shape[0]
    #         relaxed_pred_transition_BR = relax_edge_binary_optimized(
    #             pred_transition,
    #             graph_nn["source"],
    #             graph_nn["target"],
    #             n_ver,
    #             tolerance=1
    #         )
    #
    #         relaxed_pred_transition_BP = relax_edge_binary_optimized(
    #             is_transition,
    #             graph_nn["source"],
    #             graph_nn["target"],
    #             n_ver,
    #             tolerance=1
    #         )
    #
    #         relaxed_pred_transition_object_level_BR = relax_edge_binary_optimized(
    #             pred_transition_object_level,
    #             graph_nn["source"],
    #             graph_nn["target"],
    #             n_ver,
    #             tolerance=1
    #         )
    #
    #         relaxed_pred_transition_object_level_BP = relax_edge_binary_optimized(
    #             is_transition,
    #             graph_nn["source"],
    #             graph_nn["target"],
    #             n_ver,
    #             tolerance=1
    #         )
    #         # 计算 BR 和 BP
    #         BR = compute_boundary_recall_optimized(is_transition, relaxed_pred_transition_BR)
    #         BP = compute_boundary_precision_optimized(relaxed_pred_transition_BP, pred_transition)
    #         BR_Level0 = compute_boundary_recall_optimized(is_transition, relaxed_pred_transition_object_level_BR)
    #         BP_Level0 = compute_boundary_precision_optimized(relaxed_pred_transition_object_level_BP,
    #                                                          pred_transition_object_level)
    #
    #         F1 = 2 * (BR * BP) / (BR + BP)
    #         F1_level0 = 2 * (BR_Level0 * BP_Level0) / (BR_Level0 + BP_Level0)
    #         # # 在外部实现加权逻辑
    #         # # 对当前batch计算权重
    #         # weight_BR = is_transition.sum().item()  # 用真实边界体素数加权BR
    #         # weight_BP = pred_transition.sum().item()  # 用预测边界体素数加权BP
    #         # weight_BR_level0 = is_transition.sum().item()  # 同样对Level0 BR加权
    #         # weight_BP_level0 = pred_transition_object_level.sum().item()  # 对Level0 BP加权
    #         #
    #         # # 累积权重
    #         # total_weight_BR += weight_BR
    #         # total_weight_BP += weight_BP
    #         # total_weight_BR_level0 += weight_BR_level0
    #         # total_weight_BP_level0 += weight_BP_level0
    #         #
    #         # # 累积加权和
    #         # weighted_sum_BR += weight_BR * BR.item()
    #         # weighted_sum_BP += weight_BP * BP.item()
    #         # weighted_sum_BR_level0 += weight_BR_level0 * BR_Level0.item()
    #         # weighted_sum_BP_level0 += weight_BP_level0 * BP_Level0.item()
    #         #############################compute BR and BP for SuperPoint Matrix##############################
    #         # *********************** 优化后的 OOA 计算 ************************
    #         # segment: GT标签, shape [N],  pred_in_component: superpoint id, shape [N]
    #         # 利用张量操作快速计算每个superpoint对应的标签频率分布并找到主频标签计数
    #
    #         num_classes = self.trainer.cfg.data.num_classes
    #
    #         # 计算 OOA
    #         sp_unique, sp_inverse = torch.unique(pred_in_component, sorted=True, return_inverse=True)
    #         M = sp_unique.size(0)
    #         counts = torch.zeros((M, num_classes), device=segment.device, dtype=torch.int64)
    #         counts.index_put_((sp_inverse, segment), torch.ones_like(segment, dtype=torch.int64), accumulate=True)
    #         max_freq, _ = counts.max(dim=1)
    #         correct_labels = max_freq.sum().item()
    #         OOA = 100.0 * correct_labels / segment.size(0)
    #         self.trainer.storage.put_scalar("val_OOA", OOA)
    #
    #         # 计算 OOA_L0（基于pred_in_component_object_level）
    #         sp_unique_l0, sp_inverse_l0 = torch.unique(pred_in_component_object_level, sorted=True, return_inverse=True)
    #         M_l0 = sp_unique_l0.size(0)
    #         counts_l0 = torch.zeros((M_l0, num_classes), device=segment.device, dtype=torch.int64)
    #         counts_l0.index_put_((sp_inverse_l0, segment), torch.ones_like(segment, dtype=torch.int64), accumulate=True)
    #         max_freq_l0, _ = counts_l0.max(dim=1)
    #         correct_labels_l0 = max_freq_l0.sum().item()
    #         OOA_L0 = 100.0 * correct_labels_l0 / segment.size(0)
    #         self.trainer.storage.put_scalar("val_OOA_L0", OOA_L0)
    #         # *********************** 优化后的 OOA 计算结束 **********************
    #
    #         # Store results
    #         self.trainer.storage.put_scalar("val_intersection", intersection)
    #         self.trainer.storage.put_scalar("val_union", union)
    #         self.trainer.storage.put_scalar("val_target", target)
    #         self.trainer.storage.put_scalar("val_loss", loss.item())
    #         self.trainer.storage.put_scalar("val_BR", BR)
    #         self.trainer.storage.put_scalar("val_BP", BP)
    #         self.trainer.storage.put_scalar("val_BR_Level0", BR_Level0)
    #         self.trainer.storage.put_scalar("val_BP_Level0", BP_Level0)
    #         self.trainer.storage.put_scalar("val_F1", F1)
    #         self.trainer.storage.put_scalar("val_F1_level0", F1_level0)
    #
    #         # Logging each iteration
    #         info = "Test: [{iter}/{max_iter}] ".format(
    #             iter=i + 1, max_iter=len(self.trainer.val_loader)
    #         )
    #         if "origin_coord" in input_dict.keys():
    #             info = "Interp. " + info
    #         self.trainer.logger.info(
    #             info
    #             + "Loss {loss:.4f} ".format(
    #                 iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
    #             )
    #         )
    #
    #         # 在所有batch结束后计算全局加权平均BR和BP
    #     # global_BR = weighted_sum_BR / total_weight_BR if total_weight_BR > 0 else 0.0
    #     # global_BP = weighted_sum_BP / total_weight_BP if total_weight_BP > 0 else 0.0
    #     # global_BR_level0 = weighted_sum_BR_level0 / total_weight_BR_level0 if total_weight_BR_level0 > 0 else 0.0
    #     # global_BP_level0 = weighted_sum_BP_level0 / total_weight_BP_level0 if total_weight_BP_level0 > 0 else 0.0
    #     # F1 = 2*(global_BR*global_BP)/(global_BR + global_BP)
    #     # F1_level0 = 2*(global_BR_level0*global_BP_level0)/(global_BR_level0 + global_BP_level0)
    #
    #     global_BR = self.trainer.storage.history("val_BR").avg
    #     global_BP = self.trainer.storage.history("val_BP").avg
    #     global_BR_level0 = self.trainer.storage.history("val_BR_Level0").avg
    #     global_BP_level0 = self.trainer.storage.history("val_BP_Level0").avg
    #     # F1 = 2*(global_BR*global_BP)/(global_BR + global_BP)
    #     # F1_level0 = 2*(global_BR_level0*global_BP_level0)/(global_BR_level0 + global_BP_level0)
    #     # 存储全局加权结果
    #     # self.trainer.storage.put_scalar("val_global_BR", global_BR)
    #     # self.trainer.storage.put_scalar("val_global_BP", global_BP)
    #     # self.trainer.storage.put_scalar("val_global_BR_Level0", global_BR_level0)
    #     # self.trainer.storage.put_scalar("val_global_BP_Level0", global_BP_level0)
    #
    #     # Retrieve historical data
    #     loss_avg = self.trainer.storage.history("val_loss").avg
    #     intersection = self.trainer.storage.history("val_intersection").total
    #     union = self.trainer.storage.history("val_union").total
    #     target = self.trainer.storage.history("val_target").total
    #     # br_avg = self.trainer.storage.history("val_BR").avg
    #     # bp_avg = self.trainer.storage.history("val_BP").avg
    #     # br_level0_avg = self.trainer.storage.history("val_BR_Level0").avg
    #     # bp_level0_avg = self.trainer.storage.history("val_BP_Level0").avg
    #
    #     # Retrieve global BR and BP (Weighted metrics)
    #     # global_br_avg = self.trainer.storage.history("val_global_BR").avg
    #     # global_bp_avg = self.trainer.storage.history("val_global_BP").avg
    #     # global_br_level0_avg = self.trainer.storage.history("val_global_BR_Level0").avg
    #     # global_bp_level0_avg = self.trainer.storage.history("val_global_BP_Level0").avg
    #     F1_avg = self.trainer.storage.history("val_F1").avg
    #     F1_level0_avg = self.trainer.storage.history("val_F1_level0").avg
    #     OOA = self.trainer.storage.history("val_OOA").avg
    #     OOA_l0 = self.trainer.storage.history("val_OOA_L0").avg
    #     # Calculate metrics
    #     iou_class = intersection / (union + 1e-10)
    #     acc_class = intersection / (target + 1e-10)
    #     m_iou = np.mean(iou_class)
    #     m_acc = np.mean(acc_class)
    #     all_acc = sum(intersection) / (sum(target) + 1e-10)
    #
    #     # Log final results
    #     self.trainer.logger.info(
    #         "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f} BR/BP {:.4f}/{:.4f} BR_L0/BP_L0 {:.4f}/{:.4f} F1/F1_L0 {:.4f}/{:.4f} OOA {:.4f} OOA_l0 {:.4f}".format(
    #             m_iou, m_acc, all_acc, global_BR, global_BP, global_BR_level0, global_BP_level0, F1_avg, F1_level0_avg,
    #             OOA, OOA_l0
    #         )
    #     )
    #
    #     # Log class-wise results
    #     for i in range(self.trainer.cfg.data.num_classes):
    #         self.trainer.logger.info(
    #             "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
    #                 idx=i,
    #                 name=self.trainer.cfg.data.names[i],
    #                 iou=iou_class[i],
    #                 accuracy=acc_class[i],
    #             )
    #         )
    #
    #     # Write to TensorBoard
    #     current_epoch = self.trainer.epoch + 1
    #     if self.trainer.writer is not None:
    #         self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
    #         self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
    #         self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
    #         self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
    #         self.trainer.writer.add_scalar("val/BR", global_BR, current_epoch)
    #         self.trainer.writer.add_scalar("val/BP", global_BP, current_epoch)
    #         self.trainer.writer.add_scalar("val/BR_Level0", global_BR_level0, current_epoch)
    #         self.trainer.writer.add_scalar("val/BP_Level0", global_BP_level0, current_epoch)
    #         self.trainer.writer.add_scalar("val/F1", F1_avg, current_epoch)
    #         self.trainer.writer.add_scalar("val/F1_Level0", F1_level0_avg, current_epoch)
    #         self.trainer.writer.add_scalar("val/OOA", OOA, current_epoch)
    #         self.trainer.writer.add_scalar("val/OOA_l0", OOA_l0, current_epoch)
    #
    #     # Save metrics for monitoring
    #     self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
    #     self.trainer.comm_info["current_metric_value"] = m_iou
    #     self.trainer.comm_info["current_metric_name"] = "mIoU"
    #     # 增加其他指标的记录
    #
    #     self.trainer.comm_info["current_BR"] = global_BR
    #     self.trainer.comm_info["current_BP"] = global_BP
    #     self.trainer.comm_info["current_OOA"] = OOA
    #     self.trainer.comm_info["current_F1"] = F1_avg  # 增加其他指标的记录

    # def eval(self):
    #     self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    #     self.trainer.model.eval()
    #
    #     # 初始化加权累积变量
    #     total_weight_BR = 0.0
    #     total_weight_BP = 0.0
    #     weighted_sum_BR = 0.0
    #     weighted_sum_BP = 0.0
    #
    #     total_weight_BR_level0 = 0.0
    #     total_weight_BP_level0 = 0.0
    #     weighted_sum_BR_level0 = 0.0
    #     weighted_sum_BP_level0 = 0.0
    #
    #     # 新增用于统计OOA的列表（如果需要对多个batch求平均OOA，可以累积后求平均）
    #     OOA_values = []
    #
    #     for i, input_dict in enumerate(self.trainer.val_loader):
    #         for key in input_dict.keys():
    #             if isinstance(input_dict[key], torch.Tensor):
    #                 input_dict[key] = input_dict[key].cuda(non_blocking=True)
    #         with torch.no_grad():
    #             output_dict = self.trainer.model(input_dict)
    #         output = output_dict["seg_logits"]
    #         loss = output_dict["loss"]
    #         pred = output.max(1)[1]
    #         pred_in_component = output_dict["pred_in_component"]
    #         pred_in_component_object_level = output_dict["pred_in_component_object_level"]
    #         segment = input_dict["segment"]
    #         objects = input_dict["segment"]  # instance or segment. dependant on the task
    #
    #         if "origin_coord" in input_dict.keys():
    #             idx, _ = pointops.knn_query(
    #                 1,
    #                 input_dict["coord"].float(),
    #                 input_dict["offset"].int(),
    #                 input_dict["origin_coord"].float(),
    #                 input_dict["origin_offset"].int(),
    #             )
    #             pred = pred[idx.flatten().long()]
    #             segment = input_dict["origin_segment"]
    #         intersection, union, target = intersection_and_union_gpu(
    #             pred,
    #             segment,
    #             self.trainer.cfg.data.num_classes,
    #             self.trainer.cfg.data.ignore_index,
    #         )
    #         if comm.get_world_size() > 1:
    #             dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
    #         intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
    #
    #         #############################compute BR and BP for SuperPoint Matrix##############################
    #         graph_nn, local_neighbors = compute_graph_nn_optimized(input_dict["coord"], 5, 20)
    #
    #         # 确保所有张量都在同一设备上（通常是 GPU）
    #         device = graph_nn["source"].device
    #         objects = objects.to(device)
    #         pred_in_component = pred_in_component.to(device)
    #         pred_in_component_object_level = pred_in_component_object_level.to(device)
    #
    #         # 计算有效的边（排除包含未标注点的边）
    #         is_transition = objects[graph_nn["source"]] != objects[graph_nn["target"]]
    #
    #         # 计算 pred_transition
    #         pred_transition = pred_in_component[graph_nn["source"]] != pred_in_component[graph_nn["target"]]
    #
    #         pred_transition_object_level = pred_in_component_object_level[graph_nn["source"]] != \
    #                                        pred_in_component_object_level[graph_nn["target"]]
    #
    #         # 使用优化后的 relax_edge_binary 函数
    #         n_ver = input_dict["coord"].shape[0]
    #         relaxed_pred_transition_BR = relax_edge_binary_optimized(
    #             pred_transition,
    #             graph_nn["source"],
    #             graph_nn["target"],
    #             n_ver,
    #             tolerance=1
    #         )
    #
    #         relaxed_pred_transition_BP = relax_edge_binary_optimized(
    #             is_transition,
    #             graph_nn["source"],
    #             graph_nn["target"],
    #             n_ver,
    #             tolerance=1
    #         )
    #
    #         relaxed_pred_transition_object_level_BR = relax_edge_binary_optimized(
    #             pred_transition_object_level,
    #             graph_nn["source"],
    #             graph_nn["target"],
    #             n_ver,
    #             tolerance=1
    #         )
    #
    #         relaxed_pred_transition_object_level_BP = relax_edge_binary_optimized(
    #             is_transition,
    #             graph_nn["source"],
    #             graph_nn["target"],
    #             n_ver,
    #             tolerance=1
    #         )
    #
    #         # 计算 BR 和 BP
    #         BR = compute_boundary_recall_optimized(is_transition, relaxed_pred_transition_BR)
    #         BP = compute_boundary_precision_optimized(relaxed_pred_transition_BP, pred_transition)
    #         BR_Level0 = compute_boundary_recall_optimized(is_transition, relaxed_pred_transition_object_level_BR)
    #         BP_Level0 = compute_boundary_precision_optimized(relaxed_pred_transition_object_level_BP,
    #                                                          pred_transition_object_level)
    #
    #         # 对当前batch计算权重
    #         weight_BR = is_transition.sum().item()  # 用真实边界体素数加权BR
    #         weight_BP = pred_transition.sum().item()  # 用预测边界体素数加权BP
    #         weight_BR_level0 = is_transition.sum().item()  # 同样对Level0 BR加权
    #         weight_BP_level0 = pred_transition_object_level.sum().item()  # 对Level0 BP加权
    #
    #         # 累积权重
    #         total_weight_BR += weight_BR
    #         total_weight_BP += weight_BP
    #         total_weight_BR_level0 += weight_BR_level0
    #         total_weight_BP_level0 += weight_BP_level0
    #
    #         # 累积加权和
    #         weighted_sum_BR += weight_BR * BR.item()
    #         weighted_sum_BP += weight_BP * BP.item()
    #         weighted_sum_BR_level0 += weight_BR_level0 * BR_Level0.item()
    #         weighted_sum_BP_level0 += weight_BP_level0 * BP_Level0.item()
    #
    #         #############################compute BR and BP for SuperPoint Matrix##############################
    #
    #         # *********************** 优化后的 OOA 计算 ************************
    #         num_classes = self.trainer.cfg.data.num_classes
    #
    #         # 计算 OOA
    #         sp_unique, sp_inverse = torch.unique(pred_in_component, sorted=True, return_inverse=True)
    #         M = sp_unique.size(0)
    #         counts = torch.zeros((M, num_classes), device=segment.device, dtype=torch.int64)
    #         counts.index_put_((sp_inverse, segment), torch.ones_like(segment, dtype=torch.int64), accumulate=True)
    #         max_freq, _ = counts.max(dim=1)
    #         correct_labels = max_freq.sum().item()
    #         OOA = 100.0 * correct_labels / segment.size(0)
    #         self.trainer.storage.put_scalar("val_OOA", OOA)
    #
    #         # 计算 OOA_L0（基于pred_in_component_object_level）
    #         sp_unique_l0, sp_inverse_l0 = torch.unique(pred_in_component_object_level, sorted=True, return_inverse=True)
    #         M_l0 = sp_unique_l0.size(0)
    #         counts_l0 = torch.zeros((M_l0, num_classes), device=segment.device, dtype=torch.int64)
    #         counts_l0.index_put_((sp_inverse_l0, segment), torch.ones_like(segment, dtype=torch.int64), accumulate=True)
    #         max_freq_l0, _ = counts_l0.max(dim=1)
    #         correct_labels_l0 = max_freq_l0.sum().item()
    #         OOA_L0 = 100.0 * correct_labels_l0 / segment.size(0)
    #         self.trainer.storage.put_scalar("val_OOA_L0", OOA_L0)
    #         # *********************** 优化后的 OOA 计算结束 **********************
    #
    #         # Store results
    #         self.trainer.storage.put_scalar("val_intersection", intersection)
    #         self.trainer.storage.put_scalar("val_union", union)
    #         self.trainer.storage.put_scalar("val_target", target)
    #         self.trainer.storage.put_scalar("val_loss", loss.item())
    #         self.trainer.storage.put_scalar("val_BR", BR)
    #         self.trainer.storage.put_scalar("val_BP", BP)
    #         self.trainer.storage.put_scalar("val_BR_Level0", BR_Level0)
    #         self.trainer.storage.put_scalar("val_BP_Level0", BP_Level0)
    #
    #         # Logging each iteration
    #         info = "Test: [{iter}/{max_iter}] ".format(
    #             iter=i + 1, max_iter=len(self.trainer.val_loader)
    #         )
    #         if "origin_coord" in input_dict.keys():
    #             info = "Interp. " + info
    #         self.trainer.logger.info(
    #             info
    #             + "Loss {loss:.4f} ".format(
    #                 iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
    #             )
    #         )
    #
    #     # 在所有batch结束后计算全局加权BR和BP
    #     global_BR = weighted_sum_BR / total_weight_BR if total_weight_BR > 0 else 0.0
    #     global_BP = weighted_sum_BP / total_weight_BP if total_weight_BP > 0 else 0.0
    #     global_BR_level0 = weighted_sum_BR_level0 / total_weight_BR_level0 if total_weight_BR_level0 > 0 else 0.0
    #     global_BP_level0 = weighted_sum_BP_level0 / total_weight_BP_level0 if total_weight_BP_level0 > 0 else 0.0
    #
    #     F1 = 2 * (global_BR * global_BP) / (global_BR + global_BP)
    #     F1_level0 = 2 * (global_BR_level0 * global_BP_level0) / (global_BR_level0 + global_BP_level0)
    #     self.trainer.storage.put_scalar("val_F1", F1)
    #     self.trainer.storage.put_scalar("val_F1_level0", F1_level0)
    #     # Retrieve historical data
    #     loss_avg = self.trainer.storage.history("val_loss").avg
    #     intersection = self.trainer.storage.history("val_intersection").total
    #     union = self.trainer.storage.history("val_union").total
    #     target = self.trainer.storage.history("val_target").total
    #     F1_avg = self.trainer.storage.history("val_F  1").avg
    #     F1_level0_avg = self.trainer.storage.history("val_F1_level0").avg
    #     OOA = self.trainer.storage.history("val_OOA").avg
    #     OOA_l0 = self.trainer.storage.history("val_OOA_L0").avg
    #
    #     # Calculate metrics
    #     iou_class = intersection / (union + 1e-10)
    #     acc_class = intersection / (target + 1e-10)
    #     m_iou = np.mean(iou_class)
    #     m_acc = np.mean(acc_class)
    #     all_acc = sum(intersection) / (sum(target) + 1e-10)
    #
    #     # Log final results
    #     self.trainer.logger.info(
    #         "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f} BR/BP {:.4f}/{:.4f} BR_L0/BP_L0 {:.4f}/{:.4f} F1/F1_L0 {:.4f}/{:.4f} OOA {:.4f} OOA_l0 {:.4f}".format(
    #             m_iou, m_acc, all_acc, global_BR, global_BP, global_BR_level0, global_BP_level0, F1, F1_level0, OOA,
    #             OOA_l0
    #         )
    #     )
    #
    #     # Log class-wise results
    #     for i in range(self.trainer.cfg.data.num_classes):
    #         self.trainer.logger.info(
    #             "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
    #                 idx=i,
    #                 name=self.trainer.cfg.data.names[i],
    #                 iou=iou_class[i],
    #                 accuracy=acc_class[i],
    #             )
    #         )
    #
    #     # Write to TensorBoard
    #     current_epoch = self.trainer.epoch + 1
    #     if self.trainer.writer is not None:
    #         self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
    #         self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
    #         self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
    #         self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
    #         self.trainer.writer.add_scalar("val/BR", global_BR, current_epoch)
    #         self.trainer.writer.add_scalar("val/BP", global_BP, current_epoch)
    #         self.trainer.writer.add_scalar("val/BR_Level0", global_BR_level0, current_epoch)
    #         self.trainer.writer.add_scalar("val/BP_Level0", global_BP_level0, current_epoch)
    #         self.trainer.writer.add_scalar("val/F1", F1_avg, current_epoch)
    #         self.trainer.writer.add_scalar("val/F1_Level0", F1_level0_avg, current_epoch)
    #         self.trainer.writer.add_scalar("val/OOA", OOA, current_epoch)
    #         self.trainer.writer.add_scalar("val/OOA_l0", OOA_l0, current_epoch)
    #
    #     # Save metrics for monitoring
    #     self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
    #     self.trainer.comm_info["current_metric_value"] = m_iou
    #     self.trainer.comm_info["current_metric_name"] = "mIoU"

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(self, segment_ignore_index=(-1,), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.segment_ignore_index)

        assert (
                pred["pred_classes"].shape[0]
                == pred["pred_scores"].shape[0]
                == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        # get gt instances
        gt_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                gt_instances[self.trainer.cfg.data.names[i]] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.trainer.cfg.data.names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for i in range(self.trainer.cfg.data.num_classes):
            if i not in self.segment_ignore_index:
                pred_instances[self.trainer.cfg.data.names[i]] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.trainer.cfg.data.names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
                zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                               and gt["med_dist"] <= distance_thresh
                               and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                        gt["vert_count"]
                                        + pred["vert_count"]
                                        - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                        gt["vert_count"]
                                        + pred["vert_count"]
                                        - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                            gt["vert_count"] < min_region_size
                                            or gt["med_dist"] > distance_thresh
                                            or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                        float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                    else:
                        ap_current = float("nan")
                    ap_table[di, li, oi] = ap_current
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
        return ap_scores

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = []
        for i, input_dict in enumerate(self.trainer.val_loader):
            assert (
                    len(input_dict["offset"]) == 1
            )  # currently only support bs 1 for each GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            # map to origin
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                idx = idx.cpu().flatten().long()
                output_dict["pred_masks"] = output_dict["pred_masks"][:, idx]
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes.append(dict(gt=gt_instances, pred=pred_instance))

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        self.trainer.logger.info(
            "Val result: mAP/AP50/AP25 {:.4f}/{:.4f}/{:.4f}.".format(
                all_ap, all_ap_50, all_ap_25
            )
        )
        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: AP/AP50/AP25 {AP:.4f}/{AP50:.4f}/{AP25:.4f}".format(
                    idx=i, name=label_name, AP=ap, AP50=ap_50, AP25=ap_25
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mAP", all_ap, current_epoch)
            self.trainer.writer.add_scalar("val/AP50", all_ap_50, current_epoch)
            self.trainer.writer.add_scalar("val/AP25", all_ap_25, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_ap_50  # save for saver
        self.trainer.comm_info["current_metric_name"] = "AP50"  # save for saver
