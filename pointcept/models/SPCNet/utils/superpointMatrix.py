import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
import torch
from numpy import linalg as LA
from torch_geometric.nn import knn_graph


# Generate a random (1000, 3) point cloud tensor (xyz coordinates)

# np.random.seed(42)  # For reproducibility
#
# xyz = np.random.rand(1000, 3).astype('float32')
# k_nn1 = 5
# k_nn2 = 20


def compute_graph_nn_2(xyz, k_nn1, k_nn2, voronoi=0.0):
    """Compute simultaneously 2 knn structures. Only saves target for knn2.
    Assumption: knn1 <= knn2."""
    assert k_nn1 <= k_nn2, "k_nn1 must be smaller than or equal to k_nn2"

    # Convert PyTorch tensor to NumPy if necessary
    if isinstance(xyz, torch.Tensor):
        xyz = xyz.cpu().numpy()

    n_ver = xyz.shape[0]
    # Compute nearest neighbors
    graph = {"is_nn": True}
    nn = NearestNeighbors(n_neighbors=k_nn2 + 1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    del nn
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]

    # --- knn2 ---
    target2 = neighbors.flatten().astype('int64')  # Changed from uint32 to int64

    # --- knn1 ---
    if voronoi > 0:
        tri = Delaunay(xyz)
        graph["source"] = np.hstack((
            tri.vertices[:, 0], tri.vertices[:, 0], tri.vertices[:, 0],
            tri.vertices[:, 1], tri.vertices[:, 1], tri.vertices[:, 2]
        )).astype('int64')  # Changed from uint64 to int64
        graph["target"] = np.hstack((
            tri.vertices[:, 1], tri.vertices[:, 2], tri.vertices[:, 3],
            tri.vertices[:, 2], tri.vertices[:, 3], tri.vertices[:, 3]
        )).astype('int64')  # Changed from uint64 to int64
        graph["distances"] = ((xyz[graph["source"]] - xyz[graph["target"]]) ** 2).sum(1)
        keep_edges = graph["distances"] < voronoi
        graph["source"] = graph["source"][keep_edges]
        graph["target"] = graph["target"][keep_edges]

        graph["source"] = np.hstack((graph["source"],
                                     np.repeat(np.arange(n_ver), k_nn1).astype('int64')))
        neighbors = neighbors[:, :k_nn1]
        graph["target"] = np.hstack((graph["target"],
                                     np.transpose(neighbors.flatten(order='C')).astype('int64')))

        edg_id = graph["source"] + n_ver * graph["target"]
        _, unique_edges = np.unique(edg_id, return_index=True)
        graph["source"] = graph["source"][unique_edges]
        graph["target"] = graph["target"][unique_edges]

        graph["distances"] = graph["distances"][keep_edges]
    else:
        neighbors = neighbors[:, :k_nn1]
        distances = distances[:, :k_nn1]
        graph["source"] = np.repeat(np.arange(n_ver), k_nn1).astype('int64')  # Changed to int64
        graph["target"] = np.transpose(neighbors.flatten(order='C')).astype('int64')  # Changed to int64
        graph["distances"] = distances.flatten().astype('float32')

    return graph, target2


def compute_graph_nn_optimized(xyz, k_nn1, k_nn2):
    """
    使用 PyTorch Geometric 的 knn_graph 函数计算 k-NN 图。
    假设 k_nn1 <= k_nn2。
    """
    assert k_nn1 <= k_nn2, "k_nn1 must be smaller than or equal to k_nn2"

    # 确保 xyz 是 PyTorch 张量，且在合适的设备上
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.tensor(xyz, dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xyz = xyz.to(device)

    n_ver = xyz.shape[0]

    edge_index_knn1 = knn_graph(xyz, k=k_nn1, loop=False, flow='source_to_target')
    source_knn1 = edge_index_knn1[0]  # 源节点索引
    target_knn1 = edge_index_knn1[1]  # 目标节点索引
    distances_knn1 = torch.norm(xyz[source_knn1] - xyz[target_knn1], p=2, dim=1)  # 计算边的距离

    edge_index_knn2 = knn_graph(xyz, k=k_nn2, loop=False, flow='source_to_target')
    target2 = edge_index_knn2[1]  # 仅需要目标节点索引

    graph = {
        "is_nn": True,
        "source": source_knn1,
        "target": target_knn1,
        "distances": distances_knn1
    }

    return graph, target2


# Test the function again
# graph_nn, local_neighbors = compute_graph_nn_2(xyz, k_nn1, k_nn2)
# print("graph_nn[source]:", graph_nn["source"])
# print("graph_nn[target]:", graph_nn["target"])

# ------------------------------ superpoint metrics ------------------------------

def relax_edge_binary(edg_binary, edg_source, edg_target, n_ver, tolerance):
    if torch.is_tensor(edg_binary):
        relaxed_binary = edg_binary.cpu().numpy().copy()
    else:
        relaxed_binary = edg_binary.copy()
    transition_vertex = np.full((n_ver,), 0, dtype='uint8')
    for i_tolerance in range(tolerance):
        transition_vertex[edg_source[relaxed_binary.nonzero()]] = True
        transition_vertex[edg_target[relaxed_binary.nonzero()]] = True
        relaxed_binary[transition_vertex[edg_source]] = True
        relaxed_binary[transition_vertex[edg_target] > 0] = True
    return relaxed_binary


# def relax_edge_binary_optimized(edg_binary, edg_source, edg_target, n_ver, tolerance):
#     """
#     使用 PyTorch 优化的 relax_edge_binary 函数，避免循环和数据转换。
#     """
#     device = edg_binary.device
#     relaxed_binary = edg_binary.clone()
#     transition_vertex = torch.zeros(n_ver, dtype=torch.bool, device=device)
#
#     for _ in range(tolerance):
#         transition_vertex.index_fill_(0, edg_source[relaxed_binary], True)
#         transition_vertex.index_fill_(0, edg_target[relaxed_binary], True)
#         relaxed_binary = relaxed_binary | transition_vertex[edg_source] | transition_vertex[edg_target]
#
#     return relaxed_binary
def relax_edge_binary_optimized(edg_binary, edg_source, edg_target, n_ver, tolerance):
    """
    优化后的 relax_edge_binary 函数，通过向量化操作和避免使用 index_fill_ 提高效率。

    参数:
    - edg_binary (torch.Tensor): 二进制边缘张量，形状为 [num_edges]。
    - edg_source (torch.Tensor): 边缘的源节点索引，形状为 [num_edges]。
    - edg_target (torch.Tensor): 边缘的目标节点索引，形状为 [num_edges]。
    - n_ver (int): 顶点数量。
    - tolerance (int): 放宽的容差次数。

    返回:
    - relaxed_binary (torch.Tensor): 放宽后的二进制边缘张量，形状为 [num_edges]。
    """
    device = edg_binary.device
    relaxed_binary = edg_binary.clone()
    transition_vertex = torch.zeros(n_ver, dtype=torch.bool, device=device)

    for _ in range(tolerance):
        # 获取当前 relaxed_binary 为 True 的边缘对应的源和目标节点
        selected_sources = edg_source[relaxed_binary]
        selected_targets = edg_target[relaxed_binary]

        # 重置 transition_vertex
        transition_vertex.zero_()

        # 设置 transition_vertex 中对应的源和目标节点为 True
        transition_vertex[selected_sources] = True
        transition_vertex[selected_targets] = True

        # 使用向量化操作更新 relaxed_binary
        relaxed_binary |= transition_vertex[edg_source] | transition_vertex[edg_target]

    return relaxed_binary


def compute_boundary_recall(is_transition, pred_transitions):
    """
    计算边界召回率 (Boundary Recall, BR)
    参数:
        is_transition: 真实边界标志 (张量, dtype=torch.bool 或 torch.int)
        pred_transitions: 预测边界标志 (张量, dtype=torch.bool 或 torch.int)
    返回:
        Boundary Recall (BR) 的百分比值
    """

    # 确保输入的张量类型是布尔型并形状一致
    assert is_transition.shape == pred_transitions.shape, "Shape mismatch between is_transition and pred_transitions!"
    is_transition = is_transition.bool()

    # 如果 pred_transitions 是 numpy 数组，则转换为 torch 张量，并确保它是布尔类型
    if isinstance(pred_transitions, np.ndarray):
        pred_transitions = torch.tensor(pred_transitions, dtype=torch.bool)

    # 确保 pred_transitions 是 torch 张量，并转换为布尔型
    pred_transitions = pred_transitions.bool()

    # 确保两个张量在相同的设备上
    if is_transition.device != pred_transitions.device:
        pred_transitions = pred_transitions.to(is_transition.device)

    # 打印输入的 is_transition 和 pred_transitions 的统计值
    # print(f"is_transition: True = {is_transition.sum().item()}, False = {(~is_transition).sum().item()}")
    # print(f"pred_transitions: True = {pred_transitions.sum().item()}, False = {(~pred_transitions).sum().item()}")

    # 逐元素比较 is_transition == pred_transitions
    equality = (is_transition == pred_transitions)  # 逐元素比较
    # print(f"(is_transition == pred_transitions): True = {equality.sum().item()}, False = {(~equality).sum().item()}")
    # print(f"equality tensor (逐元素比较): {equality.cpu().numpy()}")  # 打印逐元素的比较结果

    # 逐元素计算 (is_transition == pred_transitions) & is_transition
    boundary_recall_component = equality & is_transition
    # print(f"boundary_recall_component (equality & is_transition): True = {boundary_recall_component.sum().item()}, False = {(~boundary_recall_component).sum().item()}")
    # print(f"boundary_recall_component tensor (逐元素计算): {boundary_recall_component.cpu().numpy()}")  # 打印逐元素的计算结果

    # 计算 is_transition.sum()
    is_transition_sum = is_transition.sum().item()
    # print(f"is_transition.sum(): {is_transition_sum}")

    # 检查分母为零的情况
    if is_transition_sum == 0:
        # print("Warning: No transitions in is_transition. Returning 0 as BR.")
        return torch.tensor(0.0, device=is_transition.device)

    # 计算并返回 Boundary Recall (BR)
    br = (boundary_recall_component.sum().item() / is_transition_sum) * 100.0
    # print(f"BR: {br}")

    return br


def compute_boundary_recall_optimized(is_transition, pred_transitions):
    """
       计算边界召回率 (Boundary Recall, BR)。
       """
    assert is_transition.shape == pred_transitions.shape, "Shape mismatch between is_transition and pred_transitions!"

    is_transition = is_transition.bool()
    pred_transitions = pred_transitions.bool()

    equal_transitions = (is_transition == pred_transitions)

    numerator = (equal_transitions & is_transition).sum().float()
    denominator = is_transition.sum().float()

    if denominator == 0:
        return torch.tensor(0.0, device=is_transition.device)

    br = (numerator / denominator) * 100.0
    return br


def compute_boundary_precision(is_transition, pred_transitions):
    """
    计算边界精确率 (Boundary Precision, BP)
    参数:
        is_transition: 真实边界标志 (张量, dtype=torch.bool 或 torch.int)
        pred_transitions: 预测边界标志 (张量, dtype=torch.bool 或 torch.int)
    返回:
        Boundary Precision (BP) 的百分比值
    """
    # 确保输入的张量形状一致
    assert is_transition.shape == pred_transitions.shape, "Shape mismatch between is_transition and pred_transitions!"

    # 将 is_transition 转换为布尔类型
    is_transition = is_transition.bool()

    # 如果 pred_transitions 是 numpy 数组，则转换为 torch 张量，并确保它是布尔类型
    if isinstance(pred_transitions, np.ndarray):
        pred_transitions = torch.tensor(pred_transitions, dtype=torch.bool)

    # 确保 pred_transitions 是 torch 张量，并转换为布尔型
    pred_transitions = pred_transitions.bool()

    # 确保两个张量在相同的设备上
    if is_transition.device != pred_transitions.device:
        pred_transitions = pred_transitions.to(is_transition.device)

    # 计算 pred_transitions.sum()，并检查分母为零的情况
    pred_transitions_sum = pred_transitions.sum().item()
    if pred_transitions_sum == 0:
        # 如果 pred_transitions 全为 False，则返回 0
        # print("Warning: No predicted transitions in pred_transitions. Returning 0 as BP.")
        return torch.tensor(0.0, device=is_transition.device)

    # 逐元素比较 is_transition == pred_transitions，并与 pred_transitions 按位与
    boundary_precision_component = (is_transition == pred_transitions) & pred_transitions

    # 计算 Boundary Precision (BP)
    bp = (boundary_precision_component.sum().item() / pred_transitions_sum) * 100.0

    return bp


def compute_boundary_precision_optimized(is_transition, pred_transitions):
    """
        计算边界精确率 (Boundary Precision, BP)。
        """
    assert is_transition.shape == pred_transitions.shape, "Shape mismatch between is_transition and pred_transitions!"

    is_transition = is_transition.bool()
    pred_transitions = pred_transitions.bool()
    equal_transitions = (is_transition == pred_transitions)
    numerator = (equal_transitions & pred_transitions).sum().float()
    denominator = pred_transitions.sum().float()

    if denominator == 0:
        return torch.tensor(0.0, device=is_transition.device)

    bp = (numerator / denominator) * 100.0
    return bp


def mode(array, only_freq=False):
    value, counts = np.unique(array, return_counts=True)
    if only_freq:
        return np.amax(counts)
    else:
        return value[np.argmax(counts)], np.amax(counts)


def compute_OOA(components, gt_labels):
    """
    计算 OOA (Oracle Overall Accuracy)
    components: list，每个元素是一个 superpoint 包含的点的index数组。
    gt_labels: numpy array，长度为点的个数，表示每个点的真实类标签ID。
    """
    correct_labels = 0
    for comp in components:
        # 对该superpoint内的所有点标签求众数
        _, freq = mode(gt_labels[comp])
        correct_labels += freq
    return 100.0 * correct_labels / len(gt_labels)