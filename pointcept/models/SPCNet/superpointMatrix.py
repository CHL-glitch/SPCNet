import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
import torch
from numpy import linalg as LA


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


def compute_graph_nn_2_torch_opt(xyz, k_nn1, k_nn2, voronoi=0.0):
    """Compute simultaneously 2 knn structures using PyTorch.
    Assumption: knn1 <= knn2."""
    assert k_nn1 <= k_nn2, "k_nn1 must be smaller than or equal to k_nn2"

    # Ensure xyz is a PyTorch tensor and move to appropriate device
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.tensor(xyz, dtype=torch.float32)
    xyz = xyz.to("cuda" if torch.cuda.is_available() else "cpu")

    n_ver = xyz.shape[0]
    # Compute pairwise distances
    distances = torch.cdist(xyz, xyz, p=2)  # Pairwise Euclidean distance, shape (n_ver, n_ver)

    # Get nearest neighbors
    knn_distances, knn_indices = torch.topk(distances, k=k_nn2 + 1, largest=False, dim=-1)  # Keep smallest distances
    knn_distances = knn_distances[:, 1:]  # Exclude self-distance
    knn_indices = knn_indices[:, 1:]  # Exclude self-index

    # --- knn2 ---
    target2 = knn_indices.flatten().cpu().numpy().astype('int64')  # Ensure compatible type

    # --- knn1 ---
    graph = {"is_nn": True}
    if voronoi > 0:
        raise NotImplementedError("Voronoi-based filtering is not implemented in the PyTorch version.")
    else:
        knn1_indices = knn_indices[:, :k_nn1]
        knn1_distances = knn_distances[:, :k_nn1]
        graph["source"] = torch.repeat_interleave(torch.arange(n_ver, device=xyz.device), k_nn1).cpu().numpy().astype(
            'int64')
        graph["target"] = knn1_indices.flatten().cpu().numpy().astype('int64')  # Ensure compatible type
        graph["distances"] = knn1_distances.flatten().cpu().numpy().astype('float32')

    return graph, target2


# Test the function again
# graph_nn, local_neighbors = compute_graph_nn_2(xyz, k_nn1, k_nn2)
# print("graph_nn[source]:", graph_nn["source"])
# print("graph_nn[target]:", graph_nn["target"])

def compute_partition(args, embeddings, edg_source, edg_target, diff, xyz=0):
    edge_weight = np.ones_like(edg_source).astype('f4')
    if args.edge_weight_threshold > 0:
        edge_weight[diff > 1] = args.edge_weight_threshold
    if args.edge_weight_threshold < 0:
        edge_weight = torch.exp(diff * args.edge_weight_threshold).detach().cpu().numpy() / np.exp(
            args.edge_weight_threshold)

    ver_value = np.zeros((embeddings.shape[0], 0), dtype='f4')
    use_spatial = 0
    ver_value = np.hstack((ver_value, embeddings.detach().cpu().numpy()))
    if args.spatial_emb > 0:
        ver_value = np.hstack((ver_value, args.spatial_emb * xyz))  # * math.sqrt(args.reg_strength)))
        # ver_value = xyz * args.spatial_emb
        use_spatial = 1  # !!!

    pred_components, pred_in_component = libcp.cutpursuit(ver_value, \
                                                          edg_source.astype('uint32'), edg_target.astype('uint32'),
                                                          edge_weight, \
                                                          args.reg_strength / (4 * args.k_nn_adj),
                                                          cutoff=args.CP_cutoff, spatial=use_spatial, weight_decay=0.7)
    # emb2 = libcp.cutpursuit2(ver_value, edg_source.astype('uint32'), edg_target.astype('uint32'), edge_weight, args.reg_strength, cutoff=0, spatial =0)
    # emb2 = emb2.reshape(ver_value.shape)
    # ((ver_value-emb2)**2).sum(0)
    # cut = pred_in_component[edg_source]!=pred_in_component[edg_target]
    return pred_components, pred_in_component
