import numpy as np
import networkx as nx
import itertools
from .graphs import ConstructGraph, GetSubGraph
from .core import (
    PartitionData,
    # PartitionData_cp,
    Encode2ElasticMatrix,
)

# -----extract oriented branches and associated data
def bf_search(dict_branches, root_node):
    """ breadth-first tree search """
    flat_tree = nx.Graph()
    flat_tree.add_nodes_from(
        list(set(itertools.chain.from_iterable(dict_branches.keys())))
    )
    flat_tree.add_edges_from(dict_branches.keys())
    edges = list(nx.bfs_edges(flat_tree, root_node))
    nodes = [root_node] + [v for u, v in edges]
    return edges, nodes


def get_tree(Edges, root_node):
    # ----get branches
    net = ConstructGraph({"Edges": [Edges]})
    branches = GetSubGraph(net, "branches")
    _dict_branches = {
        (b[0], b[-1]): b for i, b in enumerate(branches)
    }  # temporary branch node lists (not in order)

    # ----check validity of the root node
    root_branch = [k for k in _dict_branches.keys() if root_node in k]
    if len(root_branch) > 1:
        raise ValueError(f"Multiple root branches {root_branch}")

    # ----reorder branches
    # find ordered relations between branches
    ordered_edges, ordered_nodes = bf_search(_dict_branches, root_node)

    # ----create ordered dicts
    dict_tree = {}  # branch parent-child relations
    dict_branches = {}  # branch node lists
    dict_branches_single_end = (
        {}
    )  # branch node lists with no overlapping terminal nodes
    # visited_nodes = []
    for i, e in enumerate(ordered_edges):  # for each branch
        # store branch in order (both the key and the list)
        if e not in _dict_branches:
            dict_branches[e] = _dict_branches[e[::-1]][::-1]
        else:
            dict_branches[e] = _dict_branches[e]

        # store children
        dict_tree[e] = [
            _e for _e in ordered_edges[:i] + ordered_edges[i + 1 :] if e[-1] in _e
        ]

        # store single ended branch
        if i == 0:
            dict_branches_single_end[e] = dict_branches[e]  # if n not in visited_nodes]
        else:
            dict_branches_single_end[e] = dict_branches[e][1:]
        # dict_branches_single_end[e] = [n for n in dict_branches[e] if n not in visited_nodes]
        # visited_nodes.extend(dict_branches[e])
    return dict_tree, dict_branches, dict_branches_single_end


def partition_data_by_branch(X, NodePositions, branches):
    partition, dists = PartitionData(
        X, NodePositions, 10 ** 8, np.sum(X ** 2, axis=1, keepdims=1)
    )
    branches_dataidx = {k: np.isin(partition[:, 0], b) for k, b in branches.items()}
    return branches_dataidx


# ------generate pseudotime centroid branches
def nNodes_pseudotime(bX, bpseudotime, bnNodes):
    blocksize = int(len(bX) / bnNodes)
    argsort_pseudotime = np.argsort(pseudotime)
    PseudotimeNodePositions = np.zeros((bnNodes, bX.shape[1]))
    for idx_curve, idx_data in enumerate(np.arange(0, len(bX), blocksize)):
        PseudotimeNodePositions[idx_curve] = bX[
            argsort_pseudotime[idx_data : idx_data + blocksize]
        ].mean(axis=0)
    return PseudotimeNodePositions


def bin_pseudotime(bX, bpseudotime, bnNodes):
    # create nodes with uniformly spread pseudotime
    count, bins = np.histogram(bpseudotime, bins=bnNodes)
    clusters = np.digitize(bpseudotime, bins[1:], right=True)
    PseudotimeNodePositions = np.zeros((bnNodes, bX.shape[1]))
    MeanPseudotime = np.zeros(bnNodes)
    # for each branch node
    for j in range(bnNodes):
        # index associated data
        data_idx = clusters == j
        # generate node
        PseudotimeNodePositions[j] = bX[data_idx].mean(axis=0)
        MeanPseudotime[j] = bpseudotime[data_idx].mean()
    return PseudotimeNodePositions, MeanPseudotime


def gen_pseudotime_centroids(X, pseudotime, branches_single_end, branches_dataidx):
    """generate pseudotime centroids for each branch of the graph"""
    for i, (k, bdata) in enumerate(
        branches_dataidx.items()
    ):  # for data associated with each branch
        # branch data points, data pseudotime
        bX, bpseudotime = X[bdata], pseudotime[bdata]
        bnNodes = len(branches_single_end[k])
        # generate node positions
        if i == 0:
            PseudotimeNodePositions, MeanPseudotime = bin_pseudotime(
                bX, bpseudotime, bnNodes
            )
        else:
            _PseudotimeNodePositions, _MeanPseudotime = bin_pseudotime(
                bX, bpseudotime, bnNodes
            )
            PseudotimeNodePositions = np.concatenate(
                (PseudotimeNodePositions, _PseudotimeNodePositions)
            )
            MeanPseudotime = np.concatenate((MeanPseudotime, _MeanPseudotime))
    return PseudotimeNodePositions, MeanPseudotime


# ------for each branch, create elastic edges between pseudotime nodes & elpigraph nodes and merge pseudotime and elpigraph nodesp, elasticmatrix
def pseudotime_augmented_graph(
    NodePositions,
    Edges,
    PseudotimeNodePositions,
    branches_single_end,
    Mus,
    Lambdas,
    LinkMu,
    LinkLambda,
):
    """
    generate a graph merging node positions and pseudotime node positions
    with one edge between each of their nodes. 
    pseudotime node positions and edges are placed as the top rows of the matrices
    """
    # ------for each branch, create elastic edges between pseudotime nodes & elpigraph nodes
    # ordering of nodespositions in the graph (corresponding to pseudotime nodes)
    NodesOrder = np.array([n for b in branches_single_end.values() for n in b])
    PseudotimeNodes = np.arange(len(PseudotimeNodePositions))
    # link pseudotime nodes and graph nodes
    LinkEdges = np.array(
        list(zip(PseudotimeNodes, NodesOrder + len(PseudotimeNodePositions)))
    )
    LinkMus = np.repeat(LinkMu, len(PseudotimeNodePositions))
    LinkLambdas = np.repeat(LinkLambda, len(PseudotimeNodePositions))

    # -----merge pseudotime and graph nodepositions, elasticmatrix
    MergedNodePositions = np.concatenate((PseudotimeNodePositions, NodePositions))
    MergedEdges = np.concatenate((LinkEdges, Edges + len(PseudotimeNodePositions)))
    MergedLambdas = np.concatenate((LinkLambdas, Lambdas))
    MergedMus = np.concatenate((LinkMus, Mus))
    MergedElasticMatrix = Encode2ElasticMatrix(MergedEdges, MergedLambdas, MergedMus)

    return (
        MergedNodePositions,
        MergedElasticMatrix,
        MergedEdges,
        MergedLambdas,
        MergedMus,
    )

