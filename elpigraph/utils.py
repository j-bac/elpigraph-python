import numpy as np
import pandas as pd
import networkx as nx
import scipy
import networkx as nx
from .src.core import PartitionData
from .src.reporting import project_point_onto_graph
from sklearn.neighbors import NearestNeighbors, KernelDensity


def getProjection(X, PG):
    """Compute graph projection from principal graph dict
    (result stored in PG dict under 'projection' key)
    """

    G = nx.Graph()
    G.add_edges_from(PG["Edges"][0].tolist(), weight=1)
    mat_conn = nx.to_scipy_sparse_matrix(
        G, nodelist=np.arange(len(PG["NodePositions"])), weight="weight"
    )

    # partition points
    node_id, node_dist = PartitionData(
        X=X,
        NodePositions=PG["NodePositions"],
        MaxBlockSize=len(PG["NodePositions"]) ** 4,
        SquaredX=np.sum(X ** 2, axis=1, keepdims=1),
    )
    # project points onto edges
    dict_proj = project_point_onto_graph(
        X=X,
        NodePositions=PG["NodePositions"],
        Edges=PG["Edges"][0],
        Partition=node_id,
    )

    PG["projection"] = {}
    PG["projection"]["node_id"] = node_id.flatten()
    PG["projection"]["node_dist"] = node_dist
    PG["projection"]["edge_id"] = dict_proj["EdgeID"].astype(int)
    PG["projection"]["edge_loc"] = dict_proj["ProjectionValues"]
    PG["projection"]["X_projected"] = dict_proj["X_projected"]
    PG["projection"]["conn"] = mat_conn
    PG["projection"]["edge_len"] = dict_proj["EdgeLen"]
    PG["projection"]["MSEP"] = dict_proj["MSEP"]


def getPseudotime(
    X, PG, source, target=None, nodes_to_include=None, project=True
):
    """Compute pseudotime given
    source: int
        source node
    target: int
        optional target node
    nodes_to_include: list
        optional nodes to include in the path
        (useful for complex topologies with loops,
        where multiple paths are possible between 2 nodes)
    project: bool
        if False, will save computation time by using the projection already stored in PG dict (computed using elpigraph.utils.getProjection)
    """
    if project is True:
        getProjection(X, PG)
    elif project is False and not ("projection" in PG.keys()):
        raise ValueError(
            "key 'projection not found in PG. To use a precomputed projection"
            " run elpigraph.utils.getProjection"
        )

    epg_edge = PG["Edges"][0]
    epg_edge_len = PG["projection"]["edge_len"]
    G = nx.Graph()
    edges_weighted = list(zip(epg_edge[:, 0], epg_edge[:, 1], epg_edge_len))
    G.add_weighted_edges_from(edges_weighted, weight="len")
    if target is not None:
        if nodes_to_include is None:
            # nodes on the shortest path
            nodes_sp = nx.shortest_path(
                G, source=source, target=target, weight="len"
            )
        else:
            assert isinstance(
                nodes_to_include, list
            ), "`nodes_to_include` must be list"
            # lists of simple paths, in order from shortest to longest
            list_paths = list(
                nx.shortest_simple_paths(
                    G, source=source, target=target, weight="len"
                )
            )
            flag_exist = False
            for p in list_paths:
                if set(nodes_to_include).issubset(p):
                    nodes_sp = p
                    flag_exist = True
                    break
            if not flag_exist:
                return f"no path that passes {nodes_to_include} exists"
    else:
        nodes_sp = [source] + [v for u, v in nx.bfs_edges(G, source)]
    G_sp = G.subgraph(nodes_sp).copy()
    index_nodes = {
        x: nodes_sp.index(x) if x in nodes_sp else G.number_of_nodes()
        for x in G.nodes
    }

    if target is None:
        dict_dist_to_source = nx.shortest_path_length(
            G_sp, source=source, weight="len"
        )
    else:
        dict_dist_to_source = dict(
            zip(
                nodes_sp,
                np.cumsum(
                    np.array(
                        [0.0]
                        + [
                            G.get_edge_data(nodes_sp[i], nodes_sp[i + 1])[
                                "len"
                            ]
                            for i in range(len(nodes_sp) - 1)
                        ]
                    )
                ),
            )
        )

    cells = np.isin(PG["projection"]["node_id"], nodes_sp)
    id_edges_cell = PG["projection"]["edge_id"][cells].tolist()
    edges_cell = PG["Edges"][0][id_edges_cell, :]
    len_edges_cell = PG["projection"]["edge_len"][id_edges_cell]

    # proportion on the edge
    prop_edge = np.clip(PG["projection"]["edge_loc"][cells], a_min=0, a_max=1)

    dist_to_source = []
    for i in np.arange(edges_cell.shape[0]):
        if index_nodes[edges_cell[i, 0]] > index_nodes[edges_cell[i, 1]]:
            dist_to_source.append(dict_dist_to_source[edges_cell[i, 1]])
            prop_edge[i] = 1 - prop_edge[i]
        else:
            dist_to_source.append(dict_dist_to_source[edges_cell[i, 0]])
    dist_to_source = np.array(dist_to_source)
    dist_on_edge = len_edges_cell * prop_edge
    dist = dist_to_source + dist_on_edge

    PG["pseudotime"] = np.repeat(np.nan, len(X))
    PG["pseudotime"][cells] = dist
    PG["pseudotime_params"] = {
        "source": source,
        "target": target,
        "nodes_to_include": nodes_to_include,
    }


#### supervised knn


def _longform_knn_to_sparse(dis, idx):
    row_ind = np.tile(np.arange(len(idx))[:, None], idx.shape[1])
    col_ind = idx
    return scipy.sparse.csr_matrix((dis.flat, (row_ind.flat, col_ind.flat)))


def getWeights(
    X, bandwidth=1, griddelta=100, exponent=1, method="sklearn", **kwargs
):
    """Get point weights as the inverse density of data
    X: np.array, (n_sample x n_dims)
    bandwidth: sklearn KernelDensity bandwidth if method == 'sklearn'
    griddelta: FFTKDE grid step size if method =='fft'
    exponent: density values are raised to the power of exponent
    """
    if method == "sklearn":
        kde = KernelDensity(
            kernel="gaussian", bandwidth=bandwidth, **kwargs
        ).fit(X)
        scores = kde.score_samples(X)
        scores = np.exp(scores)[:, None]

    elif method == "fft":
        import KDEpy

        kde = KDEpy.FFTKDE(**kwargs).fit(X)
        x, y = kde.evaluate(griddelta)
        scores = scipy.interpolate.griddata(x, y, X)

    p = 1 / (scores ** exponent)
    p /= np.sum(p)
    return p


def ordinal_neighbors_stagewise_longform(
    X, stages_labels, stages=None, k=15, radius=None, m="cosine"
):
    """Supervised (ordinal) nearest-neighbor search.
    Stages is an ordered list of stages labels (low to high). If None, taken as np.unique(stages_labels)"""

    if stages is None:
        stages = np.unique(stages_labels)

    knn_distances = np.zeros((len(X), 3 * k))
    knn_idx = np.zeros((len(X), 3 * k), dtype=int)

    nn_stage = {}
    for s in stages:
        nn_stage[s] = NearestNeighbors(n_neighbors=k, metric=m).fit(
            X[stages_labels.values == s, :]
        )

    s = []
    t = []
    w = []
    for i in range(len(stages) - 1):
        dis, ind = nn_stage[stages[i]].kneighbors(
            X[stages_labels.values == stages[i + 1], :], k
        )
        knn_distances[stages_labels.values == stages[i + 1], :k] = dis
        knn_idx[stages_labels.values == stages[i + 1], :k] = np.where(
            stages_labels.values == stages[i]
        )[0][ind]

    for i in range(1, len(stages)):
        dis, ind = nn_stage[stages[i]].kneighbors(
            X[stages_labels.values == stages[i - 1], :], k
        )
        knn_distances[stages_labels.values == stages[i - 1], k : 2 * k] = dis
        knn_idx[stages_labels.values == stages[i - 1], k : 2 * k] = np.where(
            stages_labels.values == stages[i]
        )[0][ind]

    for i in range(len(stages)):
        if i == 0:
            dis, ind = nn_stage[stages[i]].kneighbors(
                X[stages_labels.values == stages[i], :], 2 * k + 1
            )
            dis, ind = dis[:, 1:], ind[:, 1:]
            knn_distances[
                np.argwhere(stages_labels.values == stages[i]),
                list(range(k)) + list(range(2 * k, 3 * k)),
            ] = dis
            knn_idx[
                np.argwhere(stages_labels.values == stages[i]),
                list(range(k)) + list(range(2 * k, 3 * k)),
            ] = np.where(stages_labels.values == stages[i])[0][ind]

        elif i == (len(stages) - 1):
            dis, ind = nn_stage[stages[i]].kneighbors(
                X[stages_labels.values == stages[i], :], 2 * k + 1
            )
            dis, ind = dis[:, 1:], ind[:, 1:]
            knn_distances[stages_labels.values == stages[i], k:] = dis
            knn_idx[stages_labels.values == stages[i], k:] = np.where(
                stages_labels.values == stages[i]
            )[0][ind]

        else:
            dis, ind = nn_stage[stages[i]].kneighbors(
                X[stages_labels.values == stages[i], :], k + 1
            )
            dis, ind = dis[:, 1:], ind[:, 1:]
            knn_distances[stages_labels.values == stages[i], 2 * k :] = dis
            knn_idx[stages_labels.values == stages[i], 2 * k :] = np.where(
                stages_labels.values == stages[i]
            )[0][ind]

    _sort = np.argsort(knn_distances, axis=1)
    knn_distances = knn_distances[
        np.arange(len(knn_distances))[:, None], _sort
    ]
    knn_idx = knn_idx[np.arange(len(knn_distances))[:, None], _sort]

    return knn_distances, knn_idx


def ordinal_neighbors_longform(
    X, stages_labels, stages=None, k=15, m="cosine"
):
    """Supervised (ordinal) nearest-neighbor search.
    Stages is an ordered list of stages labels (low to high). If None, taken as np.unique(stages_labels)"""
    if stages is None:
        stages = np.unique(stages_labels)

    knn_distances = np.zeros((len(X), k))
    knn_idx = np.zeros((len(X), k), dtype=int)
    for i in range(len(stages)):

        sel_points = (
            (stages_labels.values == stages[i])
            | (stages_labels.values == stages[max(0, i - 1)])
            | (stages_labels.values == stages[min(i + 1, len(stages) - 1)])
        )

        stage = stages_labels.values == stages[i]
        dis, ind = (
            NearestNeighbors(n_neighbors=k, metric=m)
            .fit(X[sel_points, :])
            .kneighbors(X[stage, :])
        )

        knn_distances[stage, :] = dis
        knn_idx[stage, :] = np.where(sel_points)[0][ind]
    return knn_distances, knn_idx


def supervised_knn(
    X,
    stages_labels,
    stages=None,
    n_neighbors=15,
    n_natural=0,
    m="cosine",
    method="force",
    return_sparse=False,
):
    """Supervised (ordinal) nearest-neighbor search.
    Stages is an ordered list of stages labels (low to high). If None, taken as np.unique(stages_labels)

    Parameters
    ----------
    method : str (default='force')
        if 'force', searches for each point at stage[i] n_neighbors nearest_neighbors, forcing:
            - n_neighbors/3 to be from stage[i-1]
            - n_neighbors/3 to be from stage[i]
            - n_neighbors/3 to be from stage[i+1]
            For stage[0] and stage[-1], 2*n_neighbors/3 are taken from stage[i]

        if 'guide', searches for each point at stage[i] n_neighbors nearest_neighbors
            from points in {stage[i-1], stage[i], stage[i+1]}, without constraints on proportions
    """

    if n_neighbors % 3 != 0:
        raise ValueError("Please provide n_neighbors divisible by 3")
    if stages is None:
        stages = np.unique(stages_labels)

    dis, idx = (
        NearestNeighbors(n_neighbors=n_neighbors, metric=m, n_jobs=8)
        .fit(X)
        .kneighbors()
    )

    if method == "guide":
        knn_distances, knn_idx = ordinal_neighbors_longform(
            X, stages_labels, stages=stages, k=n_neighbors, m=m
        )
    if method == "force":
        knn_distances, knn_idx = ordinal_neighbors_stagewise_longform(
            X, stages_labels, stages=stages, k=n_neighbors // 3, m=m
        )

    # ---mix natural nn with ordinal nn
    merged_idx = np.zeros((len(X), n_neighbors), dtype=np.int32)
    merged_dists = np.zeros((len(X), n_neighbors))
    for i in range(len(X)):
        merged_idx[i][:n_natural] = idx[i][:n_natural]
        merged_idx[i][n_natural:] = np.setdiff1d(
            knn_idx[i], idx[i][:n_natural], assume_unique=True
        )[: n_neighbors - n_natural]

        merged_dists[i][:n_natural] = dis[i][:n_natural]
        merged_dists[i][n_natural:] = knn_distances[i][
            np.isin(
                knn_idx[i],
                np.setdiff1d(
                    knn_idx[i], idx[i][:n_natural], assume_unique=True
                )[: n_neighbors - n_natural],
            )
        ]

    if return_sparse:
        return _longform_knn_to_sparse(merged_dists, merged_idx)
    else:
        return merged_dists, merged_idx


def geodesic_pseudotime(X, k, root, g=None):
    """pseudotime as graph distance from root point"""
    if g is None:
        nn = NearestNeighbors(n_neighbors=k, n_jobs=8).fit(X)
        g = nx.convert_matrix.from_scipy_sparse_matrix(
            nn.kneighbors_graph(mode="distance")
        )
    else:
        g = nx.convert_matrix.from_scipy_sparse_matrix(g)
    if len(list(nx.connected_components(g))) > 1:
        raise ValueError(
            f"detected more than 1 components with k={k} neighbors. Please"
            " increase k"
        )
    lengths = nx.single_source_dijkstra_path_length(g, root)
    pseudotime = np.array(pd.Series(lengths).sort_index())
    return pseudotime


def proj2embedding(X_elpi,X_embed,NodePositions):
    part,dis = PartitionData(X_elpi,NodePositions,50000,np.sum(X_elpi**2,axis=1,keepdims=1))
    # Hard assigment
    R = scipy.sparse.csr_matrix(
        (np.ones(len(X_elpi)), (range(len(X_elpi)), part.flat)), (len(X_elpi),len(NodePositions))
    ).A
    proj = (np.dot(X_embed.T, R) / R.sum(axis=0)).T
    return proj


def _propagate_labels(Xs,Xt,ys,flavor,n_neighbors,reg_e=0.01,reg_m=0.08):
    if 'ot' in flavor:
        try: import ot
        except: raise ImportError('Install Python OT to use this function'+
            '(pip install POT or conda install -c conda-forge pot)')
            
    if flavor=='ot':
        coupling = ot.sinkhorn([], [], ot.da.cost_normalization(ot.dist(Xs,Xt),'max'),reg=reg_e,numItermax=1000000)
    elif flavor=='ot_unbalanced':
        coupling = ot.sinkhorn_unbalanced([], [], ot.da.cost_normalization(ot.dist(Xs,Xt),'max'),reg_e,reg_m,numItermax=1000000)
    elif flavor=='ot_equal':
        ws=np.ones(len(ys))
        ws[ys==0]/=sum(ys==0)
        ws[ys==1]/=sum(ys==1)
        ws/=sum(ws)
        coupling = ot.sinkhorn(ws, [], ot.da.cost_normalization(ot.dist(Xs,Xt),'max'), reg=reg_e,numItermax=1000000)
    elif flavor=='knn':
        coupling = NearestNeighbors(n_neighbors=n_neighbors).fit(Xs).kneighbors_graph(Xt).toarray().T
    else: 
        raise ValueError('method must be ot, ot_unbalanced, ot_equal, knn')
        
    labels=[]
    for j,coup in enumerate(coupling.T):
        nz=coup>0
        labels.append(np.bincount(ys[nz],weights=coup[nz].flat).argmax())
        #labels.append(ys[coup.argmax()])

    return labels

def early_groups(X, PG, branch_nodes, source, target, nodes_to_include = None,
                 flavor='ot_unbalanced', n_windows = 20, n_neighbors=20, ot_reg_e=0.01,ot_reg_m=0.001):

    getPseudotime(X, PG, source, target=target, nodes_to_include=nodes_to_include, project=True)
    ix=~np.isnan(PG['pseudotime'])
    
    y = -np.ones(len(X),dtype=int)

    bnodes_ix = np.zeros(len(X),dtype=bool)
    for bn in branch_nodes:
        nix = PG['projection']['node_id']==bn
        bnodes_ix[nix] = True
        y[nix] = bn
        
    Xpath, ypath = X[ix].copy(), y[ix].copy() # points from path 
    Xbranch, ybranch = X[bnodes_ix].copy(), y[bnodes_ix].copy() # points from branch
    
    #propagate labels
    bpseudotime = PG['pseudotime'][ix]
    count, bins = np.histogram(bpseudotime, bins=n_windows)
    clusters = np.digitize(bpseudotime, bins[1:], right=True)

    #init 
    cnext = clusters==clusters.max()
    ypath[cnext] = _propagate_labels(Xs=Xbranch,Xt=Xpath[cnext],ys=ybranch,flavor=flavor,n_neighbors=n_neighbors)

    for c in np.arange(1,n_windows)[::-1]:
        cprev = clusters==c
        cnext = clusters==(c-1)
        ypath[cnext] = _propagate_labels(Xs=Xpath[cprev],Xt=Xpath[cnext],ys=ypath[cprev],
                                    flavor=flavor,n_neighbors=n_neighbors,reg_e=ot_reg_e,reg_m=ot_reg_m)
        
    s = '-'.join(str(x) for x in branch_nodes)
    
    y[ix] = ypath
    PG[f'early_groups_{source}->{s}']=y.astype(str)
    y[ix] = clusters
    PG[f'early_groups_{source}->{s}_clusters']=y.astype(str)