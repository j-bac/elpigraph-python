from mimetypes import MimeTypes
import networkx as nx
import elpigraph
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from sklearn.decomposition import PCA
from copy import deepcopy
from scipy.spatial.qhull import ConvexHull
from shapely.geometry import Point, Polygon, MultiLineString, LineString
from shapely.geometry.multipolygon import MultiPolygon
from sklearn.neighbors import NearestNeighbors


def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


@nb.njit
def _get_intersect_inner(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack((a1, a2, b1, b2))  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return np.array([np.inf, np.inf])
    return np.array([x / z, y / z])


@nb.njit
def _isBetween(a, b, c):
    """ Check if c is in between a and b """
    dotproduct = (c[0] - a[0]) * (b[0] - a[0]) + (c[1] - a[1]) * (b[1] - a[1])
    if dotproduct < 0:
        return False

    squaredlengthba = (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])
    if dotproduct > squaredlengthba:
        return False
    return True


@nb.njit
def _get_intersect(Xin, nodep, edges, cent):
    inters = np.zeros_like(Xin)
    for i, x in enumerate(Xin):
        for e in edges:
            p_inter = _get_intersect_inner(nodep[e[0]], nodep[e[1]], x, cent)
            if _isBetween(nodep[e[0]], nodep[e[1]], p_inter):
                if np.sum((x - p_inter) ** 2) < np.sum((cent - p_inter) ** 2):
                    inters[i] = p_inter
    return inters


@nb.njit
def get_weights_lineproj(Xin, nodep, edges, cent, threshold=0.2):

    Xin_lineproj = _get_intersect(Xin, nodep, edges, cent)
    distcent_Xin_lineproj = np.sqrt(np.sum((Xin_lineproj - cent) ** 2, axis=1))
    distcent_Xin = np.sqrt(np.sum((Xin - cent) ** 2, axis=1))

    w = 1 - distcent_Xin / distcent_Xin_lineproj
    idx_close = w > threshold
    w[idx_close] = 1.0
    return w, idx_close


def shrink_or_swell_shapely_polygon(coords, factor=0.10, swell=False):
    """returns the shapely polygon which is smaller or bigger by passed factor.
    If swell = True , then it returns bigger polygon, else smaller"""

    my_polygon = Polygon(coords)
    xs = list(my_polygon.exterior.coords.xy[0])
    ys = list(my_polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = Point(min(xs), min(ys))
    max_corner = Point(max(xs), max(ys))
    center = Point(x_center, y_center)
    shrink_distance = center.distance(min_corner) * factor

    if swell:
        my_polygon_resized = my_polygon.buffer(shrink_distance)  # expand
    else:
        my_polygon_resized = my_polygon.buffer(-shrink_distance)  # shrink

    return my_polygon_resized


def remove_intersections(nodep, edges):
    """ Update edges to account for possible 2d intersections in graph after adding loops """
    new_nodep = nodep.copy()
    lnodep = new_nodep.tolist()
    new_edges = edges.tolist()
    multiline = MultiLineString([LineString(new_nodep[e]) for e in new_edges])

    while not (multiline.is_simple):  # while intersections in graph

        # find an intersection, update edges, break, update graph
        for i, j in itertools.combinations(range(len(multiline)), 2):
            line1, line2 = multiline[i], multiline[j]
            if line1.intersects(line2):
                if list(np.array(line1.intersection(line2))) not in lnodep:
                    new_nodep = np.append(
                        new_nodep, np.array(line1.intersection(line2))[None], axis=0,
                    )
                    intersects_idx = [list(new_edges[i]), list(new_edges[j])]
                    new_edges.pop(new_edges.index(intersects_idx[0]))
                    new_edges.pop(new_edges.index(intersects_idx[1]))

                    for n in np.array(intersects_idx).flatten():
                        new_edges.append([n, len(new_nodep) - 1])
                    break

        multiline = MultiLineString([LineString(new_nodep[e]) for e in new_edges])
        lnodep = new_nodep.tolist()
    return new_nodep, np.array(new_edges)


@nb.njit
def mahalanobis(M, cent):
    cov = np.cov(M, rowvar=0)
    try:
        cov_inverse = np.linalg.inv(cov)
    except:
        cov_inverse = np.linalg.pinv(cov)

    M_c = M - cent
    dist = np.sqrt(np.sum((M_c) * cov_inverse.dot(M_c.T).T, axis=1))
    return dist


@nb.njit
def polygon_area(x, y):
    # coordinate shift
    x_ = x - x.mean()
    y_ = y - y.mean()
    # everything else is the same as maxb's code
    correction = x_[-1] * y_[0] - y_[-1] * x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5 * np.abs(main_area + correction)


def pp_compactness(cycle_nodep):
    """Polsby-Popper compactness"""
    area = polygon_area(cycle_nodep[:, 0], cycle_nodep[:, 1])
    length = np.sum(np.sqrt((np.diff(cycle_nodep, axis=0) ** 2).sum(axis=1)))
    return (4 * np.pi * area) / (length ** 2)


def find_all_cycles(G, source=None, cycle_length_limit=None):
    """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
    types which work with min() and > ."""
    if source is None:
        # produce edges for all components
        nodes = [list(i)[0] for i in nx.connected_components(G)]
    else:
        # produce edges for components with source
        nodes = [source]
    # extra variables for cycle detection:
    cycle_stack = []
    output_cycles = set()

    def get_hashable_cycle(cycle):
        """cycle as a tuple in a deterministic order."""
        m = min(cycle)
        mi = cycle.index(m)
        mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
        if cycle[mi - 1] > cycle[mi_plus_1]:
            result = cycle[mi:] + cycle[:mi]
        else:
            result = list(reversed(cycle[:mi_plus_1])) + list(
                reversed(cycle[mi_plus_1:])
            )
        return tuple(result)

    for start in nodes:
        if start in cycle_stack:
            continue
        cycle_stack.append(start)

        stack = [(start, iter(G[start]))]
        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)

                if child not in cycle_stack:
                    cycle_stack.append(child)
                    stack.append((child, iter(G[child])))
                else:
                    i = cycle_stack.index(child)
                    if i < len(cycle_stack) - 2:
                        output_cycles.add(get_hashable_cycle(cycle_stack[i:]))

            except StopIteration:
                stack.pop()
                cycle_stack.pop()

    return [list(i) for i in output_cycles]


# def in_hull(points, queries):
#    hull = _Qhull(
#        b"i",
#        points,
#        options=b"",
#        furthest_site=False,
#        incremental=False,
#        interior_point=None,
#    )
#    equations = hull.get_simplex_facet_array()[2].T
#    return np.all(queries @ equations[:-1] < -equations[-1], axis=1)


def in_hull(points, queries):
    equations = ConvexHull(points).equations.T
    return np.all(queries @ equations[:-1] < -equations[-1], axis=1)


def findPaths(
    X,
    PG,
    min_path_len=None,
    nnodes=None,
    max_inner_fraction=0.1,
    min_node_n_points=2,
    max_n_points=None,
    # max_empty_curve_fraction=.2,
    min_compactness=0.5,
    radius=None,
    allow_same_branch=True,
    fit_loops=True,
    Lambda=None,
    Mu=None,
    cycle_Lambda=None,
    cycle_Mu=None,
    weights=None,
    plot=False,
    verbose=False,
):
    """
    This function tries to add extra paths to the graph
    by computing a series of principal curves connecting two nodes
    and retaining plausible ones using heuristic parameters

    min_path_len: int, default=None
        Minimum distance along the graph (in number of nodes) that separates the two nodes to connect with a principal curve
    n_nodes: int, default=None
        Number of nodes in the candidate principal curves
    max_inner_fraction: float in [0,1], default=0.1
        Maximum fraction of points inside vs outside the loop Controls how empty the loop formed with the added path should be.
    min_node_n_points: int, default=1
        Minimum number of points associated to nodes of the principal curve (prevents creating paths through empty space)
    max_n_points: int, default=5% of the number of points
        Maximum number of points inside the loop
    min_compactness: float in [0,1], default=0.5
        Minimum 'roundness' of the loop (1=more round) (if very narrow loops are not desired)
    radius: float, default=None
        Max distance in space that separates the two nodes to connect with a principal curve
    allow_same_branch: bool, default=True
        Whether to allow new paths to connect two nodes from the same branch
    fit_loops: bool, default=True
        Whether to refit the graph to data after adding the new paths
    plot: bool, default=False
        Whether to plot selected candidate paths
    verbose: bool, default=False
    copy: bool, default=False
    use_weights: bool, default=False
        Whether to use point weights
    use_partition: bool or list, default=False
    """

    _PG = deepcopy(PG)
    if "projection" in _PG.keys():
        del _PG["projection"]
    init_nodes_pos, init_edges = _PG["NodePositions"], _PG["Edges"][0]

    # --- Init parameters, variables
    epg = nx.Graph(init_edges.tolist())

    SquaredX = np.sum(X ** 2, axis=1, keepdims=1)
    part, part_dist = elpigraph.src.core.PartitionData(
        X, init_nodes_pos, 10 ** 6, SquaredX=SquaredX
    )
    leaves = [k for k, v in epg.degree if v == 1]
    edge_lengths = np.sqrt(
        np.sum(
            (init_nodes_pos[init_edges[:, 0], :] - init_nodes_pos[init_edges[:, 1], :])
            ** 2,
            axis=1,
        )
    )

    if Mu is None:
        Mu = _PG["Mu"]
    if Lambda is None:
        Lambda = _PG["Lambda"]
    if cycle_Mu is None:
        cycle_Mu = Mu / 10
    if cycle_Lambda is None:
        cycle_Lambda = Lambda / 10
    if radius is None:
        radius = np.mean(edge_lengths) * len(init_nodes_pos) / 10
    if min_path_len is None:
        min_path_len = len(init_nodes_pos) // 5
    if max_n_points is None:
        max_n_points = int(len(X) * 0.05)
    if min_node_n_points is None:
        min_node_n_points = max(1, np.bincount(part.flat).min())
    if weights is None:
        weights = np.ones(len(X))[:, None]
    if nnodes is None:
        nnodes = min(16, max(6, int(radius / np.mean(edge_lengths))))
    elif nnodes < 6:
        raise ValueError("nnodes should be at least 6")

    if verbose:
        print(
            f"Using default parameters: max_n_points={max_n_points},"
            f" radius={radius:.2f}, min_node_n_points={min_node_n_points},"
            f" min_path_len={min_path_len}, nnodes={nnodes}"
        )

    # --- Get candidate nodes to connect
    dist, ind = (
        NearestNeighbors(radius=radius)
        .fit(init_nodes_pos)
        .radius_neighbors(init_nodes_pos[leaves])
    )
    net = elpigraph.src.graphs.ConstructGraph({"Edges": [init_edges]})

    if all(np.array(net.degree()) <= 2):
        branches = net.get_shortest_paths(leaves[0], leaves[-1])
    else:
        (
            dict_tree,
            dict_branches,
            dict_branches_single_end,
        ) = elpigraph.src.supervised.get_tree(init_edges, leaves[0])
        branches = list(dict_branches.values())

    candidate_nodes = []
    for i in range(len(leaves)):
        root_branch = [b for b in branches if leaves[i] in b][0]

        if allow_same_branch:
            _cand_nodes = [node for b in branches for node in b if node in ind[i]]
        else:
            _cand_nodes = [
                node
                for b in branches
                for node in b
                if not (node in root_branch) and node in ind[i]
            ]
        paths = net.get_shortest_paths(leaves[i], _cand_nodes)
        candidate_nodes.append([p[-1] for p in paths if len(p) > min_path_len])

    # --- Test each of the loops connecting a leaf to its candidate nodes,
    # --- for each leaf select the one with minimum energy and that respect constraints
    if verbose:
        print("testing", sum([len(_) for _ in candidate_nodes]), "candidates")

    new_edges = []
    new_nodep = []
    new_leaves = []
    new_part = []
    new_energy = []
    new_inner_fraction = []
    for i, l in enumerate(leaves):
        inner_fractions = []
        energies = []
        merged_edges = []
        merged_nodep = []
        merged_part = []
        loop_edges = []
        loop_nodep = []
        loop_leaves = []
        for c in candidate_nodes[i]:

            clus = (part == c) | (part == l)
            X_fit = np.vstack((init_nodes_pos[c], init_nodes_pos[l], X[clus.flat]))
            try:
                pg = elpigraph.computeElasticPrincipalCurve(
                    X_fit, nnodes, Lambda=Lambda, Mu=Mu, FixNodesAtPoints=[[0], [1]],
                )[0]
            except Exception as e:
                energies.append(np.inf)
                merged_edges.append(np.inf)
                merged_nodep.append(np.inf)
                loop_edges.append(np.inf)
                loop_nodep.append(np.inf)
                loop_leaves.append(np.inf)
                # candidate curve has infinite energy, ignore error
                if e.args == (
                    "local variable 'NewNodePositions' referenced before" " assignment",
                ):
                    continue
                else:
                    raise e

            # ---get nodep, edges, create new graph with added loop
            nodep, edges = pg["NodePositions"], pg["Edges"][0]
            # _part, _part_dist = elpigraph.src.core.PartitionData(
            #    X_fit, nodep, 10 ** 6, np.sum(X_fit ** 2, axis=1, keepdims=1)
            # )
            _edges = edges.copy()
            _edges[(edges != 0) & (edges != 1)] += init_edges.max() - 1
            _edges[edges == 0] = c
            _edges[edges == 1] = l
            _merged_edges = np.concatenate((init_edges, _edges))
            _merged_nodep = np.concatenate((init_nodes_pos, nodep[2:]))

            cycle_nodes = find_all_cycles(nx.Graph(_merged_edges.tolist()))[0]

            ElasticMatrix = elpigraph.src.core.MakeUniformElasticMatrix_with_cycle(
                _merged_edges,
                Lambda=Lambda,
                Mu=Mu,
                cycle_Lambda=cycle_Lambda,
                cycle_Mu=cycle_Mu,
                cycle_nodes=cycle_nodes,
            )

            (
                _merged_nodep,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = elpigraph.src.core.PrimitiveElasticGraphEmbedment(
                X,
                _merged_nodep,
                ElasticMatrix,
                PointWeights=weights,
                FixNodesAtPoints=[],
            )

            ### candidate validity tests ###
            valid = True
            # --- curve validity test
            # if (max_empty_curve_fraction is not None) and valid: # if X_fit projected to curve has long gaps
            #    infer_pseudotime(_adata,source=0)
            #    sorted_X_proj=_adata.obsm['X_epg_proj'][_adata.obs['epg_pseudotime'].argsort()]
            #    dist = np.sqrt((np.diff(sorted_X_proj,axis=0)**2).sum(axis=1))
            #    curve_len = np.sum(_adata.uns['epg']['edge_len'])
            #    if np.max(dist) > (curve_len * max_empty_curve_fraction):
            #        valid = False

            # --- cycle validity test
            if valid:
                G = nx.Graph(_merged_edges.tolist())
                cycle_nodes = find_all_cycles(G)[0]
                cycle_nodep = np.array([_merged_nodep[e] for e in cycle_nodes])
                cent_part, cent_dists = elpigraph.src.core.PartitionData(
                    X, _merged_nodep, 10 ** 6, SquaredX=SquaredX
                )
                cycle_points = np.isin(cent_part.flat, cycle_nodes)

                if X.shape[1] > 2:
                    pca = PCA(n_components=2, svd_solver="arpack").fit(X[cycle_points])
                    X_cycle_2d = pca.transform(X[cycle_points])
                    cycle_2d = pca.transform(cycle_nodep)
                else:
                    cycle_2d = cycle_nodep
                    X_cycle_2d = X[cycle_points]

                inside_idx = in_hull(cycle_2d, X_cycle_2d)

                if sum(inside_idx) == 0:
                    inner_fraction = 0.0
                else:
                    cycle_centroid = np.mean(cycle_2d, axis=0, keepdims=1)
                    X_inside = X_cycle_2d[inside_idx]

                    if len(X_inside) == 1:
                        w = np.ones(len(X_inside))
                    else:
                        w = mahalanobis(X_inside, cycle_centroid)

                    # points belonging to cycle shrunk by 10% or within 2 std of centroid (mahalanobis < 2)
                    shrunk_cycle_2d = shrink_or_swell_shapely_polygon(
                        cycle_2d, factor=0.1
                    )

                    # prevent shapely bugs when multi-polygon is returned. Fall back to mahalanobis
                    if type(shrunk_cycle_2d) == MultiPolygon:
                        in_shrunk_cycle = np.ones(len(X_inside), dtype=bool)
                    else:
                        shrunk_cycle_2d = np.array(shrunk_cycle_2d.exterior.coords)

                        # prevent bug when self-intersection
                        if len(shrunk_cycle_2d) == 0:
                            in_shrunk_cycle = np.ones(len(X_inside), dtype=bool)
                        else:
                            in_shrunk_cycle = in_hull(shrunk_cycle_2d, X_inside)
                    idx_close = in_shrunk_cycle | (w < 1)
                    w = 1 - w / w.max()
                    w[idx_close] = 1

                    # cycle_nodes_array = np.append(np.array(list(zip(range(len(cycle_2d)-1),
                    #                                  range(1,len(cycle_2d))))),[[len(cycle_2d)-1,0]],axis=0)
                    # w, idx_close = get_weights_lineproj(X_inside,cycle_2d,cycle_nodes_array,cycle_centroid[0],threshold=.2)

                    inner_fraction = np.sum(w) / np.sum(cycle_points)

                if init_nodes_pos.shape[1] == 2:
                    intersect = not (
                        MultiLineString(
                            [LineString(_merged_nodep[e]) for e in _merged_edges]
                        ).is_simple
                    )
                    if intersect:
                        valid = False

                # idx for min_node_n_points test: points that are away from the center
                ix_outside = np.ones(len(cycle_points), dtype=bool)
                ix_outside[
                    np.arange(len(X))[cycle_points][inside_idx][idx_close]
                ] = False
                if (
                    any(
                        np.bincount(
                            cent_part[ix_outside].flat, minlength=len(_merged_nodep),
                        )[len(init_nodes_pos) :]
                        < min_node_n_points
                    )  # if empty cycle node
                    or (
                        inner_fraction > max_inner_fraction
                    )  # if high fraction of points inside
                    or (not np.isfinite(inner_fraction))  # prevent no points error
                    or (np.sum(idx_close) > max_n_points)  # if too many points inside
                    or pp_compactness(cycle_2d) < min_compactness
                ):  # if loop is very narrow
                    valid = False

            # ---> if cycle is invalid, continue
            if not valid:
                inner_fractions.append(np.inf)
                energies.append(np.inf)
                merged_edges.append(np.inf)
                merged_nodep.append(np.inf)
                merged_part.append(np.inf)
                loop_edges.append(np.inf)
                loop_nodep.append(np.inf)
                loop_leaves.append(np.inf)
                continue

            # ---> valid cycle, compute graph energy
            else:
                (_merged_part, _merged_part_dist,) = elpigraph.src.core.PartitionData(
                    X, _merged_nodep, 10 ** 6, SquaredX=SquaredX
                )
                proj = elpigraph.src.reporting.project_point_onto_graph(
                    X, _merged_nodep, _merged_edges, _merged_part
                )
                MSE = proj["MSEP"]
                # dist2proj = np.sum(np.square(X - X_proj), axis=1)
                # ElasticMatrix = elpigraph.src.core.Encode2ElasticMatrix(_merged_edges, Lambdas=Lambda, Mus=Mu)
                # ElasticEnergy, MSE, EP, RP = elpigraph.src.core.ComputePenalizedPrimitiveGraphElasticEnergy(_merged_nodep,
                #                                                                                            ElasticMatrix,
                #                                                                                            dist2proj,alpha=0.01,beta=0)
                inner_fractions.append(inner_fraction)
                energies.append(MSE)
                merged_edges.append(_merged_edges)
                merged_nodep.append(_merged_nodep)
                merged_part.append(np.where(np.isin(_merged_part.flat, cycle_nodes))[0])
                loop_edges.append(edges)
                loop_nodep.append(nodep[2:])
                loop_leaves.append([c, l])

        # --- among all valid cycles found, retain the best
        if energies != [] and np.isfinite(energies).any():
            best = np.argmin(energies)
            if [loop_leaves[best][-1], loop_leaves[best][0]] not in new_leaves:
                # and not any(np.isin(loop_leaves[best],np.unique(np.array(new_leaves))))):

                new_edges.append(loop_edges[best])
                new_nodep.append(loop_nodep[best])
                new_leaves.append(loop_leaves[best])
                new_part.append(merged_part[best])
                new_energy.append(energies[best])
                new_inner_fraction.append(inner_fractions[best])
                _merged_edges = merged_edges[best]
                _merged_nodep = merged_nodep[best]

                if plot:
                    c = candidate_nodes[i][best]
                    clus = (part == c) | (part == l)
                    X_fit = np.vstack(
                        (init_nodes_pos[c], init_nodes_pos[l], X[clus.flat])
                    )
                    proj = elpigraph.src.reporting.project_point_onto_graph(
                        X, _merged_nodep, _merged_edges, _merged_part
                    )
                    MSE = proj["MSEP"]

                    # ----- cycle test
                    G = nx.Graph(_merged_edges.tolist())
                    cycle_nodes = find_all_cycles(G)[0]
                    cycle_nodep = np.array([_merged_nodep[e] for e in cycle_nodes])
                    cent_part, cent_dists = elpigraph.src.core.PartitionData(
                        X, _merged_nodep, 10 ** 6, SquaredX=SquaredX
                    )
                    cycle_points = np.isin(cent_part.flat, cycle_nodes)

                    if X.shape[1] > 2:
                        pca = PCA(n_components=2, svd_solver="arpack").fit(cycle_nodep)
                        cycle_2d = pca.transform(cycle_nodep)
                        X_cycle_2d = pca.transform(X[cycle_points])
                    else:
                        cycle_2d = cycle_nodep
                        X_cycle_2d = X[cycle_points]
                    inside_idx = in_hull(cycle_2d, X_cycle_2d)

                    cycle_centroid = np.mean(cycle_2d, axis=0, keepdims=1)
                    X_inside = X_cycle_2d[inside_idx]

                    if len(X_inside) == 1:
                        w = np.ones(len(X_inside))
                    else:
                        w = mahalanobis(X_inside, cycle_centroid)

                    # points belonging to cycle shrunk by 10% or within 2 std of centroid (mahalanobis < 2)
                    shrunk_cycle_2d = shrink_or_swell_shapely_polygon(
                        cycle_2d, factor=0.1
                    )

                    # prevent shapely bugs when multi-polygon is returned. Fall back to mahalanobis
                    if type(shrunk_cycle_2d) == MultiPolygon:
                        in_shrunk_cycle = np.ones(len(X_inside), dtype=bool)
                    else:
                        shrunk_cycle_2d = np.array(shrunk_cycle_2d.exterior.coords)

                        # prevent bug when self-intersection
                        if len(shrunk_cycle_2d) == 0:
                            in_shrunk_cycle = np.ones(len(X_inside), dtype=bool)
                        else:
                            in_shrunk_cycle = in_hull(shrunk_cycle_2d, X_inside)
                    idx_close = in_shrunk_cycle | (w < 1)
                    w = 1 - w / w.max()
                    w[idx_close] = 1
                    inner_fraction = np.sum(w) / np.sum(cycle_points)

                    compactness = pp_compactness(cycle_2d)

                    plt.title(
                        f"{c}, {l}, MSE={MSE:.4f}, \n"
                        f" inner%={inner_fraction:.2f},"
                        f" compactness={compactness:.2f}"
                    )
                    plt.scatter(*X[:, :2].T, alpha=0.1, s=5)
                    plt.scatter(*X_fit[:, :2].T, s=5)
                    plt.scatter(*_merged_nodep[:, :2].T, c="k")
                    for e in _merged_edges:
                        plt.plot(
                            [_merged_nodep[e[0], 0], _merged_nodep[e[1], 0]],
                            [_merged_nodep[e[0], 1], _merged_nodep[e[1], 1]],
                            c="k",
                        )

                    _ = plt.scatter(*X[cycle_points][inside_idx, :2].T, c=w.flat, s=5)
                    plt.colorbar(_)

                    plt.show()

    # ignore equivalent loops (with more than 2/3 shared points and nodes)
    valid = np.ones(len(new_part))
    for i in range(len(new_part) - 1):
        for j in range(i + 1, len(new_part)):
            if (
                len(np.intersect1d(new_part[i], new_part[j]))
                / max(len(new_part[i]), len(new_part[j]))
            ) > (2 / 3):
                if np.argmin([new_inner_fraction[i], new_inner_fraction[j]]) == 0:
                    valid[i] = 0
                else:
                    valid[j] = 0

    new_edges = [e for i, e in enumerate(new_edges) if valid[i]]
    new_nodep = [e for i, e in enumerate(new_nodep) if valid[i]]
    new_leaves = [e for i, e in enumerate(new_leaves) if valid[i]]
    new_part = [e for i, e in enumerate(new_part) if valid[i]]
    new_energy = [e for i, e in enumerate(new_energy) if valid[i]]
    new_inner_fraction = [e for i, e in enumerate(new_inner_fraction) if valid[i]]

    ### form graph with all valid loops found ###
    if (new_edges == []) or (sum(valid) == 0):
        print("Found no valid path to add")
        return None

    for i, loop_edges in enumerate(new_edges):
        if i == 0:
            loop_edges[(loop_edges != 0) & (loop_edges != 1)] += init_edges.max() - 1
            loop_edges[loop_edges == 0] = new_leaves[i][0]
            loop_edges[loop_edges == 1] = new_leaves[i][1]
            merged_edges = np.concatenate((init_edges, loop_edges))
        else:
            loop_edges[(loop_edges != 0) & (loop_edges != 1)] += merged_edges.max() - 1
            loop_edges[loop_edges == 0] = new_leaves[i][0]
            loop_edges[loop_edges == 1] = new_leaves[i][1]
            merged_edges = np.concatenate((merged_edges, loop_edges))
    merged_nodep = np.concatenate((init_nodes_pos, *new_nodep))

    ### optionally refit the entire graph ###
    if fit_loops:
        cycle_nodes = np.concatenate(find_all_cycles(nx.Graph(merged_edges.tolist())))

        ElasticMatrix = elpigraph.src.core.MakeUniformElasticMatrix_with_cycle(
            merged_edges,
            Lambda=Lambda,
            Mu=Mu,
            cycle_Lambda=cycle_Lambda,
            cycle_Mu=cycle_Mu,
            cycle_nodes=cycle_nodes,
        )

        (
            merged_nodep,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = elpigraph.src.core.PrimitiveElasticGraphEmbedment(
            X, merged_nodep, ElasticMatrix, PointWeights=weights, FixNodesAtPoints=[],
        )
        # check intersection
        if merged_nodep.shape[1] == 2:
            intersect = not (
                MultiLineString(
                    [LineString(merged_nodep[e]) for e in merged_edges]
                ).is_simple
            )
            if intersect:
                merged_nodep, merged_edges = remove_intersections(
                    merged_nodep, merged_edges
                )

    _PG["Edges"] = [merged_edges]
    _PG["NodePositions"] = merged_nodep
    _PG["Lambda"] = Lambda
    _PG["Mu"] = Mu
    _PG["cycle_Lambda"] = cycle_Lambda
    _PG["cycle_Mu"] = cycle_Mu
    _PG["addLoopsdict"] = dict(
        new_edges=new_edges,
        new_nodep=new_nodep,
        new_leaves=new_leaves,
        new_part=new_part,
        new_energy=new_energy,
        new_inner_fraction=new_inner_fraction,
    )

    if verbose:
        l = [
            _PG["addLoopsdict"]["new_inner_fraction"],
            _PG["addLoopsdict"]["new_energy"],
            [len(n) for n in _PG["addLoopsdict"]["new_part"]],
        ]
        df = pd.concat(
            [pd.DataFrame(_PG["addLoopsdict"]["new_leaves"])]
            + [pd.Series(i) for i in l],
            axis=1,
        )
        df.columns = [
            "source node",
            "target node",
            "inner fraction",
            "MSE",
            "n° of points in path",
        ]
        df.index = ["" for i in df.index]
        print("Suggested paths:")
        print(df.round(4))
    return _PG


def addPath(
    X,
    PG,
    source,
    target,
    n_nodes=None,
    weights=None,
    refit_graph=False,
    Mu=None,
    Lambda=None,
    cycle_Mu=None,
    cycle_Lambda=None,
):
    """
    Add path (principal curve) between two nodes in the principal graph

    """

    _PG = deepcopy(PG)
    if "projection" in _PG.keys():
        del _PG["projection"]
    init_nodes_pos, init_edges = _PG["NodePositions"], _PG["Edges"][0]

    # --- Init parameters, variables
    if Mu is None:
        Mu = _PG["Mu"]
    if Lambda is None:
        Lambda = _PG["Lambda"]
    if cycle_Mu is None:
        cycle_Mu = Mu
    if cycle_Lambda is None:
        cycle_Lambda = Lambda
    if n_nodes is None:
        n_nodes = min(16, max(6, len(init_nodes_pos) / 20))

    SquaredX = np.sum(X ** 2, axis=1, keepdims=1)
    part, part_dist = elpigraph.src.core.PartitionData(
        X, init_nodes_pos, 10 ** 6, SquaredX=SquaredX
    )
    clus = (part == source) | (part == target)
    X_fit = np.vstack((init_nodes_pos[source], init_nodes_pos[target], X[clus.flat]))

    # --- fit path
    PG_path = elpigraph.computeElasticPrincipalCurve(
        X_fit, NumNodes=n_nodes, Lambda=Lambda, Mu=Mu, FixNodesAtPoints=[[0], [1]],
    )[0]

    # --- get nodep, edges, create new graph with added loop
    nodep, edges = PG_path["NodePositions"], PG_path["Edges"][0]

    _edges = edges.copy()
    _edges[(edges != 0) & (edges != 1)] += init_edges.max() - 1
    _edges[edges == 0] = source
    _edges[edges == 1] = target
    _merged_edges = np.concatenate((init_edges, _edges))
    _merged_nodep = np.concatenate((init_nodes_pos, nodep[2:]))

    if refit_graph:
        cycle_nodes = elpigraph._graph_editing.find_all_cycles(
            nx.Graph(_merged_edges.tolist())
        )

        if len(cycle_nodes) > 0:
            cycle_nodes = flatten(cycle_nodes)

        ElasticMatrix = elpigraph.src.core.MakeUniformElasticMatrix_with_cycle(
            _merged_edges,
            Lambda=Lambda,
            Mu=Mu,
            cycle_Lambda=cycle_Lambda,
            cycle_Mu=cycle_Mu,
            cycle_nodes=cycle_nodes,
        )

        (
            _merged_nodep,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = elpigraph.src.core.PrimitiveElasticGraphEmbedment(
            X, _merged_nodep, ElasticMatrix, PointWeights=weights, FixNodesAtPoints=[],
        )

    # check intersection
    if _merged_nodep.shape[1] == 2:
        intersect = not (
            MultiLineString(
                [LineString(_merged_nodep[e]) for e in _merged_edges]
            ).is_simple
        )
        if intersect:
            raise ValueError("The created path would intersect existing graph")

    _PG["NodePositions"] = _merged_nodep
    _PG["Edges"] = [_merged_edges]
    _PG["Lambda"] = Lambda
    _PG["Mu"] = Mu
    _PG["cycle_Lambda"] = cycle_Lambda
    _PG["cycle_Mu"] = cycle_Mu
    return _PG


def delPath(
    X,
    PG,
    source,
    target,
    nodes_to_include=None,
    weights=None,
    refit_graph=False,
    Mu=None,
    Lambda=None,
    cycle_Mu=None,
    cycle_Lambda=None,
):
    """
    Delete path between two nodes in the principal graph

    """
    _PG = deepcopy(PG)

    # --- Init parameters, variables
    if Mu is None:
        Mu = _PG["Mu"]
    if Lambda is None:
        Lambda = _PG["Lambda"]
    if cycle_Mu is None:
        cycle_Mu = Mu
    if cycle_Lambda is None:
        cycle_Lambda = Lambda

    # --- get path to remove
    epg_edge = _PG["Edges"][0]
    epg_edge_len = _PG["projection"]["edge_len"].copy()
    del _PG["projection"]

    G = nx.Graph()
    G.add_nodes_from(range(_PG["NodePositions"].shape[0]))
    edges_weighted = list(zip(epg_edge[:, 0], epg_edge[:, 1], epg_edge_len))
    G.add_weighted_edges_from(edges_weighted, weight="len")

    if nodes_to_include is None:
        # nodes on the shortest path
        nodes_sp = nx.shortest_path(G, source=source, target=target, weight="len")
    else:
        assert isinstance(nodes_to_include, list), "`nodes_to_include` must be list"
        # lists of simple paths, in order from shortest to longest
        list_paths = list(
            nx.shortest_simple_paths(G, source=source, target=target, weight="len")
        )
        flag_exist = False
        for p in list_paths:
            if set(nodes_to_include).issubset(p):
                nodes_sp = p
                flag_exist = True
                break
        if not flag_exist:
            print(f"no path that passes {nodes_to_include} exists")

    G.remove_edges_from(np.vstack((nodes_sp[:-1], nodes_sp[1:])).T)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    Gdel = nx.relabel_nodes(G, dict(zip(G.nodes, np.arange(len(G.nodes)))))

    _PG["Edges"] = [np.array(Gdel.edges)]
    _PG["NodePositions"] = _PG["NodePositions"][
        ~np.isin(range(len(_PG["NodePositions"])), isolates)
    ]

    # --- get nodep, edges, create new graph with added loop
    if refit_graph:
        nodep, edges = _PG["NodePositions"], _PG["Edges"][0]

        cycle_nodes = elpigraph._graph_editing.find_all_cycles(nx.Graph(edges.tolist()))
        if len(cycle_nodes) > 0:
            cycle_nodes = flatten(cycle_nodes)

        ElasticMatrix = elpigraph.src.core.MakeUniformElasticMatrix_with_cycle(
            edges,
            Lambda=Lambda,
            Mu=Mu,
            cycle_Lambda=cycle_Lambda,
            cycle_Mu=cycle_Mu,
            cycle_nodes=cycle_nodes,
        )

        (
            newnodep,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = elpigraph.src.core.PrimitiveElasticGraphEmbedment(
            X, nodep, ElasticMatrix, PointWeights=weights, FixNodesAtPoints=[]
        )
        _PG["NodePositions"] = newnodep
    return _PG


def refitGraph(
    X,
    PG,
    shift_nodes_pos={},
    PointWeights=None,
    Mu=None,
    Lambda=None,
    cycle_Mu=None,
    cycle_Lambda=None,
):

    init_nodes_pos, init_edges = (PG["NodePositions"], PG["Edges"][0])

    # --- Init parameters, variables
    if Mu is None:
        Mu = PG["Mu"]
    if Lambda is None:
        Lambda = PG["Lambda"]
    if cycle_Mu is None:
        cycle_Mu = Mu
    if cycle_Lambda is None:
        cycle_Lambda = Lambda

    # ---Modify node pos order (first nodes are fixed)
    for k, v in shift_nodes_pos.items():
        init_nodes_pos[k] = v
    fix_nodes = sorted(list(shift_nodes_pos.keys()), reverse=True)
    fix_order = np.arange(len(init_nodes_pos))
    fix_edges = init_edges.copy()
    for i, ifix in enumerate(fix_nodes):
        n1, n2 = fix_order == i, fix_order == ifix
        e1, e2 = fix_edges == i, fix_edges == ifix
        fix_order[n1], fix_order[n2] = ifix, i
        fix_edges[e1], fix_edges[e2] = ifix, i
    fix_nodes_pos = init_nodes_pos[fix_order]

    SquaredX = np.sum(X ** 2, axis=1, keepdims=1)
    part, part_dist = elpigraph.src.core.PartitionData(
        X, fix_nodes_pos, 10 ** 6, SquaredX=SquaredX
    )

    cycle_nodes = find_all_cycles(nx.Graph(fix_edges.tolist()))
    if len(cycle_nodes) > 0:
        cycle_nodes = flatten(cycle_nodes)

    ElasticMatrix = elpigraph.src.core.MakeUniformElasticMatrix_with_cycle(
        fix_edges,
        Lambda=Lambda,
        Mu=Mu,
        cycle_Lambda=cycle_Lambda,
        cycle_Mu=cycle_Mu,
        cycle_nodes=cycle_nodes,
    )

    (
        new_nodes_pos,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = elpigraph.src.core.PrimitiveElasticGraphEmbedment(
        X,
        fix_nodes_pos,
        ElasticMatrix,
        PointWeights=PointWeights,
        FixNodesAtPoints=[[] for i in range(len(fix_nodes))],
    )

    # ---Revert to initial node ordering
    for i, ifix in enumerate(fix_nodes):
        e1, e2 = fix_edges == i, fix_edges == ifix
        fix_edges[e1], fix_edges[e2] = ifix, i
    new_nodes_pos = new_nodes_pos[fix_order]

    PG["NodePositions"] = new_nodes_pos

