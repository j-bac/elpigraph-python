import pandas as pd
import numpy as np
import random
import copy
import warnings
import networkx as nx
import igraph
import matplotlib.pyplot as plt

import scipy.stats
import scipy.optimize
from scipy.spatial.distance import euclidean, cdist
from scipy import signal
from scipy.stats import spearmanr
from itertools import combinations

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

from .src.PCA import PCA, TruncPCA, PCA_gpu, TruncSVD_gpu
from .src.core import PartitionData
from .src.graphs import ConstructGraph, GetSubGraph, GetBranches
from .src.distutils import PartialDistance
from .src.reporting import project_point_onto_graph, project_point_onto_edge


#### ClinTraj gbrancher funcs


def find_branches(graph, verbose=0):
    """
    Computes "branches" of the graph, i.e. paths from branch vertex (or terminal vertex)  to branch vertex (or terminal vertex)
    Can process disconnected graphs. Stand-alone point - is "branch".
    Circle is exceptional case - each circle (can be several connected components) is "branch"

    g - graph (igraph)
    verbose - details output

    @examples
    import igraph
    g = igraph.Graph.Lattice([3,3], circular = False )
    dict_output = find_branches(g, verbose = 1000)
    print( dict_output['branches'] )
    """
    # verbose = np.inf
    #
    g = graph
    n_vertices_input_graph = g.vcount()
    set_vertices_input_graph = range(n_vertices_input_graph)

    dict_output = {}
    # dict_output['branches'] = found_branches.copy()

    # Main variables for process:
    found_branches = []
    processed_edges = []
    processed_vertices = set()

    ############################################################################################################################################
    # Connected components loop:
    count_connected_components = 0
    while (
        True
    ):  # Need loop if graph has several connected components, each iteration - new component
        count_connected_components += 1

        def find_start_vertex(g, processed_vertices):
            """
            Find starting vertex for branches-search algorithm.
            It should be either branching vertex (i.e. degree >2) or terminal vertex (i.e. degree 0 or 1), in special case when unprocessed part of graph is union of circles - processed outside function
            """
            n_vertices = n_vertices_input_graph  #  = g.count()#
            if n_vertices == len(processed_vertices):
                return -1, -1  # All vertices proccessed
            flag_found_start_vertex = 0
            for v in set_vertices_input_graph:
                if v in processed_vertices:
                    continue
                if g.degree(v) != 2:
                    flag_found_start_vertex = 1
                    return v, flag_found_start_vertex
            return (
                -1,
                0,
            )  # All unprocessed vertices are of degree 2, that means graph is circle of collection or collection of circles

        ############################################################################################################################################
        # Starting point initialization. End process condtion.
        #
        # Set correctly the starting vertex for the algorithm
        # That should be branch vertex or terminal vertex, only in case graph is set of circles(disconnected) we take arbitrary vertex as initial, each circle will be a branch
        initial_vertex, flag_found_start_vertex = find_start_vertex(
            g, processed_vertices
        )
        if flag_found_start_vertex > 0:
            current_vertex = initial_vertex
        elif (
            flag_found_start_vertex == 0
        ):  # All unprocessed vertices are of degree 2, that means graph is circle of collection or collection of circles
            # Take any unprocessed element
            tmp_set = set_vertices_input_graph - processed_vertices
            current_vertex = tmp_set.pop()
        else:
            # No vertices to process
            if verbose >= 10:
                print("Process finished")
            dict_output["branches"] = found_branches.copy()
            return dict_output
            # break

        ############################################################################################################################################
        # Core function implementing "Breath First Search" like algorithm
        # with some updates in storage, since we need to arrange edges into "branches"
        def find_branches_core(
            current_vertex, previous_vertex, current_branch
        ):
            core_call_count[0] = core_call_count[0] + 1
            if verbose >= 1000:
                print(
                    core_call_count[0],
                    "core call.",
                    "current_vertex",
                    current_vertex,
                    "previous_vertex",
                    previous_vertex,
                    "found_branches",
                    found_branches,
                    "current_branch",
                    current_branch,
                )

            processed_vertices.add(current_vertex)
            neis = g.neighbors(current_vertex)
            if len(neis) == 0:  # current_vertex is standalone vertex
                found_branches.append([current_vertex])
                return
            if len(neis) == 1:  # current_vertex is terminal vertex
                if neis[0] == previous_vertex:
                    current_branch.append(current_vertex)
                    found_branches.append(current_branch.copy())
                    # processed_edges.append(  set([current_vertex , previous_vertex])  )
                    return
                else:
                    # That case may happen if we just started from that vertex
                    # Because it has one neigbour, but it is not previous_vertex, so it is None, which is only at start
                    current_branch = [
                        current_vertex
                    ]  # , neis[0] ] # .append( current_vertex  )
                    processed_edges.append(set([current_vertex, neis[0]]))
                    find_branches_core(
                        current_vertex=neis[0],
                        previous_vertex=current_vertex,
                        current_branch=current_branch,
                    )
                    return
            if len(neis) == 2:  #
                # continue the current branch:
                current_branch.append(current_vertex)
                next_vertex = neis[0]
                if next_vertex == previous_vertex:
                    next_vertex = neis[1]
                if (
                    next_vertex in processed_vertices
                ):  # Cannot happen for trees, but may happen if graph has a loop
                    if (
                        set([current_vertex, next_vertex])
                        not in processed_edges
                    ):
                        current_branch.append(next_vertex)
                        found_branches.append(current_branch.copy())
                        processed_edges.append(
                            set([current_vertex, next_vertex])
                        )
                        return
                    else:
                        return
                processed_edges.append(set([current_vertex, next_vertex]))
                find_branches_core(
                    current_vertex=next_vertex,
                    previous_vertex=current_vertex,
                    current_branch=current_branch,
                )
                return
            if len(neis) > 2:  # Branch point
                if previous_vertex is not None:
                    # Stop current branch
                    current_branch.append(current_vertex)
                    found_branches.append(current_branch.copy())
                for next_vertex in neis:
                    if next_vertex == previous_vertex:
                        continue
                    if (
                        next_vertex in processed_vertices
                    ):  # Cannot happen for trees, but may happen if graph has a loop
                        if (
                            set([current_vertex, next_vertex])
                            not in processed_edges
                        ):
                            processed_edges.append(
                                set([current_vertex, next_vertex])
                            )
                            found_branches.append(
                                [current_vertex, next_vertex]
                            )
                        continue
                    current_branch = [current_vertex]
                    processed_edges.append(set([current_vertex, next_vertex]))
                    find_branches_core(
                        current_vertex=next_vertex,
                        previous_vertex=current_vertex,
                        current_branch=current_branch,
                    )
            return

        ############################################################################################################################################
        # Core function call. It should process the whole connected component
        if verbose >= 10:
            print(
                "Start process count_connected_components",
                count_connected_components,
                "initial_vertex",
                current_vertex,
            )
        processed_vertices.add(current_vertex)
        core_call_count = [0]
        find_branches_core(
            current_vertex=current_vertex,
            previous_vertex=None,
            current_branch=[],
        )

        ############################################################################################################################################
        # Output of results for connected component
        if verbose >= 10:
            print(
                "Connected component ",
                count_connected_components,
                " processed ",
            )
            print("Final found_branches", found_branches)
            print("N Final found_branches", len(found_branches))


def branch_labler(X, graph, nodes_positions, verbose=0):
    """
    Labels points of the dataset X by "nearest"-"branches" of graph.


    @examples
    # X = np.array( [[0.1,0.1], [0.1,0.2], [1,2],[3,4],[5,0]] )
    # nodes_positions = np.array( [ [0,0], [1,0], [0,1], [1,1] ]  )
    # import igraph
    # g = igraph.Graph(); g.add_vertices(  4  )
    # g.add_edges([[0,1],[0,2],[0,3]])
    # vec_labels_by_branches = branch_labler( X , g, nodes_positions )
    """
    #####################################################################################
    # Calculate branches and clustering by vertices of graph
    dict_output = find_branches(graph, verbose=verbose)
    if verbose >= 100:
        print(
            "Function find_branches results branches:", dict_output["branches"]
        )
    vec_labels_by_vertices, dists, all_dists = PartitionData(
        X, nodes_positions, 1e6, np.sum(X ** 2, axis=1, keepdims=1)
    )  # np.array([[1,2,3,4], [1,2,3,4], [1,2,3,4], [10,20,30,40]]), [[1,2,3,4], [10,20,30,40]], 10**6)#,SquaredX)
    vec_labels_by_vertices = vec_labels_by_vertices.ravel()
    if verbose >= 100:
        print(
            "Function partition_data returns: vec_labels_by_vertices.shape,"
            " dists.shape, all_dists.shape",
            vec_labels_by_vertices.shape,
            dists.shape,
            all_dists.shape,
        )
    #####################################################################################

    n_vertices = len(nodes_positions)
    branches = dict_output["branches"]

    #####################################################################################
    # Create dictionary vertex to list of branches it belongs to
    dict_vertex2branches = {}
    for i, b in enumerate(branches):
        for v in b:
            if v in dict_vertex2branches.keys():
                dict_vertex2branches[v].append(i)
            else:
                dict_vertex2branches[v] = [i]
    if verbose >= 100:
        print("dict_vertex2branches", dict_vertex2branches)

    #####################################################################################
    # create list of branch and non-branch vertices
    list_branch_vertices = []
    list_non_branch_vertices = []
    for v in dict_vertex2branches:
        list_branches = dict_vertex2branches[v]
        if len(list_branches) == 1:
            list_non_branch_vertices.append(v)
        else:
            list_branch_vertices.append(v)
    if verbose >= 100:
        print(
            "list_branch_vertices, list_non_branch_vertices",
            list_branch_vertices,
            list_non_branch_vertices,
        )

    #####################################################################################
    # First stage of creation of final output - create labels by branches vector
    # After that step it will be only correct for non-branch points
    vec_vertex2branch = np.zeros(n_vertices)
    for i in range(n_vertices):
        vec_vertex2branch[i] = dict_vertex2branches[i][0]
    vec_labels_by_branches = vec_vertex2branch[vec_labels_by_vertices]
    if verbose >= 100:
        print("branches", branches)
        print("vec_labels_by_branches", vec_labels_by_branches)

    #####################################################################################
    # Second stage of creation of final output -
    # make correct calculation for branch-vertices create labels by correct branches
    for branch_vertex in list_branch_vertices:
        if verbose >= 100:
            print("all_dists.shape", all_dists.shape)

        def labels_for_one_branch_vertex(
            branch_vertex, vec_labels_by_vertices, all_dists
        ):
            """
            For the branch_vertex re-labels points of dataset which were labeled by it to label by "correct branch".
            "Correct branch" label is a branch 'censored'-nearest to given point.
            Where 'censored'-nearest means the minimal distance between the point  and all points of the branch except the given branch_vertex

            Function changes vec_labels_by_branches defined above
            Uses vec_labels_by_vertices defined above - vector of same length as dataset, which contains labels by vertices
            """

            mask = (
                vec_labels_by_vertices.ravel() == branch_vertex
            )  # Select part of the dataset which is closest to branch_vertex

            # Allocate memory for array: first coordinate - point of dataset[mask],  second coordinate - branch number , for all branches contianing given vertex (i.e. branch_vertex)
            # For each point of dataset[mask] it contains 'censored'-distances to "branches" adjoint to "branch_vertex",
            # 'censored' means minimal over vertices belonging to  distance to branches (excluding branch_vertex)
            dist2branches = np.zeros(
                [mask.sum(), len(dict_vertex2branches[branch_vertex])]
            )
            list_branch_ids = (
                []
            )  # that will be necessary to renumerate local number to branch_ids
            for i, branch_id in enumerate(dict_vertex2branches[branch_vertex]):
                list_branch_ids.append(branch_id)
                # Create list of vertices of current branch, with EXCLUSION of branch_vertex
                branch_vertices_wo_given_branch_vertex = [
                    v for v in branches[branch_id] if v != branch_vertex
                ]
                # For all points of dataset[mask] calculate minimal distances to given branch (with exclusion of branch_point), i.e. mininal difference for
                if verbose >= 1000:
                    print(
                        "mask.shape, all_dists.shape",
                        mask.shape,
                        all_dists.shape,
                    )
                dist2branches[:, i] = np.min(
                    all_dists[mask, :][
                        :, branch_vertices_wo_given_branch_vertex
                    ],
                    1,
                ).ravel()

            vec_labels_by_branches[mask] = np.array(list_branch_ids)[
                np.argmin(dist2branches, 1)
            ]

        labels_for_one_branch_vertex(
            branch_vertex, vec_labels_by_vertices, all_dists
        )

        if verbose >= 10:
            print("Output: vec_labels_by_branches", vec_labels_by_branches)

    return vec_labels_by_branches


#### ClinTraj epg funcs


def firstNonNan(floats):
    for i, item in enumerate(floats):
        if np.isnan(item) == False:
            return i, item


def firstNanIndex(floats):
    for i, item in enumerate(floats):
        if np.isnan(item) == True:
            return i


def lastNonNan(floats):
    for i, item in enumerate(np.flip(floats)):
        if np.isnan(item) == False:
            return len(floats) - i - 1, item


def fill_gaps_in_number_sequence(x):
    firstnonnan, val = firstNonNan(x)
    firstnan = firstNanIndex(x)
    if firstnan is not None:
        x[0:firstnonnan] = val
    lastnonnan, val = lastNonNan(x)
    if firstnan is not None:
        x[lastnonnan:-1] = val
        x[-1] = val
    # print('Processing',x)
    firstnan = firstNanIndex(x)
    while firstnan is not None:
        # print(x[firstNanIndex:])
        firstnonnan, val = firstNonNan(x[firstnan:])
        # print(val)
        firstnonnan = firstnonnan + firstnan
        # print('firstNanIndex',firstnan)
        # print('firstnonnan',firstnonnan)
        # print(np.linspace(x[firstnan-1],val,firstnonnan-firstnan+2))
        x[firstnan - 1 : firstnonnan + 1] = np.linspace(
            x[firstnan - 1], val, firstnonnan - firstnan + 2
        )
        # print('Imputed',x)
        firstnan = firstNanIndex(x)
    return x


def moving_weighted_average(
    x, y, step_size=0.1, steps_per_bin=1, weights=None
):
    # This ensures that all samples are within a bin
    number_of_bins = int(np.ceil(np.ptp(x) / step_size))
    bins = np.linspace(
        np.min(x),
        np.min(x) + step_size * number_of_bins,
        num=number_of_bins + 1,
    )
    bins -= (bins[-1] - np.max(x)) / 2
    bin_centers = bins[:-steps_per_bin] + step_size * steps_per_bin / 2

    counts, _ = np.histogram(x, bins=bins)
    # print(bin_centers)
    # print(counts)
    vals, _ = np.histogram(x, bins=bins, weights=y)
    bin_avgs = vals / counts
    # print(bin_avgs)
    n = len(bin_avgs)
    windowed_bin_avgs = np.lib.stride_tricks.as_strided(
        bin_avgs, (n - steps_per_bin + 1, steps_per_bin), bin_avgs.strides * 2
    )

    weighted_average = np.average(windowed_bin_avgs, axis=1, weights=weights)
    return bin_centers, weighted_average


def visualize_eltree_with_data(
    PG,
    X,
    X_original,
    principal_component_vectors,
    mean_vector,
    color,
    variable_names,
    showEdgeNumbers=False,
    showNodeNumbers=False,
    showBranchNumbers=False,
    showPointNumbers=False,
    Color_by_feature="",
    Feature_Edge_Width="",
    Invert_Edge_Value=False,
    Min_Edge_Width=5,
    Max_Edge_Width=5,
    Big_Point_Size=100,
    Small_Point_Size=1,
    Normal_Point_Size=20,
    Visualize_Edge_Width_AsNodeCoordinates=True,
    Color_by_partitioning=False,
    visualize_partition=[],
    Transparency_Alpha=0.2,
    Transparency_Alpha_points=1,
    verbose=False,
    Visualize_Branch_Class_Associations=[],  # list_of_branch_class_associations
    cmap="cool",
    scatter_parameter=0.03,
    highlight_subset=[],
    add_color_bar=False,
    vmin=-1,
    vmax=-1,
    percentile_contraction=20,
):

    nodep = PG["NodePositions"]
    nodep_original = (
        np.matmul(nodep, principal_component_vectors[:, 0 : X.shape[1]].T)
        + mean_vector
    )
    adjmat = PG["ElasticMatrix"]
    edges = PG["Edges"][0]
    color2 = color
    if not Color_by_feature == "":
        k = variable_names.index(Color_by_feature)
        color2 = X_original[:, k]
    if Color_by_partitioning:
        color2 = visualize_partition
        color_seq = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 0, 0.5],
            [1, 0.5, 0],
            [0.5, 0, 1],
            [0.5, 1, 0],
            [0.5, 0.5, 1],
            [0.5, 1, 0.5],
            [1, 0.5, 0.5],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0.5],
            [0, 0, 0.5],
            [0, 0.5, 0],
            [0.5, 0, 0],
            [0, 0.25, 0.5],
            [0, 0.5, 0.25],
            [0.25, 0, 0.5],
            [0.25, 0.5, 0],
            [0.5, 0, 0.25],
            [0.5, 0.25, 0],
            [0.25, 0.25, 0.5],
            [0.25, 0.5, 0.25],
            [0.5, 0.25, 0.25],
            [0.25, 0.25, 0.5],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
            [0.25, 0.5, 0.25],
            [0.5, 0, 0.25],
            [0.5, 0.25, 0.25],
        ]
        color2_unique, color2_count = np.unique(color2, return_counts=True)
        inds = sorted(
            range(len(color2_count)),
            key=lambda k: color2_count[k],
            reverse=True,
        )
        newc = []
        for i, c in enumerate(color2):
            k = np.where(color2_unique == c)[0][0]
            count = color2_count[k]
            k1 = np.where(inds == k)[0][0]
            k1 = k1 % len(color_seq)
            col = color_seq[k1]
            newc.append(col)
        color2 = newc

    plt.style.use("ggplot")
    points_size = Normal_Point_Size * np.ones(X_original.shape[0])
    if len(Visualize_Branch_Class_Associations) > 0:
        points_size = Small_Point_Size * np.ones(X_original.shape[0])
        for assc in Visualize_Branch_Class_Associations:
            branch = assc[0]
            cls = assc[1]
            indices = [i for i, x in enumerate(color) if x == cls]
            # print(branch,cls,color,np.where(color==cls))
            points_size[indices] = Big_Point_Size

    node_size = 10
    # Associate each node with datapoints
    if verbose:
        print("Partitioning the data...")
    partition, dists = PartitionData(
        X=X,
        NodePositions=nodep,
        MaxBlockSize=100000000,
        TrimmingRadius=np.inf,
        SquaredX=np.sum(X ** 2, axis=1, keepdims=1),
    )
    # col_nodes = {node: color[np.where(partition==node)[0]] for node in np.unique(partition)}

    # Project points onto the graph
    if verbose:
        print("Projecting data points onto the graph...")
    ProjStruct = project_point_onto_graph(
        X=X, NodePositions=nodep, Edges=edges, Partition=partition
    )

    projval = ProjStruct["ProjectionValues"]
    edgeid = (ProjStruct["EdgeID"]).astype(int)
    X_proj = ProjStruct["X_projected"]

    dist2proj = np.sum(np.square(X - X_proj), axis=1)
    shift = np.percentile(dist2proj, percentile_contraction)
    dist2proj = dist2proj - shift

    # Create graph
    if verbose:
        print("Producing graph layout...")
    g = nx.Graph()
    g.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(g, scale=2)
    # pos = nx.planar_layout(g)
    # pos = nx.spring_layout(g,scale=2)
    idx = np.array([pos[j] for j in range(len(pos))])

    # plt.figure(figsize=(16,16))
    if verbose:
        print("Calculating scatter aroung the tree...")
    x = np.zeros(len(X))
    y = np.zeros(len(X))
    for i in range(len(X)):
        # distance from edge
        # This is squared distance from a node
        # r = np.sqrt(dists[i])*scatter_parameter
        # This is squared distance from a projection (from edge),
        # even though the difference might be tiny
        r = 0
        if dist2proj[i] > 0:
            r = np.sqrt(dist2proj[i]) * scatter_parameter

        # get node coordinates for this edge
        x_coos = np.concatenate(
            (idx[edges[edgeid[i], 0], [0]], idx[edges[edgeid[i], 1], [0]])
        )
        y_coos = np.concatenate(
            (idx[edges[edgeid[i], 0], [1]], idx[edges[edgeid[i], 1], [1]])
        )

        projected_on_edge = False

        if projval[i] < 0:
            # project to 0% of the edge (first node)
            x_coo = x_coos[0]
            y_coo = y_coos[0]
        elif projval[i] > 1:
            # project to 100% of the edge (second node)
            x_coo = x_coos[1]
            y_coo = y_coos[1]
        else:
            # project to appropriate % of the edge
            x_coo = x_coos[0] + (x_coos[1] - x_coos[0]) * projval[i]
            y_coo = y_coos[0] + (y_coos[1] - y_coos[0]) * projval[i]
            projected_on_edge = True

        # if projected_on_edge:
        #     color2[i]=0
        # else:
        #     color2[i]=1
        # random angle
        # alpha = 2 * np.pi * np.random.random()
        # random scatter to appropriate distance
        # x[i] = r * np.cos(alpha) + x_coo
        # y[i] = r * np.sin(alpha) + y_coo
        # we rather position the point close to project and put
        # it at distance r orthogonally to the edge
        # on a random side of the edge
        # However, if projection was on a node then we scatter
        # in random direction
        vex = x_coos[1] - x_coos[0]
        vey = y_coos[1] - y_coos[0]
        if not projected_on_edge:
            vex = np.random.random() - 0.5
            vey = np.random.random() - 0.5
        vn = np.sqrt(vex * vex + vey * vey)
        vex = vex / vn
        vey = vey / vn
        rsgn = random_sign()
        x[i] = x_coo + vey * r * rsgn
        y[i] = y_coo - vex * r * rsgn
    if vmin < 0:
        vmin = min(color2)
    if vmax < 0:
        vmax = max(color2)
    plt.scatter(
        x,
        y,
        c=color2,
        cmap=cmap,
        s=points_size,
        vmin=vmin,
        vmax=vmax,
        alpha=Transparency_Alpha_points,
    )
    if showPointNumbers:
        for j in range(len(X)):
            plt.text(x[j], y[j], j)
    if len(highlight_subset) > 0:
        color_subset = [color2[i] for i in highlight_subset]
        plt.scatter(
            x[highlight_subset],
            y[highlight_subset],
            c=color_subset,
            cmap=cmap,
            s=Big_Point_Size,
        )
    if add_color_bar:
        plt.colorbar()

    # Scatter nodes
    PG["NodePositions2D"] = idx
    plt.scatter(idx[:, 0], idx[:, 1], s=node_size, c="black", alpha=0.8)

    # Associate edge width to a feature
    edge_vals = [1] * len(edges)
    if (
        not Feature_Edge_Width == ""
        and not Visualize_Edge_Width_AsNodeCoordinates
    ):
        k = variable_names.index(Feature_Edge_Width)
        for j in range(len(edges)):
            vals = X_original[np.where(edgeid == j)[0], k]
            vals = (np.array(vals) - np.min(X_original[:, k])) / (
                np.max(X_original[:, k]) - np.min(X_original[:, k])
            )
            edge_vals[j] = np.mean(vals)
        for j in range(len(edges)):
            if np.isnan(edge_vals[j]):
                e = edges[j]
                inds = [
                    ei
                    for ei, ed in enumerate(edges)
                    if ed[0] == e[0]
                    or ed[1] == e[0]
                    or ed[0] == e[1]
                    or ed[1] == e[1]
                ]
                inds.remove(j)
                evals = np.array(edge_vals)[inds]
                # print(j,inds,evals,np.mean(evals))
                edge_vals[j] = np.mean(evals[~np.isnan(evals)])
        if Invert_Edge_Value:
            edge_vals = [1 - v for v in edge_vals]

    if not Feature_Edge_Width == "" and Visualize_Edge_Width_AsNodeCoordinates:
        k = variable_names.index(Feature_Edge_Width)
        for j in range(len(edges)):
            e = edges[j]
            amp = np.max(nodep_original[:, k]) - np.min(nodep_original[:, k])
            mn = np.min(nodep_original[:, k])
            v0 = (nodep_original[e[0], k] - mn) / amp
            v1 = (nodep_original[e[1], k] - mn) / amp
            # print(v0,v1)
            edge_vals[j] = (v0 + v1) / 2
        if Invert_Edge_Value:
            edge_vals = [1 - v for v in edge_vals]

    # print(edge_vals)

    # Plot edges
    for j in range(len(edges)):
        x_coo = np.concatenate((idx[edges[j, 0], [0]], idx[edges[j, 1], [0]]))
        y_coo = np.concatenate((idx[edges[j, 0], [1]], idx[edges[j, 1], [1]]))
        plt.plot(
            x_coo,
            y_coo,
            c="k",
            linewidth=Min_Edge_Width
            + (Max_Edge_Width - Min_Edge_Width) * edge_vals[j],
            alpha=Transparency_Alpha,
        )
        if showEdgeNumbers:
            plt.text(
                (x_coo[0] + x_coo[1]) / 2,
                (y_coo[0] + y_coo[1]) / 2,
                j,
                FontSize=20,
                bbox=dict(facecolor="grey", alpha=0.5),
            )

    if showBranchNumbers:
        branch_vals = list(set(visualize_partition))
        for i, val in enumerate(branch_vals):
            ind = visualize_partition == val
            xbm = np.mean(x[ind])
            ybm = np.mean(y[ind])
            plt.text(
                xbm,
                ybm,
                int(val),
                FontSize=20,
                bbox=dict(facecolor="grey", alpha=0.5),
            )

    if showNodeNumbers:
        for i in range(nodep.shape[0]):
            plt.text(
                idx[i, 0],
                idx[i, 1],
                str(i),
                FontSize=20,
                bbox=dict(facecolor="grey", alpha=0.5),
            )

    # plt.axis('off')


def convert_elpigraph_to_igraph(elpigraph):
    edges = elpigraph["Edges"][0]
    nodes_positions = elpigraph["NodePositions"]
    g = igraph.Graph()
    g.add_vertices(len(nodes_positions))
    g.add_edges(edges)
    return g


def partition_data_by_tree_branches(X, PG):
    edges = PG["Edges"][0]
    nodes_positions = PG["NodePositions"]
    g = igraph.Graph()
    g.add_vertices(len(nodes_positions))
    g.add_edges(edges)
    vec_labels_by_branches = branch_labler(X, g, nodes_positions)
    return vec_labels_by_branches


def random_sign():
    return 1 if random.random() < 0.5 else -1


def pseudo_time(root_node, point_index, traj, projval, edgeid, edges):
    xi = int(point_index)
    proj_val_x = projval[xi]
    # print(proj_val_x)
    if proj_val_x < 0:
        proj_val_x = 0
    if proj_val_x > 1:
        proj_val_x = 1
    edgeid_x = edgeid[xi]
    # print(edges[edgeid_x])
    traja = np.array(traj)
    i1 = 1000000
    i2 = 1000000
    if edges[edgeid_x][0] in traja:
        i1 = np.where(traja == edges[edgeid_x][0])[0][0]
    if edges[edgeid_x][1] in traja:
        i2 = np.where(traja == edges[edgeid_x][1])[0][0]
    i = min(i1, i2)
    pstime = i + proj_val_x
    return pstime


def pseudo_time_trajectory(traj, ProjStruct):
    projval = ProjStruct["ProjectionValues"]
    edgeid = (ProjStruct["EdgeID"]).astype(int)
    edges = ProjStruct["Edges"]
    partition = ProjStruct["Partition"]
    traj_points = np.zeros(0, "int32")
    for p in traj:
        traj_points = np.concatenate(
            (traj_points, np.where(partition == p)[0])
        )
    # print(len(traj_points))
    pst = np.zeros(len(traj_points))
    for i, p in enumerate(traj_points):
        pst[i] = pseudo_time(traj[0], p, traj, projval, edgeid, edges)
    return pst, traj_points


def extract_trajectories(tree, root_node, verbose=False):
    """
    Extracting trajectories from ElPiGraph result object tree,
    starting from a root_node.
    Extracting trajectories is a required step for quantifying pseudotime
    after.
    Example:
        all_trajectories,all_trajectories_edges = extract_trajectories(tree,root_node)
        print(len(all_trajectories),' trajectories found.')
        ProjStruct = project_on_tree(X,tree)
        PseudoTimeTraj = quantify_pseudotime(all_trajectories,all_trajectories_edges,ProjStruct)
    """
    edges = tree["Edges"][0]
    nodes_positions = tree["NodePositions"]
    g = igraph.Graph()
    g.add_vertices(len(nodes_positions))
    g.add_edges(edges)
    degs = g.degree()
    leaf_nodes = [i for i, d in enumerate(degs) if d == 1]
    if verbose:
        print(len(leaf_nodes), "trajectories found")
    all_trajectories_vertices = []
    all_trajectories_edges = []
    for lf in leaf_nodes:
        path_vertices = g.get_shortest_paths(root_node, to=lf, output="vpath")
        all_trajectories_vertices.append(path_vertices[0])
        path_edges = g.get_shortest_paths(root_node, to=lf, output="epath")
        all_trajectories_edges.append(path_edges[0])
        if verbose:
            print("Vertices:", path_vertices)
            print("Edges:", path_edges)
        ped = []
        for ei in path_edges[0]:
            ped.append((g.get_edgelist()[ei][0], g.get_edgelist()[ei][1]))
        if verbose:
            print("Edges:", ped)
        # compute pseudotime along each path
    return all_trajectories_vertices, all_trajectories_edges


def correlation_of_variable_with_trajectories(
    PseudoTimeTraj,
    var,
    var_names,
    X_original,
    verbose=False,
    producePlot=False,
    Correlation_Threshold=0.5,
):
    List_of_Associations = []
    for i, pstt in enumerate(PseudoTimeTraj):
        inds = pstt["Trajectory"]
        # traj_nodep = nodep_original[inds,:]
        points = pstt["Points"]
        pst = pstt["Pseudotime"]
        TrajName = (
            "Trajectory:"
            + str(pstt["Trajectory"][0])
            + "--"
            + str(pstt["Trajectory"][-1])
        )
        k = var_names.index(var)
        vals = X_original[:, k]
        spcorr = spearmanr(pst, vals[points]).correlation
        asstup = (TrajName, var, spcorr)
        if abs(spcorr) > Correlation_Threshold:
            List_of_Associations.append(asstup)
            if verbose:
                print(i, asstup)
            if producePlot:
                x = pst
                y = vals[points]
                bincenters, wav = moving_weighted_average(x, y, step_size=1.5)
                plt.plot(pst, y, "ro")
                plt.plot(
                    bincenters,
                    fill_gaps_in_number_sequence(wav),
                    "bo-",
                    linewidth=10,
                    markersize=10,
                )
                # plt.plot(np.linspace(0,len(inds)-1,len(inds)),traj_nodep[:,k])
                # plt.ylim(min(y)-np.ptp(y)*0.05,max(y)+np.ptp(y)*0.05)
                plt.xlabel("Pseudotime", fontsize=20)
                plt.ylabel(var, fontsize=20)
                plt.title(TrajName + ", r={:2.2f}".format(spcorr), fontsize=20)
                plt.show()
    return List_of_Associations


def regress_variable_on_pseudotime(
    pseudotime,
    vals,
    TrajName,
    var_name,
    var_type,
    producePlot=True,
    verbose=False,
    Continuous_Regression_Type="linear",
    R2_Threshold=0.5,
    max_sample=-1,
    alpha_factor=2,
):
    # Continuous_Regression_Type can be 'linear','gpr' for Gaussian Process, 'kr' for kernel ridge
    if var_type == "BINARY":
        # convert back to binary vals
        mn = min(vals)
        mx = max(vals)
        vals[np.where(vals == mn)] = 0
        vals[np.where(vals == mx)] = 1
        if len(np.unique(vals)) == 1:
            regressor = None
        else:
            regressor = LogisticRegression(
                random_state=0, max_iter=1000, penalty="none"
            ).fit(pseudotime, vals)
    if var_type == "CATEGORICAL":
        if len(np.unique(vals)) == 1:
            regressor = None
        else:
            regressor = LogisticRegression(
                random_state=0, max_iter=1000, penalty="none"
            ).fit(pseudotime, vals)
    if var_type == "CONTINUOUS" or var_type == "ORDINAL":
        if len(np.unique(vals)) == 1:
            regressor = None
        else:
            if Continuous_Regression_Type == "gpr":
                # subsampling if needed
                pst = pseudotime.copy()
                vls = vals.copy()
                if max_sample > 0:
                    l = list(range(len(vals)))
                    random.shuffle(l)
                    index_value = random.sample(l, min(max_sample, len(vls)))
                    pst = pst[index_value]
                    vls = vls[index_value]
                if len(np.unique(vls)) > 1:
                    gp_kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
                    # gp_kernel =  RBF(np.std(vals))
                    regressor = GaussianProcessRegressor(
                        kernel=gp_kernel, alpha=np.var(vls) * alpha_factor
                    )
                    regressor.fit(pst, vls)
                else:
                    regressor = None
            if Continuous_Regression_Type == "linear":
                regressor = LinearRegression()
                regressor.fit(pseudotime, vals)

    r2score = 0
    if regressor is not None:
        r2score = r2_score(vals, regressor.predict(pseudotime))

        if producePlot and r2score > R2_Threshold:
            plt.plot(pseudotime, vals, "ro", label="data")
            unif_pst = np.linspace(min(pseudotime), max(pseudotime), 100)
            pred = regressor.predict(unif_pst)
            if var_type == "BINARY" or var_type == "CATEGORICAL":
                prob = regressor.predict_proba(unif_pst)
                plt.plot(
                    unif_pst, prob[:, 1], "g-", linewidth=2, label="proba"
                )
            if var_type == "CONTINUOUS" or var_type == "ORDINAL":
                plt.plot(unif_pst, pred, "g-", linewidth=2, label="predicted")
            bincenters, wav = moving_weighted_average(
                pseudotime, vals.reshape(-1, 1), step_size=1.5
            )
            plt.plot(
                bincenters,
                fill_gaps_in_number_sequence(wav),
                "b-",
                linewidth=2,
                label="sliding av",
            )
            plt.xlabel("Pseudotime", fontsize=20)
            plt.ylabel(var_name, fontsize=20)
            plt.title(TrajName + ", r2={:2.2f}".format(r2score), fontsize=20)
            plt.legend(fontsize=15)
            plt.show()

    return r2score, regressor


def regression_of_variable_with_trajectories(
    PseudoTimeTraj,
    var,
    var_names,
    variable_types,
    X_original,
    verbose=False,
    producePlot=True,
    R2_Threshold=0.5,
    Continuous_Regression_Type="linear",
    max_sample=1000,
    alpha_factor=2,
):
    List_of_Associations = []
    for i, pstt in enumerate(PseudoTimeTraj):
        inds = pstt["Trajectory"]
        # traj_nodep = nodep_original[inds,:]
        points = pstt["Points"]
        pst = pstt["Pseudotime"]
        pst = pst.reshape(-1, 1)
        TrajName = (
            "Trajectory:"
            + str(pstt["Trajectory"][0])
            + "--"
            + str(pstt["Trajectory"][-1])
        )
        k = var_names.index(var)
        vals = X_original[points, k]
        r2, regressor = regress_variable_on_pseudotime(
            pst,
            vals,
            TrajName,
            var,
            variable_types[k],
            producePlot=producePlot,
            verbose=verbose,
            R2_Threshold=R2_Threshold,
            Continuous_Regression_Type=Continuous_Regression_Type,
            max_sample=max_sample,
            alpha_factor=alpha_factor,
        )
        pstt[var + "_regressor"] = regressor
        asstup = (TrajName, var, r2)
        # if verbose:
        #    print(var,'R2',r2)
        if r2 > R2_Threshold:
            List_of_Associations.append(asstup)
            if verbose:
                print(i, asstup)
    return List_of_Associations


def quantify_pseudotime(all_trajectories, ProjStruct, producePlot=False):
    projval = ProjStruct["ProjectionValues"]
    edgeid = (ProjStruct["EdgeID"]).astype(int)
    edges = ProjStruct["Edges"]
    partition = ProjStruct["Partition"]
    PseudoTimeTraj = []
    for traj in all_trajectories:
        pst, points = pseudo_time_trajectory(traj, ProjStruct)
        pstt = {}
        pstt["Trajectory"] = traj
        pstt["Points"] = points
        pstt["Pseudotime"] = pst
        PseudoTimeTraj.append(pstt)
        if producePlot:
            plt.plot(np.sort(pst))
    return PseudoTimeTraj


def project_on_tree(X, tree):
    nodep = tree["NodePositions"]
    edges = tree["Edges"][0]
    partition, dists = PartitionData(
        X=X,
        NodePositions=nodep,
        MaxBlockSize=100000000,
        TrimmingRadius=np.inf,
        SquaredX=np.sum(X ** 2, axis=1, keepdims=1),
    )
    ProjStruct = project_point_onto_graph(
        X=X, NodePositions=nodep, Edges=edges, Partition=partition
    )
    # projval = ProjStruct['ProjectionValues']
    # edgeid = (ProjStruct['EdgeID']).astype(int)
    ProjStruct["Partition"] = partition
    return ProjStruct


def draw_pseudotime_dependence(
    trajectory,
    variable_name,
    variable_names,
    variable_types,
    X_original,
    color_line,
    linewidth=1,
    fontsize=20,
    draw_datapoints=False,
    label=None,
    linestyle=None,
):
    regressor = trajectory[variable_name + "_regressor"]
    k = variable_names.index(variable_name)
    mn = min(X_original[:, k])
    mx = max(X_original[:, k])
    vals = None
    if regressor is not None:
        pst = trajectory["Pseudotime"]
        # pst = np.unique(pst).reshape(-1,1)
        unif_pst = np.linspace(min(pst), max(pst), 100).reshape(-1, 1)
        var_type = variable_types[k]
        if var_type == "BINARY":
            vals = regressor.predict_proba(unif_pst)[:, 1]
        else:
            vals = regressor.predict(unif_pst)
            vals = (vals - mn) / (mx - mn)
        if draw_datapoints:
            plt.plot(
                pst,
                (X_original[trajectory["Points"], k] - mn) / (mx - mn),
                "ko",
                color=color_line,
            )
        if label is None:
            label = variable_name
        if linestyle is None:
            linestyle = "-"
        plt.plot(
            unif_pst,
            vals,
            color=color_line,
            linewidth=linewidth,
            label=label,
            linestyle=linestyle,
        )
        plt.xlabel("Pseudotime", fontsize=fontsize)
    return vals


def add_pie_charts_tree(ax, tree, partition, values, color_seq, scale=1):
    nodep = tree["NodePositions"]
    edges = tree["Edges"][0]
    g = nx.Graph()
    g.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(g, scale=2)
    idx = np.array([pos[j] for j in range(len(pos))])
    add_pie_charts(ax, idx, partition, values, color_seq, scale=scale)


def add_pie_charts(
    ax, node_positions2d, partition, values, color_seq, scale=1
):
    df = pd.DataFrame({"CLASS": values})
    vals_unique_df = df.CLASS.value_counts()
    vals_unique = vals_unique_df.index.to_list()
    vals_unique_freq = vals_unique_df.to_numpy()
    print(vals_unique, vals_unique_freq)

    for i in range(node_positions2d.shape[0]):
        inode = np.where(partition == i)
        dfi = df.loc[partition == i]
        node_valunique_df = dfi.CLASS.value_counts()
        node_valunique = node_valunique_df.index.to_list()
        node_valunique_freq = node_valunique_df.to_numpy()
        freq_sum = np.sum(node_valunique_freq)
        freq = len(vals_unique) * [0]
        for j, v in enumerate(node_valunique):
            freq[vals_unique.index(v)] = node_valunique_freq[j] / freq_sum
        # print(i,':',node_valunique,node_valunique_freq)
        # print(i,':',freq)
        draw_pie(
            ax,
            freq,
            color_seq,
            X=node_positions2d[i, 0],
            Y=node_positions2d[i, 1],
            size=scale * len(inode[0]),
        )


def draw_pie(ax, ratios, colors, X=0, Y=0, size=1000):
    N = len(ratios)
    xy = []
    start = 0.0
    for ratio in ratios:
        x = [0] + np.cos(
            np.linspace(2 * np.pi * start, 2 * np.pi * (start + ratio), 30)
        ).tolist()
        y = [0] + np.sin(
            np.linspace(2 * np.pi * start, 2 * np.pi * (start + ratio), 30)
        ).tolist()
        xy1 = list(zip(x, y))
        xy.append(xy1)
        start += ratio
    for i, xyi in enumerate(xy):
        ax.scatter([X], [Y], marker=xyi, s=size, facecolor=colors[i])


####################


def PlotPG(
    X,
    PG,
    X_color="r",
    Node_color="k",
    Do_PCA=True,
    DimToPlot=[0, 1],
    show_node=True,
    show_text=True,
    ax=None,
):

    if Do_PCA:
        # Perform PCA on the nodes
        mv = PG["NodePositions"].mean(axis=0)
        data_centered = PG["NodePositions"] - mv
        vglobal, nodesp, explainedVariances = PCA(data_centered)
        # Rotate the data using eigenvectors
        BaseData = np.dot((X - mv), vglobal)
        DataVarPerc = np.var(BaseData, axis=0) / np.sum(np.var(X, axis=0))

    else:
        nodesp = PG["NodePositions"]
        BaseData = X
        DataVarPerc = np.var(X, axis=0) / np.sum(np.var(X, axis=0))

    if ax is None:
        ax = plt.subplot()

    # scatter data
    ax.scatter(
        BaseData[:, DimToPlot[0]],
        BaseData[:, DimToPlot[1]],
        c=X_color,
        alpha=0.15,
    )

    # scatter nodes
    if show_node:
        ax.scatter(
            nodesp[:, DimToPlot[0]],
            nodesp[:, DimToPlot[1]],
            c=Node_color,
            s=20,
        )

    if show_text:
        for i in np.arange(nodesp.shape[0]):
            ax.text(
                nodesp[i, DimToPlot[0]],
                nodesp[i, DimToPlot[1]],
                i,
                color="black",
                ha="left",
                va="bottom",
            )

    Edges = PG["Edges"][0].T

    # plot edges
    for j in range(Edges.shape[1]):
        x_coo = np.concatenate(
            (nodesp[Edges[0, j], [0]], nodesp[Edges[1, j], [0]])
        )
        y_coo = np.concatenate(
            (nodesp[Edges[0, j], [1]], nodesp[Edges[1, j], [1]])
        )
        ax.plot(x_coo, y_coo, c="black", linewidth=1, alpha=0.6)

    if Do_PCA:
        TarPGVarPerc = explainedVariances / explainedVariances.sum() * 100
    else:
        TarPGVarPerc = np.var(PG["NodePositions"], axis=0) / np.sum(
            np.var(PG["NodePositions"], axis=0)
        )
    ax.set_xlabel(f"PG % var: {TarPGVarPerc[DimToPlot[0]]:.2f}")
    ax.set_ylabel(f"PG % var: {TarPGVarPerc[DimToPlot[1]]:.2f}")

    if ax is None:
        plt.show()


# def old_PlotPG(
#    X,
#    TargetPG,
#    BootPG=None,
#    PGCol="",
#    PlotProjections="none",
#    GroupsLab=None,
#    PointViz="points",
#    Main="",
#    p_alpha=0.3,
#    PointSize=None,
#    NodeLabels=None,
#    LabMult=1,
#    Do_PCA=True,
#    DimToPlot=[0, 1],
#    VizMode=("Target", "Boot"),
# ):
#    """
#    work in progress, only basic plotting supported
#    Plot data and principal graph(s)
#
#    X numerical 2D matrix, the n-by-m matrix with the position of n m-dimensional points
#    TargetPG the main principal graph to plot
#    BootPG A list of principal graphs that will be considered as bostrapped curves
#    PGCol string, the label to be used for the main principal graph
#    PlotProjections string, the plotting mode for the node projection on the principal graph.
#    It can be "none" (no projections will be plotted), "onNodes" (the projections will indicate how points are associated to nodes),
#    and "onEdges" (the projections will indicate how points are projected on edges or nodes of the graph)
#    GroupsLab factor or numeric vector. A vector indicating either a category or a numeric value associted with
#    each data point
#    PointViz string, the modality to show points. It can be 'points' (data will be represented a dot) or
#    'density' (the data will be represented by a field)
#    Main string, the title of the plot
#    p.alpha numeric between 0 and 1, the alpha value of the points. Lower values will prodeuce more transparet points
#    PointSize numeric vector, a vector indicating the size to be associted with each node of the graph.
#    If NA points will have size 0.
#    NodeLabels string vector, a vector indicating the label to be associted with each node of the graph
#    LabMult numeric, a multiplier controlling the size of node labels
#    Do_PCA bolean, should the node of the principal graph be used to derive principal component projections and
#    rotate the space? If TRUE the plots will use the "EpG PC" as dimensions, if FALSE, the original dimensions will be used.
#    DimToPlot a integer vector specifing the PCs (if Do_PCA=TRUE) or dimension (if Do_PCA=FALSE) to plot. All the
#    combination will be considered, so, for example, if DimToPlot = 1:3, three plot will be produced.
#    VizMode vector of string, describing the ElPiGraphs to visualize. Any combination of "Target" and "Boot".
#
#    @return
#
#
#    @examples"""
#
#    if len(PGCol) == 1:
#        PGCol = [PGCol] * len(TargetPG["NodePositions"])
#
#    if GroupsLab is None:
#        GroupsLab = ["N/A"] * len(X)
#
#    #    levels(GroupsLab) = c(levels(GroupsLab), unique(PGCol))
#
#    if PointSize is not None:
#        if len(PointSize) == 1:
#            PointSize = [PointSize] * len(TargetPG["NodePositions"])
#
#    if Do_PCA:
#        # Perform PCA on the nodes
#        mv = TargetPG["NodePositions"].mean(axis=0)
#        data_centered = TargetPG["NodePositions"] - mv
#        vglobal, NodesPCA, explainedVariances = PCA(data_centered)
#        # Rotate the data using eigenvectors
#        BaseData = np.dot((X - mv), vglobal)
#        DataVarPerc = np.var(BaseData, axis=0) / np.sum(np.var(X, axis=0))
#
#    else:
#        NodesPCA = TargetPG["NodePositions"]
#        BaseData = X
#        DataVarPerc = np.var(X, axis=0) / np.sum(np.var(X, axis=0))
#
#    # Base Data
#
#    AllComb = list(combinations(DimToPlot, 2))
#
#    PlotList = list()
#
#    for i in range(len(AllComb)):
#
#        Idx1 = AllComb[i][0]
#        Idx2 = AllComb[i][1]
#
#        df1 = pd.DataFrame.from_dict(
#            dict(PCA=BaseData[:, Idx1], PCB=BaseData[:, Idx2], Group=GroupsLab)
#        )
#        # Initialize plot
#
#        Initialized = False
#
#        if PointViz == "points":
#            p = plotnine.ggplot(
#                data=df1, mapping=plotnine.aes(x="PCA", y="PCB")
#            ) + plotnine.geom_point(alpha=p_alpha, mapping=plotnine.aes(color="Group"))
#            Initialized = True
#
#        if PointViz == "density":
#            p = plotnine.ggplot(
#                data=df1, mapping=plotnine.aes(x="PCA", y="PCB")
#            ) + plotnine.stat_density_2d(
#                contour=True,
#                alpha=0.5,
#                geom="polygon",
#                mapping=plotnine.aes(fill="..level.."),
#            )
#            Initialized = True
#
#        #             p = sns.kdeplot(df1['PCA'], df1['PCB'], cmap="Reds", shade=True, bw=.15)
#
#        if not Initialized:
#            raise ValueError("Invalid point representation selected")
#
#        # Target graph
#
#        tEdg = dict(x=[], y=[], xend=[], yend=[], Col=[])
#        for i in range(len(TargetPG["Edges"][0])):
#            Node_1 = TargetPG["Edges"][0][i][0]
#            Node_2 = TargetPG["Edges"][0][i][1]
#            if PGCol:
#                if PGCol[Node_1] == PGCol[Node_2]:
#                    tCol = "ElPiG" + str(PGCol[Node_1])
#
#                if PGCol[Node_1] != PGCol[Node_2]:
#                    tCol = "ElPiG Multi"
#
#                if any(PGCol[(Node_1, Node_2)] == "None"):
#                    tCol = "ElPiG None"
#
#            tEdg["x"].append(NodesPCA[Node_1, Idx1])
#            tEdg["y"].append(NodesPCA[Node_1, Idx2])
#            tEdg["xend"].append(NodesPCA[Node_2, Idx1])
#            tEdg["yend"].append(NodesPCA[Node_2, Idx2])
#            if PGCol:
#                tEdg["Col"].append(tCol)
#            else:
#                tEdg["Col"].append(1)
#        if Do_PCA:
#            TarPGVarPerc = explainedVariances.sum() / explainedVariances.sum() * 100
#        else:
#            TarPGVarPerc = np.var(TargetPG["NodePositions"], axis=0) / np.sum(
#                np.var(TargetPG["NodePositions"], axis=0)
#            )
#
#        df2 = pd.DataFrame.from_dict(tEdg)
#
#        # Replicas
#
#        #         if(BootPG is not None) and ("Boot" is in VizMode):
#        #             AllEdg = lapply(1:length(BootPG), function(i){
#        #             tTree = BootPG[[i]]
#
#        #             if(Do_PCA):
#        #                 RotData = t(t(tTree$NodePositions) - NodesPCA$center) %*% NodesPCA$rotation
#        #             else: {
#        #                 RotData = tTree$NodePositions
#        #             }
#
#        #             tEdg = t(sapply(1:nrow(tTree$Edges$Edges), function(i){
#        #               c(RotData[tTree$Edges$Edges[i, 1],c(Idx1, Idx2)], RotData[tTree$Edges$Edges[i, 2],c(Idx1, Idx2)])
#        #             }))
#
#        #             cbind(tEdg, i)
#        #             })
#
#        #             AllEdg = do.call(rbind, AllEdg)
#
#        #             df3 = data.frame(x = AllEdg[,1], y = AllEdg[,2], xend = AllEdg[,3], yend = AllEdg[,4], Rep = AllEdg[,5])
#
#        #             p = p + plotnine.geom_segment(data = df3, mapping = plotnine.aes(x=x, y=y, xend=xend, yend=yend),
#        #                                          inherit.aes = False, alpha = .2, color = "black")
#
#        # Plot projections
#
#        if PlotProjections == "onEdges":
#
#            if Do_PCA:
#                Partition = PartitionData(
#                    X=BaseData,
#                    NodePositions=NodesPCA,
#                    MaxBlockSize=100000000,
#                    SquaredX=np.sum(BaseData ** 2, axis=1, keepdims=1),
#                    TrimmingRadius=float("inf"),
#                )[0]
#                OnEdgProj = project_point_onto_graph(
#                    X=BaseData,
#                    NodePositions=NodesPCA,
#                    Edges=TargetPG["Edges"],
#                    Partition=Partition,
#                )
#            else:
#                Partition = PartitionData(
#                    X=BaseData,
#                    NodePositions=TargetPG["NodePositions"],
#                    MaxBlockSize=100000000,
#                    SquaredX=np.sum(BaseData ** 2, axis=1, keepdims=1),
#                    TrimmingRadius=float("inf"),
#                )[0]
#                OnEdgProj = project_point_onto_graph(
#                    X=BaseData,
#                    NodePositions=TargetPG["NodePositions"],
#                    Edges=TargetPG["Edges"],
#                    Partition=Partition,
#                )
#
#            ProjDF = pd.DataFrame.from_dict(
#                dict(
#                    X=BaseData[:, Idx1],
#                    Y=BaseData[:, Idx2],
#                    Xend=OnEdgProj["X_projected"][:, Idx1],
#                    Yend=OnEdgProj["X_projected"][:, Idx2],
#                    Group=GroupsLab,
#                )
#            )
#
#            p = p + plotnine.geom_segment(
#                data=ProjDF,
#                mapping=plotnine.aes(
#                    x="X", y="Y", xend="Xend", yend="Yend", col="Group"
#                ),
#                inherit_aes=False,
#            )
#
#        elif PlotProjections == "onNodes":
#
#            if Do_PCA:
#                Partition = PartitionData(
#                    X=BaseData,
#                    NodePositions=NodesPCA,
#                    MaxBlockSize=100000000,
#                    SquaredX=np.sum(BaseData ** 2, axis=1, keepdims=1),
#                    TrimmingRadius=float("inf"),
#                )[0]
#                ProjDF = pd.DataFrame.from_dict(
#                    dict(
#                        X=BaseData[:, Idx1],
#                        Y=BaseData[:, Idx2],
#                        Xend=NodesPCA[Partition, Idx1],
#                        Yend=NodesPCA[Partition, Idx2],
#                        Group=GroupsLab,
#                    )
#                )
#            else:
#                Partition = PartitionData(
#                    X=BaseData,
#                    NodePositions=TargetPG["NodePositions"],
#                    MaxBlockSize=100000000,
#                    SquaredX=np.sum(BaseData ** 2, axis=1, keepdims=1),
#                    TrimmingRadius=float("inf"),
#                )[0]
#                ProjDF = pd.DataFrame.from_dict(
#                    dict(
#                        X=BaseData[:, Idx1],
#                        Y=BaseData[:, Idx2],
#                        Xend=TargetPG["NodePositions"][Partition, Idx1],
#                        Yend=TargetPG["NodePositions"][Partition, Idx2],
#                        Group=GroupsLab,
#                    )
#                )
#
#            p = p + plotnine.geom_segment(
#                data=ProjDF,
#                mapping=plotnine.aes(
#                    x="X", y="Y", xend="Xend", yend="Yend", col="Group"
#                ),
#                inherit_aes=False,
#                alpha=0.3,
#            )
#
#        if "Target" in VizMode:
#            if GroupsLab is not None:
#                p = (
#                    p
#                    + plotnine.geom_segment(
#                        data=df2,
#                        mapping=plotnine.aes(
#                            x="x", y="y", xend="xend", yend="yend", col="Col"
#                        ),
#                        inherit_aes=True,
#                    )
#                    + plotnine.labs(linetype="")
#                )
#            else:
#                p = p + plotnine.geom_segment(
#                    data=df2,
#                    mapping=plotnine.aes(x="x", y="y", xend="xend", yend="yend"),
#                    inherit_aes=False,
#                )
#
#        if Do_PCA:
#            df4 = pd.DataFrame.from_dict(
#                dict(PCA=NodesPCA[:, Idx1], PCB=NodesPCA[:, Idx2])
#            )
#        else:
#            df4 = pd.DataFrame.from_dict(
#                dict(
#                    PCA=TargetPG["NodePositions"][:, Idx1],
#                    PCB=TargetPG["NodePositions"][:, Idx2],
#                )
#            )
#
#        if "Target" in VizMode:
#            if PointSize is not None:
#
#                p = p + plotnine.geom_point(
#                    mapping=plotnine.aes(x="PCA", y="PCB", size=PointSize),
#                    data=df4,
#                    inherit_aes=False,
#                )
#
#            else:
#                p = p + plotnine.geom_point(
#                    mapping=plotnine.aes(x="PCA", y="PCB"), data=df4, inherit_aes=False
#                )
#
#        #         if(NodeLabels):
#
#        #             if(Do_PCA){
#        #                 df4 = data.frame(PCA = NodesPCA$x[,Idx1], PCB = NodesPCA$x[,Idx2], Lab = NodeLabels)
#        #             else {
#        #                 df4 = data.frame(PCA = TargetPG$NodePositions[,Idx1], PCB = TargetPG$NodePositions[,Idx2], Lab = NodeLabels)
#        #           }
#
#        #           p = p + plotnine.geom_text(mapping = plotnine.aes(x = PCA, y = PCB, label = Lab),
#        #                                       data = df4, hjust = 0,
#        #                                       inherit.aes = False, na.rm = True,
#        #                                       check_overlap = True, color = "black", size = LabMult)
#
#        #         }
#
#        #         if(Do_PCA){
#        #             LabX = "EpG PC", Idx1, " (Data var = ",  np.round(100*DataVarPerc[Idx1], 3), "% / PG var = ", signif(100*TarPGVarPerc[Idx1], 3), "%)"
#        #             LabY = "EpG PC", Idx2, " (Data var = ",  np.round(100*DataVarPerc[Idx2], 3), "% / PG var = ", signif(100*TarPGVarPerc[Idx2], 3), "%)"
#        #         else {
#        #             LabX = paste0("Dimension ", Idx1, " (Data var = ",  np.round(100*DataVarPerc[Idx1], 3), "% / PG var = ", np.round(100*TarPGVarPerc[Idx1], 3), "%)")
#        #             LabY = paste0("Dimension ", Idx2, " (Data var = ",  np.round(100*DataVarPerc[Idx2], 3), "% / PG var = ", np.round(100*TarPGVarPerc[Idx2], 3), "%)")
#        #         }
#
#        #         if(!is.na(TargetPG$FinalReport$FVEP)){
#        #             p = p + plotnine.labs(x = LabX,
#        #                                  y = LabY,
#                                  title = paste0(Main,
#                                                 "/ FVE=",
#                                                 signif(as.numeric(TargetPG$FinalReport$FVE), 3),
#                                                 "/ FVEP=",
#                                                 signif(as.numeric(TargetPG$FinalReport$FVEP), 3))
#           ) +
#             plotnine.theme(plot.title = plotnine.element_text(hjust = 0.5))
#         else {
#           p = p + plotnine.labs(x = LabX,
#                                  y = LabY,
#                                  title = paste0(Main,
#                                                 "/ FVE=",
#                                                 signif(as.numeric(TargetPG$FinalReport$FVE), 3))
#           ) +
#             plotnine.theme(plot.title = plotnine.element_text(hjust = 0.5))
#         }

#        PlotList.append(p)

#    return PlotList


# Plotting Functions (Diagnostic) --------------------------------------------

#' Plot the MSD VS Energy plot
#'
#' PrintGraph a struct returned by computeElasticPrincipalGraph
#' Main string, title of the plot
#'
#' Return-------a ggplot plot
#'
#'
#' @examples
# plotMSDEnergyPlot <- function(ReportTable, Main = ''){

#   df <- rbind(data.frame(Nodes = as.integer(ReportTable[,"NNODES"]),
#                    Value = as.numeric(ReportTable[,"ENERGY"]), Type = "Energy"),
#               data.frame(Nodes = as.integer(ReportTable[,"NNODES"]),
#                          Value = as.numeric(ReportTable[,"MSEP"]), Type = "MSEP")
#   )

#   p <- ggplot2::ggplot(data = df, mapping = ggplot2::aes(x = Nodes, y = Value, color = Type, shape = Type),
#                        environment = environment()) +
#     ggplot2::geom_point() + ggplot2::geom_line() + ggplot2::facet_grid(Type~., scales = "free_y") +
#     ggplot2::guides(color = "none", shape = "none") + ggplot2::ggtitle(Main)

#   return(p)

# }


#' Accuracy-Complexity plot
#'
#' Main string, tht title of the plot
#' Mode integer or string, the mode used to identify minima: if 'LocMin', the code of the
#' local minima will be plotted, if the number n, the code will be plotted each n configurations.
#' If NULL, no code will be plotted
#' Xlims a numeric vector of length 2 indicating the minimum and maximum of the x axis. If NULL (the default)
#' the rage of the data will be used
#' ReportTable A report table as returned from an ElPiGraph computation function
#' AdjFactor numeric, the factor used to adjust the values on the y axis (computed as UR*NNODE^AdjFactor)
#'
#' Return-------a ggplot plot
#'
#'
#' @examples
# accuracyComplexityPlot <- function(ReportTable, AdjFactor=1, Main = '', Mode = 'LocMin', Xlims = NULL){

#   if(is.null(Xlims)){
#     Xlims <- range(as.numeric(ReportTable[,"FVEP"]))
#   }

#   YVal <- as.numeric(ReportTable[,"UR"])*(as.integer(ReportTable[,"NNODES"])^AdjFactor)


#   df <- data.frame(FVEP = as.numeric(ReportTable[,"FVEP"]), Comp = YVal)
#   p <- ggplot2::ggplot(data = df, ggplot2::aes(x = FVEP, y = Comp), environment = environment()) +
#     ggplot2::geom_point() + ggplot2::geom_line() +
#     ggplot2::labs(title = Main, x = "Fraction of Explained Variance", y = "Geometrical Complexity") +
#     ggplot2::coord_cartesian(xlim = Xlims)

#   TextMat <- NULL

#   if(Mode == 'LocMin'){
#     for(i in 2:(length(YVal)-1)){
#       xp = YVal[i-1]
#       x = YVal[i]
#       xn = YVal[i+1]
#       if(x < min(c(xp,xn))){
#         diff = abs(x-(xp+xn)/2);
#         if(diff>0.01){
#           TextMat <- rbind(TextMat, c(ReportTable[i,"FVEP"], y = YVal[i], labels = ReportTable[i,"BARCODE"]))
#         }
#       }
#     }
#   }

#   if(is.numeric(Mode)){
#     Mode = round(Mode)

#     TextMat <- rbind(TextMat, c(ReportTable[2,"FVEP"], y = YVal[2], labels = ReportTable[2,"BARCODE"]))
#     TextMat <- rbind(TextMat, c(ReportTable[length(YVal),"FVEP"], y = YVal[length(YVal)], labels = ReportTable[length(YVal),"BARCODE"]))

#     if(Mode > 2){
#       Mode <- Mode - 1
#       Step <- (length(YVal) - 2)/Mode

#       for (i in seq(from=2+Step, to = length(YVal)-1, by = Step)) {
#         TextMat <- rbind(TextMat, c(ReportTable[round(i),"FVEP"], y = YVal[round(i)], labels = ReportTable[round(i),"BARCODE"]))
#       }
#     }

#   }

#   if(!is.null(TextMat)){
#     df2 <- data.frame(FVEP = as.numeric(TextMat[,1]), Comp = as.numeric(TextMat[,2]), Label = TextMat[,3])

#     p <- p + ggplot2::geom_text(data = df2, mapping = ggplot2::aes(x = FVEP, y = Comp, label = Label),
#                                 check_overlap = TRUE, inherit.aes = FALSE, nudge_y = .005)
#   }

#   return(p)
# }


# Plotting Functions (2D plots) --------------------------------------------


#' Plot a graph with pie chart associated with each node
#'
#' X numerical 2D matrix, the n-by-m matrix with the position of n m-dimensional points
#' TargetPG the main principal graph to plot
#' Nodes integer, the vector of nodes to plot. If NULL, all the nodes will be plotted.
#' Graph a igraph object of the ElPiGraph, if NULL (the default) it will be computed by the function
#' LayOut the global layout of yhe final network. It can be
#' \itemize{
#'  \item 'tree', a tree
#'  \item 'circle', a closed circle
#'  \item 'circle_line', a line arranged as a circle
#'  \item 'kk', a topology generated by the Kamada-Kawai layout algorithm
#'  \item 'mds', a topology generated by multidimensional scaling on the node positions
#'  \item 'fr', a topology generated by the Fruchterman-Reingold layout algorithm
#'  \item 'nicely', the topology will be inferred by igraph
#' }
#' TreeRoot the id of the node to use as the root of the tree when LayOut = 'tree', multiple nodes are allowed.
#' Main string, the title of the plot
#' ScaleFunction function, a function used to scale the nuumber of points (sqrt by default)
#' NodeSizeMult integer, an adjustment factor to control the size of the pies
#' ColCat string vector, a vector of colors to associate to each category
#' GroupsLab string factor, a vector indicating the category of each data point
#' Partition A vector associating each point to a node of the ElPiGraph. If NULL (the default), this will be computed
#' TrimmingRadius numeric, the trimming radius to use when associting points to nodes when Partition = NULL
#' Leg.cex numeric, a value to adjust the size of the legend
#' distMeth the matric used to compute the distance if LayOut = 'mds'
#' Arrow.size numeric, the size of the arrow
#' LabSize numeric, the size of the node labels
#' LayoutIter numeric, the number of interation of the layout algorithm
#' Leg.pos character, the position of the legend (see the help of the legend function)
#' Leg.horiz boolean, should the legend be plotted in horizontal
#' NodeLabels character vector, the names of the nodes
#' RootLevel numeric, the level of the root(s)
#'
#' Return-------NULL
#'
#'
#' @examples
# plotPieNet <- function(X,
#                        TargetPG,
#                        GroupsLab = NULL,
#                        Nodes = NULL,
#                        Partition = NULL,
#                        TrimmingRadius = Inf,
#                        Graph = NULL,
#                        LayOut = 'nicely',
#                        LayoutIter = 500,
#                        TreeRoot = numeric(),
#                        RootLevel = numeric(),
#                        distMeth = "manhattan",
#                        Main="",
#                        ScaleFunction = sqrt,
#                        NodeSizeMult = 1,
#                        ColCat = NULL,
#                        Leg.cex = 1,
#                        Leg.pos = "bottom",
#                        Leg.horiz = TRUE,
#                        Arrow.size = 1,
#                        NodeLabels = NULL,
#                        LabSize = 1) {

#   if(!is.factor(GroupsLab)){
#     GroupsLab <- factor(GroupsLab)
#   }

#   if(is.null(ColCat)){
#     ColCat <- c(rainbow(length(unique(GroupsLab))), NA)
#     names(ColCat) <- c(levels(droplevels(GroupsLab)), NA)
#   } else {
#     if(sum(names(ColCat) %in% levels(GroupsLab)) < length(unique(GroupsLab))){
#       print("Reassigning colors to categories")
#       names(ColCat) <- c(levels(GroupsLab))
#     }
#     ColCat <- c(ColCat[levels(GroupsLab)], NA)
#     # ColCat <- c(ColCat, NA)
#   }

#   if(is.null(Partition)){
#     print("Partition will be computed. Consider do that separetedly")
#     Partition <- PartitionData(X = X, NodePositions = TargetPG$NodePositions,
#                                SquaredX = rowSums(X^2), TrimmingRadius = TrimmingRadius,
#                                nCores = 1)$Partition
#   }

#   GroupPartTab <- matrix(0, nrow = nrow(TargetPG$NodePositions), ncol = length(ColCat))
#   colnames(GroupPartTab) <- c(levels(GroupsLab), "None")

#   TTab <- table(Partition[Partition > 0], GroupsLab[Partition > 0])
#   GroupPartTab[as.integer(rownames(TTab)), colnames(TTab)] <- TTab

#   Missing <- setdiff(1:nrow(TargetPG$NodePositions), unique(Partition))

#   if(length(Missing)>0){
#     GroupPartTab[Missing, "None"] <- 1
#   }

#   if(is.null(Graph)){
#     print("A graph will be constructed. Consider do that separatedly")
#     Net <- ConstructGraph(PrintGraph = TargetPG)
#   } else {
#     Net <- Graph
#   }

#   if(is.null(NodeLabels)){
#     igraph::V(Net)$lab <- 1:igraph::vcount(Net)
#   } else {
#     igraph::V(Net)$label <- NodeLabels
#   }
#   PieList <- apply(GroupPartTab, 1, list)
#   PieList <- lapply(PieList, function(x){x[[1]]})

#   PieColList <- lapply(PieList, function(x){ColCat})

#   if(!is.null(Nodes)){
#     Net <- igraph::induced.subgraph(Net, Nodes)
#     PieList <- PieList[as.integer(names(igraph::V(Net)))]
#     # NodePos <- TargetPG$NodePositions[as.integer(names(igraph::V(tNet))), ]
#   } else {
#     # NodePos <- TargetPG$NodePositions
#   }

#   if(!is.null(ScaleFunction)){
#     if(is.null(Nodes)){
#       PieSize <- NodeSizeMult*do.call(what = ScaleFunction,
#                                       list(table(factor(x = Partition, levels = 1:nrow(TargetPG$NodePositions)))))
#     } else {
#       PieSize <- NodeSizeMult*do.call(what = ScaleFunction,
#                                       list(table(factor(x = Partition[Partition %in% as.integer(names(igraph::V(Net)))],
#                                                         levels = as.integer(names(igraph::V(Net)))
#                                                         ))))
#     }

#   } else {
#     PieSize <- rep(NodeSizeMult, igraph::vcount(Net))
#   }

#   PieSize[sapply(PieList, "[[", "None")>0] <- 0

#   LayOutDONE <- FALSE

#   if(LayOut == 'tree'){
#     RestrNodes <- igraph::layout_as_tree(graph = igraph::as.undirected(Net, mode = 'collapse'), root = TreeRoot,
#                                          rootlevel = RootLevel);
#     LayOutDONE <- TRUE
#   }

#   if(LayOut == 'circle'){
#     IsoGaph <- igraph::graph.ring(n = igraph::vcount(Net), directed = FALSE, circular = TRUE)
#     Iso <- igraph::graph.get.isomorphisms.vf2(igraph::as.undirected(Net, mode = 'collapse'), IsoGaph)
#     if(length(Iso)>0){
#       VerOrder <- igraph::V(Net)[Iso[[1]]]
#       RestrNodes <- igraph::layout_in_circle(graph = Net, order = VerOrder)
#       LayOutDONE <- TRUE
#     } else {
#       Net1 <- ConstructGraph(PrintGraph = TargetPG)
#       IsoGaph <- igraph::graph.ring(n = igraph::vcount(Net1), directed = FALSE, circular = TRUE)
#       Iso <- igraph::graph.get.isomorphisms.vf2(igraph::as.undirected(Net1, mode = 'collapse'), IsoGaph)
#       VerOrder <- igraph::V(Net1)[Iso[[1]]]
#       RestrNodes <- igraph::layout_in_circle(graph = Net, order = VerOrder$name)
#       LayOutDONE <- TRUE
#     }
#   }

#   if(LayOut == 'circle_line'){
#     IsoGaph <- igraph::graph.ring(n = igraph::vcount(Net), directed = FALSE, circular = FALSE)
#     Iso <- igraph::graph.get.isomorphisms.vf2(igraph::as.undirected(Net, mode = 'collapse'), IsoGaph)
#     if(length(Iso) > 0){
#       VerOrder <- igraph::V(Net)[Iso[[1]]]
#       RestrNodes <- igraph::layout_in_circle(graph = Net, order = VerOrder)
#       LayOutDONE <- TRUE
#     } else {
#       Net1 <- ConstructGraph(PrintGraph = TargetPG)
#       IsoGaph <- igraph::graph.ring(n = igraph::vcount(Net1), directed = FALSE, circular = FALSE)
#       Iso <- igraph::graph.get.isomorphisms.vf2(igraph::as.undirected(Net1, mode = 'collapse'), IsoGaph)
#       VerOrder <- igraph::V(Net1)[Iso[[1]]]
#       RestrNodes <- igraph::layout_in_circle(graph = Net, order = VerOrder$name)
#       LayOutDONE <- TRUE
#     }

#   }

#   if(LayOut == 'nicely'){
#     RestrNodes <- igraph::layout_nicely(graph = Net)
#     LayOutDONE <- TRUE
#   }
#   if(LayOut == 'kk'){
#     tNet <- Net
#     igraph::E(tNet)$weight <- NA
#     for(edg in igraph::E(tNet)){
#       Nodes <- igraph::ends(tNet, edg)
#       InC1 <- Partition == Nodes[1,1]
#       InC2 <- Partition == Nodes[1,2]

#       if(any(InC1) & any(InC2)){

#         if(sum(InC1)>1){
#           C1 <- colMeans(X[InC1,])
#         } else {
#           C1 <- X[InC1,]
#         }

#         if(sum(InC2)>1){
#           C2 <- colMeans(X[InC2,])
#         } else {
#           C2 <- X[InC2,]
#         }

#         igraph::E(tNet)[edg]$weight <- sum(abs(C1 - C2))
#       }

#     }

#     igraph::E(tNet)$weight[is.na(igraph::E(tNet)$weight)] <- min(igraph::E(tNet)$weight, na.rm = TRUE)/10

#     RestrNodes <- igraph::layout_with_kk(graph = igraph::as.undirected(tNet, mode = 'collapse'),
#                                          weights = igraph::E(tNet)$weight, maxiter = LayoutIter)
#     LayOutDONE <- TRUE
#   }


#   if(LayOut == 'mds'){

#     NodeCentr <- matrix(NA, nrow = igraph::vcount(Net), ncol = ncol(X))
#     rownames(NodeCentr) <- names(igraph::V(Net))

#     SelNodeCentr <- t(sapply(split(data.frame(X), Partition), colMeans))
#     NodeCentr <- SelNodeCentr[rownames(NodeCentr), ]

#     NodeCentr_1 <- NodeCentr

#     for(i in which(rowSums(is.na(NodeCentr))>0)){
#       for(j in 1:igraph::vcount(Net)){
#         NP <- colMeans(NodeCentr[as.integer(igraph::neighborhood(Net, order = j, nodes = i)[[1]]),], na.rm = TRUE)
#         if(all(is.finite(NP))){
#           NodeCentr_1[i, ] <- NP
#           break()
#         }
#       }
#     }

#     RestrNodes <- igraph::layout_with_mds(graph = igraph::as.undirected(Net, mode = 'collapse'),
#                                           dist = as.matrix(dist(NodeCentr_1, method = distMeth)))
#     LayOutDONE <- TRUE
#   }


#   if(LayOut == 'fr'){

#     tNet <- Net
#     igraph::E(tNet)$weight <- NA
#     for(edg in igraph::E(tNet)){
#       Nodes <- igraph::ends(tNet, edg)
#       InC1 <- Partition == Nodes[1,1]
#       InC2 <- Partition == Nodes[1,2]

#       if(any(InC1) & any(InC2)){

#         if(sum(InC1)>1){
#           C1 <- colMeans(X[InC1,])
#         } else {
#           C1 <- X[InC1,]
#         }

#         if(sum(InC2)>1){
#           C2 <- colMeans(X[InC2,])
#         } else {
#           C2 <- X[InC2,]
#         }

#         igraph::E(tNet)[edg]$weight <- sum(abs(C1 - C2))
#       }

#     }

#     igraph::E(tNet)$weight[is.na(igraph::E(tNet)$weight)] <- 10/min(igraph::E(tNet)$weight, na.rm = TRUE)

#     RestrNodes <- igraph::layout_with_fr(graph = tNet, niter = LayoutIter)
#     LayOutDONE <- TRUE
#   }
#   if(!LayOutDONE){
#     print(paste("LayOut =", LayOut, "unrecognised"))
#     return(NULL)
#   }
#   igraph::plot.igraph(Net, layout = RestrNodes[,1:2], main = Main,
#                       vertex.shape="pie", vertex.pie.color = PieColList,
#                       vertex.pie=PieList, vertex.pie.border = NA,
#                       vertex.size=PieSize,
#                       edge.color = "black", vertex.label.dist = 0.7,
#                       vertex.label.color = "black", vertex.label.cex = LabSize)

#   if(Leg.cex>0){

#     legend(x = Leg.pos, legend = names(ColCat)[names(ColCat) != "" & !is.na(ColCat)],
#            fill = ColCat[names(ColCat) != "" & !is.na(ColCat)], horiz = Leg.horiz, cex = Leg.cex)
#   }


# }
