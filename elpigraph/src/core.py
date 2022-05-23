try:
    import cupy
except:
    pass
import numpy as np
import numba as nb
from .distutils import *

# Base functions: Distance and energy computation --------------------------


def PartitionData_cp(
    Xcp,
    NodePositions,
    MaxBlockSize,
    SquaredXcp,
    TrimmingRadius=float("inf"),
    precomp=False,
):
    """
    # Partition the data by proximity to graph nodes
    # (same step as in K-means EM procedure)
    #
    # Inputs:
    #   X is n-by-m matrix of datapoints with one data point per row. n is
    #       number of data points and m is dimension of data space.
    #   NodePositions is k-by-m matrix of embedded coordinates of graph nodes,
    #       where k is number of nodes and m is dimension of data space.
    #   MaxBlockSize integer number which defines maximal number of
    #       simultaneously calculated distances. Maximal size of created matrix
    #       is MaxBlockSize-by-k, where k is number of nodes.
    #   SquaredX is n-by-1 vector of data vectors length: SquaredX = sum(X.^2,2);
    #   TrimmingRadius (optional) is squared trimming radius.
    #
    # Outputs
    #   partition is n-by-1 vector. partition[i] is number of the node which is
    #       associated with data point X[i, ].
    #   dists is n-by-1 vector. dists[i] is squared distance between the node with
    #       number partition[i] and data point X[i, ].
    """
    NodePositionscp = cupy.asarray(NodePositions)
    n = Xcp.shape[0]
    partition = cupy.zeros((n, 1), dtype=int)
    dists = cupy.zeros((n, 1))
    d = cupy.zeros((n, len(NodePositions)))

    # Calculate squared length of centroids
    cent = NodePositionscp.T
    centrLength = (cent ** 2).sum(axis=0)
    # Process partitioning without trimming

    for i in range(0, n, MaxBlockSize):
        # Define last element for calculation
        last = i + MaxBlockSize
        if last > n:
            last = n
        # Calculate distances
        _d = (
            SquaredXcp[i:last]
            + centrLength
            - 2
            * cupy.dot(
                Xcp[
                    i:last,
                ],
                cent,
            )
        )
        tmp = _d.argmin(axis=1)
        partition[i:last] = tmp[:, cupy.newaxis]
        dists[i:last] = _d[cupy.arange(_d.shape[0]), tmp][:, cupy.newaxis]
        d[i:last] = _d

    # Apply trimming
    if not cupy.isinf(TrimmingRadius):
        ind = dists > (TrimmingRadius ** 2)
        partition[ind] = -1
        dists[ind] = TrimmingRadius ** 2
    if precomp:
        return cupy.asnumpy(partition), cupy.asnumpy(dists), cupy.asnumpy(d)
    else:
        return cupy.asnumpy(partition), cupy.asnumpy(dists)


def PartitionData(
    X,
    NodePositions,
    MaxBlockSize,
    SquaredX,
    TrimmingRadius=float("inf"),
    precomp=False,
):
    """
    # Partition the data by proximity to graph nodes
    # (same step as in K-means EM procedure)
    #
    # Inputs:
    #   X is n-by-m matrix of datapoints with one data point per row. n is
    #       number of data points and m is dimension of data space.
    #   NodePositions is k-by-m matrix of embedded coordinates of graph nodes,
    #       where k is number of nodes and m is dimension of data space.
    #   MaxBlockSize integer number which defines maximal number of
    #       simultaneously calculated distances. Maximal size of created matrix
    #       is MaxBlockSize-by-k, where k is number of nodes.
    #   SquaredX is n-by-1 vector of data vectors length: SquaredX = sum(X.^2,2);
    #   TrimmingRadius (optional) is squared trimming radius.
    #
    # Outputs
    #   partition is n-by-1 vector. partition[i] is number of the node which is
    #       associated with data point X[i, ].
    #   dists is n-by-1 vector. dists[i] is squared distance between the node with
    #       number partition[i] and data point X[i, ].
    """
    n = X.shape[0]
    partition = np.zeros((n, 1), dtype=int)
    dists = np.zeros((n, 1))
    d = np.zeros((n, len(NodePositions)))
    # Calculate squared length of centroids
    cent = NodePositions.T
    centrLength = (cent ** 2).sum(axis=0)
    # Process partitioning without trimming
    for i in range(0, n, MaxBlockSize):
        # Define last element for calculation
        last = i + MaxBlockSize
        if last > n:
            last = n
        # Calculate distances
        _d = (
            SquaredX[i:last]
            + centrLength
            - 2
            * np.dot(
                X[
                    i:last,
                ],
                cent,
            )
        )
        tmp = _d.argmin(axis=1)
        partition[i:last] = tmp[:, np.newaxis]
        dists[i:last] = _d[np.arange(_d.shape[0]), tmp][:, np.newaxis]
        d[i:last] = _d

    # Apply trimming
    if not np.isinf(TrimmingRadius):
        ind = dists > (TrimmingRadius ** 2)
        partition[ind] = -1
        dists[ind] = TrimmingRadius ** 2
    if precomp:
        return partition, dists, d
    else:
        return partition, dists


def RePartitionData(
    X,
    NodePositions,
    NewNodePositions,
    NodeIndicesArray,
    opType,
    precomp_d,
    SquaredX,
    TrimmingRadius=np.inf,
):

    if "remove" in opType:
        d = precomp_d[:, NodeIndicesArray]
    elif "shrink" in opType:
        repositioned_node_idx = np.all(
            NodePositions[NodeIndicesArray] != NewNodePositions, axis=1
        )
        newcent = NewNodePositions[repositioned_node_idx].T
        newcentrLength = (newcent ** 2).sum(axis=0)
        new_d = SquaredX + newcentrLength - 2 * np.dot(X, newcent)
        d = precomp_d[:, NodeIndicesArray]
        d[:, repositioned_node_idx] = new_d

    else:  # grow grammar:
        newcent = NewNodePositions[[-1]].T
        newcentrLength = (newcent ** 2).sum(axis=0)
        new_d = SquaredX + newcentrLength - 2 * np.dot(X, newcent)
        d = np.hstack((precomp_d, new_d))

    tmp = d.argmin(axis=1)
    partition = tmp[:, np.newaxis]
    dists = d[np.arange(d.shape[0]), tmp][:, np.newaxis]
    # Apply trimming
    if not np.isinf(TrimmingRadius):
        ind = dists > (TrimmingRadius ** 2)
        partition[ind] = -1
        dists[ind] = TrimmingRadius ** 2

    return partition, dists, d


def PartitionDataBarycenter(
    X,
    NodePositions,
    MaxBlockSize,
    SquaredX,
    TrimmingRadius=float("inf"),
    previous_partition=None,
):
    """
    # Partition the data by barycentric proximity to graph nodes
    # https://arxiv.org/pdf/1902.10288.pdf
    # (same step as in barycentric K-means procedure)
    #
    # Inputs:
    #   X is n-by-m matrix of datapoints with one data point per row. n is
    #       number of data points and m is dimension of data space.
    #   NodePositions is k-by-m matrix of embedded coordinates of graph nodes,
    #       where k is number of nodes and m is dimension of data space.
    #   MaxBlockSize integer number which defines maximal number of
    #       simultaneously calculated distances. Maximal size of created matrix
    #       is MaxBlockSize-by-k, where k is number of nodes.
    #   SquaredX is n-by-1 vector of data vectors length: SquaredX = sum(X.^2,2);
    #   TrimmingRadius (optional) is squared trimming radius.
    #
    # Outputs
    #   partition is n-by-1 vector. partition[i] is number of the node which is
    #       associated with data point X[i, ].
    #   dists is n-by-1 vector. dists[i] is squared distance between the node with
    #       number partition[i] and data point X[i, ].
    """
    # ------clusters standard deviations
    nodes = np.arange(len(NodePositions))
    sds = np.zeros(len(nodes))

    if previous_partition is not None:
        previous_partition = previous_partition.ravel()
        for i, node in enumerate(nodes):
            node_data = X[previous_partition == node]
            if len(node_data) < 2:
                # adding epsilon for clusters with single point or no point
                sds[i] = 1e-100
            else:
                sds[i] = np.std(node_data)

    n = X.shape[0]
    partition = np.zeros((n, 1), dtype=int)
    dists = np.zeros((n, 1))
    # Calculate squared length of centroids
    cent = NodePositions.T
    centrLength = (cent ** 2).sum(axis=0)
    # Process partitioning without trimming
    for i in range(0, n, MaxBlockSize):
        # Define last element for calculation
        last = i + MaxBlockSize
        if last > n:
            last = n
        # Calculate distances
        d = (
            SquaredX[i:last]
            + centrLength
            - 2
            * np.dot(
                X[
                    i:last,
                ],
                cent,
            )
        )
        if previous_partition is not None:
            d = d / sds + sds
            tmp = d.argmin(axis=1)
        else:
            tmp = d.argmin(axis=1)
            for j, node in enumerate(nodes):
                node_data = X[tmp == node]
                if len(node_data) < 2:
                    # adding epsilon for clusters with single point or no point
                    sds[j] = 1e-100
                else:
                    sds[j] = np.std(node_data)
            d = d / sds + sds
            tmp = d.argmin(axis=1)

        partition[i:last] = tmp[:, np.newaxis]
        dists[i:last] = d[np.arange(d.shape[0]), tmp][:, np.newaxis]
    # Apply trimming
    if not np.isinf(TrimmingRadius):
        ind = dists > (TrimmingRadius ** 2)
        partition[ind] = -1
        dists[ind] = TrimmingRadius ** 2
    return partition, dists


def MakeUniformElasticMatrix(Edges, Lambda, Mu):
    """
    # Base function: Function to deal with elastic matrices --------------------------
    Create a uniform elastic matrix from a set of edges

    Edges an e-by-2 matrix containing the index of the edges connecting the nodes
    Lambda the lambda parameter. It can be a real value or a vector of lenght e
    Mu the mu parameter. It can be a real value or a vector with a length equal to the number of nodes

    Return
    -------
    the elastic matrix



    @examples
    """
    NumberOfNodes = Edges.max() + 1
    ElasticMatrix = np.zeros((NumberOfNodes, NumberOfNodes))
    for i in range(Edges.shape[0]):
        ElasticMatrix[Edges[i][0], Edges[i][1]] = Lambda
        ElasticMatrix[Edges[i][1], Edges[i][0]] = Lambda
    Connect = (ElasticMatrix > 0).sum(axis=0)
    ind = Connect > 1
    Mus = np.zeros((NumberOfNodes, 1))
    Mus[ind] = Mu
    ElasticMatrix = ElasticMatrix + np.diag(Mus.ravel())
    return ElasticMatrix


def MakeUniformElasticMatrix_with_cycle(
    Edges, Lambda, Mu, cycle_Lambda, cycle_Mu, cycle_nodes
):

    ElasticMatrix = MakeUniformElasticMatrix(Edges, Lambda=Lambda, Mu=Mu)
    for ei, ej in Edges:
        if ei in cycle_nodes or ej in cycle_nodes:
            ElasticMatrix[ei, ej] = ElasticMatrix[ej, ei] = cycle_Lambda
    ElasticMatrix[cycle_nodes, cycle_nodes] = cycle_Mu
    return ElasticMatrix


def Encode2ElasticMatrix(Edges, Lambdas, Mus):
    """
    Create an Elastic matrix from a set of edges

    Lambdas the lambda parameters. Either a single value (which will be used for all the edges),
    or a vector containing the values for each edge
    Mus the mu parameters. Either a single value (which will be used for all the nodes),
    or a vector containing the values for each node
    Edges an e-by-2 matrix containing the index of the edges connecting the nodes

    Return
    -------
    the elastic matrix



    @examples"""

    NumberOfNodes = np.max(Edges) + 1
    NumberOfEdges = Edges.shape[0]

    EM = np.zeros((NumberOfNodes, NumberOfNodes))

    if isinstance(Lambdas, int) or isinstance(Lambdas, float):
        Lambdas = np.array([Lambdas] * NumberOfEdges)

    if isinstance(Mus, int) or isinstance(Mus, float):
        Mus = np.array([Mus] * NumberOfNodes)

    for i in range(NumberOfEdges):
        EM[Edges[i, 0], Edges[i, 1]] = Lambdas[i]
        EM[Edges[i, 1], Edges[i, 0]] = Lambdas[i]

    return EM + np.diag(Mus)


@nb.njit(cache=True)
def ComputeSpringLaplacianMatrix(ElasticMatrix):
    NumberOfNodes = ElasticMatrix.shape[0]
    # first, make the vector of mu coefficients
    Mu = np.diag(ElasticMatrix)
    # create the matrix with edge elasticity moduli at non-diagonal elements
    Lambda = ElasticMatrix - np.diag(Mu)
    # set mu of endpoints to 0.
    # endpoints = np.where((np.bincount(np.concatenate(Lambda.nonzero())) // 2) == 1)[0]
    # Mu[endpoints] = 0.0
    # Diagonal matrix of edge elasticities
    LambdaSums = Lambda.sum(axis=0)
    # E matrix (contribution from edges) is simply weighted Laplacian
    E = np.diag(LambdaSums) - Lambda
    # matrix S (contribution from stars) is composed of Laplacian for
    # positive strings (star edges) with elasticities mu/k, where k is the
    # order of the star, and Laplacian for negative strings with
    # elasticities -mu/k^2. Negative springs connect all star leafs in a
    # clique.
    StarCenterIndices = (Mu > 0).nonzero()[0]
    S = np.zeros((NumberOfNodes, NumberOfNodes))
    for i in range(StarCenterIndices.size):
        Spart = np.zeros((NumberOfNodes, NumberOfNodes))
        # leaves indices
        leaves = Lambda[StarCenterIndices[i]] > 0
        ind_leaves = (leaves).nonzero()[0]
        # order of the star
        K = leaves.sum()
        Spart[StarCenterIndices[i], StarCenterIndices[i]] = Mu[
            StarCenterIndices[i]
        ]
        for j in ind_leaves:
            Spart[StarCenterIndices[i], j] = -Mu[StarCenterIndices[i]] / K
            Spart[j, StarCenterIndices[i]] = -Mu[StarCenterIndices[i]] / K
        tmp = np.repeat(leaves, ElasticMatrix.shape[0]).reshape(
            (-1, ElasticMatrix.shape[0])
        )
        tmp_r, tmp_c = np.where(np.logical_and(tmp, tmp.transpose()))
        K2 = K ** 2
        for it in range(len(tmp_r)):
            Spart[tmp_r[it], tmp_c[it]] = Mu[StarCenterIndices[i]] / K2
        S = S + Spart
    return E + S


def DecodeElasticMatrix(ElasticMatrix):
    """
    Converts ElasticMatrix into a set of edges and vectors of elasticities for edges and stars

    ElasticMatrix an e-by-e elastic matrix

    Return
    -------
    a list with three elements: a matrix with the edges (Edges), a vector of lambdas (Lambdas), and a vector of Mus (Mus)



    @examples
    """
    Mus = ElasticMatrix.diagonal()
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)

    Edges_i, Edges_j = np.where(Lambda > 0)
    Edges = np.concatenate((Edges_j[:, None], Edges_i[:, None]), axis=1)
    inds = np.where(Edges_i > Edges_j)

    Edges = Edges[inds]

    Lambdas = Lambda[Edges[:, 0], Edges[:, 1]]

    return Edges, Lambdas, Mus


def DecodeElasticMatrix2(ElasticMatrix):
    Mus = ElasticMatrix.diagonal()
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)

    Edges_i, Edges_j = np.where(Lambda > 0)
    Edges = np.concatenate((Edges_i[:, None], Edges_j[:, None]), axis=1)
    inds = np.where(Edges_i < Edges_j)

    Edges = Edges[inds]

    Lambdas = Lambda[Edges[:, 0], Edges[:, 1]]

    return Edges, Lambdas, Mus


@nb.njit(cache=True)
def ComputeRelativeChangeOfNodePositions(NodePositions, NewNodePositions):
    """
    Estimates the relative difference between two node configurations

    NodePositions a k-by-m numeric matrix with the coordiantes of the nodes in the old configuration
    NewNodePositions a k-by-m numeric matrix with the coordiantes of the nodes in the new configuration
    Mode an integer indicating the modality used to compute the difference (currently only 1 is an accepted value)
    X an n-by-m numeric matrix with the coordinates of the data points

    @return


    @examples
    """
    return (
        ((NodePositions - NewNodePositions) ** 2).sum(axis=1)
        / (NewNodePositions ** 2).sum(axis=1)
    ).max()


def PrimitiveElasticGraphBarycentricEmbedment(
    X,
    NodePositions,
    ElasticMatrix,
    MaxNumberOfIterations=10,
    eps=0.01,
    Mode=1,
    FinalEnergy="Base",
    alpha=0,
    beta=0,
    prob=1,
    DisplayWarnings=True,
    PointWeights=None,
    MaxBlockSize=100000000,
    verbose=False,
    TrimmingRadius=float("inf"),
    SquaredX=None,
    FixNodesAtPoints=[],
):

    """
    Function fitting a primitive elastic graph to the data

    X is n-by-m matrix containing the positions of the n points in the m-dimensional space
    NodePositions is k-by-m matrix of positions of the graph nodes in the same space as X
    ElasticMatrix is a k-by-k symmetric matrix describing the connectivity and the elastic
    properties of the graph. Star elasticities (mu coefficients) are along the main diagonal
    (non-zero entries only for star centers), and the edge elasticity moduli are at non-diagonal elements.
    MaxNumberOfIterations is an integer number indicating the maximum number of iterations for the EM algorithm
    TrimmingRadius is a real value indicating the trimming radius, a parameter required for robust principal graphs
    (see https://github.com/auranic/Elastic-principal-graphs/wiki/Robust-principal-graphs)
    eps a real number indicating the minimal relative change in the nodenpositions
    to be considered the graph embedded (convergence criteria)
    verbose is a boolean indicating whether diagnostig informations should be plotted
    Mode integer, the energy mode. It can be 1 (difference is computed using the position of the nodes) and
    2 (difference is computed using the changes in elestic energy of the configuraztions)
    SquaredX the sum (by node) of X squared. It not specified, it will be calculated by the fucntion
    FastSolve boolean, shuold the Fastsolve of Armadillo by enabled?
    DisplayWarnings boolean, should warning about convergence be displayed?
    FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    beta positive numeric, the value of the beta parameter of the penalized elastic energy
    prob numeric between 0 and 1. If less than 1 point will be sampled at each iteration. Prob indicate the probability of
    using each points. This is an *experimental* feature, which may helps speeding up the computation if a large number of points is present.

    @return


    @examples
    """

    if prob < 1:
        raise ValueError("probPoint < 1 option not implemented yet")
    N = X.shape[0]

    # Auxiliary computations
    if PointWeights is None:
        PointWeights = np.ones((N, 1))
    if SquaredX is None:
        SquaredX = (X ** 2).sum(axis=1).reshape((N, 1))
    SpringLaplacianMatrix = ComputeSpringLaplacianMatrix(ElasticMatrix)

    if FixNodesAtPoints != []:
        # -----Index data and graph into moving nodes and fixed nodes------#
        flat_FixNodesAtPoints = [
            item for sublist in FixNodesAtPoints for item in sublist
        ]  # fixed datapoints

        # partition, dists = PartitionData(
        #    X, NodePositions, MaxBlockSize, SquaredX, TrimmingRadius,
        # )
        # move_data_idx = np.where(~np.isin(partition, range(len(FixNodesAtPoints))))[0]

        move_data_idx = [
            i for i in range(len(X)) if i not in flat_FixNodesAtPoints
        ]
        move_nodes_idx = np.arange(len(FixNodesAtPoints), len(NodePositions))
    else:
        move_data_idx = np.arange(len(X))
        move_nodes_idx = np.arange(len(NodePositions))

    # fitted subset
    move_X, move_PointWeights, move_SquaredX, move_NodePositions = (
        X[move_data_idx, :],
        PointWeights[move_data_idx],
        SquaredX[move_data_idx],
        NodePositions[move_nodes_idx],
    )

    # Main iterative EM cycle: partition, fit given the partition, repeat
    move_partition, move_dists = PartitionDataBarycenter(
        move_X,
        move_NodePositions,
        MaxBlockSize,
        move_SquaredX,
        TrimmingRadius,
        previous_partition=None,
    )

    if verbose or Mode == 2:
        # if FixNodesAtPoints != []:
        #    dists = np.concatenate((FixedNodesDists, move_dists))
        OldElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
            NodePositions, ElasticMatrix, move_dists
        )

    ElasticEnergy = 0
    for i in range(MaxNumberOfIterations):
        # Updated positions
        if FixNodesAtPoints != []:
            NewNodePositions = FitSubGraph2DataGivenPartition(
                move_X,
                move_PointWeights,
                SpringLaplacianMatrix,
                NodePositions,
                move_partition,
                move_nodes_idx,
            )
        else:
            NewNodePositions = FitGraph2DataGivenPartition(
                move_X,
                move_PointWeights,
                SpringLaplacianMatrix,
                move_partition,
            )

        # Look at differences
        if verbose or Mode == 2:
            # if FixNodesAtPoints != []:
            #    dists = np.concatenate((FixedNodesDists, move_dists))
            ElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
                NewNodePositions, ElasticMatrix, move_dists
            )

        if Mode == 1:
            diff = ComputeRelativeChangeOfNodePositions(
                NodePositions, NewNodePositions
            )
        elif Mode == 2:
            diff = (OldElasticEnergy - ElasticEnergy) / ElasticEnergy

        # Print Info
        if verbose:
            print(
                "Iteration ",
                (i + 1),
                " difference of node position=",
                np.around(diff, 5),
                ", Energy=",
                np.around(ElasticEnergy, 5),
                ", MSE=",
                np.around(MSE, 5),
                ", EP=",
                np.around(EP, 5),
                ", RP=",
                np.around(RP, 5),
            )

        # Have we converged?
        if not np.isfinite(diff):
            diff = 0
        if diff < eps:
            break

        elif i < MaxNumberOfIterations - 1:
            move_NodePositions = NewNodePositions[move_nodes_idx]
            move_partition, move_dists = PartitionDataBarycenter(
                move_X,
                move_NodePositions,
                MaxBlockSize,
                move_SquaredX,
                TrimmingRadius,
                previous_partition=move_partition,
            )

            NodePositions = NewNodePositions
            OldElasticEnergy = ElasticEnergy

    if DisplayWarnings and not (diff < eps):
        print(
            "Maximum number of iterations (",
            MaxNumberOfIterations,
            ") has been reached. diff = ",
            diff,
        )

    # If we
    # 1) Didn't use use energy during the embedment
    # 2) Didn't compute energy step by step due to verbose being false
    # or
    # 3) FinalEnergy != "Penalized"

    if FixNodesAtPoints != []:
        partition, dists = PartitionData(
            X,
            NewNodePositions,
            MaxBlockSize,
            SquaredX,
            TrimmingRadius,
        )
    else:
        partition, dists = move_partition, move_dists
    # if FixNodesAtPoints != []:
    #    move_partition[move_partition >= 0] += len(FixNodesAtPoints)
    #    partition = np.concatenate((FixedNodesPart, move_partition))
    #    dists = np.concatenate((FixedNodesDists, move_dists))

    if (FinalEnergy != "Base") or (not (verbose) and (Mode != 2)):
        if FinalEnergy == "Base":
            ElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
                NewNodePositions, ElasticMatrix, dists
            )

        elif FinalEnergy == "Penalized":
            (
                ElasticEnergy,
                MSE,
                EP,
                RP,
            ) = ComputePenalizedPrimitiveGraphElasticEnergy(
                NewNodePositions, ElasticMatrix, dists, alpha, beta
            )

    return NewNodePositions, ElasticEnergy, partition, dists, MSE, EP, RP


def PrimitiveElasticGraphEmbedment(
    X,
    NodePositions,
    ElasticMatrix,
    MaxNumberOfIterations=10,
    eps=0.01,
    Mode=1,
    FinalEnergy="Base",
    alpha=0,
    beta=0,
    prob=1,
    DisplayWarnings=True,
    PointWeights=None,
    MaxBlockSize=100000000,
    verbose=False,
    TrimmingRadius=float("inf"),
    SquaredX=None,
    FixNodesAtPoints=[],
):

    """
    Function fitting a primitive elastic graph to the data

    X is n-by-m matrix containing the positions of the n points in the m-dimensional space
    NodePositions is k-by-m matrix of positions of the graph nodes in the same space as X
    ElasticMatrix is a k-by-k symmetric matrix describing the connectivity and the elastic
    properties of the graph. Star elasticities (mu coefficients) are along the main diagonal
    (non-zero entries only for star centers), and the edge elasticity moduli are at non-diagonal elements.
    MaxNumberOfIterations is an integer number indicating the maximum number of iterations for the EM algorithm
    TrimmingRadius is a real value indicating the trimming radius, a parameter required for robust principal graphs
    (see https://github.com/auranic/Elastic-principal-graphs/wiki/Robust-principal-graphs)
    eps a real number indicating the minimal relative change in the nodenpositions
    to be considered the graph embedded (convergence criteria)
    verbose is a boolean indicating whether diagnostig informations should be plotted
    Mode integer, the energy mode. It can be 1 (difference is computed using the position of the nodes) and
    2 (difference is computed using the changes in elestic energy of the configuraztions)
    SquaredX the sum (by node) of X squared. It not specified, it will be calculated by the fucntion
    FastSolve boolean, shuold the Fastsolve of Armadillo by enabled?
    DisplayWarnings boolean, should warning about convergence be displayed?
    FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    beta positive numeric, the value of the beta parameter of the penalized elastic energy
    prob numeric between 0 and 1. If less than 1 point will be sampled at each iteration. Prob indicate the probability of
    using each points. This is an *experimental* feature, which may helps speeding up the computation if a large number of points is present.

    @return


    @examples
    """

    if prob < 1:
        raise ValueError("probPoint < 1 option not implemented yet")
    N = X.shape[0]

    # Auxiliary computations
    if PointWeights is None:
        PointWeights = np.ones((N, 1))
    if SquaredX is None:
        SquaredX = (X ** 2).sum(axis=1).reshape((N, 1))
    SpringLaplacianMatrix = ComputeSpringLaplacianMatrix(ElasticMatrix)

    if FixNodesAtPoints != []:
        # -----Index data and graph into moving nodes and fixed nodes------#
        flat_FixNodesAtPoints = [
            item for sublist in FixNodesAtPoints for item in sublist
        ]  # fixed datapoints

        # partition, dists = PartitionData(
        #    X, NodePositions, MaxBlockSize, SquaredX, TrimmingRadius,
        # )
        # move_data_idx = np.where(~np.isin(partition, range(len(FixNodesAtPoints))))[0]

        move_data_idx = [
            i for i in range(len(X)) if i not in flat_FixNodesAtPoints
        ]
        move_nodes_idx = np.arange(len(FixNodesAtPoints), len(NodePositions))
    else:
        move_data_idx = np.arange(len(X))
        move_nodes_idx = np.arange(len(NodePositions))

    # fitted subset
    move_X, move_PointWeights, move_SquaredX, move_NodePositions = (
        X[move_data_idx, :],
        PointWeights[move_data_idx],
        SquaredX[move_data_idx],
        NodePositions[move_nodes_idx],
    )

    # Main iterative EM cycle: partition, fit given the partition, repeat
    move_partition, move_dists = PartitionData(
        move_X,
        move_NodePositions,
        MaxBlockSize,
        move_SquaredX,
        TrimmingRadius,
    )

    if verbose or Mode == 2:
        # if FixNodesAtPoints != []:
        #    dists = np.concatenate((FixedNodesDists, move_dists))
        OldElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
            NodePositions, ElasticMatrix, move_dists
        )

    ElasticEnergy = 0
    for i in range(MaxNumberOfIterations):
        # Updated positions
        if FixNodesAtPoints != []:
            NewNodePositions = FitSubGraph2DataGivenPartition(
                move_X,
                move_PointWeights,
                SpringLaplacianMatrix,
                NodePositions,
                move_partition,
                move_nodes_idx,
            )
        else:
            NewNodePositions = FitGraph2DataGivenPartition(
                move_X,
                move_PointWeights,
                SpringLaplacianMatrix,
                move_partition,
            )

        # Look at differences
        if verbose or Mode == 2:
            # if FixNodesAtPoints != []:
            #    dists = np.concatenate((FixedNodesDists, move_dists))
            ElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
                NewNodePositions, ElasticMatrix, move_dists, move_PointWeights
            )

        if Mode == 1:
            diff = ComputeRelativeChangeOfNodePositions(
                NodePositions, NewNodePositions
            )
        elif Mode == 2:
            diff = (OldElasticEnergy - ElasticEnergy) / ElasticEnergy

        # Print Info
        if verbose:
            print(
                "Iteration ",
                (i + 1),
                " difference of node position=",
                np.around(diff, 5),
                ", Energy=",
                np.around(ElasticEnergy, 5),
                ", MSE=",
                np.around(MSE, 5),
                ", EP=",
                np.around(EP, 5),
                ", RP=",
                np.around(RP, 5),
            )

        # Have we converged?
        if not np.isfinite(diff):
            diff = 0
        if diff < eps:
            break

        elif i < MaxNumberOfIterations - 1:
            move_NodePositions = NewNodePositions[move_nodes_idx]
            move_partition, move_dists = PartitionData(
                move_X,
                move_NodePositions,
                MaxBlockSize,
                move_SquaredX,
                TrimmingRadius,
            )

            NodePositions = NewNodePositions
            OldElasticEnergy = ElasticEnergy

    if DisplayWarnings and not (diff < eps):
        print(
            "Maximum number of iterations (",
            MaxNumberOfIterations,
            ") has been reached. diff = ",
            diff,
        )

    # If we
    # 1) Didn't use use energy during the embedment
    # 2) Didn't compute energy step by step due to verbose being false
    # or
    # 3) FinalEnergy != "Penalized"

    if FixNodesAtPoints != []:
        partition, dists = PartitionData(
            X,
            NewNodePositions,
            MaxBlockSize,
            SquaredX,
            TrimmingRadius,
        )
    else:
        partition, dists = move_partition, move_dists
    # if FixNodesAtPoints != []:
    #    move_partition[move_partition >= 0] += len(FixNodesAtPoints)
    #    partition = np.concatenate((FixedNodesPart, move_partition))
    #    dists = np.concatenate((FixedNodesDists, move_dists))

    if (FinalEnergy != "Base") or (not (verbose) and (Mode != 2)):
        if FinalEnergy == "Base":
            ElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
                NewNodePositions, ElasticMatrix, dists, PointWeights
            )

        elif FinalEnergy == "Penalized":
            (
                ElasticEnergy,
                MSE,
                EP,
                RP,
            ) = ComputePenalizedPrimitiveGraphElasticEnergy_v2(
                NewNodePositions,
                ElasticMatrix,
                dists,
                alpha,
                beta,
                PointWeights=PointWeights,
            )

    return NewNodePositions, ElasticEnergy, partition, dists, MSE, EP, RP


def PrimitiveElasticGraphEmbedment_cp(
    X,
    NodePositions,
    ElasticMatrix,
    MaxNumberOfIterations=10,
    eps=0.01,
    Mode=1,
    FinalEnergy="Base",
    alpha=0,
    beta=0,
    prob=1,
    DisplayWarnings=True,
    PointWeights=None,
    MaxBlockSize=100000000,
    verbose=False,
    TrimmingRadius=float("inf"),
    SquaredX=None,
    Xcp=None,
    SquaredXcp=None,
    FixNodesAtPoints=[],
):

    """
    Function fitting a primitive elastic graph to the data

    X is n-by-m matrix containing the positions of the n points in the m-dimensional space
    NodePositions is k-by-m matrix of positions of the graph nodes in the same space as X
    ElasticMatrix is a k-by-k symmetric matrix describing the connectivity and the elastic
    properties of the graph. Star elasticities (mu coefficients) are along the main diagonal
    (non-zero entries only for star centers), and the edge elasticity moduli are at non-diagonal elements.
    MaxNumberOfIterations is an integer number indicating the maximum number of iterations for the EM algorithm
    TrimmingRadius is a real value indicating the trimming radius, a parameter required for robust principal graphs
    (see https://github.com/auranic/Elastic-principal-graphs/wiki/Robust-principal-graphs)
    eps a real number indicating the minimal relative change in the nodenpositions
    to be considered the graph embedded (convergence criteria)
    verbose is a boolean indicating whether diagnostig informations should be plotted
    Mode integer, the energy mode. It can be 1 (difference is computed using the position of the nodes) and
    2 (difference is computed using the changes in elestic energy of the configuraztions)
    SquaredX the sum (by node) of X squared. It not specified, it will be calculated by the fucntion
    FastSolve boolean, shuold the Fastsolve of Armadillo by enabled?
    DisplayWarnings boolean, should warning about convergence be displayed?
    FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    beta positive numeric, the value of the beta parameter of the penalized elastic energy
    prob numeric between 0 and 1. If less than 1 point will be sampled at each iteration. Prob indicate the probability of
    using each points. This is an *experimental* feature, which may helps speeding up the computation if a large number of points is present.

    @return


    @examples
    """

    if prob < 1:
        raise ValueError("probPoint < 1 option not implemented yet")
    N = X.shape[0]

    # Auxiliary computations
    if PointWeights is None:
        PointWeights = np.ones((N, 1))

    SpringLaplacianMatrix = ComputeSpringLaplacianMatrix(ElasticMatrix)

    if FixNodesAtPoints != []:
        # -----Index data and graph into moving nodes and fixed nodes------#
        flat_FixNodesAtPoints = [
            item for sublist in FixNodesAtPoints for item in sublist
        ]  # fixed datapoints

        # partition, dists = PartitionData(
        #    X, NodePositions, MaxBlockSize, SquaredX, TrimmingRadius,
        # )
        # move_data_idx = np.where(~np.isin(partition, range(len(FixNodesAtPoints))))[0]

        move_data_idx = np.array(
            [i for i in range(len(X)) if i not in flat_FixNodesAtPoints]
        )
        move_nodes_idx = np.arange(len(FixNodesAtPoints), len(NodePositions))
    else:
        move_data_idx = np.arange(len(X))
        move_nodes_idx = np.arange(len(NodePositions))

    # fitted subset
    (
        move_X,
        move_Xcp,
        move_PointWeights,
        move_SquaredXcp,
        move_NodePositions,
    ) = (
        X[move_data_idx, :],
        Xcp[move_data_idx, :],
        PointWeights[move_data_idx],
        SquaredXcp[move_data_idx],
        NodePositions[move_nodes_idx],
    )

    # Main iterative EM cycle: partition, fit given the partition, repeat
    move_partition, move_dists = PartitionData_cp(
        move_Xcp,
        move_NodePositions,
        MaxBlockSize,
        move_SquaredXcp,
        TrimmingRadius,
    )

    if verbose or Mode == 2:
        # if FixNodesAtPoints != []:
        #    dists = np.concatenate((FixedNodesDists, move_dists))
        OldElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
            NodePositions, ElasticMatrix, move_dists
        )

    ElasticEnergy = 0
    for i in range(MaxNumberOfIterations):
        # Updated positions
        if FixNodesAtPoints != []:
            NewNodePositions = FitSubGraph2DataGivenPartition(
                move_X,
                move_PointWeights,
                SpringLaplacianMatrix,
                NodePositions,
                move_partition,
                move_nodes_idx,
            )
        else:
            NewNodePositions = FitGraph2DataGivenPartition(
                move_X,
                move_PointWeights,
                SpringLaplacianMatrix,
                move_partition,
            )

        # Look at differences
        if verbose or Mode == 2:
            # if FixNodesAtPoints != []:
            #    dists = np.concatenate((FixedNodesDists, move_dists))
            ElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
                NewNodePositions, ElasticMatrix, move_dists, move_PointWeights
            )

        if Mode == 1:
            diff = ComputeRelativeChangeOfNodePositions(
                NodePositions, NewNodePositions
            )
        elif Mode == 2:
            diff = (OldElasticEnergy - ElasticEnergy) / ElasticEnergy

        # Print Info
        if verbose:
            print(
                "Iteration ",
                (i + 1),
                " difference of node position=",
                np.around(diff, 5),
                ", Energy=",
                np.around(ElasticEnergy, 5),
                ", MSE=",
                np.around(MSE, 5),
                ", EP=",
                np.around(EP, 5),
                ", RP=",
                np.around(RP, 5),
            )

        # Have we converged?
        if not np.isfinite(diff):
            diff = 0
        if diff < eps:
            break

        elif i < MaxNumberOfIterations - 1:
            move_NodePositions = NewNodePositions[move_nodes_idx]
            move_partition, move_dists = PartitionData_cp(
                move_Xcp,
                move_NodePositions,
                MaxBlockSize,
                move_SquaredXcp,
                TrimmingRadius,
            )

            NodePositions = NewNodePositions
            OldElasticEnergy = ElasticEnergy

    if DisplayWarnings and not (diff < eps):
        print(
            "Maximum number of iterations (",
            MaxNumberOfIterations,
            ") has been reached. diff = ",
            diff,
        )

    # If we
    # 1) Didn't use use energy during the embedment
    # 2) Didn't compute energy step by step due to verbose being false
    # or
    # 3) FinalEnergy != "Penalized"

    if FixNodesAtPoints != []:
        partition, dists = PartitionData_cp(
            Xcp,
            NewNodePositions,
            MaxBlockSize,
            SquaredXcp,
            TrimmingRadius,
        )
    else:
        partition, dists = move_partition, move_dists
    # if FixNodesAtPoints != []:
    #    move_partition[move_partition >= 0] += len(FixNodesAtPoints)
    #    partition = np.concatenate((FixedNodesPart, move_partition))
    #    dists = np.concatenate((FixedNodesDists, move_dists))

    if (FinalEnergy != "Base") or (not (verbose) and (Mode != 2)):
        if FinalEnergy == "Base":
            ElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
                NewNodePositions, ElasticMatrix, dists, PointWeights
            )

        elif FinalEnergy == "Penalized":
            (
                ElasticEnergy,
                MSE,
                EP,
                RP,
            ) = ComputePenalizedPrimitiveGraphElasticEnergy_v2(
                NewNodePositions,
                ElasticMatrix,
                dists,
                alpha,
                beta,
                PointWeights=PointWeights,
            )

    return NewNodePositions, ElasticEnergy, partition, dists, MSE, EP, RP


def PrimitiveElasticGraphEmbedment_v2(
    X,
    NodePositions,
    ElasticMatrix,
    MaxNumberOfIterations=10,
    eps=0.01,
    Mode=1,
    FinalEnergy="Base",
    alpha=0,
    beta=0,
    prob=1,
    DisplayWarnings=True,
    PointWeights=None,
    MaxBlockSize=100000000,
    verbose=False,
    TrimmingRadius=float("inf"),
    SquaredX=None,
    FixNodesAtPoints=[],
    PseudotimeNodePositions=None,
    PseudotimeLambda=None,
    partition=None,
    dists=None,
    precomp_d=None,
):

    """
    Function fitting a primitive elastic graph to the data

    X is n-by-m matrix containing the positions of the n points in the m-dimensional space
    NodePositions is k-by-m matrix of positions of the graph nodes in the same space as X
    ElasticMatrix is a k-by-k symmetric matrix describing the connectivity and the elastic
    properties of the graph. Star elasticities (mu coefficients) are along the main diagonal
    (non-zero entries only for star centers), and the edge elasticity moduli are at non-diagonal elements.
    MaxNumberOfIterations is an integer number indicating the maximum number of iterations for the EM algorithm
    TrimmingRadius is a real value indicating the trimming radius, a parameter required for robust principal graphs
    (see https://github.com/auranic/Elastic-principal-graphs/wiki/Robust-principal-graphs)
    eps a real number indicating the minimal relative change in the nodenpositions
    to be considered the graph embedded (convergence criteria)
    verbose is a boolean indicating whether diagnostig informations should be plotted
    Mode integer, the energy mode. It can be 1 (difference is computed using the position of the nodes) and
    2 (difference is computed using the changes in elestic energy of the configuraztions)
    SquaredX the sum (by node) of X squared. It not specified, it will be calculated by the fucntion
    FastSolve boolean, shuold the Fastsolve of Armadillo by enabled?
    DisplayWarnings boolean, should warning about convergence be displayed?
    FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    beta positive numeric, the value of the beta parameter of the penalized elastic energy
    prob numeric between 0 and 1. If less than 1 point will be sampled at each iteration. Prob indicate the probability of
    using each points. This is an *experimental* feature, which may helps speeding up the computation if a large number of points is present.

    @return


    @examples
    """

    if prob < 1:
        raise ValueError("probPoint < 1 option not implemented yet")
    N = X.shape[0]

    # Auxiliary computations
    if PointWeights is None:
        PointWeights = np.ones((N, 1))
    if SquaredX is None:
        SquaredX = (X ** 2).sum(axis=1).reshape((N, 1))
    SpringLaplacianMatrix = ComputeSpringLaplacianMatrix(ElasticMatrix)

    if FixNodesAtPoints != []:
        # -----Index data and graph into moving nodes and fixed nodes------#
        # flat_FixNodesAtPoints = [
        #    item for sublist in FixNodesAtPoints for item in sublist
        # ]  # fixed datapoints

        # partition, dists = PartitionData(
        #    X,
        #    NodePositions,
        #    MaxBlockSize,
        #    SquaredX,
        #    TrimmingRadius,
        # )
        move_data_idx = np.where(
            ~np.isin(partition, range(len(FixNodesAtPoints)))
        )[0]

        # move_data_idx = [i for i in range(len(X)) if i not in flat_FixNodesAtPoints]
        move_nodes_idx = np.arange(len(FixNodesAtPoints), len(NodePositions))
        partition, dists, _ = RePartitionData(
            None,
            None,
            None,
            NodeIndicesArray=move_nodes_idx,
            opType="removenode",
            precomp_d=precomp_d,
            SquaredX=None,
        )
        move_partition, move_dists = (
            partition[move_data_idx],
            dists[move_data_idx],
        )
    else:
        move_data_idx = np.arange(len(X))
        move_nodes_idx = np.arange(len(NodePositions))
        move_partition = partition
        move_dists = dists

    # fitted subset
    move_X, move_PointWeights, move_SquaredX, move_NodePositions = (
        X[move_data_idx, :],
        PointWeights[move_data_idx],
        SquaredX[move_data_idx],
        NodePositions[move_nodes_idx],
    )

    if verbose or Mode == 2:
        # if FixNodesAtPoints != []:
        #    dists = np.concatenate((FixedNodesDists, move_dists))
        OldElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
            NodePositions, ElasticMatrix, move_dists
        )

    ElasticEnergy = 0
    for i in range(MaxNumberOfIterations):
        # Updated positions
        if FixNodesAtPoints != []:
            NewNodePositions = FitSubGraph2DataGivenPartition_v2(
                move_X,
                move_PointWeights,
                SpringLaplacianMatrix,
                NodePositions,
                move_partition,
                move_nodes_idx,
                PseudotimeNodePositions=PseudotimeNodePositions,
                PseudotimeLambda=PseudotimeLambda,
            )
        else:
            NewNodePositions = FitGraph2DataGivenPartition(
                move_X,
                move_PointWeights,
                SpringLaplacianMatrix,
                move_partition,
            )

        # Look at differences
        if verbose or Mode == 2:
            # if FixNodesAtPoints != []:
            #    dists = np.concatenate((FixedNodesDists, move_dists))
            ElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
                NewNodePositions, ElasticMatrix, move_dists
            )

        if Mode == 1:
            diff = ComputeRelativeChangeOfNodePositions(
                NodePositions, NewNodePositions
            )
        elif Mode == 2:
            diff = (OldElasticEnergy - ElasticEnergy) / ElasticEnergy

        # Print Info
        if verbose:
            print(
                "Iteration ",
                (i + 1),
                " difference of node position=",
                np.around(diff, 5),
                ", Energy=",
                np.around(ElasticEnergy, 5),
                ", MSE=",
                np.around(MSE, 5),
                ", EP=",
                np.around(EP, 5),
                ", RP=",
                np.around(RP, 5),
            )

        # Have we converged?
        if not np.isfinite(diff):
            diff = 0
        if diff < eps:
            break

        elif i < MaxNumberOfIterations - 1:
            move_NodePositions = NewNodePositions[move_nodes_idx]
            move_partition, move_dists = PartitionData(
                move_X,
                move_NodePositions,
                MaxBlockSize,
                move_SquaredX,
                TrimmingRadius,
            )

            NodePositions = NewNodePositions
            OldElasticEnergy = ElasticEnergy

    if DisplayWarnings and not (diff < eps):
        print(
            "Maximum number of iterations (",
            MaxNumberOfIterations,
            ") has been reached. diff = ",
            diff,
        )

    # If we
    # 1) Didn't use use energy during the embedment
    # 2) Didn't compute energy step by step due to verbose being false
    # or
    # 3) FinalEnergy != "Penalized"

    if FixNodesAtPoints != []:
        partition, dists = PartitionData(
            X,
            NewNodePositions,
            MaxBlockSize,
            SquaredX,
            TrimmingRadius,
        )
    else:
        partition, dists = move_partition, move_dists
    # if FixNodesAtPoints != []:
    #    move_partition[move_partition >= 0] += len(FixNodesAtPoints)
    #    partition = np.concatenate((FixedNodesPart, move_partition))
    #    dists = np.concatenate((FixedNodesDists, move_dists))

    if (FinalEnergy != "Base") or (not (verbose) and (Mode != 2)):
        if FinalEnergy == "Base":
            ElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
                NewNodePositions, ElasticMatrix, dists, PointWeights
            )

        elif FinalEnergy == "Penalized":
            (
                ElasticEnergy,
                MSE,
                EP,
                RP,
            ) = ComputePenalizedPrimitiveGraphElasticEnergy_v2(
                NewNodePositions,
                ElasticMatrix,
                dists,
                alpha,
                beta,
                PointWeights,
                PseudotimeNodePositions,
            )

    return NewNodePositions, ElasticEnergy, partition, dists, MSE, EP, RP


def PrimitiveElasticGraphEmbedment_v3(
    X,
    NodePositions,
    ElasticMatrix,
    MaxNumberOfIterations=10,
    eps=0.01,
    Mode=1,
    FinalEnergy="Base",
    alpha=0,
    beta=0,
    prob=1,
    DisplayWarnings=True,
    PointWeights=None,
    MaxBlockSize=100000000,
    verbose=False,
    TrimmingRadius=float("inf"),
    SquaredX=None,
    FixNodesAtPoints=[],
    PseudotimeNodePositions=None,
    PseudotimeLambda=None,
    Label=None,
    LabelLambda=None,
):

    """
    Function fitting a primitive elastic graph to the data

    X is n-by-m matrix containing the positions of the n points in the m-dimensional space
    NodePositions is k-by-m matrix of positions of the graph nodes in the same space as X
    ElasticMatrix is a k-by-k symmetric matrix describing the connectivity and the elastic
    properties of the graph. Star elasticities (mu coefficients) are along the main diagonal
    (non-zero entries only for star centers), and the edge elasticity moduli are at non-diagonal elements.
    MaxNumberOfIterations is an integer number indicating the maximum number of iterations for the EM algorithm
    TrimmingRadius is a real value indicating the trimming radius, a parameter required for robust principal graphs
    (see https://github.com/auranic/Elastic-principal-graphs/wiki/Robust-principal-graphs)
    eps a real number indicating the minimal relative change in the nodenpositions
    to be considered the graph embedded (convergence criteria)
    verbose is a boolean indicating whether diagnostig informations should be plotted
    Mode integer, the energy mode. It can be 1 (difference is computed using the position of the nodes) and
    2 (difference is computed using the changes in elestic energy of the configuraztions)
    SquaredX the sum (by node) of X squared. It not specified, it will be calculated by the fucntion
    FastSolve boolean, shuold the Fastsolve of Armadillo by enabled?
    DisplayWarnings boolean, should warning about convergence be displayed?
    FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    beta positive numeric, the value of the beta parameter of the penalized elastic energy
    prob numeric between 0 and 1. If less than 1 point will be sampled at each iteration. Prob indicate the probability of
    using each points. This is an *experimental* feature, which may helps speeding up the computation if a large number of points is present.

    @return


    @examples
    """

    if prob < 1:
        raise ValueError("probPoint < 1 option not implemented yet")
    N = X.shape[0]

    # Auxiliary computations
    if PointWeights is None:
        PointWeights = np.ones((N, 1))
    if SquaredX is None:
        SquaredX = (X ** 2).sum(axis=1).reshape((N, 1))
    SpringLaplacianMatrix = ComputeSpringLaplacianMatrix(ElasticMatrix)

    if FixNodesAtPoints != []:
        # -----Index data and graph into moving nodes and fixed nodes------#
        # flat_FixNodesAtPoints = [
        #    item for sublist in FixNodesAtPoints for item in sublist
        # ]  # fixed datapoints

        partition, dists = PartitionData(
            X,
            NodePositions,
            MaxBlockSize,
            SquaredX,
            TrimmingRadius,
        )
        move_data_idx = np.where(
            ~np.isin(partition, range(len(FixNodesAtPoints)))
        )[0]

        # move_data_idx = [i for i in range(len(X)) if i not in flat_FixNodesAtPoints]
        move_nodes_idx = np.arange(len(FixNodesAtPoints), len(NodePositions))
    else:
        move_data_idx = np.arange(len(X))
        move_nodes_idx = np.arange(len(NodePositions))

    # fitted subset
    move_X, move_PointWeights, move_SquaredX, move_NodePositions = (
        X[move_data_idx, :],
        PointWeights[move_data_idx],
        SquaredX[move_data_idx],
        NodePositions[move_nodes_idx],
    )

    # Main iterative EM cycle: partition, fit given the partition, repeat
    move_partition, move_dists = PartitionData(
        move_X,
        move_NodePositions,
        MaxBlockSize,
        move_SquaredX,
        TrimmingRadius,
    )

    if verbose or Mode == 2:
        # if FixNodesAtPoints != []:
        #    dists = np.concatenate((FixedNodesDists, move_dists))
        OldElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
            NodePositions, ElasticMatrix, move_dists
        )

    ElasticEnergy = 0
    for i in range(MaxNumberOfIterations):
        # Updated positions
        if FixNodesAtPoints != []:
            NewNodePositions = FitSubGraph2DataGivenPartition_v2(
                move_X,
                move_PointWeights,
                SpringLaplacianMatrix,
                NodePositions,
                move_partition,
                move_nodes_idx,
                PseudotimeNodePositions=PseudotimeNodePositions,
                PseudotimeLambda=PseudotimeLambda,
            )
        else:
            NewNodePositions = FitGraph2DataGivenPartition(
                move_X,
                move_PointWeights,
                SpringLaplacianMatrix,
                move_partition,
            )

        # Look at differences
        if verbose or Mode == 2:
            # if FixNodesAtPoints != []:
            #    dists = np.concatenate((FixedNodesDists, move_dists))
            ElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
                NewNodePositions, ElasticMatrix, move_dists
            )

        if Mode == 1:
            diff = ComputeRelativeChangeOfNodePositions(
                NodePositions, NewNodePositions
            )
        elif Mode == 2:
            diff = (OldElasticEnergy - ElasticEnergy) / ElasticEnergy

        # Print Info
        if verbose:
            print(
                "Iteration ",
                (i + 1),
                " difference of node position=",
                np.around(diff, 5),
                ", Energy=",
                np.around(ElasticEnergy, 5),
                ", MSE=",
                np.around(MSE, 5),
                ", EP=",
                np.around(EP, 5),
                ", RP=",
                np.around(RP, 5),
            )

        # Have we converged?
        if not np.isfinite(diff):
            diff = 0
        if diff < eps:
            break

        elif i < MaxNumberOfIterations - 1:
            move_NodePositions = NewNodePositions[move_nodes_idx]
            move_partition, move_dists = PartitionData(
                move_X,
                move_NodePositions,
                MaxBlockSize,
                move_SquaredX,
                TrimmingRadius,
            )

            NodePositions = NewNodePositions
            OldElasticEnergy = ElasticEnergy

    if DisplayWarnings and not (diff < eps):
        print(
            "Maximum number of iterations (",
            MaxNumberOfIterations,
            ") has been reached. diff = ",
            diff,
        )

    # If we
    # 1) Didn't use use energy during the embedment
    # 2) Didn't compute energy step by step due to verbose being false
    # or
    # 3) FinalEnergy != "Penalized"

    if FixNodesAtPoints != []:
        partition, dists = PartitionData(
            X,
            NewNodePositions,
            MaxBlockSize,
            SquaredX,
            TrimmingRadius,
        )
    else:
        partition, dists = move_partition, move_dists
    # if FixNodesAtPoints != []:
    #    move_partition[move_partition >= 0] += len(FixNodesAtPoints)
    #    partition = np.concatenate((FixedNodesPart, move_partition))
    #    dists = np.concatenate((FixedNodesDists, move_dists))

    if (FinalEnergy != "Base") or (not (verbose) and (Mode != 2)):
        if FinalEnergy == "Base":
            ElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
                NewNodePositions, ElasticMatrix, dists
            )

        elif FinalEnergy == "Penalized":
            (
                ElasticEnergy,
                MSE,
                EP,
                RP,
            ) = ComputePenalizedPrimitiveGraphElasticEnergy_v3(
                NewNodePositions,
                ElasticMatrix,
                partition.flatten(),
                dists,
                alpha,
                beta,
                PseudotimeNodePositions,
                Label,
                LabelLambda,
            )

    return NewNodePositions, ElasticEnergy, partition, dists, MSE, EP, RP


def PrimitiveElasticGraphEmbedment_cp_v2(
    X,
    NodePositions,
    ElasticMatrix,
    MaxNumberOfIterations=10,
    eps=0.01,
    Mode=1,
    FinalEnergy="Base",
    alpha=0,
    beta=0,
    prob=1,
    DisplayWarnings=True,
    PointWeights=None,
    MaxBlockSize=100000000,
    verbose=False,
    TrimmingRadius=float("inf"),
    SquaredX=None,
    FixNodesAtPoints=[],
    PseudotimeNodePositions=None,
    PseudotimeLambda=None,
    partition=None,
    dists=None,
    precomp_d=None,
    Xcp=None,
    SquaredXcp=None,
):

    """
    Function fitting a primitive elastic graph to the data

    X is n-by-m matrix containing the positions of the n points in the m-dimensional space
    NodePositions is k-by-m matrix of positions of the graph nodes in the same space as X
    ElasticMatrix is a k-by-k symmetric matrix describing the connectivity and the elastic
    properties of the graph. Star elasticities (mu coefficients) are along the main diagonal
    (non-zero entries only for star centers), and the edge elasticity moduli are at non-diagonal elements.
    MaxNumberOfIterations is an integer number indicating the maximum number of iterations for the EM algorithm
    TrimmingRadius is a real value indicating the trimming radius, a parameter required for robust principal graphs
    (see https://github.com/auranic/Elastic-principal-graphs/wiki/Robust-principal-graphs)
    eps a real number indicating the minimal relative change in the nodenpositions
    to be considered the graph embedded (convergence criteria)
    verbose is a boolean indicating whether diagnostig informations should be plotted
    Mode integer, the energy mode. It can be 1 (difference is computed using the position of the nodes) and
    2 (difference is computed using the changes in elestic energy of the configuraztions)
    SquaredX the sum (by node) of X squared. It not specified, it will be calculated by the fucntion
    FastSolve boolean, shuold the Fastsolve of Armadillo by enabled?
    DisplayWarnings boolean, should warning about convergence be displayed?
    FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    beta positive numeric, the value of the beta parameter of the penalized elastic energy
    prob numeric between 0 and 1. If less than 1 point will be sampled at each iteration. Prob indicate the probability of
    using each points. This is an *experimental* feature, which may helps speeding up the computation if a large number of points is present.

    @return


    @examples
    """

    if prob < 1:
        raise ValueError("probPoint < 1 option not implemented yet")
    N = X.shape[0]

    # Auxiliary computations
    if PointWeights is None:
        PointWeights = np.ones((N, 1))
    if SquaredX is None:
        SquaredX = (X ** 2).sum(axis=1).reshape((N, 1))
    SpringLaplacianMatrix = ComputeSpringLaplacianMatrix(ElasticMatrix)

    if FixNodesAtPoints != []:
        # -----Index data and graph into moving nodes and fixed nodes------#
        # flat_FixNodesAtPoints = [
        #    item for sublist in FixNodesAtPoints for item in sublist
        # ]  # fixed datapoints

        # partition, dists = PartitionData(
        #    X,
        #    NodePositions,
        #    MaxBlockSize,
        #    SquaredX,
        #    TrimmingRadius,
        # )
        move_data_idx = np.where(
            ~np.isin(partition, range(len(FixNodesAtPoints)))
        )[0]

        # move_data_idx = [i for i in range(len(X)) if i not in flat_FixNodesAtPoints]
        move_nodes_idx = np.arange(len(FixNodesAtPoints), len(NodePositions))
        partition, dists, _ = RePartitionData(
            None,
            None,
            None,
            NodeIndicesArray=move_nodes_idx,
            opType="removenode",
            precomp_d=precomp_d,
            SquaredX=None,
        )
        move_partition, move_dists = (
            partition[move_data_idx],
            dists[move_data_idx],
        )
    else:
        move_data_idx = np.arange(len(X))
        move_nodes_idx = np.arange(len(NodePositions))
        move_partition = partition
        move_dists = dists

    # fitted subset
    (
        move_X,
        move_Xcp,
        move_PointWeights,
        move_SquaredXcp,
        move_NodePositions,
    ) = (
        X[move_data_idx, :],
        Xcp[move_data_idx, :],
        PointWeights[move_data_idx],
        SquaredXcp[move_data_idx],
        NodePositions[move_nodes_idx],
    )

    if verbose or Mode == 2:
        # if FixNodesAtPoints != []:
        #    dists = np.concatenate((FixedNodesDists, move_dists))
        OldElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
            NodePositions,
            ElasticMatrix,
            move_dists,
            move_PointWeights,
        )

    ElasticEnergy = 0
    for i in range(MaxNumberOfIterations):
        # Updated positions
        if FixNodesAtPoints != []:
            NewNodePositions = FitSubGraph2DataGivenPartition_v2(
                move_X,
                move_PointWeights,
                SpringLaplacianMatrix,
                NodePositions,
                move_partition,
                move_nodes_idx,
                PseudotimeNodePositions=PseudotimeNodePositions,
                PseudotimeLambda=PseudotimeLambda,
            )
        else:
            # NewNodePositions = FitGraph2DataGivenPartition(
            #    move_X,
            #    move_PointWeights,
            #    SpringLaplacianMatrix,
            #    move_partition,
            # )
            NewNodePositions = FitSubGraph2DataGivenPartition_v2(
                move_X,
                move_PointWeights,
                SpringLaplacianMatrix,
                NodePositions,
                move_partition,
                move_nodes_idx,
                PseudotimeNodePositions=PseudotimeNodePositions,
                PseudotimeLambda=PseudotimeLambda,
            )

        # Look at differences
        if verbose or Mode == 2:
            # if FixNodesAtPoints != []:
            #    dists = np.concatenate((FixedNodesDists, move_dists))
            ElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
                NewNodePositions, ElasticMatrix, move_dists, move_PointWeights
            )

        if Mode == 1:
            diff = ComputeRelativeChangeOfNodePositions(
                NodePositions, NewNodePositions
            )
        elif Mode == 2:
            diff = (OldElasticEnergy - ElasticEnergy) / ElasticEnergy

        # Print Info
        if verbose:
            print(
                "Iteration ",
                (i + 1),
                " difference of node position=",
                np.around(diff, 5),
                ", Energy=",
                np.around(ElasticEnergy, 5),
                ", MSE=",
                np.around(MSE, 5),
                ", EP=",
                np.around(EP, 5),
                ", RP=",
                np.around(RP, 5),
            )

        # Have we converged?
        if not np.isfinite(diff):
            diff = 0
        if diff < eps:
            break

        elif i < MaxNumberOfIterations - 1:
            move_NodePositions = NewNodePositions[move_nodes_idx]
            move_partition, move_dists = PartitionData_cp(
                move_Xcp,
                move_NodePositions,
                MaxBlockSize,
                move_SquaredXcp,
                TrimmingRadius,
            )

            NodePositions = NewNodePositions
            OldElasticEnergy = ElasticEnergy

    if DisplayWarnings and not (diff < eps):
        print(
            "Maximum number of iterations (",
            MaxNumberOfIterations,
            ") has been reached. diff = ",
            diff,
        )

    # If we
    # 1) Didn't use use energy during the embedment
    # 2) Didn't compute energy step by step due to verbose being false
    # or
    # 3) FinalEnergy != "Penalized"

    if FixNodesAtPoints != []:
        partition, dists = PartitionData_cp(
            Xcp,
            NewNodePositions,
            MaxBlockSize,
            SquaredXcp,
            TrimmingRadius,
        )
    else:
        partition, dists = move_partition, move_dists
    # if FixNodesAtPoints != []:
    #    move_partition[move_partition >= 0] += len(FixNodesAtPoints)
    #    partition = np.concatenate((FixedNodesPart, move_partition))
    #    dists = np.concatenate((FixedNodesDists, move_dists))

    if (FinalEnergy != "Base") or (not (verbose) and (Mode != 2)):
        if FinalEnergy == "Base":
            ElasticEnergy, MSE, EP, RP = ComputePrimitiveGraphElasticEnergy(
                NewNodePositions,
                ElasticMatrix,
                dists,
                PointWeights,
            )

        elif FinalEnergy == "Penalized":
            (
                ElasticEnergy,
                MSE,
                EP,
                RP,
            ) = ComputePenalizedPrimitiveGraphElasticEnergy_v2(
                NewNodePositions,
                ElasticMatrix,
                dists,
                alpha,
                beta,
                PointWeights,
                PseudotimeNodePositions,
            )

    return NewNodePositions, ElasticEnergy, partition, dists, MSE, EP, RP
