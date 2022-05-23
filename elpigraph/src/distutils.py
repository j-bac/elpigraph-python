import numpy as np
import numba as nb
import igraph
from .graphs import GetSubGraph

# def ComputePrimitiveGraphElasticEnergy(NodePositions, ElasticMatrix, dists):
#     MSE = dists.sum() / np.size(dists)
#     Mu = ElasticMatrix.diagonal()
#     Lambda = np.triu(ElasticMatrix, 1)
#     StarCenterIndices = np.nonzero(Mu > 0)[0]
#     (row, col) = Lambda.nonzero()
#     dev = NodePositions.take(row, axis=0) - NodePositions.take(col, axis=0)
#     L = Lambda[Lambda > 0]
#     EP = sum(L.flatten('F')*np.sum(dev**2, axis=1))
#     indL = Lambda+Lambda.transpose() > 0
#     RP = 0
#     for i in range(np.size(StarCenterIndices)):
#         leaves = indL.take(StarCenterIndices[i], axis=1)
#         K = sum(leaves)
#         dev = (NodePositions.take(StarCenterIndices[i], axis=0) -
#                sum(NodePositions[leaves])/K)
#         RP = RP + Mu[StarCenterIndices[i]] * sum(dev ** 2)
#     ElasticEnergy = MSE + EP + RP
#     return ElasticEnergy, MSE, EP, RP


@nb.njit(cache=True)
def ComputePrimitiveGraphElasticEnergy(
    NodePositions, ElasticMatrix, dists, PointWeights=None
):
    """
        //' Compute the elastic energy associated with a particular configuration
    //'
    //' This function computes the elastic energy associated to a set of points and graph embedded
    //' into them.
    //'
    //' NodePositions A numeric k-by-m matrix containing the position of the k nodes of the embedded graph
    //' ElasticMatrix A numeric l-by-l matrix containing the elastic parameters associates with the edge
    //' of the embedded graph
    //' dists A numeric vector containind the squared distance of the data points to the closest node of the graph
    //'
    //' Return
    -------
    A list with four elements:
    //' * ElasticEnergy is the total energy
    //' * MSE is the MSE component of the energy
    //' * EP is the EP component of the energy
    //' * RP is the RP component of the energy
    """

    MSE = np.sum(dists * PointWeights) / np.sum(PointWeights)

    Mu = np.diag(ElasticMatrix)
    Lambda = np.triu(ElasticMatrix, 1)
    StarCenterIndices = (Mu > 0).nonzero()[0]
    (row, col) = Lambda.nonzero()
    dev = NodePositions[row] - NodePositions[col]

    dev2 = np.sum(dev ** 2, axis=1)
    L = np.zeros((len(row)))
    for i in range(len(row)):
        L[i] = Lambda[row[i], col[i]]
    EP = np.dot(L, dev2)

    indL = (Lambda + Lambda.transpose()) > 0
    RP = 0
    for i in range(StarCenterIndices.size):
        leaves = indL[StarCenterIndices[i]]
        ind_leaves = leaves.nonzero()[0]
        K = ind_leaves.size
        dev_ = NodePositions[StarCenterIndices[i]] - (
            NodePositions[ind_leaves] / K
        ).sum(axis=0)
        RP = RP + Mu[StarCenterIndices[i]] * (dev_ ** 2).sum()
    ElasticEnergy = MSE + EP + RP
    return ElasticEnergy, MSE, EP, RP


def RadialCount(A, B, A_squared, Dvect):
    Dvect = Dvect ** 2
    distances = (
        -2 * np.matmul(A, np.transpose(B))
        + np.sum(np.square(B), axis=1)
        + A_squared
    )

    idx = np.array([np.where(distances < dist)[0] for dist in Dvect])
    count = np.array([len(i) for i in idx])
    return count, idx[-1]


# def ComputePenalizedPrimitiveGraphElasticEnergy(NodePositions, ElasticMatrix, dists,alpha=.1,beta=.1):
#     MSE = dists.sum() / np.size(dists)
#     Mu = ElasticMatrix.diagonal()
#     Lambda = np.triu(ElasticMatrix, 1)
#     StarCenterIndices = np.nonzero(Mu > 0)[0]
#     (row, col) = Lambda.nonzero()
#     dev = NodePositions.take(row, axis=0) - NodePositions.take(col, axis=0)
#     L = Lambda[Lambda > 0]
#     ### diff compared to base function
#     BinEM = ElasticMatrix.copy()
#     np.fill_diagonal(BinEM,0)
#     BinEM[BinEM > 0] = 1
#     Ks = np.sum(BinEM,axis=0)
#     lp = np.maximum(Ks[row], Ks[col])
#     lp = lp-2
#     lp[lp<0] = 0
#     Lpenalized = L + alpha*lp
#     EP = sum(Lpenalized * np.sum(dev**2, axis=1))
#     # ###
#     indL = Lambda+Lambda.transpose() > 0
#     RP = 0
#     for i in range(np.size(StarCenterIndices)):
#         leaves = indL.take(StarCenterIndices[i], axis=1)
#         K = sum(leaves)
#         dev = (NodePositions.take(StarCenterIndices[i], axis=0) -
#                sum(NodePositions[leaves])/K)
#         RP += Mu[StarCenterIndices[i]]* np.power(K,beta) * sum(dev ** 2)

#     ElasticEnergy = MSE + EP + RP

#     return ElasticEnergy, MSE, EP, RP


@nb.njit(cache=True)
def ComputePenalizedPrimitiveGraphElasticEnergy(
    NodePositions, ElasticMatrix, dists, alpha=0.01, beta=0.01
):
    """
        //' Compute the penalized elastic energy associated with a particular configuration
    //'
    //' This function computes the elastic energy associated to a set of points and graph embedded
    //' into them.
    //'
    //' NodePositions A numeric k-by-m matrix containing the position of the k nodes of the embedded graph
    //' ElasticMatrix A numeric l-by-l matrix containing the elastic parameters associates with the edge
    //' of the embedded graph
    //' dists A numeric vector containing the squared distance of the data points to the closest node of the graph
    //' alpha
    //' beta
    //'
    //' Return
    -------
    A list with four elements:
    //' * ElasticEnergy is the total energy
    //' * MSE is the MSE component of the energy
    //' * EP is the EP component of the energy
    //' * RP is the RP component of the energy
    """
    MSE = dists.sum() / dists.size
    Mu = np.diag(ElasticMatrix)
    Lambda = np.triu(ElasticMatrix, 1)
    StarCenterIndices = (Mu > 0).nonzero()[0]
    (row, col) = Lambda.nonzero()
    dev = NodePositions[row] - NodePositions[col]
    L = np.zeros((len(row)))
    for i in range(len(row)):
        L[i] = Lambda[row[i], col[i]]
    ### diff compared to base function
    BinEM = (Lambda + Lambda.transpose()) > 0
    Ks = BinEM.sum(axis=0)
    lp = np.maximum(Ks[row], Ks[col])
    lp = lp - 2
    lp[np.where(lp < 0)] = 0

    Lpenalized = L + alpha * lp
    EP = np.dot(Lpenalized, np.sum(dev ** 2, axis=1))
    ####
    indL = Lambda + Lambda.transpose() > 0
    RP = 0
    for i in range(StarCenterIndices.size):
        leaves = indL[StarCenterIndices[i]]
        ind_leaves = leaves.nonzero()[0]
        K = ind_leaves.size
        dev_ = NodePositions[StarCenterIndices[i]] - (
            NodePositions[ind_leaves] / K
        ).sum(axis=0)
        RP += Mu[StarCenterIndices[i]] * (K ** beta) * (dev_ ** 2).sum()

    ElasticEnergy = MSE + EP + RP

    return ElasticEnergy, MSE, EP, RP


@nb.njit(cache=True)
def ComputePenalizedPrimitiveGraphElasticEnergy_v2(
    NodePositions,
    ElasticMatrix,
    dists,
    alpha=0.01,
    beta=0.01,
    PointWeights=None,
    PseudotimeNodePositions=None,
):
    """
        //' Compute the penalized elastic energy associated with a particular configuration
    //'
    //' This function computes the elastic energy associated to a set of points and graph embedded
    //' into them.
    //'
    //' NodePositions A numeric k-by-m matrix containing the position of the k nodes of the embedded graph
    //' ElasticMatrix A numeric l-by-l matrix containing the elastic parameters associates with the edge
    //' of the embedded graph
    //' dists A numeric vector containing the squared distance of the data points to the closest node of the graph
    //' alpha
    //' beta
    //'
    //' Return
    -------
    A list with four elements:
    //' * ElasticEnergy is the total energy
    //' * MSE is the MSE component of the energy
    //' * EP is the EP component of the energy
    //' * RP is the RP component of the energy
    """
    MSE = np.sum(dists * PointWeights) / np.sum(PointWeights)

    Mu = np.diag(ElasticMatrix)
    Lambda = np.triu(ElasticMatrix, 1)
    StarCenterIndices = (Mu > 0).nonzero()[0]
    (row, col) = Lambda.nonzero()
    dev = NodePositions[row] - NodePositions[col]
    L = np.zeros((len(row)))
    for i in range(len(row)):
        L[i] = Lambda[row[i], col[i]]
    ### diff compared to base function
    BinEM = (Lambda + Lambda.transpose()) > 0
    Ks = BinEM.sum(axis=0)
    lp = np.maximum(Ks[row], Ks[col])
    lp = lp - 2
    lp[np.where(lp < 0)] = 0

    Lpenalized = L + alpha * lp
    EP = np.dot(Lpenalized, np.sum(dev ** 2, axis=1))

    if PseudotimeNodePositions is not None:
        ps_dev = NodePositions - PseudotimeNodePositions
        pseudotimeEP = np.dot(
            np.repeat(L.min(), len(NodePositions)), np.sum(ps_dev ** 2, axis=1)
        )
        EP += pseudotimeEP

    ####
    indL = Lambda + Lambda.transpose() > 0
    RP = 0
    for i in range(StarCenterIndices.size):
        leaves = indL[StarCenterIndices[i]]
        ind_leaves = leaves.nonzero()[0]
        K = ind_leaves.size
        dev_ = NodePositions[StarCenterIndices[i]] - (
            NodePositions[ind_leaves] / K
        ).sum(axis=0)
        RP += Mu[StarCenterIndices[i]] * (K ** beta) * (dev_ ** 2).sum()

    ElasticEnergy = MSE + EP + RP

    return ElasticEnergy, MSE, EP, RP


def ComputePenalizedPrimitiveGraphElasticEnergy_v3(
    NodePositions,
    ElasticMatrix,
    part,
    dists,
    alpha=0.01,
    beta=0.01,
    PseudotimeNodePositions=None,
    Label=None,
    LabelLambda=None,
):
    """
        //' Compute the penalized elastic energy associated with a particular configuration
    //'
    //' This function computes the elastic energy associated to a set of points and graph embedded
    //' into them.
    //'
    //' NodePositions A numeric k-by-m matrix containing the position of the k nodes of the embedded graph
    //' ElasticMatrix A numeric l-by-l matrix containing the elastic parameters associates with the edge
    //' of the embedded graph
    //' dists A numeric vector containing the squared distance of the data points to the closest node of the graph
    //' alpha
    //' beta
    //'
    //' Return
    -------
    A list with four elements:
    //' * ElasticEnergy is the total energy
    //' * MSE is the MSE component of the energy
    //' * EP is the EP component of the energy
    //' * RP is the RP component of the energy
    """
    MSE = dists.sum() / dists.size
    Mu = np.diag(ElasticMatrix)
    Lambda = np.triu(ElasticMatrix, 1)
    StarCenterIndices = (Mu > 0).nonzero()[0]
    (row, col) = Lambda.nonzero()
    dev = NodePositions[row] - NodePositions[col]
    L = np.zeros((len(row)))
    for i in range(len(row)):
        L[i] = Lambda[row[i], col[i]]
    ### diff compared to base function
    BinEM = (Lambda + Lambda.transpose()) > 0
    Ks = BinEM.sum(axis=0)
    lp = np.maximum(Ks[row], Ks[col])
    lp = lp - 2
    lp[np.where(lp < 0)] = 0

    Lpenalized = L + alpha * lp
    EP = np.dot(Lpenalized, np.sum(dev ** 2, axis=1))

    # ---pseudotime penalty
    if PseudotimeNodePositions is not None:
        ps_dev = NodePositions - PseudotimeNodePositions
        pseudotimeEP = np.dot(
            np.repeat(L.min(), len(NodePositions)), np.sum(ps_dev ** 2, axis=1)
        )
        EP += pseudotimeEP

    ####
    indL = Lambda + Lambda.transpose() > 0
    RP = 0
    for i in range(StarCenterIndices.size):
        leaves = indL[StarCenterIndices[i]]
        ind_leaves = leaves.nonzero()[0]
        K = ind_leaves.size
        dev_ = NodePositions[StarCenterIndices[i]] - (
            NodePositions[ind_leaves] / K
        ).sum(axis=0)
        RP += Mu[StarCenterIndices[i]] * (K ** beta) * (dev_ ** 2).sum()

    ElasticEnergy = MSE + EP + RP

    # ---categorical penalty
    if Label is not None:
        try:
            LL = LabelLoss(Label, LabelLambda, part, ElasticMatrix)
        except Exception as e:
            print(e)
            import pdb

            pdb.set_trace()
        ElasticEnergy = ElasticEnergy * (1 + LL)

    return ElasticEnergy, MSE, EP, RP


@nb.njit(cache=True)
def sum_squares_2d_array_along_axis1(arr):
    res = np.empty(arr.shape[0], dtype=arr.dtype)
    for o_idx in range(arr.shape[0]):
        sum_ = 0
        for i_idx in range(arr.shape[1]):
            sum_ += arr[o_idx, i_idx] * arr[o_idx, i_idx]
        res[o_idx] = sum_
    return res


@nb.njit(cache=True)
def euclidean_distance_square_numba(x1, x2):
    distances = np.sqrt(
        -2 * np.dot(x1, x2.T)
        + np.expand_dims(sum_squares_2d_array_along_axis1(x1), axis=1)
        + sum_squares_2d_array_along_axis1(x2)
    )
    return distances


# def PartialDistance(c1,c2):
#     with np.errstate(invalid='ignore'):

#         distances = np.sqrt(-2 * np.matmul(c1, np.transpose(c2)) + np.sum(np.square(c2), axis=1)
#                         + np.expand_dims(np.sum(np.square(c1), axis=1), axis=1))
#         distances[np.isnan(distances)]=0
#     return distances


def PartialDistance(A, B):
    a = euclidean_distance_square_numba(A, B)
    a[np.isnan(a)] = 0
    return a


def ComputeWeightedAverage(X, partition, PointWeights, NumberOfNodes):
    X = X * PointWeights
    # Auxiliary calculations
    M = X.shape[1]
    part = partition.ravel() + 1
    # Calculate total weights
    TotalWeight = PointWeights.sum()
    # Calculate weights for Relative size
    tmp = np.bincount(
        part, weights=PointWeights.ravel(), minlength=NumberOfNodes + 1
    )
    NodeClusterRelativeSize = tmp[1:] / TotalWeight
    # To prevent dividing by 0
    tmp[tmp == 0] = 1
    NodeClusterCenters = np.zeros((NumberOfNodes + 1, X.shape[1]))
    for k in range(M):
        NodeClusterCenters[:, k] = (
            np.bincount(part, weights=X[:, k], minlength=NumberOfNodes + 1)
            / tmp
        )
    return (
        NodeClusterCenters[
            1:,
        ],
        NodeClusterRelativeSize[np.newaxis].T,
    )


def FitGraph2DataGivenPartition(
    X, PointWeights, SpringLaplacianMatrix, partition
):
    """
    # Solves the SLAU to find new node positions
    """
    NumberOfNodes = SpringLaplacianMatrix.shape[0]
    NodeClusterCenters, NodeClusterRelativeSize = ComputeWeightedAverage(
        X, partition, PointWeights, NumberOfNodes
    )
    SLAUMatrix = (
        np.diag(NodeClusterRelativeSize.transpose()[0]) + SpringLaplacianMatrix
    )
    NewNodePositions = np.linalg.solve(
        SLAUMatrix, NodeClusterRelativeSize * NodeClusterCenters
    )
    return NewNodePositions


def FitSubGraph2DataGivenPartition(
    move_X,
    move_PointWeights,
    SpringLaplacianMatrix,
    NodePositions,
    move_partition,
    move_nodes_idx,
):
    """
    Fits moving subpart of the graph to data while constraining some nodes to remain fixed
    """

    # params
    NumberOfNodes = len(NodePositions)
    NumberOfNodesToFit = len(move_nodes_idx)
    fixed_nodes_idx = list(set(range(NumberOfNodes)) - set(move_nodes_idx))
    # new node positions
    NewNodePositions = np.zeros(NodePositions.shape)

    # weighted average of moving nodes
    (
        move_NodeClusterCenters,
        move_NodeClusterRelativeSize,
    ) = ComputeWeightedAverage(
        move_X, move_partition, move_PointWeights, NumberOfNodesToFit
    )

    # SLAUMatrices
    rs = np.zeros((NumberOfNodes))
    rs[move_nodes_idx] = move_NodeClusterRelativeSize.flatten()
    SLAUMatrix = np.diag(rs) + SpringLaplacianMatrix
    move_SLAUMatrix = SLAUMatrix[move_nodes_idx][:, move_nodes_idx]
    fixed_SLAUMatrix = SLAUMatrix[move_nodes_idx][:, fixed_nodes_idx]

    # Fit
    RightHandSide = move_NodeClusterRelativeSize * move_NodeClusterCenters
    rhs1 = NodePositions[fixed_nodes_idx]
    RightHandSide -= np.dot(fixed_SLAUMatrix, rhs1)
    move_NewNodePositions = np.linalg.solve(move_SLAUMatrix, RightHandSide)

    # Store
    NewNodePositions[fixed_nodes_idx, :] = NodePositions[fixed_nodes_idx, :]
    NewNodePositions[move_nodes_idx, :] = move_NewNodePositions

    return NewNodePositions


def FitSubGraph2DataGivenPartition_v2(
    move_X,
    move_PointWeights,
    SpringLaplacianMatrix,
    NodePositions,
    move_partition,
    move_nodes_idx,
    PseudotimeNodePositions,
    PseudotimeLambda,
):
    """
    Fits moving subpart of the graph to data while constraining some nodes to remain fixed
    """

    # params
    NumberOfNodes = len(NodePositions)
    NumberOfNodesToFit = len(move_nodes_idx)
    fixed_nodes_idx = list(set(range(NumberOfNodes)) - set(move_nodes_idx))
    # new node positions
    NewNodePositions = np.zeros(NodePositions.shape)

    # weighted average of moving nodes
    (
        move_NodeClusterCenters,
        move_NodeClusterRelativeSize,
    ) = ComputeWeightedAverage(
        move_X, move_partition, move_PointWeights, NumberOfNodesToFit
    )

    if PseudotimeNodePositions is not None:
        move_NodeClusterCenters = (
            move_NodeClusterCenters
            + PseudotimeNodePositions[move_nodes_idx] * PseudotimeLambda
        ) / (1 + PseudotimeLambda)

    # SLAUMatrices
    rs = np.zeros((NumberOfNodes))
    rs[move_nodes_idx] = move_NodeClusterRelativeSize.flatten()
    SLAUMatrix = np.diag(rs) + SpringLaplacianMatrix
    move_SLAUMatrix = SLAUMatrix[move_nodes_idx][:, move_nodes_idx]
    fixed_SLAUMatrix = SLAUMatrix[move_nodes_idx][:, fixed_nodes_idx]

    # Fit
    RightHandSide = move_NodeClusterRelativeSize * move_NodeClusterCenters
    rhs1 = NodePositions[fixed_nodes_idx]
    RightHandSide -= np.dot(fixed_SLAUMatrix, rhs1)
    move_NewNodePositions = np.linalg.solve(move_SLAUMatrix, RightHandSide)

    # Store
    NewNodePositions[fixed_nodes_idx, :] = NodePositions[fixed_nodes_idx, :]
    NewNodePositions[move_nodes_idx, :] = move_NewNodePositions

    return NewNodePositions


#####################################
### Categorical supervision utils ###
#####################################


# @nb.njit
def normalized_entropy_times_cardinality(num_labels):
    """Calculates the normalized entropy of labels

    Parameters
    ----------
    num_labels : int array, shape = [n_samples]
        The labels
    """
    if len(num_labels) == 0:
        return 1.0
    pi = np.bincount(num_labels).astype(np.float64)
    pi = pi[pi > 0]
    pi_sum = np.sum(pi)
    # log(a / b) should be calculated as log(a) - log(b) for
    # possible loss of precision
    return (
        -np.sum((pi / pi_sum) * (np.log(pi) - np.log(pi_sum)))
        / np.log(len(num_labels))
        * len(num_labels)
    )


# @nb.njit
def LabelLoss(Label, LabelLambda, part, EM):
    valid_nodes = np.unique(part)
    n_nodes = int(np.sum(valid_nodes >= 0))
    _EM = EM.copy()
    np.fill_diagonal(_EM, 0)
    net = igraph.Graph.Adjacency((_EM > 0).tolist(), mode="undirected")
    branches = GetSubGraph(net, "branches")
    branches_dataidx = [np.isin(part, b) for b in branches]

    entropy = 0
    for i in range(len(branches)):
        entropy += normalized_entropy_times_cardinality(
            Label[branches_dataidx[i]]
        )
        # normalize 0/1
    LL = LabelLambda * entropy / len(part)
    # ElasticEn * (1 + LabelLoss)
    return LL