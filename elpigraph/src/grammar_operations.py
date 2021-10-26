import numpy as np
import multiprocessing as mp

from .core import (
    PartitionData,
    PartitionData_cp,
    PrimitiveElasticGraphEmbedment,
    PrimitiveElasticGraphEmbedment_cp,
    DecodeElasticMatrix2,
)
from .._EMAdjustment import AdjustByConstant


def proxy(Dict):
    return PrimitiveElasticGraphEmbedment(**Dict)


def proxy_cp(Dict):
    return PrimitiveElasticGraphEmbedment_cp(**Dict)


# Some elementary graph transformations -----------------------------------

# def f_RemoveNode(NodePositions, ElasticMatrix,NodeNumber):
#    '''
#    remove from the graph node number NodeNumber
#    '''
#    idx = np.arange(len(NodePositions))!= NodeNumber
#    return NodePositions[idx,:],ElasticMatrix[idx,:][:,idx]
#
# def f_reattach_edges(ElasticMatrix, NodeNumber1, NodeNumber2):
#    '''
#    # reattaches all edges connected with NodeNumber2 to NodeNumber1
#    # and make a new star with an elasticity average of two merged stars
#    '''
#    ElasticMatrix2 = ElasticMatrix.copy()
#    Lambda = ElasticMatrix.copy()
#    np.fill_diagonal(Lambda, 0)
#
#    ElasticMatrix2[NodeNumber1,:] = np.max(np.concatenate((Lambda[[NodeNumber1],:],Lambda[[NodeNumber2],:])), axis=0)
#    ElasticMatrix2[:,NodeNumber1] = np.max(np.concatenate((Lambda[:,[NodeNumber1]],Lambda[:,[NodeNumber2]]),axis=1), axis=1)
#    ElasticMatrix2[NodeNumber1,NodeNumber1] =(
#    ElasticMatrix[NodeNumber1,NodeNumber1]+ElasticMatrix[NodeNumber2,NodeNumber2]/2)
#
#    return ElasticMatrix2
#
#
# def f_add_nonconnected_node(NodePositions, ElasticMatrix, NewNodePosition):
#    '''
#    # add a new non-connected node
#    '''
#    NodePositions2 = np.concatenate((NodePositions, NewNodePosition.squeeze()[None]))
#    length_em = ElasticMatrix.shape[0]
#    ElasticMatrix2 = np.zeros((length_em+1, length_em+1))
#    ElasticMatrix2[:-1, :-1] = ElasticMatrix
#
#    return NodePositions2, ElasticMatrix2
#
#
#
# def f_removeedge(ElasticMatrix, Node1, Node2):
#    '''
#    # remove edge connecting Node1 and Node 2
#    '''
#    ElasticMatrix[Node1,Node2] = 0
#    ElasticMatrix[Node2,Node1] = 0
#
#    return ElasticMatrix
#
#
# def f_add_edge(ElasticMatrix, Node1, Node2, _lambda):
#    '''
#    # connects Node1 and Node 2 by an edge with elasticity lambda
#    '''
#
#    ElasticMatrix[Node1,Node2] = _lambda
#    ElasticMatrix[Node2,Node1] = _lambda
#
#    return ElasticMatrix
#
#
#
# def f_get_star(NodePositions, ElasticMatrix, NodeCenter):
#    '''
#    # extracts a star from the graph with the center in NodeCenter
#    '''
#    NodeIndices = np.where(ElasticMatrix[NodeCenter,:]>0)[0]
#    ElasticMatrix[NodeIndices,:][:,NodeIndices]
#    return NodePositions[NodeIndices,:], ElasticMatrix[NodeIndices,:][:,NodeIndices], NodeIndices
#
#
#
# Grammar function wrapper ------------------------------------------------


def GraphGrammarOperation(X, NodePositions, ElasticMatrix, AdjustVect, Type, partition):
    if Type == "addnode2node":
        return AddNode2Node(X, NodePositions, ElasticMatrix, partition, AdjustVect)
    elif Type == "addnode2node_1":
        return AddNode2Node(
            X, NodePositions, ElasticMatrix, partition, AdjustVect, Max_K=1
        )
    elif Type == "addnode2node_2":
        return AddNode2Node(
            X, NodePositions, ElasticMatrix, partition, AdjustVect, Max_K=2
        )
    elif Type == "removenode":
        return RemoveNode(NodePositions, ElasticMatrix, AdjustVect)
    elif Type == "bisectedge":
        return BisectEdge(NodePositions, ElasticMatrix, AdjustVect)
    elif Type == "bisectedge_3":
        return BisectEdge(NodePositions, ElasticMatrix, AdjustVect, Min_K=3)
    elif Type == "shrinkedge":
        return ShrinkEdge(NodePositions, ElasticMatrix, AdjustVect)
    elif Type == "shrinkedge_3":
        return ShrinkEdge(NodePositions, ElasticMatrix, AdjustVect, Min_K=3)
    else:
        raise ValueError("Operation " + Type + " is not defined")


# Grammar functions ------------------------------------------------


def AddNode2Node(
    X, NodePositions, ElasticMatrix, partition, AdjustVect, Max_K=float("inf")
):
    """
    #' Adds a node to each graph node
    #'
    #' This grammar operation adds a node to each graph node. The position of the node
    #' is chosen as a linear extrapolation for a leaf node (in this case the elasticity of
    #' a newborn star is chosed as in BisectEdge operation), or as the data point giving
    #' the minimum local MSE for a star (without any optimization).
    #'
    #' @param X
    #' @param NodePositions
    #' @param ElasticMatrix
    #' @return
    #' @export
    #'
    #' @details
    #'
    #'
    #'
    #' @examples
    """
    nNodes = NodePositions.shape[0]
    Mus = ElasticMatrix.diagonal()
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    indL = Lambda > 0
    Connectivities = indL.sum(axis=0)
    # add pointweights here if added
    assoc = np.bincount(partition[partition > -1].ravel(), minlength=nNodes)
    # Create prototypes for new NodePositions, ElasticMatrix and inds
    npProt = np.vstack((NodePositions, np.zeros((1, NodePositions.shape[1]))))
    emProt = np.vstack(
        (np.hstack((Lambda, np.zeros((nNodes, 1)))), np.zeros((1, nNodes + 1)))
    )
    #     niProt = np.arange(nNodes+1)
    #     niProt[nNodes] = 0

    MuProt = np.zeros(nNodes + 1)
    MuProt[:-1] = Mus

    if not np.isinf(Max_K):
        Degree = np.sum(ElasticMatrix > 0, axis=1)
        Degree[Degree > 1] = Degree[Degree > 1] - 1

        if np.sum(Degree <= Max_K) > 1:
            idx_nodes = np.where(Degree <= Max_K)[0]
        else:
            raise ValueError("AddNode2Node impossible with the current parameters!")
    else:
        idx_nodes = np.array(range(nNodes))

    # Put prototypes to corresponding places
    NodePositionsArray = [npProt.copy() for i in range(len(idx_nodes))]
    ElasticMatrices = [emProt.copy() for i in range(len(idx_nodes))]
    #     NodeIndicesArray = np.repeat(niProt[:, np.newaxis], len(idx_nodes), axis=1)
    AdjustVectArray = [AdjustVect + [False] for i in range(len(idx_nodes))]

    for j, i in enumerate(idx_nodes):
        MuProt[-1] = 0
        # Compute mean edge elasticity for edges with node i
        meanL = Lambda[i, indL[i,]].mean(axis=0)
        # Add edge to elasticity matrix
        ElasticMatrices[j][nNodes, i] = ElasticMatrices[j][i, nNodes] = meanL

        if Connectivities[i] == 1:
            # Add node to terminal node
            ineighbour = np.nonzero(indL[i,])[0]
            # Calculate new node position
            NewNodePosition = (
                2 * NodePositions[i,] - NodePositions[ineighbour,]
            )
            # Complete Elasticity Matrix
            MuProt[nNodes] = Mus[ineighbour]
        else:
            # Add node to a star
            # if 0 data points associated with this star
            if assoc[i] == 0:
                # then select mean of all leaves as new position
                NewNodePosition = NodePositions[indL[:, i]].mean(axis=0)

            else:
                # Otherwise take the mean of the points associated with the
                # central node
                NewNodePosition = X[(partition == i).ravel()].mean(axis=0)
        # fill node position
        NodePositionsArray[j][nNodes, :] = NewNodePosition
        np.fill_diagonal(ElasticMatrices[j], MuProt)

    return NodePositionsArray, ElasticMatrices, AdjustVectArray


def BisectEdge(NodePositions, ElasticMatrix, AdjustVect, Min_K=1):
    """
    # % This grammar operation inserts a node inside the middle of each edge
    # % The elasticity of the edges do not change
    # % The elasticity of the newborn star is chosen as
    # % mean over the neighbour stars if the edge connects two star centers
    # % or
    # % the one of the single neigbour star if this is a dangling edge
    # % or
    # % if one starts from a single edge, the star elasticities should be on
    # % one of two elements in the diagoal of the ElasticMatrix
    """
    # Decompose Elastic Matrix: Mus
    Mus = ElasticMatrix.diagonal()
    # Get list of edges
    Edges, _, _ = DecodeElasticMatrix2(ElasticMatrix)

    # Define some constants
    nNodes = NodePositions.shape[0]
    if Min_K > 1:
        Degree = np.bincount(Edges.flatten())
        EdgDegree = np.max(Degree[Edges], axis=1)
        nGraphs = np.where(EdgDegree >= Min_K)[0]
    else:
        nGraphs = np.array(range(Edges.shape[0]))
    # Create prototypes for new NodePositions, ElasticMatrix and inds
    npProt = np.vstack((NodePositions, np.zeros((1, NodePositions.shape[1]))))
    emProt = np.vstack(
        (np.hstack((ElasticMatrix, np.zeros((nNodes, 1)))), np.zeros((1, nNodes + 1)))
    )
    #     niProt = np.arange(nNodes+1)
    #     niProt[nNodes] = 0

    # Allocate arrays and put prototypes in place
    NodePositionsArray = [npProt.copy() for i in range(len(nGraphs))]
    ElasticMatrices = [emProt.copy() for i in range(len(nGraphs))]
    #     NodeIndicesArray = np.repeat(niProt[:, np.newaxis], len(nGraphs), axis=1)
    AdjustVectArray = [AdjustVect + [False] for i in range(len(nGraphs))]

    for j, i in enumerate(nGraphs):
        NewNodePosition = (
            NodePositions[Edges[i, 0],] + NodePositions[Edges[i, 1],]
        ) / 2

        # Fill NodePosition
        NodePositionsArray[j][nNodes, :] = NewNodePosition
        # correct elastic matrix
        Lambda = ElasticMatrix[Edges[i, 0], Edges[i, 1]]
        # remove edge
        ElasticMatrices[j][Edges[i, 0], Edges[i, 1]] = 0
        ElasticMatrices[j][Edges[i, 1], Edges[i, 0]] = 0
        # add 2 edges
        ElasticMatrices[j][Edges[i, 0], nNodes] = Lambda
        ElasticMatrices[j][nNodes, Edges[i, 0]] = Lambda
        ElasticMatrices[j][nNodes, Edges[i, 1]] = Lambda
        ElasticMatrices[j][Edges[i, 1], nNodes] = Lambda
        # Define mus of edges
        mu1 = Mus[Edges[i, 0]]
        mu2 = Mus[Edges[i, 1]]
        if mu1 > 0 and mu2 > 0:
            ElasticMatrices[j][nNodes, nNodes] = (mu1 + mu2) / 2
        else:
            ElasticMatrices[j][nNodes, nNodes] = max(mu1, mu2)

    return NodePositionsArray, ElasticMatrices, AdjustVectArray  # , NodeIndicesArray


def RemoveNode(NodePositions, ElasticMatrix, AdjustVect):
    """    
    ##  This grammar operation removes a leaf node (connectivity==1)
    """
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    Connectivities = (Lambda > 0).sum(axis=0)
    # Define sizes
    nNodes = ElasticMatrix.shape[0]
    nGraphs = (Connectivities == 1).sum()
    # Preallocate arrays
    NodePositionsArray = [
        np.zeros((nNodes - 1, NodePositions.shape[1])) for i in range(nGraphs)
    ]
    ElasticMatrices = [np.zeros((nNodes - 1, nNodes - 1)) for i in range(nGraphs)]
    #     NodeIndicesArray = np.zeros((nNodes-1, nGraphs))
    AdjustVectArray = [[] for i in range(nGraphs)]

    k = 0
    for i in range(Connectivities.shape[0]):
        if Connectivities[i] == 1:
            # if terminal node remove it
            newInds = np.concatenate(
                (np.arange(0, i, dtype=int), np.arange(i + 1, nNodes, dtype=int))
            )
            AdjustVectArray[k] = [AdjustVect[j] for j in newInds]
            NodePositionsArray[k] = NodePositions[newInds, :]
            tmp = np.repeat(False, nNodes)
            tmp[newInds] = True
            tmp2 = ElasticMatrix[tmp, :]
            ElasticMatrices[k] = tmp2[:, tmp]
            #             NodeIndicesArray[:, k] = newInds
            k += 1
    return NodePositionsArray, ElasticMatrices, AdjustVectArray  # , NodeIndicesArray


def ShrinkEdge(NodePositions, ElasticMatrix, AdjustVect, Min_K=1):
    """
    # %
    # % This grammar operation removes an edge from the graph
    # % If this is an edge connecting a leaf node then it is equivalent to
    # % RemoveNode. So we remove only internal edges.
    # % If this is an edge connecting two stars then their leaves are merged,
    # % and the star is placed in the middle of the shrinked edge.
    # % The elasticity of the new formed star is the average of two star
    # % elasticities.
    # %
    """
    ## Shrink edge
    Mus = ElasticMatrix.diagonal()
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    Connectivities = (Lambda > 0).sum(axis=0)
    # get list of edges
    start, stop = np.triu(ElasticMatrix, 1).nonzero()
    # Edges = np.concatenate((start[None], stop[None]))
    # define size
    nNodes = NodePositions.shape[0]
    # identify edges with minimal connectivity > 1
    Degree = np.hstack((Connectivities[start[None].T], Connectivities[stop[None].T]))

    ind_sup1 = np.min(Degree, axis=1) > 1
    ind_min_K = np.max(Degree, axis=1) >= Min_K
    ind = ind_sup1 & ind_min_K

    start = start[ind]
    stop = stop[ind]
    # calculate nb of graphs
    nGraphs = start.shape[0]

    # preallocate array
    NodePositionsArray = [
        np.zeros((nNodes - 1, NodePositions.shape[1])) for i in range(nGraphs)
    ]
    ElasticMatrices = [np.zeros((nNodes - 1, nNodes - 1)) for i in range(nGraphs)]
    #     NodeIndicesArray = np.zeros((nNodes-1, nGraphs))
    AdjustVectArray = [[] for i in range(nGraphs)]

    for i in range(nGraphs):
        # create copy of elastic matrix
        em = ElasticMatrix.copy()
        # Reattaches all edges connected with stop[i] to start[i]
        # and make a new star with an elasticity average of two merged stars
        em[start[i],] = np.maximum(Lambda[start[i],], Lambda[stop[i],])
        em[:, start[i]] = np.maximum(Lambda[:, start[i]], Lambda[:, stop[i]])
        em[start[i], start[i]] = (Mus[start[i]] + Mus[stop[i]]) / 2
        #         em[start[i], start[i]] = Mus[start[i]] + Mus[stop[i]] / 2  #### R bug ????
        # Create copy of node positions
        nodep = NodePositions.copy()
        # modify node start[i]
        nodep[start[i], :] = (nodep[start[i], :] + nodep[stop[i], :]) / 2
        # Form index for retained nodes and extract corresponding part of
        # node positions and elastic matrix
        newInds = np.concatenate(
            (np.arange(0, stop[i]), np.arange(stop[i] + 1, nNodes))
        )

        AdjustVectArray[i] = [AdjustVect[j] for j in newInds]

        NodePositionsArray[i] = nodep[newInds, :]
        ElasticMatrices[i] = em.take(newInds, axis=0).take(newInds, axis=1)
    #         NodeIndicesArray[:, i] = newInds

    return NodePositionsArray, ElasticMatrices, AdjustVectArray  # ,NodeIndicesArray


def ApplyOptimalGraphGrammarOperation(
    X,
    NodePositions,
    ElasticMatrix,
    opTypes,
    AdjustVect=None,
    SquaredX=None,
    verbose=False,
    MaxBlockSize=100000000,
    MaxNumberOfIterations=100,
    eps=0.01,
    TrimmingRadius=float("inf"),
    Mode=1,
    FinalEnergy="Base",
    alpha=1,
    beta=1,
    EmbPointProb=1,
    AvoidSolitary=False,
    AdjustElasticMatrix=None,
    DisplayWarnings=True,
    n_cores=1,
    MinParOp=20,
    multiproc_shared_variables=None,
    Xcp=None,
    SquaredXcp=None,
):

    """
    # Multiple grammar application --------------------------------------------
    #' Application of the grammar operation. This in an internal function that should not be used in by the end-user
    #'
    #' @param X numerical 2D matrix, the n-by-m matrix with the position of n m-dimensional points
    #' @param NodePositions numerical 2D matrix, the k-by-m matrix with the position of k m-dimensional points
    #' @param ElasticMatrix numerical 2D matrix, the k-by-k elastic matrix
    #' @param operationtypes string vector containing the operation to use
    #' @param SquaredX rowSums(X^2), if NULL it will be computed
    #' @param verbose boolean. Should addition information be displayed
    #' @param n.cores integer. How many cores to use. If EnvCl is not NULL, that cliuster setup will be used,
    #' otherwise a SOCK cluster willbe used
    #' @param EnvCl a cluster structure returned, e.g., by makeCluster.
    #' If a cluster structure is used, all the nodes must be able to access all the variable needed by PrimitiveElasticGraphEmbedment
    #' @param MaxNumberOfIterations is an integer number indicating the maximum number of iterations for the EM algorithm
    #' @param TrimmingRadius is a real value indicating the trimming radius, a parameter required for robust principal graphs
    #' (see https://github.com/auranic/Elastic-principal-graphs/wiki/Robust-principal-graphs)
    #' @param eps a real number indicating the minimal relative change in the nodenpositions
    #' to be considered the graph embedded (convergence criteria)
    #' @param Mode integer, the energy mode. It can be 1 (difference is computed using the position of the nodes) and
    #' 2 (difference is computed using the changes in elestic energy of the configuraztions)
    #' @param FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    #' @param alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    #' @param beta positive numeric, the value of the beta parameter of the penalized elastic energy
    #' @param gamma 
    #' @param FastSolve boolean, should FastSolve be used when fitting the points to the data?
    #' @param AvoidSolitary boolean, should configurations with "solitary nodes", i.e., nodes without associted points be discarded?
    #' @param EmbPointProb numeric between 0 and 1. If less than 1 point will be sampled at each iteration. Prob indicate the probability of
    #' using each points. This is an *experimental* feature, which may helps speeding up the computation if a large number of points is present.
    #' @param AdjustVect 
    #' @param AdjustElasticMatrix 
    #' @param ... 
    #' @param MinParOp integer, the minimum number of operations to use parallel computation
    #'
    #' @return
    #'
    #' @examples
    """

    NodePositionsArrayAll = []
    ElasticMatricesAll = []
    AdjustVectAll = []

    #    if SquaredX is None and SquaredXcp is None:
    #        SquaredX = (X**2).sum(axis=1,keepdims=1)

    if Xcp is None:
        partition, _ = PartitionData(
            X, NodePositions, MaxBlockSize, SquaredX, TrimmingRadius
        )
    else:
        partition, _ = PartitionData_cp(
            Xcp, NodePositions, MaxBlockSize, SquaredXcp, TrimmingRadius
        )

    for i in range(len(opTypes)):
        if verbose:
            print(" Operation type : ", opTypes[i])

        NodePositionsArray, ElasticMatrices, AdjustVectArray = GraphGrammarOperation(
            X, NodePositions, ElasticMatrix, AdjustVect, opTypes[i], partition
        )

        #         NodePositionsArrayAll = np.concatenate((NodePositionsArrayAll,
        #                                                NodePositionsArray), axis=2)
        #         ElasticMatricesAll = np.concatenate((ElasticMatricesAll,
        #                                              ElasticMatrices), axis=2)
        #     NodeIndicesArrayAll = np.concatenate((NodeIndicesArrayAll,
        #                                           NodeIndicesArray), axis=1)
        NodePositionsArrayAll.extend(NodePositionsArray)
        ElasticMatricesAll.extend(ElasticMatrices)
        AdjustVectAll.extend(AdjustVectArray)

    if verbose:
        print("Optimizing graphs")

    Valid_configurations = range(len(NodePositionsArrayAll))

    if AvoidSolitary:
        Valid_configurations = []
        if Xcp is None:
            for i in range(len(NodePositionsArrayAll)):
                partition = PartitionData(
                    X=X,
                    MaxBlockSize=MaxBlockSize,
                    NodePositions=NodePositionsArrayAll[i],
                    SquaredX=SquaredX,
                    TrimmingRadius=TrimmingRadius,
                )[0]
                if all(
                    np.isin(
                        np.array(range(NodePositionsArrayAll[i].shape[0])),
                        partition[partition > -1],
                    )
                ):
                    Valid_configurations.append(i)

        else:
            for i in range(len(NodePositionsArrayAll)):
                partition = PartitionData_cp(
                    Xcp,
                    MaxBlockSize,
                    NodePositionsArrayAll[i],
                    SquaredXcp,
                    TrimmingRadius=TrimmingRadius,
                )[0]
                if all(
                    np.isin(
                        np.array(range(NodePositionsArrayAll[i].shape[0])),
                        partition[partition > -1],
                    )
                ):
                    Valid_configurations.append(i)

        if verbose:
            print(
                len(Valid_configurations),
                "configurations out of ",
                len(NodePositionsArrayAll),
                "used",
            )
        if Valid_configurations == []:
            return "failed operation"
    #     NodePositionArrayAll = NodePositionArrayAll[...,Valid_configurations]
    #     ElasticMatricesAll = ElasticMatricesAll[...,Valid_configurations]
    #     AdjustVectAll = AdjustVectAll[Valid_configurations]

    if AdjustElasticMatrix:
        for i in Valid_configurations:
            ElasticMatricesAll[i], AdjustVectAll[i] = AdjustByConstant(
                ElasticMatricesAll[i], AdjustVectAll[i]
            )

    if n_cores > 1 and len(Valid_configurations) // (MinParOp + 1) > 1:

        #             X_remote, X_shape, SquaredX_remote, SquaredX_shape = multiproc_shared_variables
        #             ####### Multiprocessing
        #             with Pool(processes=n_cores,
        #                       initializer=init_worker,
        #                       initargs=(X_remote,
        #                                 X_shape,
        #                                 SquaredX_remote,
        #                                 SquaredX_shape,
        #                                 NodePositionsArrayAll,
        #                                 ElasticMatricesAll,
        #                                 MaxNumberOfIterations,
        #                                 TrimmingRadius,
        #                                 eps,
        #                                 Mode,
        #                                 FinalEnergy,
        #                                 alpha,
        #                                 beta,
        #                                 EmbPointProb,
        #                                 DisplayWarnings,
        #                                 None,
        #                                 MaxBlockSize,
        #                                 False)) as pool:

        #                 results = pool.map(proxy_multiproc,Valid_configurations)

        with mp.Pool(n_cores) as pool:
            if Xcp is None:
                results = pool.map(
                    proxy,
                    [
                        dict(
                            X=X,
                            NodePositions=NodePositionsArrayAll[i],
                            ElasticMatrix=ElasticMatricesAll[i],
                            MaxNumberOfIterations=MaxNumberOfIterations,
                            eps=eps,
                            Mode=Mode,
                            FinalEnergy=FinalEnergy,
                            alpha=alpha,
                            beta=beta,
                            prob=EmbPointProb,
                            DisplayWarnings=DisplayWarnings,
                            PointWeights=None,
                            MaxBlockSize=MaxBlockSize,
                            verbose=False,
                            TrimmingRadius=TrimmingRadius,
                            SquaredX=SquaredX,
                        )
                        for i in Valid_configurations
                    ],
                )
            else:
                results = pool.map(
                    proxy_cp,
                    [
                        dict(
                            X=X,
                            NodePositions=NodePositionsArrayAll[i],
                            ElasticMatrix=ElasticMatricesAll[i],
                            MaxNumberOfIterations=MaxNumberOfIterations,
                            eps=eps,
                            Mode=Mode,
                            FinalEnergy=FinalEnergy,
                            alpha=alpha,
                            beta=beta,
                            prob=EmbPointProb,
                            DisplayWarnings=DisplayWarnings,
                            PointWeights=None,
                            MaxBlockSize=MaxBlockSize,
                            verbose=False,
                            TrimmingRadius=TrimmingRadius,
                            SquaredX=SquaredX,
                            Xcp=Xcp,
                            SquaredXcp=SquaredXcp,
                        )
                        for i in Valid_configurations
                    ],
                )

        list_energies = [r[1] for r in results]
        idx = list_energies.index(min(list_energies))
        NewNodePositions, minEnergy, partition, Dist, MSE, EP, RP = results[idx]
        AdjustVect = AdjustVectAll[idx]
        NewElasticMatrix = ElasticMatricesAll[idx]

        ########################

        ######### Ray multiprocessing
    #             ray.init(num_cpus=n_cores)
    #             Xrm = ray.put(X)
    #             SquaredXrm=ray.put(SquaredX)
    #             NodePositionsArrayAllrm = ray.put(NodePositionsArrayAll)
    #             ElasticMatricesAllrm = ray.put(ElasticMatricesAll)
    #             MaxNumberOfIterationsrm = ray.put(MaxNumberOfIterations)
    #             epsrm = ray.put(eps)
    #             Moderm= ray.put(Mode)
    #             FinalEnergyrm= ray.put(FinalEnergy)
    #             alpharm= ray.put(alpha)
    #             betarm= ray.put(beta)
    #             EmbPointProbrm= ray.put(EmbPointProb)
    #             DisplayWarningsrm= ray.put(DisplayWarnings)
    #             MaxBlockSizerm= ray.put(MaxBlockSize)
    #             TrimmingRadiusrm= ray.put(TrimmingRadius)
    #             result_ids = [ray_PrimitiveElasticGraphEmbedment.remote(Xrm,
    #                                                NodePositionsArrayAllrm[i],
    #                                                ElasticMatricesAll[i], MaxNumberOfIterationsrm, epsrm,
    #                                                Mode=Moderm, FinalEnergy=FinalEnergyrm,
    #                                                alpha=alpharm,beta=betarm,prob=EmbPointProbrm,
    #                                                DisplayWarnings=DisplayWarningsrm,
    #                                                PointWeights=None,
    #                                                MaxBlockSize=MaxBlockSizerm,
    #                                                verbose=False,
    #                                                TrimmingRadius=TrimmingRadiusrm, SquaredX=SquaredXrm) for i in Valid_configurations]

    #             result_ids = [ray_PrimitiveElasticGraphEmbedment.remote(X,
    #                                                NodePositionsArrayAll[i],
    #                                                ElasticMatricesAll[i], MaxNumberOfIterations, eps,
    #                                                Mode=Mode, FinalEnergy=FinalEnergy,
    #                                                alpha=alpha,beta=beta,prob=EmbPointProb,
    #                                                DisplayWarnings=DisplayWarnings,
    #                                                PointWeights=None,
    #                                                MaxBlockSize=MaxBlockSize,
    #                                                verbose=False,
    #                                                TrimmingRadius=TrimmingRadius, SquaredX=SquaredX) for i in Valid_configurations]

    #             resultlist = ray.get([i[1] for i in result_ids])
    #             idx = resultlist.index(min(resultlist))
    #             NewNodePositions, minEnergy, partition, Dist, MSE, EP, RP = ray.get(result_ids[idx])
    #             AdjustVect = AdjustVectAll[idx]
    #             NewElasticMatrix = ElasticMatricesAll[idx]
    #             ray.shutdown()
    #########################

    else:
        minEnergy = np.inf

        if Xcp is None:
            for i in Valid_configurations:
                # TODO add pointweights ?
                (
                    nodep,
                    ElasticEnergy,
                    part,
                    dist,
                    mse,
                    ep,
                    rp,
                ) = PrimitiveElasticGraphEmbedment(
                    X,
                    NodePositionsArrayAll[i],
                    ElasticMatricesAll[i],
                    MaxNumberOfIterations,
                    eps,
                    Mode=Mode,
                    FinalEnergy=FinalEnergy,
                    alpha=alpha,
                    beta=beta,
                    prob=EmbPointProb,
                    DisplayWarnings=DisplayWarnings,
                    PointWeights=None,
                    MaxBlockSize=MaxBlockSize,
                    verbose=False,
                    TrimmingRadius=TrimmingRadius,
                    SquaredX=SquaredX,
                )

                if ElasticEnergy < minEnergy:
                    NewNodePositions = nodep
                    NewElasticMatrix = ElasticMatricesAll[i]
                    partition = part
                    AdjustVect = AdjustVectAll[i]
                    minEnergy = ElasticEnergy
                    MSE = mse
                    EP = ep
                    RP = rp
                    Dist = dist

        else:
            for i in Valid_configurations:
                # TODO add pointweights ?
                (
                    nodep,
                    ElasticEnergy,
                    part,
                    dist,
                    mse,
                    ep,
                    rp,
                ) = PrimitiveElasticGraphEmbedment_cp(
                    X,
                    NodePositionsArrayAll[i],
                    ElasticMatricesAll[i],
                    MaxNumberOfIterations,
                    eps,
                    Mode=Mode,
                    FinalEnergy=FinalEnergy,
                    alpha=alpha,
                    beta=beta,
                    prob=EmbPointProb,
                    DisplayWarnings=DisplayWarnings,
                    PointWeights=None,
                    MaxBlockSize=MaxBlockSize,
                    verbose=False,
                    TrimmingRadius=TrimmingRadius,
                    SquaredX=SquaredX,
                    Xcp=Xcp,
                    SquaredXcp=SquaredXcp,
                )

                if ElasticEnergy < minEnergy:
                    NewNodePositions = nodep
                    NewElasticMatrix = ElasticMatricesAll[i]
                    partition = part
                    AdjustVect = AdjustVectAll[i]
                    minEnergy = ElasticEnergy
                    MSE = mse
                    EP = ep
                    RP = rp
                    Dist = dist

    return dict(
        NodePositions=NewNodePositions,
        ElasticMatrix=NewElasticMatrix,
        ElasticEnergy=minEnergy,
        MSE=MSE,
        EP=EP,
        RP=RP,
        AdjustVect=AdjustVect,
        Dist=Dist,
    )

