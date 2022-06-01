import numpy as np


try:
    import cupy
except:
    pass
import multiprocessing as mp
import datetime
import time
import copy
from .PCA import PCA, TruncPCA, PCA_gpu, TruncSVD_gpu
from .core import (
    PrimitiveElasticGraphEmbedment,
    PrimitiveElasticGraphEmbedment_v2,
    PrimitiveElasticGraphEmbedment_cp,
    PrimitiveElasticGraphEmbedment_cp_v2,
    PartitionData,
    PartitionData_cp,
    MakeUniformElasticMatrix,
    Encode2ElasticMatrix,
    DecodeElasticMatrix,
)
from .grammar_operations import (
    ApplyOptimalGraphGrammarOperation,
    ApplyOptimalGraphGrammarOperation_v2,
    ApplyOptimalGraphGrammarOperation_v3,
)

from .reporting import ReportOnPrimitiveGraphEmbedment


def isnumeric(obj):
    try:
        obj + 0
        return True
    except:
        return False


def ElPrincGraph(
    X,
    Lambda,
    Mu,
    ElasticMatrix,
    NodePositions,
    AdjustVect,
    NumNodes=100,
    NumEdges=float("inf"),
    verbose=False,
    n_cores=1,
    #                 ClusType = "Sock",
    MinParOp=20,
    CompileReport=True,
    ShowTimer=False,
    ComputeMSEP=True,
    FinalEnergy="Base",
    alpha=0,
    beta=0,
    Mode=1,
    MaxBlockSize=100000000,
    MaxNumberOfIterations=10,
    MaxFailedOperations=float("inf"),
    MaxSteps=float("inf"),
    GrammarOptimization=True,
    eps=0.01,
    TrimmingRadius=float("Inf"),
    GrowGrammars=np.array([]),
    ShrinkGrammars=np.array([]),
    GrammarOrder=["Grow", "Shrink"],
    #                 FastSolve = False,
    AvoidSolitary=False,
    EmbPointProb=1,
    PointWeights=None,
    AdjustElasticMatrix=None,
    AdjustElasticMatrix_Initial=None,
    DisplayWarnings=False,
    StoreGraphEvolution=False,
    GPU=False,
    FixNodesAtPoints=[],
    pseudotime=None,
    pseudotimeLambda=0.01,
    label=None,
    labelLambda=0.01,
    MaxNumberOfGraphCandidatesDict={
        "AddNode2Node": float("inf"),
        "BisectEdge": float("inf"),
        "RemoveNode": float("inf"),
        "ShrinkEdge": float("inf"),
    },
):
    """
    Core function to construct a principal elastic graph

    The core function that takes the n m-dimensional points and construct a principal elastic graph using the
    grammars provided.

    X numerical 2D matrix, the n-by-m matrix with the position of n m-dimensional points
    NumNodes integer, the number of nodes of the principal graph
    Lambda real, the lambda parameter used the compute the elastic energy
    Mu real, the lambda parameter used the compute the elastic energy
    CompileReport boolean, should a step-by-step report with various information on the
    contruction of the principal graph be compiled?
    ShowTimer boolean, should the time to construct the graph be computed and reported for each step?
    ComputeMSEP boolean, should MSEP be computed when building the report?
    GrowGrammars list of strings, the grammar to be used in the growth step
    ShrinkGrammars list of strings, the grammar to be used in the shrink step
    GrammarOrder vector of strings, the order of application of the grammars. Can be any combination of "Grow" and "Shrink"
    NodesPositions numerical 2D matrix, the k-by-m matrix with k m-dimensional positions of the nodes
    in the initial step
    ElasticMatrix numerical 2D matrix, the k-by-k elastic matrix
    n.cores either an integer (indicating the number of cores to used for the creation of a cluster) or
    cluster structure returned, e.g., by makeCluster. If a cluster structure is used, all the nodes must contains X
    (this is done using clusterExport)
    MinParOP integer, the minimum number of operations to use parallel computation
    MaxNumberOfIterations integer, maximum number of steps to embed the nodes in the data
    eps real, minimal relative change in the position of the nodes to stop embedment
    TrimmingRadius real, maximal distance of point from a node to affect its embedment
    NumEdges integer, the maximum nulber of edges
    Mode integer, the energy computation mode
    FastSolve boolean, should FastSolve be used when fitting the points to the data?
    ClusType string, the type of cluster to use. It can gbe either "Sock" or "Fork".
    Currently fork clustering only works in Linux
    AvoidSolitary boolean, should configurations with "solitary nodes", i.e., nodes without associted points be discarded?
    FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    beta positive numeric, the value of the beta parameter of the penalized elastic energy
    EmbPointProb numeric between 0 and 1. If less than 1 point will be sampled at each iteration.
    EmbPointProb indicates the probability of using each points. This is an *experimental* feature, which may
    helps speeding up the computation if a large number of points is present.
    MaxFailedOperations integer, the maximum allowed number of consecutive failed grammar operations,
    i.e. appplication of the single grammar operations, that did not produce any valid configuration
    MaxSteps integer, the maximum allowed number of steps of the algorithm. Each step is composed by the application of
    all the specified grammar operations
    GrammarOptimization boolean, should the grammar be used to optimize the graph? If True a number MaxSteps of operations will be applied.
    AdjustElasticMatrix a penalization function to adjust the elastic matrices after a configuration has been chosen (e.g., AdjustByConstant).
    If None (the default), no penalization will be used.
    AdjustVect boolean vector keeping track of the nodes for which the elasticity parameters have been adjusted.
    When True for a node its elasticity parameters will not be adjusted.
    gamma
    verbose
    AdjustElasticMatrix.Initial a penalization function to adjust the elastic matrices of the initial configuration (e.g., AdjustByConstant).
    If None (the default), no penalization will be used.

    Return
    -------
    a named list with a number of elements:
    \describe{
      \item{NodePositions}{A numeric matrix containing the positions of the nodes}
      \item{ElasticMatrix}{The elastic matrix of the graph}
      \item{ReportTable}{The report table for the graph construction}
      \item{FinalReport}{The report table associated with the final graph configuration}
      \item{Lambda}{The lambda parameter used during the graph construction}
      \item{Mu}{The mu parameter used during the graph construction}
      \item{FastSolve}{was FastSolve being used?}
    }



    @examples

    This is a low level function. See  \code{\link{computeElasticPrincipalCircle}},
    \code{\link{computeElasticPrincipalCurve}}, or \code{\link{computeElasticPrincipalTree}}


    """

    if PointWeights is None:
        PointWeights = np.ones((len(X), 1))

    if GrammarOptimization:
        print("Using grammar optimization")
        if np.isinf(MaxSteps):
            print(
                "When setting GrammarOptimization to TRUE, MaxSteps must be"
                " finite. Using MaxSteps = 1"
            )
            MaxSteps = 1

    if not isinstance(X, np.ndarray):
        raise TypeError("Please provide data matrix as an np array")

    if not CompileReport:
        verbose = False

    if "RemoveNode" in MaxNumberOfGraphCandidatesDict:
        if np.isfinite(MaxNumberOfGraphCandidatesDict["RemoveNode"]):
            raise ValueError(
                "MaxNumberOfGraphCandidates limitation not yet available for"
                " RemoveNode. Keep it to infinity"
            )

    if isinstance(ElasticMatrix, np.ndarray):
        if np.any(ElasticMatrix != ElasticMatrix.T):
            raise ValueError("Elastic matrix must be square and symmetric")

    if AdjustElasticMatrix_Initial is not None:
        ElasticMatrix, _ = AdjustElasticMatrix_Initial(
            ElasticMatrix, AdjustVect, verbose=True
        )

    ReportTable = []
    SquaredX = (X ** 2).sum(axis=1, keepdims=1)
    if GPU:
        Xcp = cupy.asarray(X)
        SquaredXcp = (Xcp ** 2).sum(axis=1, keepdims=1)
        InitNodePositions = PrimitiveElasticGraphEmbedment_cp(
            X=X,
            NodePositions=NodePositions,
            MaxNumberOfIterations=MaxNumberOfIterations,
            TrimmingRadius=TrimmingRadius,
            eps=eps,
            ElasticMatrix=ElasticMatrix,
            Mode=Mode,
            Xcp=Xcp,
            SquaredXcp=SquaredXcp,
            FixNodesAtPoints=FixNodesAtPoints,
            PointWeights=PointWeights,
        )[0]
    else:
        Xcp = None
        SquaredXcp = None
        InitNodePositions = PrimitiveElasticGraphEmbedment(
            X=X,
            NodePositions=NodePositions,
            MaxNumberOfIterations=MaxNumberOfIterations,
            TrimmingRadius=TrimmingRadius,
            eps=eps,
            ElasticMatrix=ElasticMatrix,
            Mode=Mode,
            FixNodesAtPoints=FixNodesAtPoints,
            PointWeights=PointWeights,
        )[0]

    UpdatedPG = dict(
        ElasticMatrix=ElasticMatrix,
        NodePositions=InitNodePositions,
        AdjustVect=AdjustVect,
    )

    #     if n_cores > 1:
    #         print('Copying data to shared memory for parallel processing...',end='')
    # #         ray.init(num_cpus=n_cores)
    # #         Xremote = ray.put(X)
    # #         SquaredXremote = ray.put(SquaredX)
    #         X_shape = X.shape
    #         SquaredX_shape = SquaredX.shape

    #         X_remote = multiprocessing.RawArray('d', X_shape[0] * X_shape[1])
    #         SquaredX_remote = multiprocessing.RawArray('d', SquaredX_shape[0] * SquaredX_shape[1])

    #         # Wrap remote objects as numpy arrays so we can easily manipulate their data.
    #         X_np = np.frombuffer(X_remote).reshape(X_shape)
    #         SquaredX_np = np.frombuffer(SquaredX_remote).reshape(SquaredX_shape)

    #         # Copy data to our shared array.
    #         np.copyto(X_np, X)
    #         np.copyto(SquaredX_np, SquaredX)
    #         multiproc_shared_variables = (X_remote,X_shape,SquaredX_remote,SquaredX_shape)
    #         # Initialize dictionary storing the variables passed from the init_worker.
    #         var_dict = {}

    #         print('Done')

    if verbose:
        print(
            "\nBARCODE\tENERGY\tNNODES\tNEDGES\tNRIBS\tNSTARS\tNRAYS\tNRAYS2\tMSE\tMSEP\tFVE\tFVEP\tUE\tUR\tURN\tURN2\tURSD\n"
        )

    # now we grow the graph up to NumNodes

    if (UpdatedPG["NodePositions"].shape[0] >= NumNodes) and not (GrammarOptimization):
        FinalReport = ReportOnPrimitiveGraphEmbedment(
            X=X,
            NodePositions=UpdatedPG["NodePositions"],
            ElasticMatrix=UpdatedPG["ElasticMatrix"],
            PartData=PartitionData(
                X=X,
                NodePositions=UpdatedPG["NodePositions"],
                MaxBlockSize=100000000,
                SquaredX=SquaredX,
                TrimmingRadius=TrimmingRadius,
            ),
            ComputeMSEP=ComputeMSEP,
            PointWeights=PointWeights,
        )

        return dict(
            NodePositions=UpdatedPG["NodePositions"],
            ElasticMatrix=UpdatedPG["ElasticMatrix"],
            ReportTable=FinalReport,
            FinalReport=FinalReport,
            Lambda=Lambda,
            Mu=Mu,
            Mode=Mode,
            MaxNumberOfIterations=MaxNumberOfIterations,
            eps=eps,
            times={},
            AllNodePositions={},
            AllElasticMatrices={},
            AllMergedElasticMatrices={},
            AllMergedNodePositions={},
        )

    FailedOperations = 0
    Steps = 0

    start = time.time()
    times = {}

    AllMergedElasticMatrices = {}
    AllMergedNodePositions = {}
    AllNodePositions = {}
    AllElasticMatrices = {}

    while (UpdatedPG["NodePositions"].shape[0] < NumNodes) or GrammarOptimization:
        nEdges = len(np.triu(UpdatedPG["ElasticMatrix"], 1).nonzero()[0])
        if (
            ((UpdatedPG["NodePositions"].shape[0]) >= NumNodes) or (nEdges >= NumEdges)
        ) and not GrammarOptimization:
            break

        OldPG = copy.deepcopy(UpdatedPG)

        for OpType in GrammarOrder:
            if OpType == "Grow" and len(GrowGrammars) > 0:

                for k in range(GrowGrammars.shape[0]):
                    if ShowTimer:
                        print("Growing")
                        t = time.time()

                    UpdatedPG = ApplyOptimalGraphGrammarOperation_v2(
                        X,
                        UpdatedPG["NodePositions"],
                        UpdatedPG["ElasticMatrix"],
                        GrowGrammars[k],
                        MaxBlockSize=MaxBlockSize,
                        AdjustVect=UpdatedPG["AdjustVect"],
                        SquaredX=SquaredX,
                        verbose=False,
                        MaxNumberOfIterations=MaxNumberOfIterations,
                        eps=eps,
                        TrimmingRadius=TrimmingRadius,
                        Mode=Mode,
                        FinalEnergy=FinalEnergy,
                        alpha=alpha,
                        beta=beta,
                        EmbPointProb=EmbPointProb,
                        PointWeights=PointWeights,
                        AvoidSolitary=AvoidSolitary,
                        AdjustElasticMatrix=AdjustElasticMatrix,
                        DisplayWarnings=DisplayWarnings,
                        n_cores=n_cores,
                        MinParOp=MinParOp,
                        Xcp=Xcp,
                        SquaredXcp=SquaredXcp,
                        FixNodesAtPoints=FixNodesAtPoints,
                        pseudotime=pseudotime,
                        pseudotimeLambda=pseudotimeLambda,
                        # label=label,
                        # labelLambda=labelLambda,
                        MaxNumberOfGraphCandidatesDict=MaxNumberOfGraphCandidatesDict,
                    )

                    if UpdatedPG == "failed operation":
                        print("failed operation")
                        FailedOperations += 1
                        UpdatedPG = copy.deepcopy(OldPG)
                        break
                    else:
                        FailedOperations = 0
                        if len(UpdatedPG["NodePositions"]) == 3:
                            # this is needed to erase the star elasticity coefficient which was initially assigned to both leaf nodes,
                            # one can erase this information after the number of nodes in the graph is > 2

                            inds = np.where(
                                np.sum(
                                    UpdatedPG["ElasticMatrix"]
                                    - np.diag(np.diag(UpdatedPG["ElasticMatrix"]))
                                    > 0,
                                    axis=0,
                                )
                                == 1
                            )

                            UpdatedPG["ElasticMatrix"][inds, inds] = 0

                    if ShowTimer:
                        elapsed = time.time() - t
                        print(np.round(elapsed, 4))

            if OpType == "Shrink" and len(ShrinkGrammars) > 0:
                for k in range(ShrinkGrammars.shape[0]):
                    if ShowTimer:
                        print("Shrinking")
                        t = time.time()
                    UpdatedPG = ApplyOptimalGraphGrammarOperation_v2(
                        X,
                        UpdatedPG["NodePositions"],
                        UpdatedPG["ElasticMatrix"],
                        ShrinkGrammars[k],
                        MaxBlockSize=MaxBlockSize,
                        AdjustVect=UpdatedPG["AdjustVect"],
                        SquaredX=SquaredX,
                        verbose=False,
                        MaxNumberOfIterations=MaxNumberOfIterations,
                        eps=eps,
                        TrimmingRadius=TrimmingRadius,
                        Mode=Mode,
                        FinalEnergy=FinalEnergy,
                        alpha=alpha,
                        beta=beta,
                        EmbPointProb=EmbPointProb,
                        PointWeights=PointWeights,
                        AvoidSolitary=AvoidSolitary,
                        AdjustElasticMatrix=AdjustElasticMatrix,
                        DisplayWarnings=DisplayWarnings,
                        n_cores=n_cores,
                        MinParOp=MinParOp,
                        Xcp=Xcp,
                        SquaredXcp=SquaredXcp,
                        FixNodesAtPoints=FixNodesAtPoints,
                        pseudotime=pseudotime,
                        pseudotimeLambda=pseudotimeLambda,
                        # label=label,
                        # labelLambda=labelLambda,
                        MaxNumberOfGraphCandidatesDict=MaxNumberOfGraphCandidatesDict,
                    )

                    if UpdatedPG == "failed operation":
                        print("failed operation")
                        FailedOperations += 1
                        UpdatedPG = copy.deepcopy(OldPG)
                        break
                    else:
                        FailedOperations = 0

                    if ShowTimer:
                        elapsed = time.time() - t
                        print(np.round(elapsed, 4))

        if CompileReport:
            if GPU:
                PartData = PartitionData_cp(
                    Xcp,
                    NodePositions=UpdatedPG["NodePositions"],
                    MaxBlockSize=1000000000,
                    SquaredXcp=SquaredXcp,
                    TrimmingRadius=TrimmingRadius,
                )
            else:
                PartData = PartitionData(
                    X,
                    NodePositions=UpdatedPG["NodePositions"],
                    MaxBlockSize=1000000000,
                    SquaredX=SquaredX,
                    TrimmingRadius=TrimmingRadius,
                )
            tReport = ReportOnPrimitiveGraphEmbedment(
                X=X,
                NodePositions=UpdatedPG["NodePositions"],
                ElasticMatrix=UpdatedPG["ElasticMatrix"],
                PartData=PartData,
                ComputeMSEP=ComputeMSEP,
            )

            FinalReport = copy.deepcopy(tReport)
            for k, v in tReport.items():
                if isnumeric(v):
                    tReport[k] = str(np.round(v, 4))
            ReportTable.append(tReport)

            if verbose:
                print("\t".join(tReport.values()))
        #                 print("\n")

        # Count the execution steps
        Steps += 1

        # If the number of execution steps is larger than MaxSteps stop the algorithm
        if Steps > MaxSteps or FailedOperations > MaxFailedOperations:
            break

        times[UpdatedPG["NodePositions"].shape[0]] = time.time() - start
        if StoreGraphEvolution:
            AllNodePositions[UpdatedPG["NodePositions"].shape[0]] = UpdatedPG[
                "NodePositions"
            ]
            AllElasticMatrices[UpdatedPG["NodePositions"].shape[0]] = UpdatedPG[
                "ElasticMatrix"
            ]
            AllMergedElasticMatrices[UpdatedPG["NodePositions"].shape[0]] = UpdatedPG[
                "StoreMergedElasticMatrix"
            ]
            AllMergedNodePositions[UpdatedPG["NodePositions"].shape[0]] = UpdatedPG[
                "StoreMergedNodePositions"
            ]

    if not verbose:
        if not CompileReport:
            if GPU:
                tReport = ReportOnPrimitiveGraphEmbedment(
                    X=X,
                    NodePositions=UpdatedPG["NodePositions"],
                    ElasticMatrix=UpdatedPG["ElasticMatrix"],
                    PartData=PartitionData_cp(
                        Xcp=Xcp,
                        NodePositions=UpdatedPG["NodePositions"],
                        SquaredXcp=SquaredXcp,
                        TrimmingRadius=TrimmingRadius,
                        MaxBlockSize=MaxBlockSize,
                    ),
                    ComputeMSEP=ComputeMSEP,
                )
            else:
                tReport = ReportOnPrimitiveGraphEmbedment(
                    X=X,
                    NodePositions=UpdatedPG["NodePositions"],
                    ElasticMatrix=UpdatedPG["ElasticMatrix"],
                    PartData=PartitionData(
                        X=X,
                        NodePositions=UpdatedPG["NodePositions"],
                        SquaredX=SquaredX,
                        TrimmingRadius=TrimmingRadius,
                        MaxBlockSize=MaxBlockSize,
                    ),
                    ComputeMSEP=ComputeMSEP,
                )

            FinalReport = copy.deepcopy(tReport)
            for k, v in tReport.items():
                if isnumeric(v):
                    tReport[k] = str(np.round(v, 4))

        else:
            tReport = ReportTable[-1]
        if verbose:
            print("\n")
            print(
                "BARCODE\tENERGY\tNNODES\tNEDGES\tNRIBS\tNSTARS\tNRAYS\tNRAYS2\tMSE\tMSEP\tFVE\tFVEP\tUE\tUR\tURN\tURN2\tURSD\n"
            )
            print("\t".join(tReport.values()))
            print("\n")

    if CompileReport:
        ReportTable = {k: [d[k] for d in ReportTable] for k in ReportTable[0]}

    #     if n_cores > 1:
    #         ray.shutdown()

    return dict(
        NodePositions=UpdatedPG["NodePositions"],
        ElasticMatrix=UpdatedPG["ElasticMatrix"],
        ReportTable=ReportTable,
        FinalReport=FinalReport,
        Lambda=Lambda,
        Mu=Mu,
        Mode=Mode,
        MaxNumberOfIterations=MaxNumberOfIterations,
        eps=eps,
        times=times,
        AllNodePositions=AllNodePositions,
        AllElasticMatrices=AllElasticMatrices,
        AllMergedElasticMatrices=AllMergedElasticMatrices,
        AllMergedNodePositions=AllMergedNodePositions,
    )


def computeElasticPrincipalGraph(
    Data,
    InitNodePositions,
    AdjustVect,
    InitEdges,
    NumNodes,
    NumEdges=float("inf"),
    ElasticMatrix=None,
    Lambda=0.01,
    Mu=0.1,
    MaxNumberOfIterations=100,
    eps=0.01,
    TrimmingRadius=float("inf"),
    Do_PCA=True,
    CenterData=True,
    ComputeMSEP=True,
    verbose=False,
    ShowTimer=False,
    ReduceDimension=None,
    # drawAccuracyComplexity = True,
    # drawPCAView = True,
    # drawEnergy = True,
    n_cores=1,
    # ClusType = "Sock",
    MinParOp=20,
    Mode=1,
    FinalEnergy="Base",
    alpha=0,
    beta=0,
    # gamma = 0,
    GrowGrammars=np.array([]),
    ShrinkGrammars=np.array([]),
    GrammarOptimization=False,
    MaxSteps=float("inf"),
    GrammarOrder=["Grow", "Shrink"],
    #                                 FastSolve = False,
    AvoidSolitary=False,
    EmbPointProb=1,
    PointWeights=None,
    AdjustElasticMatrix=None,
    AdjustElasticMatrix_Initial=None,
    Lambda_Initial=None,
    Mu_Initial=None,
    DisplayWarnings=False,
    StoreGraphEvolution=False,
    GPU=False,
    FixNodesAtPoints=[],
    pseudotime=None,
    pseudotimeLambda=0.01,
    label=None,
    labelLambda=0.01,
    MaxNumberOfGraphCandidatesDict={
        "AddNode2Node": float("inf"),
        "BisectEdge": float("inf"),
        "RemoveNode": float("inf"),
        "ShrinkEdge": float("inf"),
    },
):

    """
    Regularize data and construct a principal elastic graph

    This allow to perform basic data regularization before constructing a principla elastic graph.


    Data numerical 2D matrix, the n-by-m matrix with the position of n m-dimensional points
    NumNodes integer, the number of nodes of the principal graph
    Lambda real, the lambda parameter used the compute the elastic energy
    Mu real, the lambda parameter used the compute the elastic energy
    Do_PCA boolean, should data and initial node positions be PCA trnasformed?
    CenterData boolean, should data and initial node positions be centered?
    ComputeMSEP boolean, should MSEP be computed when building the report?
    ReduceDimension integer vector, vector of principal components to retain when performing
    dimensionality reduction. If None all the components will be used
    InitNodePositions numerical 2D matrix, the k-by-m matrix with k m-dimensional positions of the nodes
    in the initial step
    InitEdges numerical 2D matrix, the e-by-2 matrix with e end-points of the edges connecting the nodes
    ElasticMatrix numerical 2D matrix, the e-by-e matrix containing the elasticity parameters of the edges
    MaxNumberOfIterations integer, maximum number of steps to embed the nodes in the data
    eps real, minimal relative change in the position of the nodes to stop embedment
    TrimmingRadius real, maximal distance of point from a node to affect its embedment
    verbose boolean, should debugging information be reported?
    ShowTimer boolean, should the time to construct the graph be computed and reported for each step?
    n_cores either an integer (indicating the number of cores to used for the creation of a cluster) or
    cluster structure returned, e.g., by makeCluster. If a cluster structure is used, all the nodes must contains X
    (this is done using clusterExport)
    MinParOp integer, the minimum number of operations to use parallel computation
    GrowGrammars list of strings, the grammar to be used in the growth step
    ShrinkGrammars list of strings, the grammar to be used in the shrink step
    NumEdges integer, the maximum nulber of edges
    Mode integer, the energy computation mode
    AvoidSolitary boolean, should configurations with "solitary nodes", i.e., nodes without associated points be discarded?
    EmbPointProb numeric between 0 and 1. If less than 1 point will be sampled at each iteration.
    EmbPointProb indicates the probability of using each points. This is an *experimental* feature, which may
    helps speeding up the computation if a large number of points is present.
    FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    beta positive numeric, the value of the beta parameter of the penalized elastic energy
    ... optional parameter that will be passed to the AdjustHOS function
    AdjustVect boolean vector keeping track of the nodes for which the elasticity parameters have been adjusted.
    When True for a node its elasticity parameters will not be adjusted.
    AdjustElasticMatrix a penalization function to adjust the elastic matrices after a configuration has been chosen (e.g., AdjustByConstant).
    If None (the default), no penalization will be used.
    AdjustElasticMatrix.Initial a penalization function to adjust the elastic matrices of the initial configuration (e.g., AdjustByConstant).
    If None (the default), no penalization will be used.
    Lambda.Initial
    Mu.Initial

    Return
    -------
    a named list with a number of elements:
    \describe{
      \item{NodePositions}{A numeric matrix containing the positions of the nodes}
      \item{Edges}{A numeric matrix containing the pairs of nodes connected by edges}
      \item{ElasticMatrix}{The elastic matrix of the graph}
      \item{ReportTable}{The report table for the graph construction}
      \item{FinalReport}{The report table associated with the final graph configuration}
      \item{Lambda}{The lambda parameter used during the graph construction}
      \item{Mu}{The mu parameter used during the graph construction}
      \item{FastSolve}{was FastSolve being used?}
    }



    @examples

    This is a low level function. See  \code{\link{computeElasticPrincipalCircle}},
    \code{\link{computeElasticPrincipalCurve}}, or \code{\link{computeElasticPrincipalTree}}
    for examples


    """
    ST = datetime.datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
    t = time.time()

    if ReduceDimension is None:
        ReduceDimension = np.array(range(np.min(Data.shape)))

    elif not Do_PCA:
        if verbose:
            print("Cannot reduce dimensionality witout doing PCA (parameter" " Do_PCA)")
            print("Dimensionality reduction will be ignored")
        ReduceDimension = np.array(range(np.min(Data.shape)))

    DataCenters = np.mean(Data, axis=0)
    if CenterData:
        Data = Data - DataCenters
        InitNodePositions = InitNodePositions - DataCenters

    if Do_PCA:
        if verbose:
            print("Performing PCA")

        if isinstance(ReduceDimension, float):
            if ReduceDimension < 1:
                if verbose:
                    print(
                        "Dimensionality reduction via ratio of explained"
                        " variance (full PCA will be computed)"
                    )
                vglobal, PCAData, explainedVariances = PCA(Data)
                ReduceDimension = range(
                    np.min(
                        np.where(
                            np.cumsum(explainedVariances) / explainedVariances.sum()
                            >= ReduceDimension
                        )
                    )
                    + 1
                )
                perc = (
                    explainedVariances[ReduceDimension].sum()
                    / explainedVariances.sum()
                    * 100
                )

                InitNodePositions = InitNodePositions.dot(vglobal)
            else:
                raise ValueError("if ReduceDimension is a single value it must be < 1")

        else:
            if max(ReduceDimension + 1) > min(Data.shape):
                if verbose:
                    print(
                        "Selected dimensions are outside of the available"
                        " range. ReduceDimension will be updated"
                    )
                ReduceDimension = [
                    i for i in ReduceDimension if i in range(min(Data.shape))
                ]
            if max(ReduceDimension + 1) > min(Data.shape) * 0.75:
                if verbose:
                    print("Using standard PCA")
                vglobal, PCAData, explainedVariances = PCA(Data)
                perc = (
                    explainedVariances[ReduceDimension].sum()
                    / explainedVariances.sum()
                    * 100
                )

                InitNodePositions = InitNodePositions.dot(vglobal)

            else:
                if verbose:
                    print("Centering data and using PCA with truncated SVD")
                if not CenterData:
                    # if data was not centered, center it (for SVD)
                    DataCenters = np.mean(Data, axis=0)
                    Data = Data - DataCenters
                    InitNodePositions = InitNodePositions - DataCenters
                PCAData, explainedVariances, U, S, Vt = TruncPCA(
                    Data,
                    algorithm="randomized",
                    n_components=max(ReduceDimension + 1),
                )
                ExpVariance = np.sum(np.var(Data, axis=0))
                perc = np.sum(explainedVariances) / ExpVariance * 100

                vglobal = Vt.T
                InitNodePositions = InitNodePositions.dot(vglobal)
        if verbose:
            print(len(ReduceDimension), "dimensions are being used")
            print(
                np.round(perc, 2),
                "% of the original variance has been retained",
            )

        X = PCAData[:, ReduceDimension]
        InitNodePositions = InitNodePositions[:, ReduceDimension]

    else:
        X = Data

    if Lambda_Initial is None:
        Lambda_Initial = Lambda

    if Mu_Initial is None:
        Mu_Initial = Mu

    if FixNodesAtPoints != []:
        nFixedNodes = len(FixNodesAtPoints)

        # fixed datapoints
        flat_FixNodesAtPoints = [
            item for sublist in FixNodesAtPoints for item in sublist
        ]
        if len(set(flat_FixNodesAtPoints)) != len(flat_FixNodesAtPoints):
            raise ValueError("FixNodesAtPoints lists contain duplicate points")

        # init fixed nodes
        FixedNodePositions = np.array(
            [X[inds].mean(axis=0) for inds in FixNodesAtPoints]
        )

        # init edges at first indices (with closest NodePosition)
        closest_node = np.zeros(nFixedNodes, dtype=int)
        for i, fn in enumerate(FixedNodePositions):
            closest_node[i] = np.argmin(
                [np.linalg.norm(fn - n) for n in InitNodePositions]
            )

        if ElasticMatrix is not None:
            InitEdges, _, _ = DecodeElasticMatrix(ElasticMatrix)
        AddEdges = np.vstack((np.arange(nFixedNodes), closest_node + nFixedNodes)).T

        # concatenate
        InitNodePositions = np.concatenate((FixedNodePositions, InitNodePositions))
        InitEdges = np.concatenate((AddEdges, InitEdges + nFixedNodes))
        ElasticMatrix = MakeUniformElasticMatrix(
            Edges=InitEdges, Lambda=Lambda_Initial, Mu=Mu_Initial
        )

    if ElasticMatrix is None:
        InitElasticMatrix = MakeUniformElasticMatrix(
            Edges=InitEdges, Lambda=Lambda_Initial, Mu=Mu_Initial
        )
    else:
        if verbose:
            print(
                "The elastic matrix is being used. Edge configuration will be"
                " ignored"
            )
        InitElasticMatrix = ElasticMatrix

    if (
        InitElasticMatrix.shape[0] != InitNodePositions.shape[0]
        or InitElasticMatrix.shape[1] != InitNodePositions.shape[0]
    ):
        raise ValueError(
            "Elastic matrix incompatible with the node number. Impossible to"
            " proceed."
        )

    # Computing the graph
    if verbose:
        print(
            "Computing EPG with ",
            NumNodes,
            " nodes on ",
            Data.shape[0],
            " points and ",
            Data.shape[1],
            " dimensions",
        )

    ElData = ElPrincGraph(
        X=X,
        NumNodes=NumNodes,
        NumEdges=NumEdges,
        Lambda=Lambda,
        Mu=Mu,
        MaxNumberOfIterations=MaxNumberOfIterations,
        eps=eps,
        TrimmingRadius=TrimmingRadius,
        NodePositions=InitNodePositions,
        ElasticMatrix=InitElasticMatrix,
        AdjustVect=AdjustVect,
        CompileReport=True,
        ShowTimer=ShowTimer,
        FinalEnergy=FinalEnergy,
        alpha=alpha,
        beta=beta,
        Mode=Mode,
        GrowGrammars=GrowGrammars,
        ShrinkGrammars=ShrinkGrammars,
        GrammarOptimization=GrammarOptimization,
        MaxSteps=MaxSteps,
        GrammarOrder=GrammarOrder,
        ComputeMSEP=ComputeMSEP,
        verbose=verbose,
        AvoidSolitary=AvoidSolitary,
        EmbPointProb=EmbPointProb,
        PointWeights=PointWeights,
        AdjustElasticMatrix=AdjustElasticMatrix,
        AdjustElasticMatrix_Initial=AdjustElasticMatrix_Initial,
        DisplayWarnings=DisplayWarnings,
        n_cores=n_cores,
        MinParOp=MinParOp,
        StoreGraphEvolution=StoreGraphEvolution,
        GPU=GPU,
        FixNodesAtPoints=FixNodesAtPoints,
        pseudotime=pseudotime,
        pseudotimeLambda=pseudotimeLambda,
        label=label,
        labelLambda=labelLambda,
        MaxNumberOfGraphCandidatesDict=MaxNumberOfGraphCandidatesDict,
    )

    Edges = DecodeElasticMatrix(ElData["ElasticMatrix"])
    NodePositions = ElData["NodePositions"]
    if "AllNodePositions" not in ElData.keys():
        AllNodePositions = {}
        AllMergedNodePositions = {}
    else:
        AllNodePositions = ElData["AllNodePositions"]
        AllMergedNodePositions = ElData["AllMergedNodePositions"]

    # if drawEnergy and ElData['ReportTable'] is not None:
    #    print('MSDEnergyPlot not yet implemented')
    #     plotMSDEnergyPlot(ReportTable = ElData['ReportTable'])

    # if drawAccuracyComplexity and ElData['ReportTable'] is not None:
    #    print('accuracyComplexityPlot not yet implemented')
    #     accuracyComplexityPlot(ReportTable = ElData['ReportTable'])

    if Do_PCA:
        NodePositions = NodePositions.dot(vglobal[:, ReduceDimension].T)
        for (k, nodep), (_, allnodep) in zip(
            AllNodePositions.items(), AllMergedNodePositions.items()
        ):
            AllNodePositions[k] = nodep.dot(vglobal[:, ReduceDimension].T)
            if AllMergedNodePositions[k] is not None:
                AllMergedNodePositions[k] = allnodep.dot(vglobal[:, ReduceDimension].T)

    EndTimer = time.time() - t
    if verbose:
        print(np.round(EndTimer, 4), " seconds elapsed")

    FinalPG = dict(
        NodePositions=NodePositions,
        Edges=Edges,
        ReportTable=ElData["ReportTable"],
        FinalReport=ElData["FinalReport"],
        ElasticMatrix=ElData["ElasticMatrix"],
        Lambda=ElData["Lambda"],
        Mu=ElData["Mu"],
        TrimmingRadius=TrimmingRadius,
        Mode=ElData["Mode"],
        MaxNumberOfIterations=ElData["MaxNumberOfIterations"],
        eps=ElData["eps"],
        Date=ST,
        TicToc=EndTimer,
        times=ElData["times"],
        AllNodePositions=AllNodePositions,
        AllElasticMatrices=ElData["AllElasticMatrices"],
        AllMergedElasticMatrices=ElData["AllMergedElasticMatrices"],
        AllMergedNodePositions=AllMergedNodePositions,
    )

    # if drawPCAView:
    #    print(PlotPG(Data, FinalPG))

    if Do_PCA or CenterData:
        FinalPG["NodePositions"] = NodePositions + DataCenters
        for (k, nodep), (_, allnodep) in zip(
            FinalPG["AllNodePositions"].items(),
            FinalPG["AllMergedNodePositions"].items(),
        ):
            FinalPG["AllNodePositions"][k] = nodep + DataCenters
            if FinalPG["AllMergedNodePositions"][k] is not None:
                FinalPG["AllMergedNodePositions"][k] = allnodep + DataCenters
    return FinalPG
