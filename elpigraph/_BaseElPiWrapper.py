import numpy as np
try:
    import cupy
except:
    pass
from ._topologies import generateInitialConfiguration
from .src.distutils import PartialDistance
from .src.core import Encode2ElasticMatrix, PrimitiveElasticGraphEmbedment, PrimitiveElasticGraphEmbedment_cp
from .src.BaseElPi import computeElasticPrincipalGraph

def computeElasticPrincipalGraphWithGrammars(X,
                                             GrowGrammars,
                                             ShrinkGrammars,
                                             NumNodes,
                                             NumEdges = float('inf'),
                                             InitNodes = 2,
                                             Lambda = 0.01,
                                             Mu = 0.1,
                                             GrammarOptimization = False,
                                             MaxSteps = float('inf'),
                                             GrammarOrder = ["Grow", "Shrink"],
                                             MaxNumberOfIterations = 10,
                                             TrimmingRadius = float('inf'),
                                             eps = .01,
                                             Do_PCA = True,
                                             InitNodePositions = None,
                                             AdjustVect = None,
                                             ElasticMatrix = None,
                                             InitEdges = None,
                                             CenterData = True,
                                             ComputeMSEP = True,
                                             verbose = False,
                                             ShowTimer = False,
                                             ReduceDimension = None,
#                                             drawAccuracyComplexity = True,
#                                             drawPCAView = True,
#                                             drawEnergy = True,
                                             n_cores = 1,
                                             #ClusType = "Sock",
                                             MinParOp = 20,
                                             nReps = 1,
#                                              ParallelRep = False,
                                             Subsets = list(),
                                             ProbPoint = 1,
                                             Mode = 1,
                                             FinalEnergy = "Base",
                                             alpha = 0,
                                             beta = 0,
                                             #gamma = 0,
                                             #FastSolve = False,
                                             Configuration = "Line",
                                             DensityRadius = None,
                                             AvoidSolitary = False,
                                             EmbPointProb = 1,
                                             SampleIC = True,
                                             AvoidResampling = True,
                                             AdjustElasticMatrix = None,
                                             AdjustElasticMatrix_Initial = None,
                                             Lambda_Initial = None, 
                                             Mu_Initial = None,
                                             DisplayWarnings = True,
                                             StoreGraphEvolution = False,
                                             GPU = False): 

    '''
    #' Construct a principal graph with the specified grammar
    #'
    #' This function is a wrapper to the computeElasticPrincipalGraph function that constructs the appropriate initial graph and
    #' apply the required grammar operations. Note that this is a generic function that is called by the topology specific functions.
    #'
    #' @param X numerical 2D matrix, the n-by-m matrix with the position of n m-dimensional points
    #' @param NumNodes integer, the number of nodes of the principal graph
    #' @param Lambda real, the lambda parameter used the compute the elastic energy
    #' @param Mu real, the lambda parameter used the compute the elastic energy
    #' @param InitNodes integer, number of points to include in the initial graph
    #' @param MaxNumberOfIterations integer, maximum number of steps to embed the nodes in the data
    #' @param TrimmingRadius real, maximal distance of point from a node to affect its embedment
    #' @param eps real, minimal relative change in the position of the nodes to stop embedment 
    #' @param Do_PCA boolean, should data and initial node positions be PCA trnasformed?
    #' @param InitNodePositions numerical 2D matrix, the k-by-m matrix with k m-dimensional positions of the nodes
    #' in the initial step
    #' @param InitEdges numerical 2D matrix, the e-by-2 matrix with e end-points of the edges connecting the nodes
    #' @param ElasticMatrix numerical 2D matrix, the k-by-k elastic matrix
    #' @param CenterData boolean, should data and initial node positions be centered?
    #' @param ComputeMSEP boolean, should MSEP be computed when building the report?
    #' @param verbose boolean, should debugging information be reported?
    #' @param ShowTimer boolean, should the time to construct the graph be computed and reported for each step?
    #' @param ReduceDimension integer vector, vector of principal components to retain when performing
    #' dimensionality reduction. If NULL all the components will be used
    #' @param drawAccuracyComplexity boolean, should the accuracy VS complexity plot be reported?
    #' @param drawPCAView boolean, should a 2D plot of the points and pricipal curve be dranw for the final configuration?
    #' @param drawEnergy boolean, should changes of evergy VS the number of nodes be reported?
    #' @param n.cores either an integer (indicating the number of cores to used for the creation of a cluster) or 
    #' cluster structure returned, e.g., by makeCluster. If a cluster structure is used, all the nodes must contains X
    #' (this is done using clusterExport)
    #' @param MinParOP integer, the minimum number of operations to use parallel computation
    #' @param nReps integer, number of replica of the construction 
    #' @param ProbPoint real between 0 and 1, probability of inclusing of a single point for each computation
    #' @param Subsets list of column names (or column number). When specified a principal tree will be computed for each of the subsets specified.
    #' @param NumEdges integer, the maximum nulber of edges
    #' @param Mode integer, the energy computation mode
    #' @param FastSolve boolean, should FastSolve be used when fitting the points to the data?
    #' @param ClusType string, the type of cluster to use. It can gbe either "Sock" or "Fork".
    #' Currently fork clustering only works in Linux
    #' @param Configuration string, initial configuration type.
    #' @param DensityRadius numeric, the radius used to estimate local density. This need to be set when ICOver is equal to "Density"
    #' @param AvoidSolitary boolean, should configurations with "solitary nodes", i.e., nodes without associted points be discarded?
    #' @param FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    #' @param alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    #' @param beta positive numeric, the value of the beta parameter of the penalized elastic energy
    #' @param EmbPointProb numeric between 0 and 1. If less than 1 point will be sampled at each iteration.
    #' EmbPointProb indicates the probability of using each points. This is an *experimental* feature, which may
    #' helps speeding up the computation if a large number of points is present.
    #' @param GrowGrammars list of strings, the grammar to be used in the growth step
    #' @param ShrinkGrammars list of strings, the grammar to be used in the shrink step
    #' @param SampleIC boolean, should the initial configuration be considered on the sampled points when applicable? 
    #' @param AdjustVect boolean vector keeping track of the nodes for which the elasticity parameters have been adjusted.
    #' When true for a node its elasticity parameters will not be adjusted.
    #' @param gamma 
    #' @param AdjustElasticMatrix a penalization function to adjust the elastic matrices after a configuration has been chosen (e.g., AdjustByConstant).
    #' If NULL (the default), no penalization will be used.
    #' @param AdjustElasticMatrix.Initial a penalization function to adjust the elastic matrices of the initial configuration (e.g., AdjustByConstant).
    #' If NULL (the default), no penalization will be used.
    #' @param Lambda.Initial real, the lambda parameter used the construct the elastic matrix associted with ther initial configuration if needed.
    #' If NULL, the value of Lambda will be used.
    #' @param Mu.Initial real, the mu parameter used the construct the elastic matrix associted with ther initial configuration if needed.
    #' If NULL, the value of Mu will be used.
    #' @param GrammarOptimization boolean, should grammar optimization be perfomred? If true grammar operations that do not increase the number of
    #' nodes will be allowed 
    #' @param MaxSteps integer, max number of applications of the grammar. This value need to be less than infinity if GrammarOptimization is set to true 
    #' @param GrammarOrder character vector, the order of application of the grammars. It can be any combination of "Grow" and "Shrink"
    #' @param AvoidResampling booleand, should the sampling of initial conditions avoid reselecting the same points
    #' (or points neighbors if DensityRadius is specified)?
    #'
    #' @return A list of principal graph strucutures containing the trees constructed during the different replica of the algorithm.
    #' If the number of replicas is larger than 1. The the final element of the list is the "average tree", which is constructed by
    #' fitting the coordinates of the nodes of the reconstructed trees
    #' @export 
    #'
    #' @examples
    #'
    #'
    '''
    # Be default we are using a predefined initial configuration
    ComputeIC = False

    # Generate a dummy subset is not specified
    if(Subsets == list()):
        Subsets = [np.array(range(X.shape[1]))]


    # Prepare the list to be returned
    ReturnList = list()

    # Copy the original matrix, this is needed in case of subsetting
    Base_X = X.copy()

    # For each subset
    for j in range(len(Subsets)):

        # Generate the appropriate matrix
        X = Base_X[:, Subsets[j]]
        SquaredX = np.sum(X**2,axis=1,keepdims=1)
        if GPU:
            Xcp = cupy.asarray(X)
            SquaredXcp = Xcp.sum(axis=1,keepdims=1)

        # Define temporary variable to avoid excessing plotting
        #Intermediate_drawPCAView = drawPCAView
        #Intermediate_drawAccuracyComplexity = drawAccuracyComplexity 
        #Intermediate_drawEnergy = drawEnergy   

        Used = np.array([False] * len(X))

        for i in range(nReps):

            # Select the points to be used
            if(ProbPoint<1 and ProbPoint>0):
                SelPoints = np.random.uniform(len(X)) <= ProbPoint
            else:
                SelPoints = np.array([True]* len(X))

            # Do we need to compute the initial conditions?
            if(InitNodePositions is None or (InitEdges is None and ElasticMatrix is None)):

                print("Generating the initial configuration")

                # We are computing the initial conditions. InitNodePositions need to be reset after each step!
                ComputeIC = True

                if(SampleIC):
                    if(AvoidResampling):
                        InitialConf = generateInitialConfiguration(X[SelPoints & ~Used,:],
                                                                    Nodes = InitNodes,
                                                                    Configuration = Configuration,
                                                                    DensityRadius = DensityRadius)

                        Dist = np.min(PartialDistance(InitialConf['NodePositions'], X), axis=0)

                        if(DensityRadius):
                            Used = (Used | (Dist < DensityRadius))
                        else:
                            Used = (Used | (Dist <= np.finfo(float).min))

                        if(np.sum(Used) < len(X)*.9):
                            print("90% of the points have been used as initial conditions. Resetting.")
                    else:
                        # Construct the initial configuration
                        InitialConf = generateInitialConfiguration(X[SelPoints,:],
                                                                Nodes = InitNodes,
                                                                Configuration = Configuration,
                                                                DensityRadius = DensityRadius)


                else:
                    if(AvoidResampling):
                        InitialConf = generateInitialConfiguration(X[~Used,],
                                                                Nodes = InitNodes,
                                                                Configuration = Configuration,
                                                                DensityRadius = DensityRadius)

                        Dist = np.min(PartialDistance(InitialConf['NodePositions'], X), axis=0)

                        if(DensityRadius):
                            Used = Used | (Dist < DensityRadius)
                        else:
                            Used = Used | (Dist < np.finfo(float).min)


                        if(np.sum(Used) > len(X)*.9):
                            print("90% or more of the points have been used as initial conditions. Resetting.")


                    else:
                        # Construct the initial configuration
                        InitialConf = generateInitialConfiguration(X,
                                                            Nodes = InitNodes,
                                                            Configuration = Configuration,
                                                            DensityRadius = DensityRadius)

                # Set the initial edge configuration
                InitEdges = InitialConf['Edges']

                # Compute the initial elastic matrix
                ElasticMatrix = Encode2ElasticMatrix(Edges = InitialConf['Edges'], Lambdas = Lambda, Mus = Mu)

                # Compute the initial node position
                if GPU:
                    InitNodePositions = PrimitiveElasticGraphEmbedment_cp(
                        X = X, NodePositions = InitialConf['NodePositions'], 
                        MaxNumberOfIterations = MaxNumberOfIterations, TrimmingRadius = TrimmingRadius, eps = eps,
                        ElasticMatrix = ElasticMatrix, Mode = Mode, Xcp = Xcp, SquaredXcp = SquaredXcp, SquaredX=SquaredX)[0]
                else:
                    InitNodePositions = PrimitiveElasticGraphEmbedment(
                        X = X, NodePositions = InitialConf['NodePositions'], 
                        MaxNumberOfIterations = MaxNumberOfIterations, TrimmingRadius = TrimmingRadius, eps = eps,
                        ElasticMatrix = ElasticMatrix, Mode = Mode, SquaredX=SquaredX)[0]


            # Do we need to compute AdjustVect?
            if(AdjustVect is None):
                AdjustVect = [False] * len(InitNodePositions)


            # Limit plotting after a few examples
            #if(len(ReturnList) == 3):
            #    print("Graphical output will be suppressed for the remaining replicas")
            #    Intermediate_drawPCAView = False
            #    Intermediate_drawAccuracyComplexity = False 
            #    Intermediate_drawEnergy = False


            print("Constructing tree", i+1, "of", nReps, "/ Subset", j+1, "of", len(Subsets))
            # Run the ElPiGraph algorithm
            ReturnList.append(computeElasticPrincipalGraph(Data = X[SelPoints,:], NumNodes = NumNodes, NumEdges = NumEdges,
                                                        InitNodePositions = InitNodePositions, InitEdges = InitEdges, ElasticMatrix = ElasticMatrix,
                                                        AdjustVect = AdjustVect,
                                                        GrowGrammars = GrowGrammars,
                                                        ShrinkGrammars = ShrinkGrammars,
                                                        GrammarOptimization = GrammarOptimization,
                                                        MaxSteps = MaxSteps,
                                                        GrammarOrder = GrammarOrder,
                                                        MaxNumberOfIterations = MaxNumberOfIterations, TrimmingRadius = TrimmingRadius, eps = eps,
                                                        Lambda = Lambda, Mu = Mu, Do_PCA = Do_PCA,
                                                        CenterData = CenterData, ComputeMSEP = ComputeMSEP,
                                                        verbose = verbose, ShowTimer = ShowTimer,
                                                        ReduceDimension = ReduceDimension, Mode = Mode,
                                                        FinalEnergy = FinalEnergy, alpha = alpha, beta = beta, #gamma = gamma,
                                                        #drawAccuracyComplexity = Intermediate_drawAccuracyComplexity,
                                                        #drawPCAView = Intermediate_drawPCAView,
                                                        #drawEnergy = Intermediate_drawEnergy,
                                                        n_cores = n_cores, 
                                                        #ClusType = ClusType, 
                                                        MinParOp = MinParOp,
                                                        #FastSolve = FastSolve,
                                                        AvoidSolitary = AvoidSolitary,
                                                        EmbPointProb = EmbPointProb, AdjustElasticMatrix = AdjustElasticMatrix,
                                                        AdjustElasticMatrix_Initial = AdjustElasticMatrix_Initial,
                                                        Lambda_Initial = Lambda_Initial, Mu_Initial = Mu_Initial,
                                                        DisplayWarnings=DisplayWarnings,
                                                        StoreGraphEvolution=StoreGraphEvolution,
                                                        GPU = GPU
                                                        )
                             )

            # Save extra information
            ReturnList[-1]['SubSetID'] = j
            ReturnList[-1]['ReplicaID'] = i
            ReturnList[-1]['ProbPoint'] = ProbPoint

            # Reset InitNodePositions for the next iteration
            if(ComputeIC):
                InitNodePositions = None

    # Are we using bootstrapping (nReps > 1). If yes we compute the consensus tree
    if(nReps>1):

        print("Constructing average tree")

        # The nodes of the principal trees will be used as points to compute the consensus tree
        AllPoints = np.concatenate(([i['NodePositions'] for i in ReturnList]))

        # De we need to compute the initial conditions?
        if(InitNodePositions is None or (InitEdges is None and ElasticMatrix is None)):

            # construct the initial configuration
            InitialConf = generateInitialConfiguration(AllPoints, Nodes = InitNodes, Configuration = Configuration, DensityRadius = DensityRadius)

            # print(InitialConf)

            # Set the initial edge configuration
            InitEdges = InitialConf['Edges']

            # Compute the initial elastic matrix
            EM = Encode2ElasticMatrix(Edges = InitialConf['Edges'], Lambdas = Lambda, Mus = Mu)

            # Compute the initial node position
            if GPU:
                InitNodePositions = PrimitiveElasticGraphEmbedment_cp(
                    X = X, NodePositions = InitialConf['NodePositions'],
                    MaxNumberOfIterations = MaxNumberOfIterations, TrimmingRadius = TrimmingRadius, eps = eps,
                    ElasticMatrix = EM, Mode = Mode, Xcp = Xcp, SquaredXcp = SquaredXcp)[0]

            else:
                InitNodePositions = PrimitiveElasticGraphEmbedment(
                    X = X, NodePositions = InitialConf['NodePositions'],
                    MaxNumberOfIterations = MaxNumberOfIterations, TrimmingRadius = TrimmingRadius, eps = eps,
                    ElasticMatrix = EM, Mode = Mode)[0]


        ReturnList.append(computeElasticPrincipalGraph(Data = AllPoints, NumNodes = NumNodes, NumEdges = NumEdges,
                                                    InitNodePositions = InitNodePositions, InitEdges = InitEdges, ElasticMatrix = ElasticMatrix,
                                                    AdjustVect = AdjustVect,
                                                    GrowGrammars = GrowGrammars,
                                                    ShrinkGrammars = ShrinkGrammars,
                                                    MaxNumberOfIterations = MaxNumberOfIterations, TrimmingRadius = TrimmingRadius, eps = eps,
                                                    Lambda = Lambda, Mu = Mu, Do_PCA = Do_PCA,
                                                    CenterData = CenterData, ComputeMSEP = ComputeMSEP,
                                                    verbose = verbose, ShowTimer = ShowTimer,
                                                    ReduceDimension = None, Mode = Mode,
                                                    FinalEnergy = FinalEnergy, alpha = alpha, beta = beta, #gamma = gamma,
                                                    #drawAccuracyComplexity = drawAccuracyComplexity,
                                                    #drawPCAView = drawPCAView, drawEnergy = drawEnergy,
                                                    n_cores = n_cores, 
                                                    #ClusType = ClusType, 
                                                    MinParOp = MinParOp,
                                                    #FastSolve = FastSolve, 
                                                    AvoidSolitary = AvoidSolitary,
                                                    EmbPointProb = EmbPointProb, AdjustElasticMatrix = AdjustElasticMatrix,
                                                    AdjustElasticMatrix_Initial = AdjustElasticMatrix_Initial,
                                                    Lambda_Initial = Lambda_Initial, Mu_Initial = Mu_Initial,
                                                    DisplayWarnings=DisplayWarnings,
                                                    StoreGraphEvolution=StoreGraphEvolution,
                                                    GPU=GPU
                                                    )
                         )

        # Run the ElPiGraph algorithm
        ReturnList[-1]['SubSetID'] = j
        ReturnList[-1]['ReplicaID'] = 0
        ReturnList[-1]['ProbPoint'] = 1



    return(ReturnList)


