import numpy as np
import elpigraph
from .src.PCA import PCA, TruncPCA, PCA_gpu, TruncSVD_gpu
from .src.distutils import PartialDistance

def computeElasticPrincipalCircle(X,
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
                                 #drawAccuracyComplexity = True,
                                 #drawPCAView = True,
                                 #drawEnergy = True,
                                 n_cores = 1,
                                 #ClusType = "Sock",
                                 MinParOp = 20,
                                 nReps = 1,
                                 #ParallelRep = False,
                                 Subsets = list(),
                                 ProbPoint = 1,
                                 Mode = 1,
                                 FinalEnergy = "Base",
                                 alpha = 0,
                                 beta = 0,
                                 #gamma = 0,
                                 #FastSolve = False,
                                 Configuration = "Line",
                                 ICOver = None,
                                 DensityRadius = None,
                                 AvoidSolitary = False,
                                 EmbPointProb = 1,
                                 SampleIC = True,
                                 AvoidResampling = True,
                                 AdjustElasticMatrix = None,
                                 AdjustElasticMatrix_Initial = None,
                                 Lambda_Initial = None, 
                                 Mu_Initial = None,
                                 DisplayWarnings = False,
                                 StoreGraphEvolution = False,
                                 GPU = False):
    
    
    '''
    #' Construct a principal elastic circle
    #'
    #' This function is a wrapper to the computeElasticPrincipalGraph function that constructs the appropriate initial graph and grammars
    #' when constructing a circle
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
    #' @param Subsets list of column names (or column number). When specified a principal circle will be computed for each of the subsets specified.
    #' @param NumEdges integer, the maximum nulber of edges
    #' @param Mode integer, the energy computation mode
    #' @param FastSolve boolean, should FastSolve be used when fitting the points to the data?
    #' @param ClusType string, the type of cluster to use. It can gbe either "Sock" or "Fork".
    #' Currently fork clustering only works in Linux
    #' @param ICOver string, initial condition overlap mode. This can be used to alter the default behaviour for the initial configuration of the
    #' principal circle
    #' @param DensityRadius numeric, the radius used to estimate local density. This need to be set when ICOver is equal to "Density"
    #' @param AvoidSolitary boolean, should configurations with "solitary nodes", i.e., nodes without associted points be discarded?
    #' @param FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    #' @param alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    #' @param beta positive numeric, the value of the beta parameter of the penalized elastic energy
    #' @param EmbPointProb numeric between 0 and 1. If less than 1 point will be sampled at each iteration.
    #' EmbPointProb indicates the probability of using each points. This is an *experimental* feature, which may
    #' helps speeding up the computation if a large number of points is present.
    #' @param AdjustElasticMatrix a penalization function to adjust the elastic matrices after a configuration has been chosen (e.g., AdjustByConstant).
    #' If NULL (the default), no penalization will be used.
    #' @param ... optional parameter that will be passed to the AdjustElasticMatrix function
    #' @param AdjustVect boolean vector keeping track of the nodes for which the elasticity parameters have been adjusted.
    #' When true for a node its elasticity parameters will not be adjusted.
    #' @param ParallelRep boolean, should parallel execution be performed on the sampling instead of the the grammar evaluations?
    #' @param AvoidResampling booleand, should the sampling of initial conditions avoid reselecting the same points
    #' (or points neighbors if DensityRadius is specified)?
    #' @param SampleIC boolean, should the initial configuration be considered on the sampled points when applicable? 
    #' @param AdjustElasticMatrix.Initial a penalization function to adjust the elastic matrices of the initial configuration (e.g., AdjustByConstant).
    #' If NULL (the default), no penalization will be used.
    #' @param Lambda.Initial real, the lambda parameter used the construct the elastic matrix associted with ther initial configuration if needed.
    #' If NULL, the value of Lambda will be used.
    #' @param Mu.Initial real, the mu parameter used the construct the elastic matrix associted with ther initial configuration if needed.
    #' If NULL, the value of Mu will be used.
    #'
    #' @return
    #'
    #' A named list 
    #'
    #' @export
    #'
    #' @examples
    #'
    #' Elastic circle with different parameters
    #' PG <- computeElasticPrincipalCircle(X = circle_data, NumNodes = 30, InitNodes = 3, verbose = TRUE)
    #' PG <- computeElasticPrincipalCircle(X = circle_data, NumNodes = 30, InitNodes = 3, verbose = TRUE, Mu = 1, Lambda = .001)
    #'
    #' Bootstrapping the construction of the circle
    #' PG <- computeElasticPrincipalCircle(X = circle_data, NumNodes = 40, InitNodes = 3,
    #' drawAccuracyComplexity = FALSE, drawPCAView = FALSE, drawEnergy = FALSE,
    #' verbose = FALSE, nReps = 50, ProbPoint = .8)
    #'
    #' PlotPG(X = circle_data, TargetPG = PG[[length(PG)]], BootPG = PG[1:(length(PG)-1)])
    #'
    '''
  
    # Define the initial configuration

    if(ICOver is None):
        Configuration = "Circle"
        
    else:
        print("WARNING : ICOver is currently ignored when constructing a circle")
        Configuration = "Circle"

    if(InitNodes < 3):
        print("The initial number of nodes must be at least 3. This will be fixed")
        InitNodes = 3

    return(elpigraph.computeElasticPrincipalGraphWithGrammars(X = X,
                                             NumNodes = NumNodes,
                                             NumEdges = NumEdges,
                                             InitNodes = InitNodes,
                                             Lambda = Lambda,
                                             Mu = Mu,
                                             GrowGrammars = np.array([['bisectedge']]),
                                             ShrinkGrammars = np.array([]),
                                             MaxNumberOfIterations = MaxNumberOfIterations,
                                             TrimmingRadius = TrimmingRadius,
                                             eps = eps,
                                             Do_PCA = Do_PCA,
                                             InitNodePositions = InitNodePositions,
                                             InitEdges = InitEdges,
                                             AdjustVect = AdjustVect,
                                             Configuration = Configuration,
                                             CenterData = CenterData,
                                             ComputeMSEP = ComputeMSEP,
                                             verbose = verbose,
                                             ShowTimer = ShowTimer,
                                             ReduceDimension = ReduceDimension,
                                             #drawAccuracyComplexity = drawAccuracyComplexity,
                                             #drawPCAView = drawPCAView,
                                             #drawEnergy = drawEnergy,
                                             n_cores = n_cores,
                                             MinParOp = MinParOp,
#                                              ClusType = ClusType,
                                             nReps = nReps,
                                             Subsets = Subsets,
                                             ProbPoint = ProbPoint,
                                             Mode = Mode,
                                             FinalEnergy = FinalEnergy,
                                             alpha = alpha,
                                             beta = beta,
#                                              FastSolve = FastSolve,
                                             DensityRadius = DensityRadius,
                                             AvoidSolitary = AvoidSolitary,
                                             EmbPointProb = EmbPointProb,
                                             SampleIC = SampleIC,
                                             AvoidResampling = AvoidResampling,
#                                              ParallelRep = ParallelRep,
                                             AdjustElasticMatrix = AdjustElasticMatrix,
                                             AdjustElasticMatrix_Initial = AdjustElasticMatrix_Initial,
                                             Lambda_Initial = Lambda_Initial, Mu_Initial = Mu_Initial,
                                             DisplayWarnings = DisplayWarnings,
                                             MaxSteps = MaxSteps,
                                             StoreGraphEvolution=StoreGraphEvolution,
                                             GPU=GPU
                                             ))



def computeElasticPrincipalTree(X,
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
                                 #drawAccuracyComplexity = True,
                                 #drawPCAView = True,
                                 #drawEnergy = True,
                                 n_cores = 1,
                                 #ClusType = "Sock",
                                 MinParOp = 20,
                                 nReps = 1,
                                 #ParallelRep = False,
                                 Subsets = list(),
                                 ProbPoint = 1,
                                 Mode = 1,
                                 FinalEnergy = "Base",
                                 alpha = 0,
                                 beta = 0,
                                 #gamma = 0,
                                 #FastSolve = False,
                                 Configuration = "Line",
                                 ICOver = None,
                                 DensityRadius = None,
                                 AvoidSolitary = False,
                                 EmbPointProb = 1,
                                 SampleIC = True,
                                 AvoidResampling = True,
                                 AdjustElasticMatrix = None,
                                 AdjustElasticMatrix_Initial = None,
                                 Lambda_Initial = None, 
                                 Mu_Initial = None,
                                 DisplayWarnings = False,
                                 StoreGraphEvolution = False,
                                 GPU = False):
    '''
    #' Construct a principal elastic tree
    #'
    #' This function is a wrapper to the computeElasticPrincipalGraph function that constructs the appropriate initial graph and grammars
    #' when constructing a tree
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
    #' @param ICOver string, initial condition overlap mode. This can be used to alter the default behaviour for the initial configuration of the
    #' principal tree.
    #' @param DensityRadius numeric, the radius used to estimate local density. This need to be set when ICOver is equal to "Density"
    #' @param AvoidSolitary boolean, should configurations with "solitary nodes", i.e., nodes without associted points be discarded?
    #' @param FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    #' @param alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    #' @param beta positive numeric, the value of the beta parameter of the penalized elastic energy
    #' @param EmbPointProb numeric between 0 and 1. If less than 1 point will be sampled at each iteration.
    #' EmbPointProb indicates the probability of using each points. This is an *experimental* feature, which may
    #' helps speeding up the computation if a large number of points is present.
    #' @param AdjustElasticMatrix a penalization function to adjust the elastic matrices after a configuration has been chosen (e.g., AdjustByConstant).
    #' If NULL (the default), no penalization will be used.
    #' @param ... optional parameter that will be passed to the AdjustElasticMatrix function
    #' @param AdjustVect boolean vector keeping track of the nodes for which the elasticity parameters have been adjusted.
    #' When true for a node its elasticity parameters will not be adjusted.
    #' @param ParallelRep boolean, should parallel execution be performed on the sampling instead of the the grammar evaluations?
    #' @param AvoidResampling booleand, should the sampling of initial conditions avoid reselecting the same points
    #' (or points neighbors if DensityRadius is specified)?
    #' @param SampleIC boolean, should the initial configuration be considered on the sampled points when applicable? 
    #' @param AdjustElasticMatrix.Initial a penalization function to adjust the elastic matrices of the initial configuration (e.g., AdjustByConstant).
    #' If NULL (the default), no penalization will be used.
    #' @param Lambda.Initial real, the lambda parameter used the construct the elastic matrix associted with ther initial configuration if needed.
    #' If NULL, the value of Lambda will be used.
    #' @param Mu.Initial real, the mu parameter used the construct the elastic matrix associted with ther initial configuration if needed.
    #' If NULL, the value of Mu will be used.
    #'
    #' @return A list of principal graph strucutures containing the trees constructed during the different replica of the algorithm.
    #' If the number of replicas is larger than 1. The the final element of the list is the "average tree", which is constructed by
    #' fitting the coordinates of the nodes of the reconstructed trees
    #' @export 
    #'
    #' @examples
    #'
    #' Elastic trees with different parameters
    #' PG <- computeElasticPrincipalTree(X = tree_data, NumNodes = 50, InitNodes = 2, verbose = TRUE)
    #'
    #' PG <- computeElasticPrincipalTree(X = tree_data, NumNodes = 50, InitNodes = 2, verbose = TRUE, Mu = 1, Lambda = .001)
    #'
    #'
    #' Bootstrapping the construction of the tree
    #' PG <- computeElasticPrincipalTree(X = tree_data, NumNodes = 40, InitNodes = 2,
    #' drawAccuracyComplexity = FALSE, drawPCAView = FALSE, drawEnergy = FALSE,
    #' verbose = FALSE, nReps = 25, ProbPoint = .9)
    #'
    #' PlotPG(X = tree_data, TargetPG = PG[[length(PG)]], BootPG = PG[1:(length(PG)-1)])
    #'
    '''
  
    # Define the initial configuration

    if(ICOver is None):
        Configuration = "Line"
        
    else:
        Configuration = ICOver

    return(elpigraph.computeElasticPrincipalGraphWithGrammars(X = X,
                                             NumNodes = NumNodes,
                                             NumEdges = NumEdges,
                                             InitNodes = InitNodes,
                                             Lambda = Lambda,
                                             Mu = Mu,
                                             GrowGrammars = np.array([['bisectedge','addnode2node'],['bisectedge','addnode2node']]),
                                             ShrinkGrammars = np.array([['shrinkedge','removenode']]),
                                             MaxNumberOfIterations = MaxNumberOfIterations,
                                             TrimmingRadius = TrimmingRadius,
                                             eps = eps,
                                             Do_PCA = Do_PCA,
                                             InitNodePositions = InitNodePositions,
                                             InitEdges = InitEdges,
                                             AdjustVect = AdjustVect,
                                             Configuration = Configuration,
                                             CenterData = CenterData,
                                             ComputeMSEP = ComputeMSEP,
                                             verbose = verbose,
                                             ShowTimer = ShowTimer,
                                             ReduceDimension = ReduceDimension,
                                             #drawAccuracyComplexity = drawAccuracyComplexity,
                                             #drawPCAView = drawPCAView,
                                             #drawEnergy = drawEnergy,
                                             n_cores = n_cores,
                                             MinParOp = MinParOp,
#                                              ClusType = ClusType,
                                             nReps = nReps,
                                             Subsets = Subsets,
                                             ProbPoint = ProbPoint,
                                             Mode = Mode,
                                             FinalEnergy = FinalEnergy,
                                             alpha = alpha,
                                             beta = beta,
#                                              FastSolve = FastSolve,
                                             DensityRadius = DensityRadius,
                                             AvoidSolitary = AvoidSolitary,
                                             EmbPointProb = EmbPointProb,
                                             SampleIC = SampleIC,
                                             AvoidResampling = AvoidResampling,
#                                              ParallelRep = ParallelRep,
                                             AdjustElasticMatrix = AdjustElasticMatrix,
                                             AdjustElasticMatrix_Initial = AdjustElasticMatrix_Initial,
                                             Lambda_Initial = Lambda_Initial, Mu_Initial = Mu_Initial,
                                             DisplayWarnings = DisplayWarnings,
                                             MaxSteps = MaxSteps,
                                             StoreGraphEvolution=StoreGraphEvolution,
                                             GPU=GPU
                                             ))



def computeElasticPrincipalCurve(X,
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
                                 #drawAccuracyComplexity = True,
                                 #drawPCAView = True,
                                 #drawEnergy = True,
                                 n_cores = 1,
                                 #ClusType = "Sock",
                                 MinParOp = 20,
                                 nReps = 1,
                                 #ParallelRep = False,
                                 Subsets = list(),
                                 ProbPoint = 1,
                                 Mode = 1,
                                 FinalEnergy = "Base",
                                 alpha = 0,
                                 beta = 0,
                                 #gamma = 0,
                                 #FastSolve = False,
                                 Configuration = "Line",
                                 ICOver = None,
                                 DensityRadius = None,
                                 AvoidSolitary = False,
                                 EmbPointProb = 1,
                                 SampleIC = True,
                                 AvoidResampling = True,
                                 AdjustElasticMatrix = None,
                                 AdjustElasticMatrix_Initial = None,
                                 Lambda_Initial = None, 
                                 Mu_Initial = None,
                                 DisplayWarnings = False,
                                 StoreGraphEvolution = False,
                                 GPU = False):
    
    ''' 
    #' Construct a princial elastic curve
    #'
    #' This function is a wrapper to the computeElasticPrincipalGraph function that constructs the appropriate initial graph and grammars
    #' when constructing a curve
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
    #' @param Subsets list of column names (or column number). When specified a principal curve will be computed for each of the subsets specified.
    #' @param NumEdges integer, the maximum nulber of edges
    #' @param Mode integer, the energy computation mode
    #' @param FastSolve boolean, should FastSolve be used when fitting the points to the data?
    #' @param ClusType string, the type of cluster to use. It can gbe either "Sock" or "Fork".
    #' Currently fork clustering only works in Linux
    #' @param ICOver string, initial condition overlap mode. This can be used to alter the default behaviour for the initial configuration of the
    #' principal curve.
    #' @param DensityRadius numeric, the radius used to estimate local density. This need to be set when ICOver is equal to "Density"
    #' @param AvoidSolitary boolean, should configurations with "solitary nodes", i.e., nodes without associted points be discarded?
    #' @param FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    #' @param alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    #' @param beta positive numeric, the value of the beta parameter of the penalized elastic energy
    #' @param EmbPointProb numeric between 0 and 1. If less than 1 point will be sampled at each iteration.
    #' EmbPointProb indicates the probability of using each points. This is an *experimental* feature, which may
    #' helps speeding up the computation if a large number of points is present.
    #' @param AdjustElasticMatrix a penalization function to adjust the elastic matrices after a configuration has been chosen (e.g., AdjustByConstant).
    #' If NULL (the default), no penalization will be used.
    #' @param ... optional parameter that will be passed to the AdjustElasticMatrix function
    #' @param AdjustVect boolean vector keeping track of the nodes for which the elasticity parameters have been adjusted.
    #' When true for a node its elasticity parameters will not be adjusted.
    #' @param AdjustElasticMatrix.Initial a penalization function to adjust the elastic matrices of the initial configuration (e.g., AdjustByConstant).
    #' If NULL (the default), no penalization will be used.
    #' @param Lambda.Initial real, the lambda parameter used the construct the elastic matrix associted with ther initial configuration if needed.
    #' If NULL, the value of Lambda will be used.
    #' @param Mu.Initial real, the mu parameter used the construct the elastic matrix associted with ther initial configuration if needed.
    #' If NULL, the value of Mu will be used.
    #' @param ParallelRep boolean, should parallel execution be performed on the sampling instead of the the grammar evaluations?
    #' @param AvoidResampling booleand, should the sampling of initial conditions avoid reselecting the same points
    #' (or points neighbors if DensityRadius is specified)?
    #' @param SampleIC boolean, should the initial configuration be considered on the sampled points when applicable? 
    #'
    #' @return A list of principal graph strucutures containing the curves constructed during the different replica of the algorithm.
    #' If the number of replicas is larger than 1. The the final element of the list is the "average curve", which is constructed by
    #' fitting the coordinates of the nodes of the reconstructed curves
    #' @export 
    #'
    #' @examples
    #'
    #' Elastic curve with different parameters
    #' PG <- computeElasticPrincipalCurve(X = tree_data, NumNodes = 30, InitNodes = 2, verbose = TRUE)
    #' PG <- computeElasticPrincipalCurve(X = circle_data, NumNodes = 30, InitNodes = 2, verbose = TRUE)
    #'
    #' PG <- computeElasticPrincipalCurve(X = tree_data, NumNodes = 30, InitNodes = 2, verbose = TRUE, Mu = 1, Lambda = .001)
    #' PG <- computeElasticPrincipalCurve(X = circle_data, NumNodes = 30, InitNodes = 2, verbose = TRUE, Mu = 1, Lambda = .001)
    #'
    #'
    #' Bootstrapping the construction of the curve
    #' PG <- computeElasticPrincipalCurve(X = tree_data, NumNodes = 40, InitNodes = 2,
    #' drawAccuracyComplexity = FALSE, drawPCAView = FALSE, drawEnergy = FALSE,
    #' verbose = FALSE, nReps = 50, ProbPoint = .8)
    #'
    #' PlotPG(X = tree_data, TargetPG = PG[[length(PG)]], BootPG = PG[1:(length(PG)-1)])
    #'
    '''
    # Define the initial configuration
    if(ICOver is None):
        Configuration = "Line"
        
    else:
        Configuration = ICOver


    return(elpigraph.computeElasticPrincipalGraphWithGrammars(X = X,
                                             NumNodes = NumNodes,
                                             NumEdges = NumEdges,
                                             InitNodes = InitNodes,
                                             Lambda = Lambda,
                                             Mu = Mu,
                                             GrowGrammars = np.array([['bisectedge']]),
                                             ShrinkGrammars = np.array([]),
                                             MaxNumberOfIterations = MaxNumberOfIterations,
                                             TrimmingRadius = TrimmingRadius,
                                             eps = eps,
                                             Do_PCA = Do_PCA,
                                             InitNodePositions = InitNodePositions,
                                             InitEdges = InitEdges,
                                             AdjustVect = AdjustVect,
                                             Configuration = Configuration,
                                             CenterData = CenterData,
                                             ComputeMSEP = ComputeMSEP,
                                             verbose = verbose,
                                             ShowTimer = ShowTimer,
                                             ReduceDimension = ReduceDimension,
                                             #drawAccuracyComplexity = drawAccuracyComplexity,
                                             #drawPCAView = drawPCAView,
                                             #drawEnergy = drawEnergy,
                                             n_cores = n_cores,
                                             MinParOp = MinParOp,
#                                              ClusType = ClusType,
                                             nReps = nReps,
                                             Subsets = Subsets,
                                             ProbPoint = ProbPoint,
                                             Mode = Mode,
                                             FinalEnergy = FinalEnergy,
                                             alpha = alpha,
                                             beta = beta,
#                                              FastSolve = FastSolve,
                                             DensityRadius = DensityRadius,
                                             AvoidSolitary = AvoidSolitary,
                                             EmbPointProb = EmbPointProb,
                                             SampleIC = SampleIC,
                                             AvoidResampling = AvoidResampling,
#                                              ParallelRep = ParallelRep,
                                             AdjustElasticMatrix = AdjustElasticMatrix,
                                             AdjustElasticMatrix_Initial = AdjustElasticMatrix_Initial,
                                             Lambda_Initial = Lambda_Initial, Mu_Initial = Mu_Initial,
                                             DisplayWarnings = DisplayWarnings,
                                             MaxSteps = MaxSteps,
                                             StoreGraphEvolution=StoreGraphEvolution,
                                             GPU=GPU
                                             ))



def fineTuneBR(X,
             NumNodes,
             NumEdges = float('inf'),
             InitNodes = 2,
             Lambda = 0.01,
             Mu = 0.1,
             GrammarOptimization = False,
             MaxSteps = 100,
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
             #drawAccuracyComplexity = True,
             #drawPCAView = True,
             #drawEnergy = True,
             n_cores = 1,
             #ClusType = "Sock",
             MinParOp = 20,
             nReps = 1,
             #ParallelRep = False,
             Subsets = list(),
             ProbPoint = 1,
             Mode = 1,
             FinalEnergy = "Base",
             alpha = 0,
             beta = 0,
             #gamma = 0,
             #FastSolve = False,
             Configuration = "Line",
             ICOver = None,
             DensityRadius = None,
             AvoidSolitary = False,
             EmbPointProb = 1,
             SampleIC = True,
             AvoidResampling = True,
             AdjustElasticMatrix = None,
             AdjustElasticMatrix_Initial = None,
             Lambda_Initial = None, 
             Mu_Initial = None,
             DisplayWarnings = False,
             StoreGraphEvolution = False,
             GPU = False):

    '''
    #' Expand the nodes around a branching point
    #'
    #' This function is a wrapper to the computeElasticPrincipalGraph function that construct the appropriate initial graph and grammars
    #' when increasing the nume number around the branching point
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
    #' @param Subsets list of column names (or column number). When specified a principal curve will be computed for each of the subsets specified.
    #' @param NumEdges integer, the maximum nulber of edges
    #' @param Mode integer, the energy computation mode
    #' @param FastSolve boolean, should FastSolve be used when fitting the points to the data?
    #' @param ClusType string, the type of cluster to use. It can gbe either "Sock" or "Fork".
    #' Currently fork clustering only works in Linux
    #' @param ICOver string, initial condition overlap mode. This can be used to alter the default behaviour for the initial configuration of the
    #' principal curve.
    #' @param DensityRadius numeric, the radius used to estimate local density. This need to be set when ICOver is equal to "Density"
    #' @param AvoidSolitary boolean, should configurations with "solitary nodes", i.e., nodes without associted points be discarded?
    #' @param FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    #' @param alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    #' @param beta positive numeric, the value of the beta parameter of the penalized elastic energy
    #' @param EmbPointProb numeric between 0 and 1. If less than 1 point will be sampled at each iteration.
    #' EmbPointProb indicates the probability of using each points. This is an *experimental* feature, which may
    #' helps speeding up the computation if a large number of points is present.
    #' @param AdjustElasticMatrix a penalization function to adjust the elastic matrices after a configuration has been chosen (e.g., AdjustByConstant).
    #' If NULL (the default), no penalization will be used.
    #' @param ... optional parameter that will be passed to the AdjustElasticMatrix function
    #' @param AdjustVect boolean vector keeping track of the nodes for which the elasticity parameters have been adjusted.
    #' When true for a node its elasticity parameters will not be adjusted.
    #' @param AdjustElasticMatrix.Initial a penalization function to adjust the elastic matrices of the initial configuration (e.g., AdjustByConstant).
    #' If NULL (the default), no penalization will be used.
    #' @param Lambda.Initial real, the lambda parameter used the construct the elastic matrix associted with ther initial configuration if needed.
    #' If NULL, the value of Lambda will be used.
    #' @param Mu.Initial real, the mu parameter used the construct the elastic matrix associted with ther initial configuration if needed.
    #' If NULL, the value of Mu will be used.
    #' @param ParallelRep 
    #' @param SampleIC boolean, should the initial configuration be considered on the sampled points when applicable? 
    #'
    #' @return A list of principal graph strucutures containing the curves constructed during the different replica of the algorithm.
    #' If the number of replicas is larger than 1. The the final element of the list is the "average curve", which is constructed by
    #' fitting the coordinates of the nodes of the reconstructed curves
    #' @export 
    #'
    #' @examples
    #'
    #' Elastic curve with different parameters
    #' PG <- computeElasticPrincipalCurve(X = tree_data, NumNodes = 30, InitNodes = 2, verbose = TRUE)
    #' PG <- computeElasticPrincipalCurve(X = circle_data, NumNodes = 30, InitNodes = 2, verbose = TRUE)
    #'
    #' PG <- computeElasticPrincipalCurve(X = tree_data, NumNodes = 30, InitNodes = 2, verbose = TRUE, Mu = 1, Lambda = .001)
    #' PG <- computeElasticPrincipalCurve(X = circle_data, NumNodes = 30, InitNodes = 2, verbose = TRUE, Mu = 1, Lambda = .001)
    #'
    #'
    #' Bootstrapping the construction of the curve
    #' PG <- computeElasticPrincipalCurve(X = tree_data, NumNodes = 40, InitNodes = 2,
    #' drawAccuracyComplexity = FALSE, drawPCAView = FALSE, drawEnergy = FALSE,
    #' verbose = FALSE, nReps = 50, ProbPoint = .8)
    #'
    #' PlotPG(X = tree_data, TargetPG = PG[[length(PG)]], BootPG = PG[1:(length(PG)-1)])
    #'
    '''
    # Define the initial configuration

    if(ICOver is None):
        Configuration = "Line"
        
    else:
        Configuration = ICOver
        
            
    if(NumNodes > len(InitNodePositions)):
        GrammarOptimization = False
        GrowGrammars = np.array([['bisectedge_3']])
        ShrinkGrammars = np.array([['shrinkedge_3']])
        GrammarOrder = ["Grow", "Shrink", "Grow"]
    else:
        GrammarOptimization = True
        GrowGrammars = np.array([['bisectedge_3']])
        ShrinkGrammars = np.array([['shrinkedge_3']])
        GrammarOrder = ["Shrink", "Grow"]

    return(elpigraph.computeElasticPrincipalGraphWithGrammars(X = X,
                                             NumNodes = NumNodes,
                                             NumEdges = NumEdges,
                                             InitNodes = InitNodes,
                                             Lambda = Lambda,
                                             Mu = Mu,
                                             GrowGrammars = GrowGrammars,
                                             ShrinkGrammars = ShrinkGrammars,
                                             GrammarOrder = GrammarOrder,
                                             GrammarOptimization = GrammarOptimization,
                                             MaxSteps = MaxSteps,
                                             MaxNumberOfIterations = MaxNumberOfIterations,
                                             TrimmingRadius = TrimmingRadius,
                                             eps = eps,
                                             Do_PCA = Do_PCA,
                                             InitNodePositions = InitNodePositions,
                                             InitEdges = InitEdges,
                                             AdjustVect = AdjustVect,
                                             Configuration = Configuration,
                                             CenterData = CenterData,
                                             ComputeMSEP = ComputeMSEP,
                                             verbose = verbose,
                                             ShowTimer = ShowTimer,
                                             ReduceDimension = ReduceDimension,
                                             #drawAccuracyComplexity = drawAccuracyComplexity,
                                             #drawPCAView = drawPCAView,
                                             #drawEnergy = drawEnergy,
                                             n_cores = n_cores,
                                             MinParOp = MinParOp,
#                                              ClusType = ClusType,
                                             nReps = nReps,
                                             Subsets = Subsets,
                                             ProbPoint = ProbPoint,
                                             Mode = Mode,
                                             FinalEnergy = FinalEnergy,
                                             alpha = alpha,
                                             beta = beta,
#                                              FastSolve = FastSolve,
                                             DensityRadius = DensityRadius,
                                             AvoidSolitary = AvoidSolitary,
                                             EmbPointProb = EmbPointProb,
                                             SampleIC = SampleIC,
                                             AvoidResampling = AvoidResampling,
#                                              ParallelRep = ParallelRep,
                                             AdjustElasticMatrix = AdjustElasticMatrix,
                                             AdjustElasticMatrix_Initial = AdjustElasticMatrix_Initial,
                                             Lambda_Initial = Lambda_Initial, Mu_Initial = Mu_Initial,
                                             DisplayWarnings = DisplayWarnings,
                                             StoreGraphEvolution=StoreGraphEvolution,
                                             GPU = GPU
                                             ))




def GrowLeaves(X,
             NumNodes,
             NumEdges = float('inf'),
             InitNodes = 2,
             Lambda = 0.01,
             Mu = 0.1,
             MaxSteps = 100,
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
             #drawAccuracyComplexity = True,
             #drawPCAView = True,
             #drawEnergy = True,
             n_cores = 1,
             #ClusType = "Sock",
             MinParOp = 20,
             nReps = 1,
             #ParallelRep = False,
             Subsets = list(),
             ProbPoint = 1,
             Mode = 1,
             FinalEnergy = "Base",
             alpha = 0,
             beta = 0,
             #gamma = 0,
             #FastSolve = False,
             Configuration = "Line",
             ICOver = None,
             DensityRadius = None,
             AvoidSolitary = False,
             EmbPointProb = 1,
             SampleIC = True,
             AvoidResampling = True,
             AdjustElasticMatrix = None,
             AdjustElasticMatrix_Initial = None,
             Lambda_Initial = None, 
             Mu_Initial = None,
             DisplayWarnings = False,
             StoreGraphEvolution = False,
             GPU=False):
    
    '''
    #' Extend the leaves of a graph
    #'
    #' This function is a wrapper to the computeElasticPrincipalGraph function that construct the appropriate initial graph and grammars
    #' when increasing the nume number around the branching point
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
    #' (this is done using clusterExport)`
    #' @param MinParOP integer, the minimum number of operations to use parallel computation
    #' @param nReps integer, number of replica of the construction 
    #' @param ProbPoint real between 0 and 1, probability of inclusing of a single point for each computation
    #' @param Subsets list of column names (or column number). When specified a principal curve will be computed for each of the subsets specified.
    #' @param NumEdges integer, the maximum nulber of edges
    #' @param Mode integer, the energy computation mode
    #' @param FastSolve boolean, should FastSolve be used when fitting the points to the data?
    #' @param ClusType string, the type of cluster to use. It can gbe either "Sock" or "Fork".
    #' Currently fork clustering only works in Linux
    #' @param ICOver string, initial condition overlap mode. This can be used to alter the default behaviour for the initial configuration of the
    #' principal curve.
    #' @param DensityRadius numeric, the radius used to estimate local density. This need to be set when ICOver is equal to "Density"
    #' @param AvoidSolitary boolean, should configurations with "solitary nodes", i.e., nodes without associted points be discarded?
    #' @param FinalEnergy string indicating the final elastic emergy associated with the configuration. Currently it can be "Base" or "Penalized"
    #' @param alpha positive numeric, the value of the alpha parameter of the penalized elastic energy
    #' @param beta positive numeric, the value of the beta parameter of the penalized elastic energy
    #' @param EmbPointProb numeric between 0 and 1. If less than 1 point will be sampled at each iteration.
    #' EmbPointProb indicates the probability of using each points. This is an *experimental* feature, which may
    #' helps speeding up the computation if a large number of points is present.
    #' @param AdjustElasticMatrix a penalization function to adjust the elastic matrices after a configuration has been chosen (e.g., AdjustByConstant).
    #' If NULL (the default), no penalization will be used.
    #' @param ... optional parameter that will be passed to the AdjustElasticMatrix function
    #' @param AdjustVect boolean vector keeping track of the nodes for which the elasticity parameters have been adjusted.
    #' When true for a node its elasticity parameters will not be adjusted.
    #' @param AdjustElasticMatrix.Initial a penalization function to adjust the elastic matrices of the initial configuration (e.g., AdjustByConstant).
    #' If NULL (the default), no penalization will be used.
    #' @param Lambda.Initial real, the lambda parameter used the construct the elastic matrix associted with ther initial configuration if needed.
    #' If NULL, the value of Lambda will be used.
    #' @param Mu.Initial real, the mu parameter used the construct the elastic matrix associted with ther initial configuration if needed.
    #' If NULL, the value of Mu will be used.
    #' @param ParallelRep 
    #' @param SampleIC boolean, should the initial configuration be considered on the sampled points when applicable? 
    #'
    #' @return A list of principal graph strucutures containing the curves constructed during the different replica of the algorithm.
    #' If the number of replicas is larger than 1. The the final element of the list is the "average curve", which is constructed by
    #' fitting the coordinates of the nodes of the reconstructed curves
    #' @export 
    #'
    #' @examples
    #'
    #' Elastic curve with different parameters
    #' PG <- computeElasticPrincipalCurve(X = tree_data, NumNodes = 30, InitNodes = 2, verbose = TRUE)
    #' PG <- computeElasticPrincipalCurve(X = circle_data, NumNodes = 30, InitNodes = 2, verbose = TRUE)
    #'
    #' PG <- computeElasticPrincipalCurve(X = tree_data, NumNodes = 30, InitNodes = 2, verbose = TRUE, Mu = 1, Lambda = .001)
    #' PG <- computeElasticPrincipalCurve(X = circle_data, NumNodes = 30, InitNodes = 2, verbose = TRUE, Mu = 1, Lambda = .001)
    #'
    #'
    #' Bootstrapping the construction of the curve
    #' PG <- computeElasticPrincipalCurve(X = tree_data, NumNodes = 40, InitNodes = 2,
    #' drawAccuracyComplexity = FALSE, drawPCAView = FALSE, drawEnergy = FALSE,
    #' verbose = FALSE, nReps = 50, ProbPoint = .8)
    #'
    #' PlotPG(X = tree_data, TargetPG = PG[[length(PG)]], BootPG = PG[1:(length(PG)-1)])
    #'
    '''

    # Define the initial configuration

    if(ICOver is None):
        Configuration = "Line"
        
    else:
        Configuration = ICOver


    return(elpigraph.computeElasticPrincipalGraphWithGrammars(X = X,
                                             NumNodes = NumNodes,
                                             NumEdges = NumEdges,
                                             InitNodes = InitNodes,
                                             Lambda = Lambda,
                                             Mu = Mu,
                                             GrowGrammars = np.array([['addnodetonode_1']]),
                                             ShrinkGrammars = np.array([]),
                                             GrammarOrder = ['Grow'],
                                             GrammarOptimization = False,
                                             MaxSteps = MaxSteps,
                                             MaxNumberOfIterations = MaxNumberOfIterations,
                                             TrimmingRadius = TrimmingRadius,
                                             eps = eps,
                                             Do_PCA = Do_PCA,
                                             InitNodePositions = InitNodePositions,
                                             InitEdges = InitEdges,
                                             AdjustVect = AdjustVect,
                                             Configuration = Configuration,
                                             CenterData = CenterData,
                                             ComputeMSEP = ComputeMSEP,
                                             verbose = verbose,
                                             ShowTimer = ShowTimer,
                                             ReduceDimension = ReduceDimension,
                                             #drawAccuracyComplexity = drawAccuracyComplexity,
                                             #drawPCAView = drawPCAView,
                                             #drawEnergy = drawEnergy,
                                             n_cores = n_cores,
                                             MinParOp = MinParOp,
#                                              ClusType = ClusType,
                                             nReps = nReps,
                                             Subsets = Subsets,
                                             ProbPoint = ProbPoint,
                                             Mode = Mode,
                                             FinalEnergy = FinalEnergy,
                                             alpha = alpha,
                                             beta = beta,
#                                              FastSolve = FastSolve,
                                             DensityRadius = DensityRadius,
                                             AvoidSolitary = AvoidSolitary,
                                             EmbPointProb = EmbPointProb,
                                             SampleIC = SampleIC,
                                             AvoidResampling = AvoidResampling,
#                                              ParallelRep = ParallelRep,
                                             AdjustElasticMatrix = AdjustElasticMatrix,
                                             AdjustElasticMatrix_Initial = AdjustElasticMatrix_Initial,
                                             Lambda_Initial = Lambda_Initial, Mu_Initial = Mu_Initial,
                                             DisplayWarnings = DisplayWarnings,
                                             StoreGraphEvolution = StoreGraphEvolution,
                                             GPU = GPU
                                             ))




def generateInitialConfiguration(X, Nodes, Configuration = "Line",
                                 DensityRadius = None, MaxPoints = 10000,
                                 PCADensity = True, CenterDataDensity = True):
    '''  
    #' Produce an initial graph with a given structure
    #'
    #' @param X numerical 2D matrix, the n-by-m matrix with the position of n m-dimensional points
    #' @param Configuration string, type of graph to return. It should be one of the following value
    #' \describe{
    #'   \item{"Line"}{Points are placed on the 1st principal component between mean-sd and mean+sd}
    #'   \item{"Circle"}{Points are placed on the the plane induced by the 1st and 2nd principal components.
    #'   In both dimensions they are placed between mean-sd and mean+sd}
    #'   \item{"Density"}{Two points are selected randomly in the neighborhood of one of the points with
    #'   the largest number of neighbour points}
    #'   \item{"DensityProb"}{Two points are selected randomly in the neighborhood of one of a point randomly
    #'   selected with a probability proportional to the number of neighbour points}
    #'   \item{"Random"}{Two points are returned. The first is selected at random, and the second is selected with
    #'   a probability inversely proportional to thr distance to the 1st point selected}
    #' }
    #'
    #' @param Nodes integer, number of nodes of the graph
    #' @param DensityRadius numeric, the radius used to estimate local density. This need to be set when Configuration is equal to "Density"
    #' @param MaxPoints integer, the maximum number of points for which the local density will be estimated. If the number of data points is
    #' larger than MaxPoints, a subset of the original points will be sampled
    #' @param PCADensity boolean, should PCA be applied to the data before computing the most dense area 
    #'
    #' @return
    #' @export
    #'
    #' @examples
    '''
    DONE = False
    
    if(Configuration == "Line"):
        # Chain of nodes along the first principal component direction
        print("Creating a chain in the 1st PC with", Nodes, "nodes")
        mv = X.mean(axis=0)
        data_centered = X - mv
        PC1, explainedVarianceRatio, U, S, Vt = TruncPCA(data_centered,n_components=1)
        # Vt = np.abs(Vt)
        mn = np.mean(PC1)
        st = np.std(PC1,ddof=1)
        NodeP = np.dot(np.linspace(mn - st, mn + st, Nodes)[:,None],Vt)
        NodePositions = NodeP+mv[None]
        # Creating edges
        edges = np.vstack((np.arange(Nodes-1),
                           np.arange(1, Nodes))).T
        
        DONE = True

    if(Configuration == "Circle"):
        # Chain of nodes along the first principal component direction
        print("Creating a circle in the plane induced by the 1st and 2nd PCs with", Nodes, "nodes")
        
        if X.shape[1] < 3:
            if CenterDataDensity:
                mv = X.mean(axis=0)
                data_centered = X - mv
            else:
                data_centered = X
            vglobal, PCAdata, explainedVariances = PCA(data_centered)
            Vt = vglobal.T
        else:
            if CenterDataDensity:
                mv = X.mean(axis=0)
                data_centered = X - mv
            else:
                data_centered = X
            PCAdata, explainedVarianceRatio, U, S, Vt = TruncPCA(data_centered,n_components=2)
        
        Nodes_X = (np.cos(np.linspace(0,2*np.pi,Nodes + 1))*np.std(PCAdata[:,0],ddof=1))[:,None]
        Nodes_Y = (np.sin(np.linspace(0,2*np.pi,Nodes + 1))*np.std(PCAdata[:,1],ddof=1))[:,None]

        NodePositions = np.dot(np.concatenate((Nodes_X[1:], Nodes_Y[1:]),axis=1), abs(Vt)) + mv[None]       
        # Creating edges
        edges = np.vstack((np.arange(Nodes-1),
                           np.arange(1, Nodes))).T
        edges = np.concatenate((edges,np.array([[int(Nodes-1),0]])))
        
        DONE = True

    if(Configuration == "Random"):
        
        # Starting from Random Points in the data
        print("Creating a line between two random points of the data. The points will have at most a distance DensityRadius if the parameter is specified")

        ID1 = np.random.choice(len(X))
        
        Dist =  PartialDistance(X[[ID1],], X).flatten()
        with np.errstate(divide='ignore'):
            Probs = 1/(Dist)
        if(DensityRadius):
            Probs[Dist > DensityRadius] = 0
        
        Probs[np.isinf(Probs)] = 0

        ID2 = np.random.choice(len(X),p=Probs/sum(Probs))

        NodePositions = X[np.array([ID1, ID2]),:]
        
        # Creating edges
        edges = np.array([[0,1]])

        DONE = True

    if(Configuration == "RandomSpace"):

        # Starting from Random Points in the data
        print("Creating a line between a point randomly chosen uniformily in the space of points and one of its neighbours. The points will have at most a distance DensityRadius if the parameter is specified")
        from sklearn.cluster import KMeans
        KM = KMeans(n_clusters=10).fit(X)
        RandClus = np.random.choice(10)
        ID1 =  np.random.choice(np.where(KM.labels_ == RandClus)[0])
        
        Dist =  PartialDistance(X[[ID1],], X).flatten()
        
        with np.errstate(divide='ignore'):
            Probs = 1/(Dist)
        if(DensityRadius):
            Probs[Dist > DensityRadius] = 0
        Probs[np.isinf(Probs)] = 0
        
        ID2 = np.random.choice(len(X),p=Probs/sum(Probs))
        
        NodePositions = X[np.array([ID1, ID2]),:]
        
        # Creating edges
        edges = np.array([[0,1]])
        
        DONE = True

    if(Configuration == "Density"):

        if(DensityRadius is None):
            raise ValueError("DensityRadius needs to be specified for density-dependent initialization!")
        
        # Starting from Random Points in the data
        print("Creating a line in the densest part of the data")
        
        if(PCADensity):
            if CenterDataDensity:
                mv = X.mean(axis=0)
                data_centered = X - mv
            else:
                data_centered = X
            vglobal, tX_PCA, explainedVariances = PCA(data_centered)
            tX = tX_PCA
        else:
            tX = X
            
        if(len(tX) > MaxPoints):

            print("Too many points, a subset of", MaxPoints, "will be sampled")

            SampledIdxs = np.random.choice(len(tX),size=MaxPoints)
            
            PartStruct =  PartialDistance(tX[SampledIdxs, ], tX[SampledIdxs, ])
            PointsInNei = np.sum(PartStruct < DensityRadius,axis=1)
            
            if(max(PointsInNei) < 2):
                raise ValueError("DensityRadius too small (Not enough points found in the neighborhood)")

            IdMax = np.argmax(PointsInNei)

            NodePositions = X[SampledIdxs[np.random.choice(np.where(PartStruct[IdMax, ] < DensityRadius)[0], size=2)],:]
            
            # Creating edges
            edges = np.array([[0,1]])

            DONE = True

        else:
            PartStruct =  PartialDistance(tX, tX)
            PointsInNei = np.sum(PartStruct < DensityRadius, axis=1)
            
            if(max(PointsInNei) < 2):
                raise ValueError("DensityRadius too small (Not enough points found in the neighborhood)")
            
            IdMax = np.argmax(PointsInNei)
            NodePositions = X[np.random.choice(np.where(PartStruct[IdMax, ] < DensityRadius)[0], size=2),:]

            
            # Creating edges
            edges = np.array([[0,1]])

            DONE = True

    if(Configuration == "DensityProb"):

        if(DensityRadius is None):
            raise ValueError("DensityRadius need to be specified for density-dependent inizialization!")
        
        # Starting from Random Points in the data
        print("Creating a line a part of the data, chosen probabilistically by its density. DensityRadius needs to be specified!")
        
        if(PCADensity):
            if CenterDataDensity:
                mv = X.mean(axis=0)
                data_centered = X - mv
            else:
                data_centered = X
            vglobal, tX_PCA, explainedVariances = PCA(data_centered)
            tX = tX_PCA
        else:
            tX = X


        if(len(tX) > MaxPoints):

            print("Too many points, a subset of", MaxPoints, "will be sampled")

            SampledIdxs = np.random.choice(len(tX),size=MaxPoints)
            
            PartStruct =  PartialDistance(tX[SampledIdxs, ], tX[SampledIdxs, ])
            PointsInNei = np.sum(PartStruct < DensityRadius,axis=1)

            if(max(PointsInNei) < 2):
                raise ValueError("DensityRadius too small (Not enough points found in the neighborhood))!!")

            PointsInNei[PointsInNei == 1] = 0
            IdMax = np.random.choice(len(PointsInNei), p = PointsInNei/sum(PointsInNei))

            NodePositions = X[np.random.choice(np.where(PartStruct[IdMax, ] < DensityRadius)[0], size=2),:]
                
            # Creating edges
            edges = np.array([[0,1]])
          
            DONE = True

        else:
            PartStruct =  PartialDistance(tX, tX)
            PointsInNei = np.sum(PartStruct < DensityRadius,axis=1)

            if(max(PointsInNei) < 2):
                raise ValueError("DensityRadius too small (Not enough points found in the neighborhood))!!")
            

            PointsInNei[PointsInNei == 1] = 0
            IdMax = np.random.choice(len(PointsInNei), p = PointsInNei/sum(PointsInNei))

            NodePositions = X[np.random.choice(np.where(PartStruct[IdMax, ] < DensityRadius)[0], size=2),:]

            # Creating edges
            edges = np.array([[0,1]])

            DONE = True

    if(DONE):
        return(dict(NodePositions = NodePositions, Edges = edges))
    else:
        raise ValueError("Unsupported configuration!")

