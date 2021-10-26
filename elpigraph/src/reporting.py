import numpy as np
from .core import DecodeElasticMatrix, PartitionData
from .distutils import ComputePrimitiveGraphElasticEnergy

def getPrimitiveGraphStructureBarCode(ElasticMatrix):
    #Mus = ElasticMatrix.diagonal()
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    indL = Lambda > 0
    Connectivities = indL.sum(axis=0)
    Mcon = np.max(Connectivities)

    counts = np.bincount(Connectivities)[1:]
    code = '||'+str(ElasticMatrix.shape[0])

    if Mcon <= 2:
        code = '0'+code
    else:
        code = '|'.join([str(c) for c in counts[2:][::-1]])+code
    return code
                         

def project_point_onto_graph(X, NodePositions, Edges, Partition = None):
    '''                           
    #' Project data points on the precipal graph
    #'
    #' @param X numerical matrix containg points on the rows and dimensions on the columns
    #' @param NodePositions numerical matrix containg the positions of the nodes on the rows
    #' (must have the same dimensionality of X)
    #' @param Edges a 2-dimensional matrix containing edges as pairs of integers. The integers much
    #' match the rows of NodePositions
    #' @param Partition a Partition vector associating points to at most one of the nodes of the graph.
    #' It can be NULL, in which case it will be computed by the algorithm
    #'
    #' @return A list with several elements:
    #' \itemize{
    #'  \item{"X_projected "}{A matrix containing the projection of the points (on rows) on the edges of the graph}
    #'  \item{"MSEP "}{The mean squared error (distance) of the points from the graph}
    #'  \item{"ProjectionValues "}{The normalized position of the point on its associted edge.
    #'  A value <0 indicates a projection before the initial position of the node.
    #'  A value >1 indicates a projection after the final position of the node.
    #'  A value betwen 0 and 1 indicates at which percentage of the edge length the point is being projected,
    #'  e.g., a value of 0.3 indicates the 30\%.}
    #'  \item{"EdgeID "}{An integer indicating the id of the edge on which each point has been projected. Note that
    #'  if a point is projected on a node, this id will indicate one of the edges connected to that node.}
    #'  \item{"EdgeLen "}{The length of the edges described by the Edges input matrix}
    #'  \item{"NodePositions "}{the NodePositions input matrix}
    #'  \item{"Edges "}{the Edges input matrix}
    #' }
    #'
    #' @export
    #'
    #' @examples
    '''
    if Partition is None:
        Partition = PartitionData(X, NodePositions, MaxBlockSize, SquaredX=np.sum(X**2,axis=1,keepdims=1))[0]

    X_projected = np.zeros(X.shape)
    ProjectionValues = np.array([np.inf]*len(X))
    Distances_squared = np.array([np.inf]*len(X))
    EdgeID = np.array([np.nan]*len(X))
    EdgeLen = np.array([np.inf]*len(Edges))

    for i in range(len(Edges)):
        Idxs = np.where(np.isin(Partition, Edges[i,:]))[0]

        PrjStruct = project_point_onto_edge(X = X[Idxs,:], NodePositions = NodePositions[Edges[i,],:], Edge = np.array([0,1]))

        if len(Idxs)> 0:
            ToFill = PrjStruct['Distance_Squared'] < Distances_squared[Idxs]
            X_projected[Idxs[ToFill],] = PrjStruct['X_Projected'][ToFill,]
            try:
                ProjectionValues[Idxs[ToFill]] = PrjStruct['Projection_Value'][ToFill]
            except:
                ### only one index and proj value
                ProjectionValues[Idxs[ToFill]] = np.array([PrjStruct['Projection_Value']])[ToFill]

            Distances_squared[Idxs[ToFill]] = PrjStruct['Distance_Squared'][ToFill]
            EdgeID[Idxs[ToFill]] = i


        EdgeLen[i] = np.sqrt(PrjStruct['EdgeLen_Squared'])


    return(dict(X_projected = X_projected,
              MSEP = np.mean(Distances_squared),
              ProjectionValues = ProjectionValues,
              EdgeID = EdgeID,
              EdgeLen = EdgeLen,
              NodePositions = NodePositions,
              Edges = Edges))


                         
def project_point_onto_edge(X, NodePositions, Edge, ExtProj = False):
    '''
    #' Title
    #'
    #' @param X
    #' @param NodePositions
    #' @param Edge
    #'
    #' @return
    #' @export
    #'
    #' @examples
    '''
    
    vec =(NodePositions[Edge[1],:] - NodePositions[Edge[0],:])[:,None]
    u = ((X.T - NodePositions[Edge[0]][:,None]).T @ vec) / (vec.T @ vec)
    u[~np.isfinite(u)] = 0
    u = u.squeeze()
    X_Projected = X.copy()
    X_Projected[:] = np.nan

    if np.any(u<0):
        X_Projected[u<0,:] = np.repeat(NodePositions[Edge[0],:][None], np.sum(u<0),axis=0)


    if np.any(u>1):
        X_Projected[u>1,:] = np.repeat(NodePositions[Edge[1],:][None], np.sum(u>1),axis=0)


    if ExtProj:
        OnEdge = np.array([True]* len(u))
    else:
        OnEdge = (u>=0) & (u<=1)


    if np.any(OnEdge):
        UExp = np.reshape(np.repeat(u[OnEdge],len(vec)),(len(vec),int(np.sum(OnEdge))),order='F')
        X_Projected[OnEdge,:] = (UExp*vec + NodePositions[Edge[0]][:,None]).T     

    distance_squared = np.sum((X_Projected - X) * (X_Projected - X),axis=1)

    return dict(X_Projected = X_Projected, Projection_Value = u,
              Distance_Squared = distance_squared,
              EdgeLen_Squared = np.sum(vec**2))



def ReportOnPrimitiveGraphEmbedment(X, NodePositions, ElasticMatrix, PartData=None, ComputeMSEP = False):
    ''' 
    # %   This function computes various measurements concerning a primitive
    # %   graph embedment
    # %
    # %           BARCODE is code in form ...S4|S3||N, where N is number of
    # %               nodes, S3 is number of 3-stars, S4 (S5,...) is number of
    # %               four (five,...) stars, etc.
    # %           ENERGY is total elastic energy of graph embedment (ENERGY = MSE + UE +
    #                                                                  %               UR)
    # %           NNODES is number of nodes.
    # %           NEDGES is number of edges
    # %           NRIBS is number of two stars (nodes with two otherr connected
    #                                           %               nodes).
    # %           NSTARS is number of stars with 3 and more leaves (nodes
    #                                                               %               connected with central node).
    # %           NRAYS2 is sum of rays minus doubled number of nodes.
    # %           MSE is mean square error or assessment of data approximation
    # %               quality.
    # %           MSEP is mean square error after piece-wise linear projection on the edges
    # %           FVE is fraction of explained variance. This value always
    # %               between 0 and 1. Greater value means higher quality of
    # %               data approximation.
    # %           FVEP is same as FVE but computed after piece-wise linear projection on the edges
    # %           UE is total sum of squared edge lengths.
    # %           UR is total sum of star deviations from harmonicity.
    # %           URN is UR * nodes
    # %           URN2 is UR * nodes^2
    # %           URSD is standard deviation of UR
    '''
    Mus = ElasticMatrix.diagonal()
    Lambda = ElasticMatrix.copy()
    np.fill_diagonal(Lambda, 0)
    indL = Lambda > 0
    Connectivities = indL.sum(axis=0)
    Mcon = np.max(Connectivities)
    counts = np.bincount(Connectivities)[1:]
    DecodedMat = DecodeElasticMatrix(ElasticMatrix)

    TotalVariance = np.sum(np.var(X,axis=0,ddof=1))
    BARCODE = getPrimitiveGraphStructureBarCode(ElasticMatrix)

    if PartData is None:
        PartData = PartitionData(X = X, 
                                 NodePositions = NodePositions,
                                 MaxBlockSize = 1000000,
                                 SquaredX= np.sum(X**2,axis=1,keepdims=1))


    Energies = ComputePrimitiveGraphElasticEnergy(NodePositions = NodePositions,
                                                  ElasticMatrix = ElasticMatrix,
                                                  dists = PartData[1])

    NNODES = len(NodePositions)
    NEDGES = len(DecodedMat[0])


    if len(counts)>1:
        NRIBS = counts[1]
    else:
        NRIBS = 0

    if len(counts)>2:
        NSTARS = counts[2]
    else:
        NSTARS = 0

    NRAYS = 0
    NRAYS2 = 0


    if ComputeMSEP:
        NodeProj = project_point_onto_graph(X, NodePositions = NodePositions,
                                        Edges = DecodedMat[0], Partition = PartData[0])
        MSEP = NodeProj['MSEP']
        FVEP = (TotalVariance-MSEP)/TotalVariance
    else:
        MSEP = np.nan
        FVEP = np.nan

    FVE = (TotalVariance-Energies[1])/TotalVariance
    URN = Energies[-1]*NNODES
    URN2 = Energies[-1]*NNODES*NNODES
    URSD = 0

    return dict(BARCODE = BARCODE, ENERGY = Energies[0], NNODES = NNODES, NEDGES = NEDGES,
           NRIBS = NRIBS, NSTARS = NSTARS, NRAYS = NRAYS, NRAYS2 = NRAYS2,
           MSE = Energies[1], MSEP = MSEP, FVE = FVE, FVEP = FVEP, UE = Energies[2], UR = Energies[-1],
           URN = URN, URN2 = URN2, URSD = URSD)
