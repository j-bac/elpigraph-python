import numpy as np
import igraph
import scipy.optimize
import copy
import warnings
from .src.graphs import ConstructGraph, GetSubGraph, GetBranches
from .src.core import PartitionData
from .src.distutils import PartialDistance
from .src.reporting import project_point_onto_graph, project_point_onto_edge


def ExtendLeaves(
    X,
    PG,
    Mode="QuantDists",
    ControlPar=0.9,
    DoSA=True,
    DoSA_maxiter=2000,
    LeafIDs=None,
    TrimmingRadius=float("inf"),
    # PlotSelected=False,
):
    """
    #' Extend leaves with additional nodes
    #'
    #' @param X numeric matrix, the data matrix
    #' @param TargetPG list, the ElPiGraph structure to extend
    #' @param LeafIDs integer vector, the id of nodes to extend. If NULL, all the vertices will be extended.
    #' @param TrimmingRadius positive numeric, the trimming radius used to control distance 
    #' @param ControlPar positive numeric, the paramter used to control the contribution of the different data points
    #' @param DoSA bollean, should optimization (via simulated annealing) be performed when Mode = "QuantDists"?
    #' @param Mode string, the mode used to extend the graph. "QuantCentroid" and "WeigthedCentroid" are currently implemented
    #' @param PlotSelected boolean, should a diagnostic plot be visualized
    #'
    #' @return The extended ElPiGraph structure
    #'
    #' The value of ControlPar has a different interpretation depending on the valus of Mode. In each case, for only the extreme points,
    #' i.e., the points associated with the leaf node that do not have a projection on any edge are considered.
    #'
    #' If Mode = "QuantCentroid", for each leaf node, the extreme points are ordered by their distance from the node
    #' and the centroid of the points farther away than the ControlPar is returned.
    #'
    #' If Mode = "WeightedCentroid", for each leaf node, a weight is computed for each points by raising the distance to the ControlPar power.
    #' Hence, larger values of ControlPar result in a larger influence of points farther from the node
    #'
    #' If Mode = "QuantDists", for each leaf node, ... will write it later
    #'
    #'
    #' @export
    #'
    #' @examples
    #'
    #' TreeEPG <- computeElasticPrincipalTree(X = tree_data, NumNodes = 50,
    #' drawAccuracyComplexity = FALSE, drawEnergy = FALSE)
    #'
    #' ExtStruct <- ExtendLeaves(X = tree_data, TargetPG = TreeEPG[[1]], Mode = "QuantCentroid", ControlPar = .5)
    #' PlotPG(X = tree_data, TargetPG = ExtStruct)
    #'
    #' ExtStruct <- ExtendLeaves(X = tree_data, TargetPG = TreeEPG[[1]], Mode = "QuantCentroid", ControlPar = .9)
    #' PlotPG(X = tree_data, TargetPG = ExtStruct)
    #'
    #' ExtStruct <- ExtendLeaves(X = tree_data, TargetPG = TreeEPG[[1]], Mode = "WeigthedCentroid", ControlPar = .2)
    #' PlotPG(X = tree_data, TargetPG = ExtStruct)
    #'
    #' ExtStruct <- ExtendLeaves(X = tree_data, TargetPG = TreeEPG[[1]], Mode = "WeigthedCentroid", ControlPar = .8)
    #' PlotPG(X = tree_data, TargetPG = ExtStruct)
    #'
    """

    TargetPG = copy.deepcopy(PG)
    # Generate net
    Net = ConstructGraph(PrintGraph=TargetPG)

    # get leafs
    if LeafIDs is None:
        LeafIDs = np.where(np.array(Net.degree()) == 1)[0]

    # check LeafIDs
    if np.any(np.array(Net.degree(LeafIDs)) > 1):
        raise ValueError("Only leaf nodes can be extended")

    # and their neigh
    Nei = Net.neighborhood(LeafIDs, order=1)

    # and put stuff together
    NeiVect = list(map(lambda x: set(Nei[x]).difference([LeafIDs[x]]), range(len(Nei))))
    NeiVect = np.array([j for i in NeiVect for j in list(i)])
    NodesMat = np.hstack((LeafIDs[:, None], NeiVect[:, None]))

    # project data on the nodes
    PD = PartitionData(
        X=X,
        NodePositions=TargetPG["NodePositions"],
        MaxBlockSize=10000000,
        TrimmingRadius=TrimmingRadius,
        SquaredX=np.sum(X ** 2, axis=1, keepdims=1),
    )

    # Keep track of the new nodes IDs
    NodeID = len(TargetPG["NodePositions"]) - 1

    init = False
    NNPos = None
    NEdgs = None
    UsedNodes = []

    # for each leaf
    for i in range(len(NodesMat)):

        if np.sum(PD[0] == NodesMat[i, 0]) == 0:
            continue

        # generate the new node id
        NodeID = NodeID + 1

        # get all the data associated with the leaf node
        tData = X[(PD[0] == NodesMat[i, 0]).flatten(), :]

        # and project them on the edge
        Proj = project_point_onto_edge(
            X=X[(PD[0] == NodesMat[i, 0]).flatten(), :],
            NodePositions=TargetPG["NodePositions"],
            Edge=NodesMat[i, :],
        )

        # Select the distances of the associated points
        Dists = PD[1][PD[0] == NodesMat[i, 0]]

        # Set distances of points projected on beyond the initial position of the edge to 0
        Dists[Proj["Projection_Value"] >= 0] = 0

        if Mode == "QuantCentroid":
            ThrDist = np.quantile(Dists[Dists > 0], ControlPar)
            SelPoints = np.where(Dists >= ThrDist)[0]

            print(
                len(SelPoints),
                " points selected to compute the centroid while extending node",
                NodesMat[i, 0],
            )

            if len(SelPoints) > 1:
                NN = np.mean(tData[SelPoints, :], axis=0, keepdims=1)

            else:
                NN = tData[SelPoints, :]

            # Initialize the new nodes and edges
            if not init:
                init = True
                NNPos = NN.copy()
                NEdgs = np.array([[NodesMat[i, 0], NodeID]])
                UsedNodes.extend(
                    np.where(PD[0].flatten() == NodesMat[i, 0])[0][SelPoints]
                )
            else:
                NNPos = np.vstack((NNPos, NN))
                NEdgs = np.vstack((NEdgs, np.array([[NodesMat[i, 0], NodeID]])))
                UsedNodes.extend(
                    list(np.where(PD[0].flatten() == NodesMat[i, 0])[0][SelPoints])
                )

        if Mode == "WeightedCentroid":
            Dist2 = Dists ** (2 * ControlPar)
            Wei = Dist2 / np.max(Dist2)

            if len(Wei) > 1:
                NN = np.sum(tData * Wei[:, None], axis=0) / np.sum(Wei)

            else:
                NN = tData

            # Initialize the new nodes and edges
            if not init:
                init = True
                NNPos = NN.copy()
                NEdgs = np.array([NodesMat[i, 0], NodeID])
                WeiVal = list(Wei)
                UsedNodes.extend(list(np.where(PD[0].flatten() == NodesMat[i, 0])[0]))
            else:
                NNPos = np.vstack((NNPos, NN))
                NEdgs = np.vstack((NEdgs, np.array([NodesMat[i, 0], NodeID])))

                UsedNodes.extend(list(np.where(PD[0].flatten() == NodesMat[i, 0])[0]))
                WeiVal.extend(list(Wei))

        if Mode == "QuantDists":

            if sum(Dists > 0) == 0:
                continue

            if sum(Dists > 0) > 1 and len(tData) > 1:
                tData_Filtered = tData[Dists > 0, :]

                def DistFun(NodePosition):
                    return np.quantile(
                        project_point_onto_edge(
                            X=tData_Filtered,
                            NodePositions=np.vstack(
                                (
                                    TargetPG["NodePositions"][NodesMat[i, 0],],
                                    NodePosition,
                                )
                            ),
                            Edge=[0, 1],
                        )["Distance_Squared"],
                        ControlPar,
                    )

                StartingPoint = tData_Filtered[
                    np.argmin(
                        np.array(
                            [
                                DistFun(tData_Filtered[i])
                                for i in range(len(tData_Filtered))
                            ]
                        )
                    ),
                    :,
                ]

                if DoSA:

                    print("Performing simulated annealing. This may take a while")
                    StartingPoint = scipy.optimize.dual_annealing(
                        DistFun,
                        bounds=list(
                            zip(
                                np.min(tData_Filtered, axis=0),
                                np.max(tData_Filtered, axis=0),
                            )
                        ),
                        x0=StartingPoint,
                        maxiter=DoSA_maxiter,
                    )["x"]

                Projections = project_point_onto_edge(
                    X=tData_Filtered,
                    NodePositions=np.vstack(
                        (TargetPG["NodePositions"][NodesMat[i, 0], :], StartingPoint)
                    ),
                    Edge=[0, 1],
                    ExtProj=True,
                )

                SelId = np.argmax(
                    PartialDistance(
                        Projections["X_Projected"],
                        np.array([TargetPG["NodePositions"][NodesMat[i, 0], :]]),
                    )
                )

                StartingPoint = Projections["X_Projected"][SelId, :]

            else:
                StartingPoint = tData[Dists > 0, :]

            if not init:
                init = True
                NNPos = StartingPoint[None].copy()
                NEdgs = np.array([[NodesMat[i, 0], NodeID]])
                UsedNodes.extend(list(np.where(PD[0].flatten() == NodesMat[i, 0])[0]))
            else:
                NNPos = np.vstack((NNPos, StartingPoint[None]))
                NEdgs = np.vstack((NEdgs, np.array([[NodesMat[i, 0], NodeID]])))
                UsedNodes.extend(list(np.where(PD[0].flatten() == NodesMat[i, 0])[0]))

    # plot(X)
    # points(TargetPG$NodePositions, col="red")
    # points(NNPos, col="blue")
    #
    try:
        print(NNPos.shape)
    except:
        print("failed")
    try:
        print(TargetPG["NodePositions"].shape)
    except:
        pass
    TargetPG["NodePositions"] = np.vstack((TargetPG["NodePositions"], NNPos))
    TargetPG["Edges"] = [
        np.vstack((TargetPG["Edges"][0], NEdgs)),  # edges
        np.append(TargetPG["Edges"][1], np.repeat(np.nan, len(NEdgs))),  # lambdas
        # np.append(TargetPG["Edges"][2], np.repeat(np.nan, len(NEdgs))),
    ]  ##### mus become lambda in R, bug ??

    #    if PlotSelected:
    #        if Mode == "QuantCentroid":
    #            Cats = ["Unused"] * len(X)
    #            if UsedNodes:
    #                Cats[UsedNodes] = "Used"
    #
    #            p = PlotPG(X=X, TargetPG=TargetPG, GroupsLab=Cats)
    #            print(p)
    #
    #        if Mode == "WeightedCentroid":
    #            Cats = np.zeros(len(X))
    #            if UsedNodes:
    #                Cats[UsedNodes] = WeiVal
    #
    #            p = PlotPG(X=X[Cats > 0, :], TargetPG=TargetPG, GroupsLab=Cats[Cats > 0])
    #            print(p)
    #
    #            p1 = PlotPG(X=X, TargetPG=TargetPG, GroupsLab=Cats)
    #            print(p1)

    return TargetPG


def CollapseBranches(
    X, PG, Mode="PointNumber", ControlPar=5, TrimmingRadius=float("inf")
):
    """
    #' Filter "small" branches 
    #'
    #' @param X numeric matrix, the data matrix
    #' @param TargetPG list, the ElPiGraph structure to extend
    #' @param TrimmingRadius positive numeric, the trimming radius used to control distance 
    #' @param ControlPar positive numeric, the paramter used to control the contribution of the different data points
    #' @param Mode string, the mode used to extend the graph. "PointNumber", "PointNumber_Extrema", "PointNumber_Leaves",
    #' "EdgesNumber", and "EdgesLength" are currently implemented
    #' @param PlotSelected boolean, should a diagnostic plot be visualized (currently not implemented)
    #'
    #' @return a list with 2 values: Nodes (a matrix containing the new nodes positions) and Edges (a matrix describing the new edge structure)
    #'
    #' The value of ControlPar has a different interpretation depending on the valus of Mode.
    #'
    #' If Mode = "PointNumber", branches with less that ControlPar points projected on the branch
    #' (points projected on the extreme points are not considered) are removed
    #'
    #' If Mode = "PointNumber_Extrema", branches with less that ControlPar points projected on the branch or the extreme
    #' points are removed
    #'
    #' If Mode = "PointNumber_Leaves", branches with less that ControlPar points projected on the branch and any leaf points
    #' (points projected on non-leaf extreme points are not considered) are removed
    #'
    #' If Mode = "EdgesNumber", branches with less that ControlPar edges are removed
    #'
    #' If Mode = "EdgesLength", branches with with a length smaller than ControlPar are removed
    #'
    #' @export
    #'
    """
    TargetPG = copy.deepcopy(PG)
    # Generate net
    Net = ConstructGraph(PrintGraph=TargetPG)

    # Set a color for the edges
    Net.es.set_attribute_values("status", "keep")

    # Get the leaves
    Leaves = np.where(np.array(Net.degree(mode="all")) == 1)[0]

    # get the partition
    PartStruct = PartitionData(
        X=X,
        NodePositions=TargetPG["NodePositions"],
        MaxBlockSize=100000000,
        TrimmingRadius=TrimmingRadius,
        SquaredX=np.sum(X ** 2, axis=1, keepdims=1),
    )

    # Project points onto the graph
    ProjStruct = project_point_onto_graph(
        X=X,
        NodePositions=TargetPG["NodePositions"],
        Edges=TargetPG["Edges"][0],
        Partition=PartStruct[0],
    )

    # get branches
    Branches = GetSubGraph(Net=Net, Structure="branches")

    # get the number of points on the different branches
    AllBrInfo = []
    for BrNodes in Branches:
        # print(BrNodes)
        PotentialPoints = np.array([False] * len(ProjStruct["EdgeID"]))
        NodeNames = np.array(BrNodes)

        # Get the points on the extrema
        StartEdg = np.where(np.any(ProjStruct["Edges"] == NodeNames[0], axis=1))[0]

        StartOnNode = np.array([False] * len(ProjStruct["EdgeID"]))

        SelPoints = np.isin(ProjStruct["EdgeID"], StartEdg)
        StartOnNode[SelPoints] = (ProjStruct["ProjectionValues"][SelPoints] > 1) | (
            ProjStruct["ProjectionValues"][SelPoints] < 0
        )

        # EndEdg = np.where(np.any(ProjStruct["Edges"] == NodeNames[-1], axis=1))[0]
        EndOnNode = np.array([False] * len(ProjStruct["EdgeID"]))

        SelPoints = np.isin(ProjStruct["EdgeID"], EndOnNode)
        EndOnNode[SelPoints] = (ProjStruct["ProjectionValues"][SelPoints] > 1) | (
            ProjStruct["ProjectionValues"][SelPoints] < 0
        )

        EdgLen = 0

        # Get the points on the branch (extrema are excluded)
        for i in range(1, len(BrNodes)):

            # Get the edge on the segment
            WorkingEdg = np.where(
                list(
                    map(
                        lambda x: all(np.isin(x, NodeNames[range((i - 1), i + 1)])),
                        ProjStruct["Edges"],
                    )
                )
            )[0].squeeze()
            # Get the len of the segment
            EdgLen = EdgLen + ProjStruct["EdgeLen"][WorkingEdg]

            # Get the points on the segment
            Points = ProjStruct["EdgeID"] == WorkingEdg
            Points[np.isnan(Points)] = False

            # Is the edge in the right direction?
            if all(
                ProjStruct["Edges"][WorkingEdg,] == NodeNames[range((i - 1), i + 1)]
            ):
                Reverse = False
            else:
                Reverse = True

            # Counting points at the beginning

            if i == 1 and len(BrNodes) > 2:
                if Reverse:
                    PotentialPoints[Points] = (
                        ProjStruct["ProjectionValues"][Points] < 1
                    ) | PotentialPoints[Points]
                else:
                    PotentialPoints[Points] = (
                        ProjStruct["ProjectionValues"][Points] > 0
                    ) | PotentialPoints[Points]
                continue

            # Counting points at the end
            if i == (len(BrNodes) - 1):
                if Reverse:
                    PotentialPoints[Points] = (
                        ProjStruct["ProjectionValues"][Points] > 0
                    ) | PotentialPoints[Points]

                else:
                    PotentialPoints[Points] = (
                        ProjStruct["ProjectionValues"][Points] < 1
                    ) | PotentialPoints[Points]
                continue

            # all the other cases
            PotentialPoints[Points] = (
                (ProjStruct["ProjectionValues"][Points] > 0)
                & (ProjStruct["ProjectionValues"][Points] < 1)
            ) | PotentialPoints[Points]

        PointsOnEdgesLeaf = PotentialPoints

        if np.isin(NodeNames[0], Leaves):
            PointsOnEdgesLeaf = PointsOnEdgesLeaf | StartOnNode

        if np.isin(NodeNames[-1], Leaves):
            PointsOnEdgesLeaf = PointsOnEdgesLeaf | EndOnNode

        AllBrInfo.append(
            dict(
                PointsOnEdges=sum(PotentialPoints),
                PointsOnEdgeExtBoth=sum(PotentialPoints | StartOnNode | EndOnNode),
                PointsOnEdgesLeaf=sum(PointsOnEdgesLeaf),
                EdgesCount=len(BrNodes) - 1,
                EdgesLen=EdgLen,
            )
        )

    # Now all the information has been pre-computed and it is possible to filter

    if Mode == "PointNumber":
        ToFilter = np.array([i["PointsOnEdges"] for i in AllBrInfo]) < ControlPar

    if Mode == "PointNumber_Extrema":
        ToFilter = np.array([i["PointsOnEdgeExtBoth"] for i in AllBrInfo]) < ControlPar

    if Mode == "PointNumber_Leaves":
        ToFilter = np.array([i["PointsOnEdgesLeaf"] for i in AllBrInfo]) < ControlPar

    if Mode == "EdgesNumber":
        ToFilter = np.array([i["EdgesCount"] for i in AllBrInfo]) < ControlPar

    if Mode == "EdgesLength":
        ToFilter = np.array([i["EdgesLen"] for i in AllBrInfo]) < ControlPar

    # Nothing to filter
    if sum(ToFilter) == 0:
        return dict(Edges=TargetPG["Edges"][0], Nodes=TargetPG["NodePositions"])

    # TargetPG_New = TargetPG
    # NodesToRemove = None

    # Keep track of all the nodes to remove
    AllNodes_InternalBranches = {}
    # For all the branches
    for i in range(len(ToFilter)):

        # If we need to filter this
        if ToFilter[i] == True:

            NodeNames = np.array(Branches[i])

            # Is it a final branch ?
            if any(np.isin(NodeNames[[0, -1]], Leaves)):
                # It's a terminal branch, we can safely take it out

                print("Removing the terminal branch with nodes:", NodeNames)

                if len(NodeNames) > 2:
                    NodeNames_Ext = [NodeNames[0]]
                    NodeNames_Ext.extend(list(np.repeat(NodeNames[1:-1], 2)))
                    NodeNames_Ext.append(NodeNames[-1])
                else:
                    NodeNames_Ext = NodeNames
                # Set edges to be removed
                for e in range(0, len(NodeNames_Ext), 2):
                    rm_eid = Net.get_eid(NodeNames_Ext[e], NodeNames_Ext[e + 1])
                    Net.es[rm_eid]["status"] = "remove"

            else:
                # It's a "bridge". We cannot simply remove nodes. Need to introduce a new one by "fusing" two stars
                print("Removing the bridge branch with nodes:", NodeNames)

                # Update the list of nodes to update
                AllNodes_InternalBranches = list(
                    set(AllNodes_InternalBranches).union(NodeNames)
                )

    # Create the network that will contain the final filtered network
    Ret_Net = copy.deepcopy(Net)

    # Get a net with all the groups of bridges to remove
    tNet = Net.induced_subgraph(AllNodes_InternalBranches)

    if tNet.vcount() > 0:
        # Get the different connected components
        CC = tNet.components()
        # Get the nodes associated with the connected components
        Member_Comps = CC.membership
        Vertex_Comps = [
            np.array(tNet.vs["name"])[np.where(Member_Comps == i)[0]]
            for i in np.unique(Member_Comps)
        ]
        # Get the centroid of the different connected components
        Centroids = np.array(
            [
                np.mean(TargetPG["NodePositions"][np.array(i)], axis=0)
                for i in Vertex_Comps
            ]
        )

        # Prepare a vector that will be used to contract vertices
        CVet = np.array(range(Net.vcount()))

        # For each centroid
        for i in range(len(Vertex_Comps)):
            # Add a new vertex
            Ret_Net.add_vertex(Ret_Net.vcount())
            # Add a new element to the contraction vector
            CVet = np.append(CVet, len(CVet))
            # specify the nodes that will collapse on the new node
            CVet[Vertex_Comps[i]] = len(CVet) - 1

        # collapse the network
        Ret_Net.contract_vertices(mapping=CVet)

    # delete edges belonging to the terminal branches
    edge_list = Ret_Net.get_edgelist()
    Ret_Net.delete_edges(
        [
            edge_list[i]
            for i in range(len(edge_list))
            if Ret_Net.es["status"][i] == "remove"
        ]
    )
    # remove loops that may have been introduced because of the collapse
    Ret_Net.simplify(loops=True)
    # # Remove empty nodes
    names = np.array(Ret_Net.vs.indices)[np.array(Ret_Net.degree()) > 0]
    Ret_Net = Ret_Net.induced_subgraph(
        np.array(Ret_Net.vs.indices)[np.array(Ret_Net.degree()) > 0]
    )

    if tNet.vcount() > 0:
        NodeMat = np.vstack((TargetPG["NodePositions"], Centroids))

    else:
        NodeMat = TargetPG["NodePositions"]

    NodeMat = NodeMat[names, :]

    return dict(Edges=np.array(Ret_Net.get_edgelist()), Nodes=NodeMat)


#' Title
#'
#' @param TargetPG
#' @param NodesToRemove
#'
#' @return
#'
#' @examples

# RemoveNodesbyIDs <- function(TargetPG, NodesToRemove) {

#   RemapNodeID <- cbind(
#     1:nrow(TargetPG$NodePositions),
#     1:nrow(TargetPG$NodePositions)
#   )

#   TargetPG_New <- TargetPG

#   # Remove nodes and edges
#   TargetPG_New$NodePositions <- TargetPG_New$NodePositions[-NodesToRemove, ]
#   TargetPG_New$ElasticMatrix <- TargetPG_New$ElasticMatrix[-NodesToRemove, -NodesToRemove]

#   # tEdges <- which(TargetPG_New_New$ElasticMatrix > 0, arr.ind = TRUE)
#   # tEdges <- tEdges[tEdges[,2] > tEdges[,1],]

#   # Remap Nodes IDs
#   RemapNodeID[RemapNodeID[,2] %in% NodesToRemove,2] <- 0
#   for(j in 1:nrow(RemapNodeID)){
#     if(RemapNodeID[j,2] == 0){
#       # the node has been removed. Remapping
#       RemapNodeID[RemapNodeID[,2] >= j, 2] <- RemapNodeID[RemapNodeID[,2] >= j, 2] - 1
#     }
#   }

#   tEdges <- TargetPG_New$Edges$Edges
#   for(j in 1:nrow(RemapNodeID)){
#     tEdges[TargetPG_New$Edges$Edges == RemapNodeID[j,1]] <- RemapNodeID[j,2]
#   }
#   EdgesToRemove <- which(rowSums(tEdges == 0) > 0)
#   tEdges <- tEdges[-EdgesToRemove, ]

#   TargetPG_New$Edges$Edges <- tEdges
#   TargetPG_New$Edges$Lambdas <- TargetPG_New$Edges$Lambdas[-EdgesToRemove]
#   TargetPG_New$Edges$Mus <- TargetPG_New$Edges$Mus[-NodesToRemove]

#   return(TargetPG_New)
# }


def ShiftBranching(
    X,
    PG,
    SelectionMode="NodePoints",
    DensityRadius=None,
    MaxShift=3,
    Compensate=False,
    BrIds=None,
    TrimmingRadius=float("inf"),
):
    """
    #' Move branching nodes to areas with higher point density
    #'
    #' @param X numeric matrix, the data matrix
    #' @param TargetPG list, the ElPiGraph structure to extend
    #' @param TrimmingRadius positive numeric, the trimming radius used to control distance 
    #' @param SelectionMode string, the mode to use to shift the branching points. The "NodePoints" and "NodeDensity" modes are currently supported
    #' @param DensityRadius positive numeric, the radius to be used when computing point density if SelectionMode = "NodeDensity"
    #' @param MaxShift positive integer, the maxium distance (as number of edges) to consider when exploring the branching point neighborhood
    #' @param Compensate booelan, should new points be included to compensate for olter one being removed (currently not implemented)
    #' @param BrIds integer vector, the id of the branching points to consider. Id not associted with node possessing degree > 2 will be ignored
    #'
    #' @return a list with two components: NodePositions (Containing the new nodes positions) and Edges (containing the new edges)
    #'
    #' The function explore the neighborhood of branching point for nodes with higher point density. It such point is found, to graph will be
    #' modified so that the new found node will be the new branching point of the neighborhood. 
    #'
    #' @examples
    #' @export
    """

    TargetPG = copy.deepcopy(PG)

    Net = ConstructGraph(PrintGraph=TargetPG)
    BrPoints = np.where(np.array(Net.degree()) > 2)[0]

    if BrIds is None:
        BrIds = BrPoints
    else:
        BrIds = set(BrIds).intersection(BrPoints)

    PD = PartitionData(
        X=X,
        NodePositions=TargetPG["NodePositions"],
        MaxBlockSize=100000000,
        TrimmingRadius=TrimmingRadius,
        SquaredX=np.sum(X ** 2, axis=1, keepdims=1),
    )

    for br in BrIds:

        Neis = Net.neighborhood(br, order=MaxShift)
        # Neis = setdiff(as.integer(Neis), br)

        if SelectionMode == "NodePoints":
            Neival = np.array(list(map(lambda x: np.sum(PD[0] == x), Neis)))

        if SelectionMode == "NodeDensity":
            Dists = PartialDistance(X, TargetPG["NodePositions"][Neis])

            if DensityRadius is None:
                raise ValueError(
                    "DensityRadius needs to be specified when SelectionMode = 'NodeDensity'"
                )

            else:
                Neival = np.sum(Dists < DensityRadius, axis=0)

        NeiDist = np.array(Net.shortest_paths(br, Neis, mode="all"))
        Neival = Neival[np.argsort(NeiDist, kind="mergesort")]
        NewBR = Neis[np.min(np.where(Neival.squeeze() == np.max(Neival))[0])]

        if NewBR != br:

            print("Moving the branching point at node", br)

            ToReconnect = Net.neighbors(br)
            # delete the edges forming the old branching point
            Net.delete_edges(Net.incident(br))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # get paths from the new branching point to the old nodes
                AllPath = Net.get_shortest_paths(NewBR, to=ToReconnect, output="epath")
                AllPath_vpath = Net.get_shortest_paths(
                    NewBR, to=ToReconnect, output="vpath"
                )

            # delete these extra nodes
            for i in range(len(AllPath)):
                if len(AllPath[i]) > 0:
                    Net.delete_edges(AllPath[i])

            # reconnect the old nodes to the new branching points, excep for any node found in the previuos operation
            ToReconnect = list(
                set(ToReconnect).difference([j for i in AllPath_vpath for j in i])
            )
            Net.add_edges([(NewBR, i) for i in ToReconnect])

    if Compensate:
        print("Warning : Node compensation is not implemented yet")

    NewNodePositions = TargetPG["NodePositions"][np.array(Net.degree()) > 0, :]

    Net.delete_vertices(np.where(np.array(Net.degree()) == 0)[0])
    NewEdges = np.array(Net.get_edgelist())

    return dict(NodePositions=NewNodePositions, Edges=NewEdges)


#' Filter cliques
#'
#' @param TargetPG
#' @param MaxClZize
#' @param DistThr
#'
#' @return
#' @export
#'
#' @examples
# CollapseCliques <- function(TargetPG, MaxClZize = NULL, DistThr = NULL, Compensate = 3) {

#   Net <- ConstructGraph(TargetPG)

#   Nodes <- TargetPG$NodePositions

#   Cliques <- igraph::cliques(Net, min = 3, max = MaxClZize)

#   ClSize <- sapply(Cliques, length)

#   print(paste(length(ClSize), "cliques detected"))

#   Tries <- 0

#   while (TRUE) {

#     if(Tries > 100 | length(ClSize) == 0){
#       return(list(
#         Nodes = TargetPG$NodePositions,
#         Edges = TargetPG$Edges$Edges
#       ))
#     }

#     Tries <- Tries + 1

#     ClToCollapse <- sample(x = which(ClSize == max(ClSize)), size = 1)
#     NodesToCollapse <- as.integer(Cliques[[ClToCollapse]])

#     NewNode <- colMeans(Nodes[NodesToCollapse, ])

#     if(!is.null(DistThr)){
#       if(mean(distutils::PartialDistance(matrix(NewNode, nrow = 1), Nodes[NodesToCollapse, ])) < DistThr){
#         break()
#       }
#     } else {
#       break()
#     }

#   }

#   Net <- igraph::add.vertices(Net, 1, attr = list(name = paste(igraph::vcount(Net)+1)))

#   ContractNet <- 1:igraph::vcount(Net)
#   ContractNet[NodesToCollapse] <- igraph::vcount(Net)

#   ContrNet <- igraph::contract.vertices(Net, ContractNet)
#   igraph::V(ContrNet)$name <- sapply(igraph::V(ContrNet)$name, function(x){
#     if(length(x)>0){
#       max(as.numeric(x))
#     } else {
#       0
#     }
#   })
#   ContrNet <- igraph::delete.vertices(ContrNet, NodesToCollapse)
#   ContrNet <- igraph::simplify(ContrNet, remove.loops = TRUE, remove.multiple = TRUE)
#   # igraph::V(ContrNet)$name <- paste(1:igraph::vcount(ContrNet))

#   UpdateNodes <- rbind(Nodes, NewNode)[as.integer(igraph::V(ContrNet)$name),]
#   rownames(UpdateNodes) <- NULL

#   if(Compensate <= max(ClSize)){
#     Neis <- unlist((igraph::adjacent_vertices(graph = ContrNet, v = paste(igraph::vcount(Net)))[[1]])$name)

#     NNodes <- t((t(Nodes[Neis, ]) + NewNode)/2)

#     ContrNet <- igraph::add.vertices(ContrNet, length(Neis), attr = list(name = paste0("Ext_", Neis)))
#     ContrNet <- igraph::delete_edges(ContrNet, igraph::incident_edges(graph = ContrNet, v = paste(igraph::vcount(Net)))[[1]])
#     ContrNet <- igraph::add.edges(graph = ContrNet, edges = rbind(paste(igraph::vcount(Net)), paste0("Ext_", Neis)))
#     ContrNet <- igraph::add.edges(graph = ContrNet, edges = rbind(paste(Neis), paste0("Ext_", Neis)))

#     UpdateNodes <- rbind(UpdateNodes, NNodes)

#   }

#   return(list(
#     Nodes = UpdateNodes,
#     Edges = igraph::get.edgelist(ContrNet, names = FALSE)
#   ))

# }


#'
#'
#' #' Title
#' #'
#' #' @param variables
#' #'
#' #' @return
#' #' @export
#' #'
#' #' @examples
#' RewireBranches <- function(X,
#'                        TargetPG,
#'                        MaxDist = 3,
#'                        Lambda,
#'                        Mu,
#'                        MaxNumberOfIterations = 10,
#'                        eps = 0.01,
#'                        FinalEnergy = "Base",
#'                        alpha = 0,
#'                        beta = 0,
#'                        Mode = 1,
#'                        TrimmingRadius = Inf,
#'                        FastSolve = FALSE,
#'                        prob = 1) {
#'
#'   TargetPG1 <- TargetPG
#'
#'   NodeDist <- distutils::PartialDistance(TargetPG$NodePositions, TargetPG$NodePositions)
#'
#'   Net <- ConstructGraph(PrintGraph = TargetPG1)
#'   BrPoint <- which(igraph::degree(Net)>2)
#'
#'   for(i in 1:length(BrPoint)){
#'
#'     Net <- ConstructGraph(PrintGraph = TargetPG1)
#'
#'     NeiBr <- as.integer(names(igraph::neighborhood(graph = Net, order = 1, nodes = BrPoint[i])[[1]]))
#'     NeiBr <- setdiff(NeiBr, BrPoint[i])
#'
#'     for(j in 1:length(NeiBr)){
#'       TargetPG1$NodePositions <- rbind(
#'         TargetPG1$NodePositions,
#'         colMeans(
#'           rbind(
#'             TargetPG$NodePositions[BrPoint[i], ],
#'             TargetPG$NodePositions[NeiBr[j], ]
#'           )
#'         )
#'       )
#'       OldEdg <- TargetPG1$Edges$Edges[,1] %in% c(BrPoint[i], NeiBr[j]) &
#'         TargetPG1$Edges$Edges[,2] %in% c(BrPoint[i], NeiBr[j])
#'       TargetPG1$Edges$Edges <- TargetPG1$Edges$Edges[!OldEdg,]
#'       TargetPG1$Edges$Edges <- rbind(TargetPG1$Edges$Edges,
#'                                      rbind(
#'                                        c(BrPoint[i], nrow(TargetPG1$NodePositions)),
#'                                        c(NeiBr[j], nrow(TargetPG1$NodePositions))
#'                                      )
#'       )
#'
#'     }
#'
#'     Net <- ConstructGraph(PrintGraph = TargetPG1)
#'     NeiBr <- as.integer(names(igraph::neighborhood(graph = Net, order = 1, nodes = BrPoint[i])[[1]]))
#'     NeiBr <- setdiff(NeiBr, BrPoint[i])
#'
#'     StarEdes <- igraph::get.edge.ids(graph = Net, vp = rbind(rep(BrPoint[i], length(NeiBr)), NeiBr), directed = FALSE)
#'
#'     tNet <- igraph::delete.edges(graph = Net, edges = StarEdes)
#'     Neigh <- igraph::neighborhood(graph = tNet, order = igraph::vcount(tNet), nodes = NeiBr)
#'     NeighDist <- lapply(Neigh, function(NeiVect){
#'       sapply(igraph::shortest_paths(graph = Net, from = BrPoint[i], to = as.vector(NeiVect))$vpath, length)
#'     })
#'
#'     Mats <- list()
#'
#'     for(j in 1:length(Neigh)){
#'
#'       tNet <- igraph::delete.edges(graph = Net, igraph::E(Net)[igraph::`%--%`(BrPoint[i], NeiBr[j])])
#'
#'       VertexToTest <- Neigh[-j]
#'       VertexToTest.Dist <- NeighDist[-j]
#'
#'       VertexToTest <- lapply(1:length(VertexToTest), function(i){
#'         (VertexToTest[[i]])[VertexToTest.Dist[[i]] <= MaxDist]
#'       })
#'       VertexToTest <- unlist(VertexToTest)
#'
#'       for(k in 1:length(VertexToTest)){
#'         TestNet <- igraph::add.edges(graph = tNet, edges = c(NeiBr[j], VertexToTest[k]))
#'         Mats[[length(Mats)+1]] <- apply(igraph::get.edgelist(TestNet), 2, as.numeric)
#'       }
#'
#'     }
#'
#'     SquaredX <- rowSums(X^2)
#'
#'     Embed <- lapply(Mats, function(mat){
#'       PrimitiveElasticGraphEmbedment(X = X,
#'                                                    NodePositions = TargetPG1$NodePositions,
#'                                                    ElasticMatrix = Encode2ElasticMatrix(mat, Lambda, Mu),
#'                                                    SquaredX = SquaredX,
#'                                                    verbose = FALSE,
#'                                                    MaxNumberOfIterations = MaxNumberOfIterations,
#'                                                    eps = eps,
#'                                                    FinalEnergy = FinalEnergy,
#'                                                    alpha = alpha,
#'                                                    beta = beta,
#'                                                    Mode = Mode,
#'                                                    TrimmingRadius = TrimmingRadius,
#'                                                    FastSolve = FastSolve,
#'                                                    prob = prob)
#'     })
#'
#'
#'     # SumDists <- sapply(Mats, function(EdgMat){
#'     #   ProjStruct <- project_point_onto_graph(X, NodePositions = TargetPG$NodePositions, Edges = EdgMat)
#'     #   OnEdes <- ProjStruct$ProjectionValues > 0 & ProjStruct$ProjectionValues < 1
#'     #   DistFromEdge <- rowSums((ProjStruct$X_projected - X)^2)
#'     #   c(sum(DistFromEdge[OnEdes]), sum(ProjStruct$EdgeLen))
#'     # })
#'
#'     # order(colSums(SumDists))
#'     # which.min(SumDists[2,])
#'
#'
#'     # TargetPG2 <- TargetPG1
#'
#'     TargetPG1$Edges$Edges <- Mats[[which.min(sapply(Embed, "[[", "ElasticEnergy"))]]
#'
#'     PlotPG(X, TargetPG1, DimToPlot = 1:3, NodeLabels = 1:nrow(TargetPG1$NodePositions), LabMult = 3)
#'
#'     which(igraph::degree(ConstructGraph(TargetPG1))>2)
#'
#'   }
#'
#'   #
#'   #
#'   #
#'   # MaxByEdg <- aggregate(DistFromEdge, by=list(ProjStruct$EdgeID), max)
#'   # ToMove <- MaxByEdg[which.max(MaxByEdg[,2]),1]
#'   #
#'   # Ends <- TargetPG$Edges$Edges[ToMove,]
#'   #
#'   #
#'   # Net <-
#'   # igraph::neighborhood(graph = Net, order = igraph::vcount(Net), nodes = Ends)
#'
#'   return(TargetPG1)
#'
#' }
#'
#'
#'
#'
