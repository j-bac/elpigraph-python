import numpy as np
import pandas as pd
import igraph
import copy

# Construct igraph objects -------------------------------------------------------------


def ConstructGraph(PrintGraph):
    """
    #' Generate an igraph object from an ElPiGraph structure
    #'
    #' @param PrintGraph A principal graph object
    #'
    #' @return An igraph network
    #' @export
    #'
    #' @examples
    """

    Net = igraph.Graph(n=np.max(PrintGraph["Edges"][0]) + 1, directed=False)
    Net.vs["name"] = list(range(np.max(PrintGraph["Edges"][0]) + 1))
    Net.add_edges(PrintGraph["Edges"][0])

    return Net


# Extract a subpath from the graph ----------------------------------------


def GetSubGraph(Net, Structure, Nodes=None, Circular=True, KeepEnds=True):
    """
    #' Extract a subgraph with a given topology from a graph
    #'
    #' @param Net an igraph network object
    #' @param Structure a string specifying the structure to return. The following options are
    #' available:
    #' \itemize{
    #'  \item 'circle', all the circles of a given length (specified by Nodes) present in the data.
    #'  If Nodes is unspecified the algorithm will look for the largest circle avaialble.
    #'  \item 'branches', all the linear path connecting the branching points
    #'  \item 'branches&bpoints', all the linear path connecting the branching points and all of
    #'  the branching points
    #'  \item 'branching', all the subtree associted with a branching point (i.e., a tree encompassing
    #' the branching points and the closests branching points and end points)
    #' \item 'end2end', all linear paths connecting end points (or leaf)
    #' }
    #' @param Circular a boolean indicating whether the circle should contain the initial points at the
    #' beginning and at the end
    #' @param Nodes the number of nodes (for cycle detection)
    #' @param KeepEnds boolean, should the end points (overlapping between structures) be included when
    #' Structure = 'branches' or 'branching'
    #'
    #' @description
    #'
    #' Note that all subgraph are returned only once. So, for example, if A and B are two end leaves of a tree
    #' and 'end2end' is being used, only the path for A to B or the path from Bt o A will be returned.
    #'
    #' @return a list of nodes defining the structures under consideration
    #' @export
    #'
    #'
    #'
    #' @examples
    """
    if Structure == "auto":
        print("Structure autodetection is not implemented yet")
        return None

    if Structure == "circle":
        print("WARNING : this mode has not been verified. Confirm with R code")

        if Nodes is None:
            print("Looking for the largest cycle")

            for i in reversed(range(3, Net.vcount())):
                RefNet = igraph.Graph.ring(n=i, directed=False, circular=True)
                if Net.subisomorphic_lad(RefNet):
                    print("A cycle of length", i, "has been found")
                    break

        else:
            i = Nodes

        RefNet = igraph.Graph.Ring(n=i, directed=False, circular=True)

        SubIsoProjList = Net.get_subisomorphisms_vf2(RefNet)
        SubIsoProjList = np.sort(SubIsoProjList, axis=0)
        duplicated = pd.DataFrame(SubIsoProjList).duplicated().to_numpy()
        SubIsoProjList = SubIsoProjList[~duplicated, :]

        # names_SubIsoProjList = ["Circle_" + str(i) for i in range(len(SubIsoProjList))]

        if Circular:
            print(list(map(lambda x: [x, x[0]], SubIsoProjList)))

        else:
            print(SubIsoProjList)

    if Structure == "branches":

        if np.any(np.array(Net.degree()) > 2) & np.any(np.array(Net.degree()) == 1):

            # Obtain the branching/end point
            BrPoints = np.where(np.array(Net.degree()) > 2)[0]
            EndPoints = np.where(np.array(Net.degree()) == 1)[0]

            AllPaths = list()

            # Keep track of the interesting nodes
            SelEp = np.union1d(BrPoints, EndPoints)

            for i in range(len(SelEp) - 1):
                AllPaths.extend(
                    Net.get_shortest_paths(
                        SelEp[i], to=SelEp[range(i + 1, len(SelEp))], output="vpath"
                    )
                )

            Valid = (
                np.array(
                    list(map(lambda x: np.sum(np.array(Net.degree(x)) != 2), AllPaths))
                )
                == 2
            )

            AllPaths = list(np.array(AllPaths)[Valid])

            if not KeepEnds:
                AllPaths = list(map(lambda x: set(x).difference(BrPoints), AllPaths))

            names_AllPaths = ["Branch" + str(i) for i in range(len(AllPaths))]

            CapturedNodes = [item for sublist in AllPaths for item in sublist]
            StillToCapture = set(range(Net.vcount())).difference(CapturedNodes)

            if len(StillToCapture) > 0:
                print(
                    "Unassigned nodes detected. This is due to the presence of loops. Additional branching assignement will be performed."
                    + "WARNING : case not verified, compare with R version"
                )

                # computing all the distances between the unassigned points and the interesting points
                AllDists = Net.shortest_paths(StillToCapture, SelEp)

                # get the closest interesting point
                EndPoint_1 = np.min(AllDists, axis=1, keepdims=1)

                EndPoint_2 = EndPoint_1.copy()
                EndPoint_2[:, :] = np.nan

                # to get the second we have to avoid passing trough the first one
                for i in range(len(StillToCapture)):
                    tNet = Net.delete_vertices(EndPoint_1[i])
                    PointDists = tNet.shortest_paths(
                        StillToCapture[i], to=set(SelEp).intersection(tNet.vs["name"])
                    )

                    EndPoint_2[i] = PointDists[np.argmin(PointDists)]

                EndPoints = np.vstack((EndPoint_1, EndPoint_2))
                EndPoints = np.sort(EndPoints, axis=0)

                NewBrEP = EndPoints[:, ~pd.DataFrame(EndPoints.T).duplicated()]
                # NewBrEp = unique_rows = np.unique(original_array, axis=1)
                for i in range(NewBrEP.shape[1]):
                    # for each pair of interesting points
                    # print(i)
                    # Create a temporary network by merging the path with the end points
                    tNet = Net.induced_subgraph(
                        set(
                            StillToCapture[
                                np.all(np.isin(EndPoints, NewBrEP[:, i]), axis=0)
                            ]
                        ).union(NewBrEP[:, i])
                    )

                    if Net.are_connected(NewBrEP[0, i], NewBrEP[1, i]):
                        tNet.delete_edges(Net.get_eids(NewBrEP[:, i]))

                    tNet = tNet.induced_subgraph(tNet.degree() > 0)

                    # plot(tNet)

                    PotentialEnds = set(NewBrEP[:, i]).intersection(tNet.vs["name"])

                    if len(PotentialEnds) == 1:
                        # it's a simple loop

                        # get all loops
                        AllLoops = tNet.get_isomorphisms_vf2(
                            igraph.Graph.Ring(
                                tNet.vcount(), directed=False, circular=True
                            )
                        )

                        # select one with the branching point at the beginning
                        Sel = np.where(AllLoops == PotentialEnds)[0][0]

                        # Add a new branch
                        AllPaths.append(AllLoops[Sel])
                        names_AllPaths.append("Branch_" + str(len(AllPaths)))

                    if len(PotentialEnds) == 2:
                        # it's either a line or a line and a loop

                        LinePath = tNet.get_shortest_paths(
                            PotentialEnds[1], PotentialEnds[2]
                        )

                        # Add a new branch
                        AllPaths.append(LinePath)
                        names_AllPaths.append("Branch_" + str(len(AllPaths)))

                        # tNet

                        if len(LinePath) < tNet.vcount():
                            # assuming that the remaining part is a loop do as above
                            AllLoops = tNet.get_subisomorphisms_vf2(
                                tNet,
                                igraph.Graph.Ring(
                                    tNet.vcount() - len(LinePath) + 1,
                                    directed=False,
                                    circular=True,
                                ),
                            )

                            if len(AllLoops) == 0:
                                raise ValueError(
                                    "Unsupported structure. Contact the package maintainer"
                                )

                            # select one with the branching point at the beginning
                            Sel = np.where(np.isin(AllLoops, PotentialEnds))[0][0]

                            if Sel == np.array([]):
                                raise ValueError(
                                    "Unsupported structure. Contact the package maintainer"
                                )

                            # Add a new branch
                            AllPaths.append(AllLoops[Sel])
                            names_AllPaths.append("Branch_" + str(len(AllPaths)))

            AllPaths = [i for i in AllPaths if len(i) > 0]

            return AllPaths

        else:
            Structure = "end2end"

    if Structure == "branches&bpoints":

        if np.any(np.array(Net.degree()) > 2) & np.any(np.array(Net.degree()) == 1):

            # Obtain the branching/end point
            BrPoints = np.where(np.array(Net.degree()) > 2)[0]
            EndPoints = np.where(np.array(Net.degree()) == 1)[0]

            Allbr = list()
            SelEp = np.union1d(BrPoints, EndPoints)

            for i in BrPoints:

                SelEp = set(SelEp).difference([i])

                for j in SelEp:
                    Path = Net.get_shortest_paths(i, to=j, output="vpath")[0]

                    if not any(
                        np.isin(
                            np.array(Path),
                            np.array(list(set(BrPoints).difference([i, j]))),
                        )
                    ):
                        Allbr.append(Path)

            Allbr = list(map(lambda x: list(set(x).difference(BrPoints)), Allbr))

            BaseNameVect = ["Branch" + str(i) for i in range(len(Allbr))]

            BrCount = 0

            for i in BrPoints:
                BrCount = BrCount + 1
                Allbr.append([i])
                BaseNameVect.append(["Branch" + str(BrCount)])

            # names_Allbr = BaseNameVect

            return Allbr

        else:
            Structure == "end2end"

    if Structure == "branching":
        raise ValueError("Not implemented")
    #        BrPoints = np.where(igraph.degree(Net)>2)
    #
    #        Allbr = list()
    #
    #        for i in BrPoints:
    #            Points = i
    #            DONE = False
    #            Terminal_Branching = None
    #            while not DONE:
    #                Nei = unlist(igraph.neighborhood(Net, 1, Points))
    #                Nei = set(Nei).difference(Points)
    #
    #                NeiDeg = igraph.degree(Net, Nei, loops = False)
    #                NewPoints = union(Points, Nei[NeiDeg < 3])
    #
    #                Terminal_Branching = union(Terminal_Branching, Nei[NeiDeg >= 3])
    #
    #                if(len(set(NewPoints).difference(Points)) == 0):
    #                    DONE = True
    #                else:
    #                    Points = NewPoints
    #
    #
    #            if(KeepEnds):
    #                Allbr.append(set(Points).union(Terminal_Branching))
    #
    #            else:
    #                Allbr.append(Points)
    #
    #
    #            names_Allbr = ["Subtree_"+str(i) for i in range(len(Allbr))]
    #
    #        return(Allbr)

    if Structure == "end2end":

        EndPoints = np.where(np.array(Net.degree()) == 1)[0]

        Allbr = list()

        for i in range(len(EndPoints) - 1):
            for j in range(i + 1, len(EndPoints)):
                Path = Net.get_shortest_paths(
                    EndPoints[i], to=EndPoints[j], output="vpath"
                )
                if len(Path) > 0:
                    Allbr.append(Path)

        # names_Allbr = ["Path_" + str(i) for i in range(len(Allbr))]

        return Allbr


def GetBranches(Net, StartingPoints=None):
    """
    #' Return a dict summarizing the branching structure
    #'
    #' @param Net an igraph network
    #' @param StartingPoint the starting points
    #'
    #' @return a data frame with three columns:
    #' \itemize{
    #'  \item VName contains the vertx names
    #'  \item Branch contains the id of the branch (numbering start from the branch
    #' containing the starting point) or 0 if a branching point
    #'  \item BrPoints the branchhing point id, or 0 if not a branching point
    #' }
    #'
    #' @export
    #'
    #' @examples
    """
    # Net = ConstructGraph(ReturnList[0])
    StartingPoint = None

    if StartingPoint is None:
        EndPoints = np.where(np.array(Net.degree()) == 1)[0]
        StartingPoint = np.random.choice(EndPoints, 1)

    if Net.degree(StartingPoint)[0] != 1:
        raise ValueError("Invalid starting point")

    Vertices = Net.vs["name"]

    Branches = np.zeros(len(Vertices))
    DiffPoints = np.zeros(len(Vertices))

    # names_Branches = Vertices
    # names_DiffPoints = Vertices

    tNet = copy.deepcopy(Net)
    CurrentEdges = StartingPoint
    Branches[StartingPoint] = 1
    NewEdges = set()

    while len(tNet.vs.indices) > 0:

        for i in range(len(CurrentEdges)):
            AllNei = [
                tNet.vs["name"][k]
                for k in tNet.neighborhood(
                    vertices=tNet.vs["name"].index(CurrentEdges[i]), order=1
                )
            ]
            AllNei = list(set(AllNei).difference([CurrentEdges[i]]))

            NewEdges = NewEdges.union(AllNei)
            if len(AllNei) > 0:
                for j in range(len(AllNei)):
                    if tNet.degree(tNet.vs["name"].index(AllNei[j])) > 2:
                        DiffPoints[AllNei[j]] = np.max(DiffPoints) + 1

                    else:
                        if DiffPoints[CurrentEdges[i]] > 0:
                            Branches[AllNei[j]] = np.max(Branches) + 1

                        else:
                            Branches[AllNei[j]] = Branches[CurrentEdges[i]]
            #                 print(DiffPoints)
            tNet.delete_vertices(tNet.vs["name"].index(CurrentEdges[i]))

        CurrentEdges = list(NewEdges)
        CurrentEdges.sort()
        NewEdges = set()

    return dict(VName=Vertices, Branch=Branches, BrPoints=DiffPoints)

