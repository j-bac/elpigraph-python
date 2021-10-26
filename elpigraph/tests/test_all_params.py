import pytest
import numpy as np
import matplotlib.pyplot as plt
import elpigraph


@pytest.fixture
def data():
    X = np.genfromtxt("./data/tree_data.csv", delimiter=",")
    return X


# test default and non-default parameters
def test_elpi_params(data):
    # Create desired list of inputs for R and Python
    input_data = [data] * 5
    epg_n_nodes = [20, 25, 15, 20, 30]
    epg_lambda = [0.1, 0.02, 0.7, 0.03, 0.08]
    epg_mu = [0.02, 0.07, 0.01, 0.04, 0.06]
    epg_trimmingradius = [float("inf"), 0.7, 0.8, 0.6, 0.5]
    epg_finalenergy = ["Penalized", "Base", "Penalized", "Base", "Base"]
    epg_alpha = [0.01, 0.03, 0.05, 0.08, 0.04]
    epg_beta = [0.03, 0.02, 0.04, 0.07, 0.01]
    epg_mode = [2, 1, 1, 2, 1]
    epg_n_processes = [1, 2, 1, 2, 1]
    epg_collapse_mode = [
        "PointNumber",
        "PointNumber_Extrema",
        "PointNumber_Leaves",
        "EdgesNumber",
        "EdgesLength",
    ]
    epg_collapse_par = [5, 7, 4, 6, 3]
    epg_maxsteps = [float("inf"), 1000, 100, 20, 200]
    # Python uses WeightedCentroid not Weigthed (corrected typo)
    epg_ext_mode = [
        "QuantDists",
        "QuantCentroid",
        "WeightedCentroid",
        "QuantCentroid",
        "WeightedCentroid",
    ]
    r_epg_ext_mode = [
        "QuantDists",
        "QuantCentroid",
        "WeigthedCentroid",
        "QuantCentroid",
        "WeigthedCentroid",
    ]
    epg_ext_par = [0.5, 0.6, 0.8, 0.9, 0.5]
    epg_shift_mode = [
        "NodeDensity",
        "NodePoints",
        "NodeDensity",
        "NodePoints",
        "NodeDensity",
    ]
    epg_shift_radius = [0.05, 0.07, 0.04, 0.08, 0.03]
    epg_shift_max = [5, 7, 4, 8, 6]

    # Results storage Python
    epg_main = []
    epg_obj_collapse = []
    epg_obj_shift = []
    epg_obj_extend = []
    epg_obj_fineTune = []

    for i in range(len(input_data)):

        ############################ Run functions, Python version ###################################

        epg_main.append(
            elpigraph.computeElasticPrincipalTree(
                X=input_data[i],
                NumNodes=epg_n_nodes[i],
                Lambda=epg_lambda[i],
                Mu=epg_mu[i],
                TrimmingRadius=epg_trimmingradius[i],
                FinalEnergy=epg_finalenergy[i],
                alpha=epg_alpha[i],
                beta=epg_beta[i],
                Do_PCA=False,
                CenterData=False,
                n_cores=epg_n_processes[i],
                nReps=1,
                EmbPointProb=1.0,
                Mode=epg_mode[i],
                MaxSteps=epg_maxsteps[i],
            )[0]
        )

        # util functions input
        epg_obj = epg_main[i]
        init_nodes_pos = epg_obj["NodePositions"]
        init_edges = epg_obj["Edges"][0]
        #########################################
        epg_obj_collapse.append(
            elpigraph.CollapseBranches(
                X=input_data[i],
                PG=epg_obj,
                Mode=epg_collapse_mode[i],
                ControlPar=epg_collapse_par[i],
            )
        )

        epg_obj_shift.append(
            elpigraph.ShiftBranching(
                X=input_data[i],
                PG=epg_obj,
                TrimmingRadius=epg_trimmingradius[i],
                SelectionMode=epg_shift_mode[i],
                DensityRadius=epg_shift_radius[i],
                MaxShift=epg_shift_max[i],
            )
        )

        epg_obj_extend.append(
            elpigraph.ExtendLeaves(
                X=input_data[i],
                PG=epg_obj,
                TrimmingRadius=epg_trimmingradius[i],
                Mode=epg_ext_mode[i],
                ControlPar=epg_ext_par[i],
                DoSA_maxiter=100,
            )
        )  # number of iterations for simulated annealing
        epg_obj_fineTune.append(
            elpigraph.fineTuneBR(
                X=input_data[i],
                MaxSteps=epg_maxsteps[i],
                Mode=2,
                NumNodes=epg_n_nodes[i],
                InitNodePositions=init_nodes_pos,
                InitEdges=init_edges,
                Lambda=epg_lambda[i],
                Mu=epg_mu[i],
                TrimmingRadius=epg_trimmingradius[i],
                FinalEnergy=epg_finalenergy[i],
                alpha=epg_alpha[i],
                beta=epg_beta[i],
                Do_PCA=False,
                CenterData=False,
                n_cores=epg_n_processes[i],
                nReps=1,
                ProbPoint=1.0,
            )[0]
        )

