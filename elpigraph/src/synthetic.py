import numpy as np

def InitializeWalkers(Number = 1,
                      Dimensions = 3,
                      MeanShift = 1,
                      SdShift = .3):
    '''
    #' Initialize a group of 'walkers'
    #'
    #' @param Number integer, the number of walkers to be initialised
    #' @param Dimensions integer, the number of dimensions of the walkers
    #' @param MeanShift numeric, the mean shift in each dimension for the walker 
    #' @param SdShift positive numeric, the standard deviation in the shift in each dimension for the walker 
    #'
    #' @return A list with the walkers
    #' @export
    #'
    #' @examples
    #'
    #' Walkers <- InitializeWalkers(Dimensions = 1000)
    #'
    '''
    RetList = list()

    for i in range(Number):
        RetList.append(dict(
          nDim = Dimensions,
          Positions = np.zeros(Dimensions),
          DiffusionShift = np.random.choice([-1, 1], 10) * np.random.normal(MeanShift, SdShift,Dimensions),
          Name = i,
          Age = 0
        ))

    return(RetList)


def GrowPath(Walkers,
             StepSD = .1,
             nSteps = 100,
             BranchProb = .01,
             MinAgeBr = 50,
             BrDim = 5):
    '''
    #' Grow a branching path using walkers
    #'
    #' @param Walkers list, a list of walker as returned from the InizializeWalkers function
    #' @param StepSize positive numeric, the standard deviation associated with the movement
    #' in each direction for each step
    #' @param nSteps integer, the number of steps 
    #' @param BranchProb numeric between 0 and 1, the probability per walker of branching at each step 
    #' @param MinAgeBr integer, the minimal number of steps before a newly introduced walker will start branhing
    #' @param BrDim integer, the number of dimensions affected during branching
    #'
    #' @return
    #' @export
    #'
    #' @examples
    #'
    #' Walkers <- InizializeWalkers(nDim = 1000)
    #'
    #' Data <- GrowPath(Walkers = Walkers, StepSize = 50, nSteps = 2000, BranchProb = .0015, MinAgeBr = 75, BrDim = 15)
    #'
    '''
    Trace = np.zeros((1,len(Walkers[0]['Pos'])))
    Branch = []
    TimeVect = []
    TimeVal = -1

    while len(Trace) < nSteps:
        nWalkers = len(Walkers)
        TimeVal = TimeVal + 1
        for j in range(nWalkers):
            # Create a new point by diffusion
            Walkers[j]['Pos'] = Walkers[j]['Pos'] + Walkers[j]['Dif'] + np.random.normal(size = Walkers[j]['nDim'], scale = StepSD)
            Trace = np.vstack(Trace, Walkers[j]['Pos'])
            # Keep track of the time
            TimeVect.append(TimeVal)
            # In crese the age of the walker
            Walkers[j]['Age'] = Walkers[j]['Age'] + 1
            # Keep Track of the originating population
            Branch.append(Walkers[j]['Name'])
            if((np.random.uniform() < BranchProb) & (Walkers[j]['Age'] >= MinAgeBr)):
                # Create a new branch
                BranchDim = np.random.uniform(size = Walkers[j]['nDim']) < BrDim/Walkers[j]['nDim']
                if (np.any(BranchDim) & (~np.all(BranchDim))):
                    print("Time=", len(Trace), " Branching in ", sum(BranchDim), " dimensions \n")

                    Walkers[j]['Age'] = 0
                    Walkers[j]['Name'] = max([i['Name'] for i in Walkers]) + 1

                    NewBuild = Walkers[j]
                    NewBuild['Age'] = 0
                    NewBuild['Dif'][BranchDim] = -NewBuild['Dif'][BranchDim]
                    NewBuild['Name'] = Walkers[j]['Name'] +1
                    Walkers.append(NewBuild)

    return(dict(UpdatedWalker = Walkers, Trace = Trace, Branch = Branch, Time = TimeVect))
