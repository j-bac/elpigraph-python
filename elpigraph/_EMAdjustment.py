import numpy as np

def AdjustByConstant(EM,AdjustVect, k = 2, divMu = 10, divLambda = 10,verbose=False):
    '''
    #' Adjust elastic matrix by dividing the values of lambda and mu of stars
    #'
    #' @param GraphInfo the elpigraph structure updated
    #' @param k the largest orger of strars to leave unadjusted. e.g., if k = 2 only branching points will be adjusted
    #' @param divLambda the value used to divide the lambda coefficients
    #' @param divMu the value used to divide the mu coefficients
    #'
    #' @return
    #' @export
    #'
    #' @examples
    '''
    EM2 = EM.copy()
    
    StarOrder = np.sum(EM>0,axis=1) - 1
    StarOrder[StarOrder == 0] = 1

    ToAdjust = np.where((~np.array(AdjustVect)) & (StarOrder > k))[0]
    n_updated = len(ToAdjust)
    if verbose:
        print(n_updated, "values of the elastic matrix have been updated")

    if(n_updated>0):
        for i in ToAdjust:
            EM2[i,:] = EM[i,:]/divLambda
            EM2[:,i] = EM[:,i]/divLambda
            EM2[i,i] = EM[i,i]/divMu
        
            AdjustVect[i] = True
    return(EM2,AdjustVect)
