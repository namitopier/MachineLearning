import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import special
import GMM_load
from sklearn.model_selection import KFold
import sklearn.datasets



'''
Gaussian mixture models

The ideia is to approximate a RV X when its density is not known.
How? --> Do a weighted sum of M Gaussians. The function receives M,S,w --> M = [µ1. . . µM] , S = [Σ1 . . . ΣM] , w = [w1 . . . wM]
'''

def mcol(v):
    return v.reshape((v.size, 1))

def logpdf_GMM(X, gmm, returnS = False):
    '''
    ## Overview:
    - The ideia is to approximate each sample of X to a gaussian distribution. In order to do it, this function calls "logpdf_GAU_ND" from lab 4.

    ## Params:
    - X = (D,N) where D is the size of a sample and N is the number of samples in X;
    - gmm = (M, 3) where M is the number of gaussians that we are considering in the weighted sum (gmm = [[w1, mu1, C1], [w2, mu2, C2], ...]) --> w1 is a value, mu = (D, 1) and C = (D,D).

    ## Returns
    If returnS = False, it returns just the log-marginal (which is the sum of S -- See the lab 10). \n
    Else, it returns both S and log marginal
    '''
    # Retrieving N:
    N = X.shape[1]
    # Retrieving M (since gmm is a list and not an array, can't use "shape"):
    M = len(gmm)

    # Creating a matrix S (M,N) to store the log Gau_ND for each mu, C for each sample
    S = []

    for i in range (M):

        w = gmm[i][0]
        mu = gmm[i][1]
        C = gmm[i][2]

        densities = logpdf_GAU_ND(X, mu, C) # Densities is a vector where each element corresponds to log Gau for the current attribute
        densities += np.log(w)
        S.append(densities)
    
    S = np.reshape(S, (M, N))

    # Now, since everything is log, need to perform a sum of the logs. This means:
    logdens = scipy.special.logsumexp(S, axis=0)

    if (returnS):
        return S, np.reshape(logdens, (1, N))
    else: 
        return np.reshape(logdens, (1, N))

def GMM_EM(X, initialGMM, stop, psi):
    '''
    ## Params
    - X = (D,N) where D is the size of a sample and N is the number of samples in X;
    - gmm = (M, 3) where M is the number of gaussians that we are considering in the weighted sum (gmm = [[w1, mu1, C1], [w2, mu2, C2], ...]) --> w1 is a value, mu = (D, 1) and C = (D,D).
    - stop = stop criterion for the algorithm iterations
    - psi = Lower bound to the eigenvalues of the covariance matrices (to make sure that that likelihood doesn't decrease)


    ## Explanation
    EM is an algorithm to calculate an estimation/approximation for a gmm. It uses an approach that is divided into 2 steps: E and M.\n
    - On the E step, it calculates the posterior probability for each component of the given gmm and for each sample. The result is called responsabilities.
    - On the M step, it calculates the new estimated parameters using the responsabilities calculated on the previous step.
    The algorithm runs until the difference of the previous likelihood and the current is lower than the "stop" parameter.
    '''

    # Regtrieving number of samples
    N = X.shape[1]
    D = X.shape[0]
    # Retrieving M (since gmm is a list and not an array, can't use "shape"):
    M = len(initialGMM)

    gmm = []

    for i in range(M):
        # Constraining the eigenvalues of the covariance matrices
        w = initialGMM[i][0]
        mu = initialGMM[i][1]
        C = initialGMM[i][2]
        
        U, s, _ = np.linalg.svd(C)
        s[s<psi] = psi
        C = np.dot(U, mcol(s)*U.T)

        gmm.append([w, mu, C])

    # Initial E step:
    S, marginal = logpdf_GMM(X, gmm, returnS=True) # S = (M, N) and marginal = (1, N)
    responsabilities = np.exp(S-marginal) # = Yg,i = (M, N)

    while(True):

        newGMM = []

        prevLikelihood = np.sum(marginal)/N

        # M step:
        for i in range(M):
            Zg = np.sum(responsabilities[i])
            Fg = np.sum(responsabilities[i]*X, 1) # Fg = (D, 1)
            Sg = np.dot(X, (responsabilities[i]*X).T)

            newMu = np.reshape(Fg/Zg, (D, 1))
            newCov = Sg/Zg - np.dot(newMu, newMu.T)
            newW = Zg/(np.sum(responsabilities))

            # Constraining the eigenvalues of the covariance matrices
            U, s, _ = np.linalg.svd(newCov)
            s[s<psi] = psi
            newCov = np.dot(U, mcol(s)*U.T)

            newGMM.append([newW, newMu, newCov])

        # New E step
        S, marginal = logpdf_GMM(X, newGMM, returnS=True) # S = (M, N) and marginal = (1, N)
        newLikelihood = np.sum(marginal)/N

        if(newLikelihood - prevLikelihood < stop):
            break
        else:
            gmm = newGMM
            responsabilities = np.exp(S-marginal)
    
    print("======================= FINAL LIKELIHOOD = ", newLikelihood)
    return newGMM

def diagonal_GMM_EM(X, initialGMM, stop, psi):
    '''
    ## Params
    - X = (D,N) where D is the size of a sample and N is the number of samples in X;
    - gmm = (M, 3) where M is the number of gaussians that we are considering in the weighted sum (gmm = [[w1, mu1, C1], [w2, mu2, C2], ...]) --> w1 is a value, mu = (D, 1) and C = (D,D).
    - stop = stop criterion for the algorithm iterations

    ## Explanation
    EM is an algorithm to calculate an estimation/approximation for a gmm. It uses an approach that is divided into 2 steps: E and M.\n
    - On the E step, it calculates the posterior probability for each component of the given gmm and for each sample. The result is called responsabilities.
    - On the M step, it calculates the new estimated parameters using the responsabilities calculated on the previous step.
    The algorithm runs until the difference of the previous likelihood and the current is lower than the "stop" parameter.
    '''

    # Regtrieving number of samples
    N = X.shape[1]
    D = X.shape[0]
    # Retrieving M (since gmm is a list and not an array, can't use "shape"):
    M = len(initialGMM)

    gmm = []

    for i in range(M):
        # Constraining the eigenvalues of the covariance matrices
        w = initialGMM[i][0]
        mu = initialGMM[i][1]
        C = initialGMM[i][2]
        
        U, s, _ = np.linalg.svd(C)
        s[s<psi] = psi
        C = np.dot(U, mcol(s)*U.T)

        gmm.append([w, mu, C])

    # Initial E step:
    S, marginal = logpdf_GMM(X, gmm, returnS=True) # S = (M, N) and marginal = (1, N)
    responsabilities = np.exp(S-marginal) # = Yg,i = (M, N)

    while(True):

        newGMM = []

        prevLikelihood = np.sum(marginal)/N

        # M step:
        for i in range(M):
            Zg = np.sum(responsabilities[i])
            Fg = np.sum(responsabilities[i]*X, 1) # Fg = (D, 1)
            Sg = np.dot(X, (responsabilities[i]*X).T)

            newMu = np.reshape(Fg/Zg, (D, 1))
            newCov = Sg/Zg - np.dot(newMu, newMu.T)
            newCov = np.multiply(newCov, np.identity(D))
            newW = Zg/(np.sum(responsabilities))

            # Constraining the eigenvalues of the covariance matrices
            U, s, _ = np.linalg.svd(newCov)
            s[s<psi] = psi
            newCov = np.dot(U, mcol(s)*U.T)

            newGMM.append([newW, newMu, newCov])

        # New E step
        S, marginal = logpdf_GMM(X, newGMM, returnS=True) # S = (M, N) and marginal = (1, N)
        newLikelihood = np.sum(marginal)/N

        if(newLikelihood - prevLikelihood < stop):
            break
        else:
            gmm = newGMM
            responsabilities = np.exp(S-marginal)
    
    return newGMM

def tied_GMM_EM(X, initialGMM, stop, psi):
    '''
    ## Params
    - X = (D,N) where D is the size of a sample and N is the number of samples in X;
    - gmm = (M, 3) where M is the number of gaussians that we are considering in the weighted sum (gmm = [[w1, mu1, C1], [w2, mu2, C2], ...]) --> w1 is a value, mu = (D, 1) and C = (D,D).
    - stop = stop criterion for the algorithm iterations

    ## Explanation
    EM is an algorithm to calculate an estimation/approximation for a gmm. It uses an approach that is divided into 2 steps: E and M.\n
    - On the E step, it calculates the posterior probability for each component of the given gmm and for each sample. The result is called responsabilities.
    - On the M step, it calculates the new estimated parameters using the responsabilities calculated on the previous step.
    The algorithm runs until the difference of the previous likelihood and the current is lower than the "stop" parameter.
    '''

    # Regtrieving number of samples
    N = X.shape[1]
    D = X.shape[0]
    # Retrieving M (since gmm is a list and not an array, can't use "shape"):
    M = len(initialGMM)

    gmm = []

    for i in range(M):
        # Constraining the eigenvalues of the covariance matrices
        w = initialGMM[i][0]
        mu = initialGMM[i][1]
        C = initialGMM[i][2]
        
        U, s, _ = np.linalg.svd(C)
        s[s<psi] = psi
        C = np.dot(U, mcol(s)*U.T)

        gmm.append([w, mu, C])

    # Initial E step:
    S, marginal = logpdf_GMM(X, gmm, returnS=True) # S = (M, N) and marginal = (1, N)
    responsabilities = np.exp(S-marginal) # = Yg,i = (M, N)

    while(True):

        newGMM = []

        prevLikelihood = np.sum(marginal)/N

        newWList = []
        newMuList = []
        sumCov = 0
        # M step:
        for i in range(M):
            Zg = np.sum(responsabilities[i])
            Fg = np.sum(responsabilities[i]*X, 1) # Fg = (D, 1)
            Sg = np.dot(X, (responsabilities[i]*X).T)

            newMu = np.reshape(Fg/Zg, (D, 1))
            sumCov += Zg*(Sg/Zg - np.dot(newMu, newMu.T))
            newW = Zg/(np.sum(responsabilities))

            newGMM.append([newW, newMu])

        newCov = sumCov/N
        # Constraining the eigenvalues of the covariance matrices
        U, s, _ = np.linalg.svd(newCov)
        s[s<psi] = psi
        newCov = np.dot(U, mcol(s)*U.T)

        for i in range(M):
            newGMM[i].append(newCov)

        # New E step
        S, marginal = logpdf_GMM(X, newGMM, returnS=True) # S = (M, N) and marginal = (1, N)
        newLikelihood = np.sum(marginal)/N

        if(newLikelihood - prevLikelihood < stop):
            break
        else:
            gmm = newGMM
            responsabilities = np.exp(S-marginal)
    
    return newGMM


def GMM_LBG(X, gmm, alpha, psi, components, stop, type = 'fullCovariance'):
    '''
    ## Explanation
    Given a initial guess for the GMM parameters ([w1, mu1, C1]), it can generate 2, 3, 4... Gs more, so it would become: [[w1, mu1, C1], [w2, mu2, C2], ...]
    With the new generated G, can use it in the EM algorithm to estimate a solution for the model.
    If it receives G, will return 2G.
    ## Params
    - gmm = (M, 3) where M is the number of gaussians that we are considering in the weighted sum (gmm = [[w1, mu1, C1], [w2, mu2, C2], ...]).
    - alpha = factor of multiplication
    - psi = Lower bound to the eigenvalues of the covariance matrices (to make sure that that likelihood doesn't decrease)
    - components = How many components to consider in the prediction
    - stop = stop criterion for the EM algorithm iterations
    '''

    # first optimization using the initial guess
    if (type == 'fullCovariance'):
        newGMM = GMM_EM(X, gmm, stop, psi)
        # print("newGMM = ", newGMM)
    elif(type == 'diagonal'):
        newGMM = diagonal_GMM_EM(X, gmm, stop, psi)
    else:
        newGMM = tied_GMM_EM(X, gmm, stop, psi)

    # newGMM = gmm

    iterations = int(np.log2(components))
    
    for i in range (iterations):
        # Retrieving M (since gmm is a list and not an array, can't use "shape"):
        M = len(newGMM)

        genGMM = []
        for i in range (M):
            w = newGMM[i][0]
            mu = newGMM[i][1]
            C = newGMM[i][2]
            
            # Constraining the eigenvalues of the covariance matrices
            U, s, _ = np.linalg.svd(C)
            s[s<psi] = psi
            C = np.dot(U, mcol(s)*U.T)

            # Calculating displacement vector dg:
            U, s, Vh = np.linalg.svd(C)
            dg = U[:, 0:1] * s[0]**0.5 * alpha

            genGMM.append([w/2, mu + dg, C])
            genGMM.append([w/2, mu - dg, C])

        if (type == 'fullCovariance'):
            newGMM = GMM_EM(X, genGMM, stop, psi)
        elif(type == 'diagonal'):
            newGMM = diagonal_GMM_EM(X, genGMM, stop, psi)
        else:
            newGMM = tied_GMM_EM(X, genGMM, stop, psi)

    return newGMM

def calculateGMM(DTR, DTE, LTR, alpha, psi, components, totalClasses, stop, type = 'fullCovariance'):
    '''
    ## Params
    - DTR
    - DTE
    - initialGMM = [w, mu, C] = Initial gmm to consider
    - alpha = factor of multiplication
    - psi = Lower bound to the eigenvalues of the covariance matrices (to make sure that that likelihood doesn't decrease)
    - components = How many components to consider in the prediction
    - totalClasses = number of classes to be considered
    - type = 'fullCovariance', 'diagonal' and 'tiedCovariance'
    '''

    # ============================================================================
    # FIRST STEP: Calcuate the empirical mean and covariance matrix for each class
    # ============================================================================
    
    # Picking samples for each class
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]
    D2 = DTR[:, LTR==2]

    # First step: Calculate mu as the mean of each attribute between all samples.
    mu0 = vcol(D0.mean(1))
    mu1 = vcol(D1.mean(1))
    mu2 = vcol(D2.mean(1))

    # Now it is needed to center the data, i.e., subtract mu (the mean) from all columns of D.
    DC0 = D0 - mu0
    DC1 = D1 - mu1
    DC2 = D2 - mu2

    # Now, it is needed to calculate the covariance matrix C = 1/N * Dc*Dc.T
    NTR = DTR.shape[1]
    NTE = DTE.shape[1]

    N0 = D0.shape[1]
    C0 = (1/N0)*np.dot(DC0, np.transpose(DC0))

    N1 = D1.shape[1]
    C1 = (1/N1)*np.dot(DC1, np.transpose(DC1))

    N2 = D2.shape[1]
    C2 = (1/N2)*np.dot(DC2, np.transpose(DC2))

    # ============================================================================
    #        SECOND STEP: Estimate the gmms for the different classes            #
    # ============================================================================

    gmm0 = GMM_LBG(D0, [[1, mu0, C0]], alpha, psi, components, stop, type)
    gmm1 = GMM_LBG(D1, [[1, mu1, C1]], alpha, psi, components, stop, type)
    gmm2 = GMM_LBG(D2, [[1, mu2, C2]], alpha, psi, components, stop, type)

    # ============================================================================
    # THIRD STEP: Calcuate the log densities for each class and all test samples
    # ============================================================================

    # It will return a score matrix S (C, N) where C = number of classes and N = number of samples

    S = []
    S.append(logpdf_GMM(DTE, gmm0))
    S.append(logpdf_GMM(DTE, gmm1))
    S.append(logpdf_GMM(DTE, gmm2))
    S = np.reshape(S, (3, NTE))


    # ============================================================================
    # FOURTH STEP: Calcuate the log joint densities (add the score by the prior prob)
    # ============================================================================

    # logSJoint = S + np.reshape(np.log(prior), (len(prior), 1))
    
    # ============================================================================
    # FOURTH STEP: Calcuate the posterior probabilities
    # ============================================================================

    # logSMarginal =  vrow(scipy.special.logsumexp(logSJoint, axis=0))
    # SPost = np.exp(logSJoint - logSMarginal)

    SPost = np.exp(S)

    # The predicted labels are obtained picking the highest probability among the classes for a given sample
    predicted = SPost.argmax(0)
    return predicted


#============================================== From other labs ============================================================

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def vrow(line):
    return np.reshape(line, (1, len(line)))

def vcol(vlist):
    '''
    Will transform a 1D list into a column vector (rows = len(vlist) // columns = 1)
     1d list = [1,2,3] --vcol--> 
     
     [[1],
     [2],
     [3]]
    '''
    return np.reshape(vlist, (len(vlist), 1))

def logpdf_GAU_ND(X, mu, C):
    
    M = X.shape[0]
    T1 = -(M/2)*np.log(2*np.pi)
    T2 = -(1/2)*np.linalg.slogdet(C)[1]
    T3 = -(1/2)*( np.dot(
        np.dot( np.transpose(X-mu), np.linalg.inv(C) ), X-mu) )
    
    T3 = np.diag(T3)
    return T1 + T2 + T3

#============================================================================================================================

def main():
    dataset = np.load("GMM_data_4D.npy")
    gmm = GMM_load.load_gmm("GMM_4D_3G_init.json")

    # TEST OF EM ALGORITHM ===================================================================== CORRECT!

    # solution = GMM_load.load_gmm("GMM_4D_3G_EM.json")
    # obtained =  GMM_EM(dataset, gmm, 10**-6, 10**-6)

    # print("Solution: ", solution)
    # print("Obtained: ", obtained)

    # ============================== TEST FOR LBG WITH EM ======================================

    # solution = GMM_load.load_gmm("GMM_4D_4G_EM_LBG.json")

    # # First step: Calculate mu as the mean of each attribute between all samples.
    # mu = vcol(dataset.mean(1))

    # # Now it is needed to center the data, i.e., subtract mu (the mean) from all columns of D.
    # DC0 = dataset - mu

    # N0 = dataset.shape[1]
    # C0 = (1/N0)*np.dot(DC0, np.transpose(DC0))

    # gmm = [[1, mu, C0]]

    # obtained =  GMM_LBG(dataset, gmm, 0.1, 0.01, 4, 10**-6)

    # print("Solution: ", solution)
    # print("Obtained: ", obtained)

    # ================================= Evaluation =============================================

    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    predicted = calculateGMM(DTR, DTE, LTR, 0.1, 0.01, 2, 3, 10**-6, 'diagonal')

    print("predicted = ", predicted)
    print("correct = ", LTE)

    correct = 0
    for i in range (LTE.shape[0]):
        if (predicted[i] == LTE[i]):
            correct += 1

    print("Percentage of correct values = ", (correct/LTE.shape[0])*100)

    # ==========================================================================================

    # calcLogDens = logpdf_GMM(dataset, gmm)
    # correctLogDens = np.load("GMM_4D_3G_init_ll.npy")

    # print ("Calculated: \n", calcLogDens)
    # print ("Correct: \n", correctLogDens)

    # correct = 0
    # print(calcLogDens[0][0])
    # for i in range (correctLogDens.shape[1]):
    #     if (correctLogDens[0][i] == calcLogDens[0][i]):
    #         correct += 1
    
    # print("Percentage of correct values = ", (correct/correctLogDens.shape[1])*100)

if __name__ == '__main__':
    main()