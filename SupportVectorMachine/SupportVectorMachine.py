from socket import IP_HDRINCL
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import sklearn.datasets

'''
SUPPORT VECTOR MACHINES

Basically: Helps finding a maximum margin separation hyperplane (i.e., a line that best sepparate 2 classes for example). Along with this approach, concepts such as "slack variables" (which represent how much a point is violating the constraint/line) arise.
Kernel functions also helps: non-linear sepparation

ATENCAO ATENCAO::::::::::::::::::::::::::::::::::::::::::::
NUMPY OF SHAPE (N,) IS NOOOOOOOOT A 2 DIMENSIONAL ARRAY!!!!!!!!! IT IS A A SIMPLE LIST MOTHER FUCKER
'''

# ============================== PROVIDED FUNCTIONS ======================================
def load_iris_binary():
    '''
    Loads the Iris dataset and ignores "Iris Setosa".\n
    The labels are: 1 (iris versicolor) and 0 (iris virginica).
    '''
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def load_iris():

    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

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

def vrow(vlist):
    '''
     Will transform a 1D list into a row vector (rows = 1 // columns = len(vlist))
     1d list = [1,2,3] --vrow--> [[1,2,3]]
    '''
    return np.reshape(vlist, (1, len(vlist)))

# =========================================================================================

def linearSVM_H(DTR, LTR, K = 1):
    '''
    ## Explanation
    Function that calculates the linear SVM. It computes the primal SVM solution through the dual SVM formulation Jb^D.
    Returns H (= zi*zj*xi.T*xj)

    ## Params
    - DTR = Matrix of training data (MxN) where M = number of attributes for each sample and N = number of samples in the training dataset.
    - LTR = Training labels (N,).
    - K = Parameter used to build the extended matrix. Increasing this number may lead to better decisions, but makes the dual problem harder to solve
    '''

    # Retrieving number of samples
    n = DTR.shape[1]

    # Building the extended matrix (adding K at the end of each feature x) and z array
    param = np.zeros((1, n))
    z = np.zeros((n,))
    for i in range(n):
        param[0][i] = K
        z[i] = 2*LTR[i] - 1

    extD = np.append(DTR, param, 0)

    # Computing H: First, will calculate G = xi.T*xj and then multiply by zi*zj
    G  = np.dot(extD.T, extD)
    zizj = z.reshape((n, 1))
    zizj = np.dot(zizj, zizj.T)
    H = np.multiply(zizj, G)

    return H

def kernelSVM_H(DTR, LTR, gamma, K = 0, RBF = True):
    '''
    ## Explanation
    Function that calculates non linear SVM using a kernel function k(xi, xj).
    Returns H (= zi*zj*k(xi, xj))

    ## Params
    - DTR = Matrix of training data (MxN) where M = number of attributes for each sample and N = number of samples in the training dataset.
    - LTR = Training labels (N,).
    - gamma = Hyperparameter used to calculate the kernel function
    - K = If want to add a regularized bias on the kernel function, the bias will be K^2
    '''

    # Retrieving number of samples
    n = DTR.shape[1]

    # Building z array
    z = np.zeros((n,))
    for i in range(n):
        z[i] = 2*LTR[i] - 1

    # Computing H: First, will calculate G = k(xi, xj) and then multiply by zi*zj
    G  = np.zeros((n,n))
    if (RBF):
        for i in range (n):
            for j in range (n):
                G[i][j] = RBFkernel(DTR[:, i], DTR[:, j], gamma, K**2)
    else:
        for i in range (n):
            for j in range (n):
                G[i][j] = quadraticKernel(DTR[:, i], DTR[:, j], K**2, c=1)

    zizj = z.reshape((n, 1))
    zizj = np.dot(zizj, zizj.T)
    H = np.multiply(zizj, G)

    return H

def quadraticKernel(xi, xj, bias, c=0):
    '''
    ## Explanation
    Radial Basis Function kernel: k(x1, x2) = (x1.T*x2 + c)^2 + bias

    ## Params
    - xi and xj = samples (M,) where M is the number of features considered for each sample
    - bias = If want to add a regularized bias on the kernel function

    '''

    # quadratic = np.exp( -gamma*(np.linalg.norm(xi - xj)**2) ) + bias
    quadratic = (np.dot(xi.T, xj) + c)**2 + bias
    return quadratic


def RBFkernel(xi, xj, gamma, bias):
    '''
    ## Explanation
    Radial Basis Function kernel: k(x1, x2) = e^( -gamma||xi-xj||^2 ) + b

    ## Params
    - xi and xj = samples (M,) where M is the number of features considered for each sample
    - gamma = Hyperparameter to use on the function
    - bias = If want to add a regularized bias on the kernel function

    '''

    RBF = np.exp( -gamma*(np.linalg.norm(xi - xj)**2) ) + bias
    return RBF

def minDualSVM(alpha, H, test = None):
    '''
    ## Explanation
    Function that calculates -J^D(alpha) = L^D(alpha) (So it can be minimized by scipy) and its gradient.
    Returns a tuple: LD and its gradient.

    ## Params
    - alpha = (N, 1) where N = number of samples
    - H = (N, N) -> Calculated on linearSVM function as zi*zj*xi.T*xj
    - test = variable that does nothing, is just here because otherwise scipy doesnt work and pass a number of arguments = size of H (bad implemented)
    '''

    # Retrieving number of samples
    n = alpha.shape[0]

    alpha = np.reshape(alpha, (n, 1))
    LD = 1/2*np.dot(np.dot(alpha.T, H),alpha) - ( np.dot(alpha.T,np.ones((n,1))) )

    gradient = np.dot(H, alpha) - np.ones((n, 1))
    gradient = np.reshape(gradient, (n,))

    return LD, gradient

def calculateSVM(DTR, LTR, C, DTE, LTE = None, K = 1, verbose = False, linear = True, gamma = 1, RBF = True):
    '''
    ## Explanation
    This functions uses "scipy.optimize.fmin_l_bfgs_b" to minimize L^D. It calls "linearSVM_H" retrieving H which will be used then by function "minDualSVM" to be minimized.
    The resulting alpha will be then used to calculate the primal solution and recover "w*"

    ## Params
    - DTR = Matrix of training data (MxN) where M = number of attributes for each sample and N = number of samples in the training dataset.
    - LTR = Training labels (N,).
    - C = Maximum value for estimated alpha (it is a hyperparameter)
    - DTE = Testing dataset (numpy array)
    - LTE = Only used if printStats == True: LTE is the correct label array
    - K = Parameter used to build the extended matrix. Increasing this number may lead to better decisions, but makes the dual problem harder to solve
    - linear = If set to False, it will use the RBF kernel function to calculate it 
    - gamma = Only used if non linear. It is a hyperparameter used on the calculation of kernel function
    '''
    
    # FIRST PART: Dual solution calculation ------------------------------------------------------------------------------

    # Retrieving number of samples
    n = DTR.shape[1]
    # Retrieving number of testing samples
    nt = DTE.shape[1]


    # Creating bounds for estimated alpha ((min, max), (min, max), ...), calculating z and extended matrix
    bounds = []
    startPoint = np.zeros((n))
    param = np.zeros((1, n))
    z = np.zeros((n,1))

    for i in range (n):
        bounds.append((0, C))
        z[i][0] = 2*LTR[i] - 1
        param[0][i] = K

    extD = np.append(DTR, param, 0) # Extended data matrix

    # Retrieving H
    if(linear):
        H = linearSVM_H(DTR, LTR, K)
    else:
        H = kernelSVM_H(DTR, LTR, gamma, K, RBF)


    # Calculating the minimum
    x,f,d = scipy.optimize.fmin_l_bfgs_b(minDualSVM, startPoint, args = (H, 1), bounds=bounds, iprint = verbose, approx_grad = False, factr=1.0)
    # HAVE TO PASS AT LEAST 2 ARGUMENTS OTHERWISE THIS STUPID FUNCTION DOESNT WORK"

    # SECOND PART ------------------------------------------------------------------------------------------------------------
    if(linear):
        # Primal solution calculation 
        alpha = np.reshape(x, (1, n))
        alphaZ = np.multiply(alpha, z.T)

        w = np.dot(alphaZ, extD.T)

        # THIRD PART: Classifying according to w and calculating the predicted classes array ---------------------------------

        param = np.zeros((1, nt))
        for i in range (nt):
            param[0][i] = K

        extT = np.append(DTE, param, 0) # Extended matrix

        # Calculating w.T * xt
        predicted = np.dot(w, extT) # In this case don't need to do do w.T because w is already of shape (1,5)
        predicted = np.reshape(predicted, (nt,))

    else:
        # Calculating for non-linear solution
        predicted = np.zeros((nt,))
        alpha = np.reshape(x, (n,))

        if(RBF):
            for i in range(nt):
                for j in range(n):
                    predicted[i] += alpha[j]*z[j][0]*RBFkernel(DTR[:, j],DTE[:, i], gamma, K**2)
        else:
            for i in range(nt):
                for j in range(n):
                    predicted[i] += alpha[j]*z[j][0]*quadraticKernel(DTR[:, j],DTE[:, i], K**2, c=1)

    for i in range (predicted.shape[0]):
        if (predicted[i] > 0):
            predicted[i] = 1
        else:
            predicted[i] = 0

    if (verbose and linear):
        print("======================== LINEAR SVM PREDICTION VERBOSE MODE ===============================")
        print("Obtained prediction of classes:\n")
        print(predicted)
        print("Percentage of correctly assigned classes:\n")
        correct = 0
        for i in range (predicted.shape[0]):
            if (predicted[i] == LTE[i]):
                correct +=1
        print( 100*(correct/LTE.shape[0]), "%" )
        
        gap = dualityGap(w, C, z, extD, f)
        print("Duality gap = ", gap)
        loss, _ = minDualSVM(x, H)
        print("Loss = ", loss)
    elif(verbose):
        print("===================== NON - LINEAR SVM PREDICTION VERBOSE MODE ============================")
        print("Obtained prediction of classes:\n")
        print(predicted)
        print("Percentage of correctly assigned classes:\n")
        correct = 0
        for i in range (predicted.shape[0]):
            if (predicted[i] == LTE[i]):
                correct +=1
        print( 100*(correct/LTE.shape[0]), "%" )
        print("Loss = ", f)

def dualityGap(w, C, z, extD, LD):
    '''
    ## Explanation
    At the optimal solution: J(w) = JD(alpha). So basically this function computes the primal objective j(w) of linear SVM and calculates the gap = j(w) - JD(alpha) (JD == -LD)
    The smaller the duality gap, the more precise is the dual (and thus the primal) solution.

    ## Params
    - w = Calculated weights of linear SVM (1,M) where M = number of attributes of each sample
    - C = Hyperparameter that determines the maximum of alpha (used on "calculateLinaerSVM")
    - z = (N, 1) where N = number of samples and z = -1 if class 0 and z = 1 otherwise
    - extD = Extended matrix of samples where last row is all equal to K (M+1, N)
    - LD = -JD and is retrieved from minDualSVM (is the value of the function at the minimum)
    '''

    # Retrieving number of samples
    n = z.shape[0]

    # Calculating j(w)
    j = 1/2*(np.linalg.norm(w)**2)
    temp = 0
    S = np.dot(w, extD)
    loss = np.maximum(np.zeros(S.shape), 1-np.reshape(z, (n,))*S).sum()
    j = j + C*loss

    gap = j + LD
    return gap
# ====================================================================================================================
def main():
    
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)    
    
    # there are 66 samples for training. 
    print("Training data has shape:", DTR.shape, "\n")
    print("Training label has shape:", LTR.shape, "\n")


    # H = linearSVM_H(DTR, LTR)
    # n = DTR.shape[1]
    # startPoint = np.zeros((n,1))

    # minDualSVM(startPoint, H)
    C = 1
    calculateSVM(DTR, LTR, C, DTE, LTE, verbose = True, linear=False, K = 1, gamma = 10, RBF = False)

if __name__ == "__main__":
    main()