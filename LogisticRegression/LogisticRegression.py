from asyncio.windows_events import NULL
from re import X
from select import select
import numpy as np
import scipy.optimize
import sklearn.datasets

'''
LOGISTIC REGRESSION
LR optimizes the Logistic Loss.


Want to minimize a function "func" starting from a point "x0". To do it, call "scipy.optimize.fmin_l_bfgs_b" -> USes second-order info (Hessian) so the convergence is faster

First task: 
Implement function (receives 1-D numpy array of shape (2,): (y,z)):
f(y, z) = (y + 3)^2 + sin(y) + (z + 1)^2.
'''

def simpleFunc(values):
    '''
    Calculates a simple convex function.
    ## Params
    Receives 1D numpy array (2,) == (y,z). \n
    Calculates: f(y, z) = (y + 3)^2 + sin(y) + (z + 1)^2.

    ## Returns
    Calculated value + gradient of f with respect to y and z as a numpy array (2,) 
    '''
    # Retrieving z and y
    y = values[0]
    z = values[1]

    # Calculating f(y,z)
    calculated = (y + 3)**2 + np.sin(y) + (z + 1)**2
    
    # Calculating gradients
    gradY = 2*(y+3) + np.cos(y)
    gradZ = 2*(z+1)
    gradients = np.zeros((2,))
    gradients[0] = gradY
    gradients[1] = gradZ

    return calculated, gradients

def findMinimum(startPoint = np.zeros((2,)), approxGrad = False):
    '''
    Calculates the estimated position of the minimum, value at the minimum and some additional info.\n
        - startPoint is by default np.array([0,0])\n
        - ApproxGrad is by default False (let it calculate an approximated gradient (not so efficient))
     '''
    #startPoint = np.zeros((2,))
    x, f, d = scipy.optimize.fmin_l_bfgs_b(simpleFunc, startPoint, approx_grad=approxGrad, iprint = 1)
    return x, f, d

def logreg_obj(v, DTR, LTR, l):
    '''
    ## Params:

    v = numpy array "(D+1,) = (w,b)" where D = dimensionality of attributes (e.g, D=4 for Iris) and the last column is the b (biases).\n
    DTR = Training data.\n
    LTR = Training labels.\n
    l = lambda (Multiplier of w).\n
    '''
    # Retrieving n (number of samples)
    n = DTR.shape[1]

    # Retrieving the weights and biases
    w = v[0:-1]
    b = v[-1]

    temp = l/2*np.linalg.norm(w)**2
    temp2 = 0
    Z = np.where(LTR > 0, 1, -1)

    temp2 = 1/n*np.sum(np.logaddexp(0, np.multiply(-Z, np.dot(np.transpose(w),DTR) + b)))
    print ("temp2 is ", temp2)
    return temp + temp2

    for i in range (n):
        z = 2*LTR[i] - 1 # Which means: z == 1 if class == 1, z == -1 otherwise
        temp2 += np.logaddexp( 0, -z*(np.dot(np.transpose(w),DTR[:,i]) + b) ) # Check if have to do transpose here...

    return temp + 1/n*temp2
        
def posteriorLikelihood(v, DTE, printStats = False, LTE = NULL):
    '''
    ## Params:
    - v = numpy array "(D+1,) = (w,b)" where D = dimensionality of attributes (e.g, D=4 for Iris) and the last column is the b (biases).\n
    - DTE = Testing dataset (numpy array)\n
    - b = bias vector
    - printStats = Verbose mode for the estimation: Show percentage of correctly assigned classes\n
    - LTE = Only used if printStats == True: LTE is the correct label array\n

    ## Return:
    Array of predicted labels
    '''

    w = v[0:-1]
    b = v[-1]

    predicted = np.dot(np.transpose(w), DTE) + b # Equivalent to likelihood
    print(np.transpose(w).shape)

    for i in range (predicted.shape[0]):
        if (predicted[i] > 0):
            predicted[i] = 1
        else:
            predicted[i] = 0

    if(printStats == True):

        print("======================== PREDICTION VERBOSE MODE ===============================")
        print("Obtained prediction array:\n")
        print(predicted)
        print("Percentage of correctly assigned classes:\n")
        correct = 0
        for i in range (predicted.shape[0]):
            if (predicted[i] == LTE[i]):
                correct +=1
        print( 100*(correct/LTE.shape[0]), "%" )



    return predicted

def multiclass_log_reg(v, DTR, LTR, l):
    '''
    ## Params:
    - v = numpy array of 1 dim but is reshaped here to ( (D+1), K ) where D is the number of attributes for each class (dimensionality of each DTR.shape[0]). K is the number of classes. Bias is the 
    last row of v (reshaped).
    - DTR = Training data.
    - LTR = Training labels.
    - l = lambda (Multiplier of w).

    ## Assumptions:
    - The classes in LTR are numbers starting from 0 to K-1 class. So in order to pick the number of classes, it is enough to pick the maximum value + 1.
    - LTR is only one dim, each value in position i represents the class of the sample i.
    '''

    # Retrieving n (number of samples)
    n = DTR.shape[1]
    # Retrieving k (number of classes)
    k = np.max(LTR)+1
    # Retrieving D (number of attributes)
    D = DTR.shape[0]
    # Reshaping v
    v = np.reshape(v, ((D + 1), k))
    # Retrieving the weights and biases
    w = v[0:-1, :]
    b = v[-1, :]

    term1 = l/2*np.linalg.norm(w)**2

    # Matrix of scores S:
    S = np.dot(np.transpose(w), DTR) + np.reshape(b, (k,1)) # Needs reshapeing because b is a row. Since we need to subtract from each class and the classes are the rows,
                                                            # it is needed to transfrom b into a column vector. (each line is a class)

    logSumExp = np.log( np.sum(np.exp(S), 0) ) # The ideia is to sum the calculation of all classes (for each sample), which means adding in axis 0.
    Ylog = S - logSumExp # Shape is k x i

    # Creating matrix of labels Tki = 1 --> ci = k
    matrixT = np.zeros((k, n))
    for i in range(n):
        classNumber = LTR[i]
        matrixT[classNumber][i] = 1

    term2 = -1/n*np.sum(np.multiply(Ylog, matrixT))

    return term1+term2


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

# =========================================================================================

def main():
    print("Which part of the lab? 1 or 2? (1 is binary and 2 is multiclass)\n")
    selection = int(input())

    if (selection == 1):
        #print(findMinimum())
        D, L = load_iris_binary()
        (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)    
        
        # there are 66 samples for training. 
        print("Training data has shape:", DTR.shape, "\n")
        print("Training label has shape:", LTR.shape, "\n")
        
        startPoint = np.zeros(DTR.shape[0] + 1)
        l = 10**(-6)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, startPoint, iprint = 1, args=(DTR, LTR, l), approx_grad=True)
        print("Estimated position of the minimum:\n")
        print(f)

        predicted = posteriorLikelihood(x, DTE, True, LTE)
    
    elif(selection == 2):
        D, L = load_iris()
        (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

        # Retrieving number of features on each sample
        D = DTR.shape[0]
        # Retrieving number of classes
        k = np.max(LTR)+1

        startPoint = np.zeros((D+1)*k)
        l = 10**(-6)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(multiclass_log_reg, startPoint, iprint = 1, args=(DTR, LTR, l), approx_grad=True)
        print("Estimated position of the minimum:\n")
        print(x)





if __name__ == '__main__':
    main()