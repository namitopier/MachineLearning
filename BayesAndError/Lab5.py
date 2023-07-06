from cgi import test
from unittest import loader
import numpy as np
from Lab2 import load
from lab3 import empirical_cov, empirical_mean # La media se calcula como vet.mean(eje)
from Lab4 import loglikelihood, vrow, logpdf_GAU_ND
import scipy.special as sp

D, L = load('iris.csv')

def calc_accuarcy(data, pred):
    bools = (data==pred)
    acc = bools.sum()/data.size
    return acc

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0) # Número de datos de entrenamiento. 2/3 para entrenar y 1/3 para testear
    np.random.seed(seed) # Se establece un valor inicial para reordenar los datos
    # Se genera un vector con D.shape[1] datos espaciados de igual manera, tipo di D.shape[1] es 3, el
    # vector es [0,1,2]. Luego este vector se permuta, lo que cambia de manera random la posición
    # de estos valores
    idx = np.random.permutation(D.shape[1]) 
    idxTrain = idx[0:nTrain] # Se separan las posiciónes de las muestras a utilizar para entrenar el modelo
    idxTest = idx[nTrain:] # Se cogen las posiciones de los datos para probar el modelo
    DTR = D[:, idxTrain] # Se cogen los datos de entrenamiento
    DTE = D[:, idxTest] # Se cogen los datos de prueba
    LTR = L[idxTrain] # Se cogen las etiquetes de entrenamiento
    LTE = L[idxTest] # Se cogen las etiquetas de testeo
    return (DTR, LTR), (DTE, LTE)

# DTR and LTR are training data and labels, DTE and LTE are evaluation data and labels
# (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

# Multivariate Gaussian Classifer <-------------------------------------------------------------------------

train, test = split_db_2to1(D,L) # Separo los datos a utilizar y encuentro su media y covarianza

def calc_likehoods(data, mu, cov): # Function to calculate the likehoood of every sample given some mu and cov
    mArr = np.array([]) # Mock array to fill with ll of each class
    ll = logpdf_GAU_ND(data, mu, cov)
    mArr = np.append(mArr, ll)
    return mArr

def log_MVG_Classifier(testData, trainData): # Multi Variate Classifier using the logarithms
    '''
    Parameters: \n
    \t -testData: Both parameters and labels of the evaluation data (DTE,LTE)\n
    \t -trainData: Both parameters and labels of the training data ((DTR,LTR))
    Returns: A matrix with the predictions for the evaluation data.
    '''
    # Here I iterate over the different classes of the training data to get the likehoods of each class
    S = np.zeros(testData[0].shape[1]) # Matrix to keep the loglikehoods. Started like this to keep shape
    for j in range(np.unique(testData[1]).size):
        mu = trainData[0][:,trainData[1]==j].mean(1)
        mu = mu.reshape((mu.size, 1))
        cov = empirical_cov(trainData[0][:,trainData[1]==j])
        ll = calc_likehoods(testData[0], mu, cov)
        S = np.vstack((S, ll)) # Contains the ll for each sample in each class
    S = np.delete(S, 0, 0) # pop the zeros
    Pc = 1/3 # Represents the probability of the class being c. It was given and its the same for all classes here.
# Now we calculate the class posterior probability as logSJoint/SMarginal. This represents the probability
# that an observation is part of a class given some attributes know a priori.
    logSJoint = S + np.log(Pc) # Creo la matriz de joint densities multiplicando la S por una prior probability DADA
    logSMarginal = vrow(sp.logsumexp(logSJoint, axis=0)) # Its the probability of a sample having its current attriutes

    logSPost = logSJoint-logSMarginal # Class posterior probability
    # Finally we got a matrix with the class probability (row) for each sample (column), 
    # for each column we have to select which row is the one with the highest value. Like this:
    logSPost = np.argmax(logSPost, axis=0); #argmax returns the index of the greatest value in a given axis
    return logSPost

# I calculate the accuarcy to evaluate the result
# prediction = log_MVG_Classifier(test, train)
# goodPred = (prediction==test[1]).sum()
# acc = goodPred/test[1].size
# print(acc)

#Naive Bayes Gaussean Classifier <---------------------------------------------------------------------------------------

def log_NVG_Classifier(testData, trainData):
    S = np.zeros(testData[0].shape[1]) 
    for j in range(np.unique(testData[1]).size):
        mu = trainData[0][:,trainData[1]==j].mean(1)
        mu = mu.reshape((mu.size, 1))
        cov = empirical_cov(trainData[0][:,trainData[1]==j])
        # Thid is the difference with the MVG one: That the covariance is just a diagonal
        idCov = np.identity(cov.shape[0])
        cov = cov*idCov
        ll = calc_likehoods(testData[0], mu, cov)
        S = np.vstack((S, ll)) 
    S = np.delete(S, 0, 0) 
    Pc = 1/3 
    logSJoint = S + np.log(Pc) 
    logSMarginal = vrow(sp.logsumexp(logSJoint, axis=0)) 

    logSPost = logSJoint-logSMarginal
    logSPost = np.argmax(logSPost, axis=0);
    return logSPost

# prediction = log_NVG_Classifier(test, train)
# goodPred = (prediction==test[1]).sum()
# acc = goodPred/test[1].size
# print(acc)

# Tied Covariance Gaussian Classifier <----------------------------------------------------------------------------------------

def tied_covariance(testData, trainData):
    cov = np.zeros((testData[0].shape[0], testData[0].shape[0])) # Square matrix of zeros to add the other covariances.
    # The shape is given by the number of attributes present.
    for j in range(np.unique(testData[1]).size):
        hold = empirical_cov(trainData[0][:,trainData[1]==j])
        hold = hold * np.unique(testData[1]).size # MUltiply by the number of classes, as per the formula
        cov = np.add(cov, hold) # Add the class covariance to the result
    cov = cov*(1/trainData[0].shape[1]) # Divide by the number of samples
    return cov

def log_TCG_Classifier(testData, trainData):
    S = np.zeros(testData[0].shape[1]) 
    # Thid is the difference with the MVG one: 
    cov = tied_covariance(testData, trainData)
    for j in range(np.unique(testData[1]).size):
        mu = trainData[0][:,trainData[1]==j].mean(1)
        mu = mu.reshape((mu.size, 1))
        ll = calc_likehoods(testData[0], mu, cov)
        S = np.vstack((S, ll)) 
    S = np.delete(S, 0, 0) 
    Pc = 1/3 
    logSJoint = S + np.log(Pc) 
    logSMarginal = vrow(sp.logsumexp(logSJoint, axis=0)) 

    logSPost = logSJoint-logSMarginal
    logSPost = np.argmax(logSPost, axis=0);
    return logSPost

if __name__ == '__main__':

    prediction = log_TCG_Classifier(test, train)
    goodPred = (prediction==test[1]).sum()
    acc = goodPred/test[1].size

    # load = np.load("SolutionsLab5/SJoint_MVG.npy")
    # print(S.shape)
    # print(np.abs(load - logSJoint).max())