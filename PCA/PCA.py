import numpy as np
import matplotlib.pyplot as plt

def loadDataset(datasetName):

    dataList = [[],[],[],[]]
    classList = []
    try:
        f = open(datasetName, 'r')

        for line in f:

            cleanedLine = line.strip()
            cleanedLine = cleanedLine.split(",")
            dataList[0].append(cleanedLine[0]) # Creating data list (without proper format yet)
            dataList[1].append(cleanedLine[1])
            dataList[2].append(cleanedLine[2])
            dataList[3].append(cleanedLine[3])

            if (cleanedLine[-1] == "Iris-setosa"):
                classList.append(0)
            elif (cleanedLine[-1] == "Iris-versicolor"):
                classList.append(1)
            else:
                classList.append(2)

        return (np.asarray(dataList).astype('float32'), np.asarray(classList))
        
    
    except:
        print("***********************************************************************************")
        print("File at path ", datasetName, "does not exist. Are you sure it is the correct name??")
        print("***********************************************************************************")

def vcol(vlist):

    '''
    Will transform a 1D list into a column vector (rows = len(vlist) // columns = 1)
     1d list = [1,2,3] --vcol--> 
     
     [[1],
     [2],
     [3]]
    '''
    return np.reshape(vlist, (len(vlist), 1))

def vrow(vlist):
    '''
     Will transform a 1D list into a row vector (rows = 1 // columns = len(vlist))
     1d list = [1,2,3] --vrow--> [[1,2,3]]
    '''
    return np.reshape(vlist, (1, len(vlist)))

def PCA(D, m, verif = False):
    '''
    PCA = Principal Component Analysis. 
    D is the data matrix where columns are the different samples and lines are the attributes of each sample.
    D.shape = MxN
    '''
    # First step: Calculate mu as the mean of each attribute between all samples.
    mu = D.mean(1)
    mu = vcol(mu)

    # Now it is needed to center the data, i.e., subtract mu (the mean) from all columns of D.
    DC = D - mu

    # Now, it is needed to calculate the covariance matrix C = 1/N * Dc*Dc.T
    N = D.shape[1]
    C = (1/N)*np.dot(DC, np.transpose(DC))

    # Next, we have to compute eigenvectors and eigenvalues:
    sortedEigenValues, eigenVectors = np.linalg.eigh(C) 

    # Note: they are sorted from smallest to largest. We need the opposite:
    # Need to pick the m largest eigenVectors "P" to project the samples into m dimensions:
    # P = eigenVectors[:, 0:m]
    P = eigenVectors[:, ::-1][:, 0:m]
    #M = D.shape[0]
    #P = eigenVectors[:, 0:M]

    if (verif == True):
        PCorrect = np.load('IRIS_PCA_matrix_m4.npy')
        print("Correct matrix of eigenvectors: \n", PCorrect)
        print("Obtained matrix:\n", P)

    # Finally, it is needed to apply the projection to a matrix of samples, in this case, "D":
    DP = np.dot(np.transpose(P), D)

    return DP



def main():

    dataMatrix, classList = loadDataset("iris.csv")
    DP = PCA(dataMatrix, 2, False)
    print(DP.shape)

    plt.figure()
    plt.scatter(DP[0, :], DP[1, :], label = "teste")
    #plt.scatter(DP[2, :], DP[3, :], label = "teste")
    #plt.scatter(DP[2, :], DP[1, :], label = "teste")
    plt.show()

if __name__ == '__main__':
    main()