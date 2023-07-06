'''
Iris dataset:
...
5.1,3.5,1.4,0.2,Iris-setosa
...

Have to write a load function to put data (csv) into a 4x150 dim array, each row corresponding to a diff attribute.

sepal length | 5.1 | 4 | ... |
sepal width  | 4.2 | 6 | ... |
petal length | 6.7 | 2 | ... |
petal width  | 2.1 | 6 | ... |


And a 1x150 dim array containing all the class labels:
 [[iris setosa = 0, iris versicolor = 1, iris virginica = 2]]
'''
import sys
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
        print("No such file exists")

def plotData(dataMatrix, classLabels):

    setosa = []
    versicolor = []
    virginica = []

    dataMatrix = np.transpose(dataMatrix)

    for i in range (len(classLabels)):
        irisType = classLabels[i]
        if (irisType == 0):
            setosa.append(dataMatrix[i])
        elif(irisType == 1):
            versicolor.append(dataMatrix[i])
        else:
            virginica.append(dataMatrix[i])

    
    setosa = np.transpose(setosa)
    versicolor = np.transpose(versicolor)
    virginica = np.transpose(virginica)

    #print(setosa)


    plt.figure()
    plt.hist(setosa[0], density=True)
    plt.hist(versicolor[0], density=True)
    plt.hist(virginica[0], density=True)
    plt.title("Sepal length")
    plt.show()


def main():
    
    datasetName = sys.argv[1]
    dataMatrix, classLabels = loadDataset(datasetName)

    plotData(dataMatrix, classLabels)

    # listaTeste = [[1,2,3,4,5,6], [1,8,9,10,11,12]]
    # listaTeste = np.transpose(listaTeste)
    # #listaTeste = np.reshape(listaTeste, (4, 3))
    # print(listaTeste[0])

if __name__ == '__main__':
    main()