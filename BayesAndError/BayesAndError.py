from __future__ import division
from xml.etree.ElementPath import prepare_predicate
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from Lab5 import log_MVG_Classifier, split_db_2to1, tied_covariance
import sklearn.datasets
import matplotlib.pyplot as plt

'''
Lab 8
General notes:
When talking about prior probability (pi), we are talking about: P(ck|xi) (probability of sample belonging to class ck given its attributes).
I.e., if pi is too high, it is going to be predicted as class 1 more often.
'''

def MVG_confusion_matrix(prediction, correct):
    '''
    ## Params:
    - prediction = List of predicted classes (assumed to be an 1 dim np array)
    - correct = List of correct classes

    ## Returns:
    Nothing, just prints the confusion matrix
    '''
    # Retrieving the number of classes (assuming the classes are labeled starting from class 0)
    classesNumber = np.max(correct) + 1
    # Creating empty confusion matrix
    confMatrix = np.zeros((classesNumber, classesNumber))

    for i in range (prediction.shape[0]):
        # print(prediction.shape)
        # print(correct.shape)
        confMatrix[prediction[i]][correct[i]] += 1
    #print(confMatrix)
    return confMatrix

def binary_optimal_bayes_decision(costMatrixAttr, priorClassProb, llrs):
    '''
    ## Params:
    - costMatrixAttr = triple (π1,Cfn,Cfp) used to contruct matrix:\n
    |0   Cfn|\n
    |Cfp   0|\n
    - priorClassProb = (1 - π1,π1)
    - llr = log likilihood ratios

    ## Returns:
    The predicted array of class
    '''

    pi1 = costMatrixAttr[0]
    Cfn = costMatrixAttr[1]
    Cfp = costMatrixAttr[2]


    # Calculating the threshold (used to decide wether it is class 1 or 0)
    thresh = -np.log( (pi1*Cfn)/(priorClassProb[0]*Cfp) )
    predicted = np.where(llrs > thresh, 1, 0)

    return predicted

def calculateRoc(resolution, llr, labels):
    
    '''
    ## Explanation:
    The ideia is to get a list of thresholds and plotting a curve FPR x TPR (where TPR = 1 - FNR)

    ## Params:
    - resolution = number of steps to calculate threshold (the resolution of the graph)
    - llr = log likilihood ratios
    - labels = correct list of labels 
    '''
    FPRvalues = []
    TPRvalues = []

    maxThresh = np.max(llr)
    minThresh = np.min(llr)

    for t in range (1, resolution, 1):

        thresh = t*( (maxThresh-minThresh)/resolution )
        thresh += minThresh
        predicted = np.where(llr > thresh, 1, 0)
        confMatrix = MVG_confusion_matrix(predicted, labels)

        # Retrieve number of false negatives and false positives
        FN = confMatrix[0][1]
        FP = confMatrix[1][0]
        # print("FP = ", FP)
        # print("FN = ", FN)

        # Retrieving the true positive and true negative values:
        TP = confMatrix[1][1]
        TN = confMatrix[0][0]
        # print("TN = ", TN)

        FNR = FN/(FN + TP)
        FPR = FP/(FP + TN)
        TPR = 1 - FNR

        FPRvalues.append(FPR)
        TPRvalues.append(FNR)

    #print("FPR values: ", FPRvalues)

    plt.figure()
    plt.plot(FPRvalues, TPRvalues)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.show()

def bayes_risk(confMatrix, costMatrixAttr, normalized = False, minimum = False, scores = None, labels = None, threshDivision = None):
    '''
    ## Quick explanation:
    Evaluation of predictions can be done using empirical Bayes risk (or detection cost function, DCF), that represents the
    cost that we pay due to our decisions c for the test data.

    ## Params:
    - confMatrix = confusion matrix for given attributes on costMatrixAttr
    - costMatrixAttr = triple (π1,Cfn,Cfp)
    - normalized = if want to normalize the errors so to get a more accurate value
    - minimum = decides if want to compute the minimum (for this, want also the divisions for threshold variation and the llr)
    - scores = What will be used to set the threshold. If it is a gaussian, use log likilihood ratios. If it is SVM or GMM, use the scores itself. Needed for the calculation of minimum.
    - threshDivision = how many tests for different threshold.

    ## Returns:
    The result of bayes risk
    '''

    pi = costMatrixAttr[0]
    Cfn = costMatrixAttr[1]
    Cfp = costMatrixAttr[2]

    # scores = np.insert(scores, 0, -np.inf)
    # scores = np.insert(scores, len(scores), np.inf)

    if (minimum and normalized):

        DCFValues = []
        # maxThresh = np.max(llr)
        # minThresh = np.min(llr)

        for t in range (len(scores)):

            # thresh = t*( (maxThresh-minThresh)/threshDivision )
            # thresh += minThresh
            # predicted = np.where(llr > thresh, 1, 0)
            # confMatrix = MVG_confusion_matrix(predicted, labels)
            thresh = scores[t]
            predicted = np.where(scores > thresh, 1, 0)
            confMatrix = MVG_confusion_matrix(predicted, labels)

            # Retrieve number of false negatives and false positives
            FN = confMatrix[0][1]
            FP = confMatrix[1][0]

            # Retrieving the true positive and true negative values:
            TP = confMatrix[1][1]
            TN = confMatrix[0][0]

            FNR = FN/(FN + TP)
            FPR = FP/(FP + TN)

            DCF = pi*Cfn*FNR + (1-pi)*Cfp*FPR
            DCFValues.append(DCF/np.min([pi*Cfn, (1-pi)*Cfp]))
        
        return np.min(DCFValues)

    elif(minimum and normalized == False):

        DCFValues = []
        maxThresh = np.max(llr)
        minThresh = np.min(llr)

        for t in range (1, threshDivision, 1):

            thresh = t*( (maxThresh-minThresh)/threshDivision )
            thresh += minThresh
            predicted = np.where(llr > thresh, 1, 0)
            confMatrix = MVG_confusion_matrix(predicted, labels)

            # Retrieve number of false negatives and false positives
            FN = confMatrix[0][1]
            FP = confMatrix[1][0]

            # Retrieving the true positive and true negative values:
            TP = confMatrix[1][1]
            TN = confMatrix[0][0]

            FNR = FN/(FN + TP)
            FPR = FP/(FP + TN)

            DCF = pi*Cfn*FNR + (1-pi)*Cfp*FPR
            DCFValues.append(DCF)
        
        return np.min(DCFValues)
    
    elif(normalized):

        # Retrieve number of false negatives and false positives
        FN = confMatrix[0][1]
        FP = confMatrix[1][0]

        # Retrieving the true positive and true negative values:
        TP = confMatrix[1][1]
        TN = confMatrix[0][0]

        FNR = FN/(FN + TP)
        FPR = FP/(FP + TN)

        # Doing the actual computation of bayes risk:
        DCF = pi*Cfn*FNR + (1-pi)*Cfp*FPR
        return DCF/np.min([pi*Cfn, (1-pi)*Cfp])

    elif(not normalized):

        pi = costMatrixAttr[0]
        Cfn = costMatrixAttr[1]
        Cfp = costMatrixAttr[2]

        # Retrieve number of false negatives and false positives
        FN = confMatrix[0][1]
        FP = confMatrix[1][0]

        # Retrieving the true positive and true negative values:
        TP = confMatrix[1][1]
        TN = confMatrix[0][0]

        FNR = FN/(FN + TP)
        FPR = FP/(FP + TN)

        # Doing the actual computation of bayes risk:
        DCF = pi*Cfn*FNR + (1-pi)*Cfp*FPR
        return DCF

def bayes_error_plot(llr, labels, pRange = (-3, 3, 21)):
    '''
    ## Ideia:
    For each p, will calculate the normalized DCF using pi = 1/( 1+e^(-p) ) by using the bayes decision for (pi, 1, 1)
    ## Params:
    - pRange = (minRange, maxRAnge, step)
    '''

    effPriorLogOdds = np.linspace(pRange[0], pRange[1], pRange[2])
    pValues = []
    DCFValues = []
    minDCFValues = []

    for i in range(effPriorLogOdds.shape[0]):

        # Retrieving p:
        p = effPriorLogOdds[i]
        pValues.append(p)

        # Calculating effective prior pi:
        pi = 1/( 1 + np.exp(-p) )

        # Calculating the prediction for pi and its confusion matrix:
        predicted = binary_optimal_bayes_decision((pi, 1, 1), (1-pi, pi), llr)
        confMatrix = MVG_confusion_matrix(predicted, labels)

        # Retrieving normalized DCF:
        DCF = bayes_risk(confMatrix, (pi, 1, 1), True)

        # Retrieving minimum normalized DCF:
        minDCF = bayes_risk(confMatrix, (pi, 1, 1), True, True, llr, labels, 100)

        DCFValues.append(DCF)
        minDCFValues.append(minDCF)

    plt.figure()
    plt.plot(effPriorLogOdds, DCFValues, label="DCF", color="r")
    plt.plot(effPriorLogOdds, minDCFValues, label="min DCF", color="b")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()
    plt.close()
        

# =============================== Provided functions ===============================
# def split_db_2to1(D, L, seed=0):
#     nTrain = int(D.shape[1]*2.0/3.0)
#     np.random.seed(seed)
#     idx = np.random.permutation(D.shape[1])
#     idxTrain = idx[0:nTrain]
#     idxTest = idx[nTrain:]
#     DTR = D[:, idxTrain]
#     DTE = D[:, idxTest]
#     LTR = L[idxTrain]
#     LTE = L[idxTest]
#     return (DTR, LTR), (DTE, LTE)

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

# ==================================================================================

def main():

    choice1 = int(input("Select which part of the lab: 1 for confusion matrix, 2 for decision, 3 for evaluation: "))
    if (choice1 == 1):
        print("========================= SELECTED CONFUSION MATRIX =========================")
        choice2 = int(input("Type 1 for MVG classifier, 2 for tied covariance: "))
        if (choice2 == 1):
            D, L = load_iris()
            (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
            prediction = log_MVG_Classifier((DTE, LTE), (DTR, LTR))
            print(MVG_confusion_matrix(prediction, LTE))

        elif(choice2 == 2):
            D, L = load_iris()
            (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
            prediction = tied_covariance((DTE, LTE), (DTR, LTR))
            print(prediction)
            #MVG_confusion_matrix(prediction, LTE)
        else:
            print("What?? It was written 1 or 2, not other number dumbass\n")
    elif(choice1 == 2):
        print("========================= SELECTED DECISION =========================")
        choice2 = int(input("Type 1 for optimal binary bayes decision, 2 for bayes error plot: "))
        if (choice2 == 1):
            '''
            Here the ideia is to understand what to do if the cost of predicting a class i as being of j is not uniform.
            I.e., the confusion matrix does not get added by 1 when a prediction is computed in it.
            '''

            llr = np.load("commedia_llr_infpar.npy")
            labels = np.load("commedia_labels_infpar.npy")

            pi1 = 0.8
            Cfn = 1
            Cfp = 1
            predicted = binary_optimal_bayes_decision((pi1, Cfn, Cfp), (1-pi1, pi1), llr)
            confMatrix = MVG_confusion_matrix(predicted, labels)
            print("Using parameters: pi = %f, Cfn = %i, Cfp = %i" %(pi1, Cfn, Cfp))
            print("Computed bayes risk: ", bayes_risk(confMatrix, (pi1, Cfn, Cfp), True))

            calculateRoc(10000, llr, labels)

        if (choice2 == 2):

            llr = np.load("commedia_llr_infpar.npy")
            labels = np.load("commedia_labels_infpar.npy")

            pi1 = 0.8
            Cfn = 1
            Cfp = 1

            bayes_error_plot(llr, labels)


if __name__ == '__main__':
    main()