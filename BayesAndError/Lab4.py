import numpy as np
import scipy.linalg as sc
from Lab2 import load, plot_scatter
from lab3 import empirical_cov
import matplotlib.pyplot as plt
import math

def vrow(mat):
    return mat.reshape((1,mat.size))

D, L = load('iris.csv') 

# Multivariate Gaussian density
# This function is used to calculate de Multivariate Gaussian density (Logarítmica) that 
# basically represents a N dimensional normal distribution. It works by passing a 1D array
# a mean of shape (M,1) and a covariance matrix of chape (M, M). M is the number of attributes 
# of the data.

def logpdf_GAU_ND(x, mu=None, cov=None):
    M = x.shape[0]
    _, logDet = np.linalg.slogdet(cov)
    invCov = np.linalg.inv(cov)
    pi = math.pi
    mul = np.dot((x-mu).T, invCov)
    mul = np.dot(mul, (x-mu))
    val = (-0.5*M*math.log(2*pi))+(-0.5*logDet)+(-0.5*mul)
    return np.diagonal(val) # Why the diagonal? Sergio knows

# Maximum Likelihood Estimate
# This is a method to estimate the value of some parameters for a given probability on some data

def loglikelihood(XND, m_ML, C_ML):
    ll = logpdf_GAU_ND(vrow(XND), m_ML, C_ML)
    return np.sum(ll)

# plt.figure()
# XPlot = np.linspace(-8, 12, 1000)
# m = np.ones((1,1)) * 1.0
# C = np.ones((1,1)) * 2.0
# plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
# plt.show()

# pdfSol = np.load('Solutions/llGAU.npy')
# pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
# print(np.abs(pdfSol - pdfGau).max())

# X1D = np.load("Solutions/X1D.npy")
# m_ML = X1D.mean(1)
# C_ML = empirical_cov(X1D)

# ll = loglikelihood(X1D, m_ML, C_ML)
# print(ll)


