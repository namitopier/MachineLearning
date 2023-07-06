import numpy as np
import scipy.linalg as sc
from Lab2 import load, plot_scatter

# Traigo los datos y las labels de las tablas dadas
D, L = load('iris.csv') 

# Calculo la media utilizando una fn, que es más rápido que usando un ciclo
# ya que estas fn estpan implementadas por np en C.
# Recordando que los datos para cada registro están ordenados como columnas (4 por registro),
# debemos especificar el eje sobre el cual calcular las medias, en este caso se hace por columna.z
mu = D.mean(1) 
# Cambio el eje del vector de medias, indicando (filas, columnas)
mu = mu.reshape((mu.size, 1))
# Procedemos a centrar los datos i.e. A restarles la diferencia con su media:
Dc = D - mu
# Primero hacemos un producto punto de Dc con su transpuesta.
C = np.dot(Dc,Dc.T)
# Luego dividimos por el número de columnas de la matriz original y obtenemos la matriz de covarianza.
C = C/D.shape[1]
# Regresa los eigen-valores ordenados de mayor a menor y los eigen-vectores, igualmente
# ordenados pero en la matriz U como sus columnas.
s, U = np.linalg.eigh(C)
# U representa los vectores de cada dimensión de los datos
# P entonces es el corte con número total de dimensiones que quiero, en este caso 2 para poderlo plotear
P = U[:,0:2]
# Formula dada: Matriz transpuesta de las direcciones de un plano por los puntos a proyectar.
DP = np.dot(U.T, D)

# Plot
# plot_scatter(DP,L)

# LDA: Es lo mismo que la PCA, hace una proyección, pero, esta mantiene la dirección mejor
# Inicializo las matrices
SW = 0
SB = 0
# Hago un ciclo que recorre las distintas clases que hay
for i in set(list(L)):
# Tomo todos los valores por clase
    Di = D[:,L==i]
# Separo la media de la clase y la vuelvo una columna
    mui = Di.mean(1)
    mui = mui.reshape((mui.size,1))
# Centro los valores con respecto a la media
    Dci = Di - mui
# Calculo SW como la formula dada
    SW = SW + np.dot(Dci,Dci.T)
# Para encontrar SB necesitamos la diferencia de la media de clase con la media global
    dmui = mui-mu
# Calculo SB como en la formula
    SB = SB + Di.shape[1]*np.dot(dmui,dmui.T)
# Divido por el número de registros como en la formula
# SW y SB son las matrices de covarianza de los datos, que respectivamente representan
# la covarianza interna de la clase y la covarianza entre clases.
SW = SW/D.shape[1]
SB = SB/D.shape[1]
# Encontramos s y U ocon
s, U = sc.eigh(SB, SW)
W = U[:,::-1][:,0:2]
DP = np.dot(W.T, D)
# Plot
# plot_scatter(DP,L)

def empirical_mean(data):
    mu = np.array([0]) # Start array like this to be constant in the shape
    for row in data:
        mup = np.sum(row)/float(data.shape[1]) #Calculate mean of a parameter 
        mu = np.vstack((mu, mup))
    mu = np.delete(mu, 0, 0) #Delete the zeros
    return mu

def empirical_cov(data): #Usá esta para la covarianza <--------------------------
    mu = data.mean(1)
    mu = mu.reshape((mu.shape[0], 1))
    Dc = data - mu
    cov = np.dot(Dc,Dc.T)
    cov = cov/float(data.shape[1])
    return cov



