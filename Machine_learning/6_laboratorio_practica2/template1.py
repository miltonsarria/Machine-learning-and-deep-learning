#Milton Orlando Sarria
#USC
#Machine Learning 

import numpy as np
import matplotlib.pyplot as plt


#### definir funciones
def center(X):
    Mu =np.mean(X, axis = 0)
    newX = X - Mu
    return newX, Mu

def standardize(X):
    sigma = np.std(X, axis = 0)
    newX = center(X)/sigma
    return newX, sigma
    
def whiten(X):
    newX, Mu = center(X)
    cov= np.cov(newX, rowvar=False, bias=True)
    # Calcular los valores y vectores propios
    eigVals, eigVecs = np.linalg.eig(cov)
    eigVals=np.real(eigVals)
    eigVecs=np.real(eigVecs)
    # Aplicar los vectores propios a los datos (rotar)
    newX = newX.dot(eigVecs)
    # Re-escalar los datos
    newX = newX / np.sqrt(eigVals + 1e-5)
    return  newX,  Mu,  eigVecs    

#################usar funciones###############
#################crear datos artificiales###############
np.random.seed(1235) 
x1 = np.random.normal(2, 1, 300) 
x2 = np.random.normal(1, 3, 300) 
X = np.array([x1, x2]).T 
print( X.shape )

#aplicar whiten

Xw,Mu,eigVecs=whiten(X) 

#graficar datos para comparar el resultado
#en rojo se muestran los datos originales, en azul el resultado


plt.figure(1)
plt.plot(X[:,0],X[:,1],'or')
plt.axis([-10,10,-10,10]) 
plt.figure(2)
plt.plot(Xw[:,0],Xw[:,1],'o')
plt.axis([-10,10,-10,10])
plt.show()
    
