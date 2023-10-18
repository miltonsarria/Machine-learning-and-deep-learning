#Milton Orlando Sarria
#USC
#Machine Learning 
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from gen_data import createData3D

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
    # Aplicar los vectores propios a los datos (rotar)
    newX = newX.dot(eigVecs)
    # Re-escalar los datos
    newX = newX / np.sqrt(eigVals + 1e-5)
    return  newX,  Mu,  eigVecs    

#################usar funciones###############
#################crear datos artificiales en 3D ###############
X,Y=createData3D()
#mostrar datos
fig = plt.figure(1)

ax = plt.axes(projection ="3d")

labels=np.unique(Y)
plt.figure(1)
color='rbk'
for ii in labels: 
        ax.scatter3D(X[Y==ii,0],X[Y==ii,1],X[Y==ii,2],c=color[ii], marker='o')

#aplicar whiten

Xw,Mu,eigVecs=whiten(X) 

#graficar datos para comparar el resultado
#


fig=plt.figure(2)
ax = plt.axes(projection ="3d")

for ii in labels: 
        ax.scatter3D(Xw[Y==ii,0],Xw[Y==ii,1],Xw[Y==ii,2],c=color[ii], marker='o')
        

plt.show()
    
