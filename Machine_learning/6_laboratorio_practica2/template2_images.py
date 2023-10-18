from process import *
import matplotlib.pyplot as plt
import numpy as np


#load data
X=np.loadtxt('cerrada.txt',delimiter=',')
print(X.shape)

X=X/255

#definir el algoritmo para procesar imagenes
#normalizer = whiten()
normalizer = zca()

#entrenar el algoritmo de normalizar
normalizer.fit(X)

#aplicar sobre los datos
X = normalizer.transform(X)

#mostar el resultado
dim1=32
dim2=32     
image = X[0].reshape((dim1,dim2))
plt.imshow(image,cmap="gray");
plt.show()

