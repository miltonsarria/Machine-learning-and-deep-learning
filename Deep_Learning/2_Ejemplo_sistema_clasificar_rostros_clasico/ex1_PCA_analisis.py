'''
Milton Orlando Sarria
USC
Ejemplo que usa PCA - analisis de componentes principales
para reducir el numero de variables de las imagenes y mostrar
las nubes de puntos de cada clase
se separa en conjunto de prueba y conjunto de entrenamiento
PCA se estima solamente con el conjunto de entrenamiento
se plica PCA tanto al conjunto de entrenamiento como prueba
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from glob import glob 
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.model_selection import train_test_split
device = 'cpu'


## leer todos los archivos
root = 'clases'
file_names = glob(root+"/**/*.png", recursive=True)
clases = glob(root+"/*")
print(clases)
clases = clases=np.array([clase.split('\\')[1] for clase in clases])
print(clases)
#leer las imagenes
frames = []
labels = []
dic_clases = {}
for file_name in tqdm(file_names):  
    frame = Image.open(file_name)
    frame = np.array(frame)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    frame = np.array(frame).ravel()
    clase = file_name.split('\\')[1]
    label = np.where(clases==clase)[0][0]
    labels.append(label)
    frames.append(frame)
    dic_clases[label]=clase
#convertir a array
names = np.array(labels)
print('Done loading')   
ACC = [] 
X=np.vstack(frames)/255
print(X.shape)
print(names.shape)

x_train, x_test, y_train, y_test = train_test_split(X,names,test_size = 0.3)
#rotacion
pca = PCA(n_components=2)
pca.fit(x_train)

x_train=pca.transform(x_train)
x_test=pca.transform(x_test)

color='rbk'
for i in names:
        key = 'o'+color[i]  
        plt.plot(x_train[y_train==i,0],x_train[y_train==i,1],key)
        
        key = '+'+color[i]  
        plt.plot(x_test[y_test==i,0],x_test[y_test==i,1],key)
        
        
plt.show()        
    
    

