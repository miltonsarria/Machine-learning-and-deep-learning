'''
Milton Orlando Sarria
USC
Ejemplo para entrenar un sistema base empleando regresion logistica
se hace uso de las imagenes que han sido recolectadas previamente
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
root = 'clases_'
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
    #frame = frame.resize((100,100))   
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


x_train, x_test, y_train, y_test = train_test_split(X,names,test_size = 0.3)
#n_components puede ser un valor entero o una porcion de 0 a 1.
#si se quiere visualizar debe ser n_components=2
pca = PCA(n_components=0.99)
pca.fit(x_train)

x_train=pca.transform(x_train)
x_test=pca.transform(x_test)

print(f"[INFO]: train set: {x_train.shape}, test set: {x_test.shape}")

print("[INFO] entrenando....")
#entrenar un clasificador simple
clf = LogisticRegression(solver='lbfgs', max_iter=3)
clf.fit(x_train,y_train)
print("[INFO] evaluando....")
y = clf.predict(x_test)
acc =(y==y_test).sum()/y.size*100
print('[INFO] Porcentaje de prediccion : ',acc)

'''
#solo se puede mostrar si se usan n_components=2
theta0=clf.intercept_ 
theta =clf.coef_

xx = np.linspace(-10, 15, 100)    
yy=(-theta[0][0]*xx-theta0)/theta[0][1]
plt.plot(xx,yy,'k',linewidth=1.5)

color='rbk'
for i in names:
        key = 'o'+color[i]  
        plt.plot(x_train[y_train==i,0],x_train[y_train==i,1],key)
        
        key = '+'+color[i]  
        plt.plot(x_test[y_test==i,0],x_test[y_test==i,1],key)
        
        
plt.show()        
'''