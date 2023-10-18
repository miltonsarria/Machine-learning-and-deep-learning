import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import cv2
from glob import glob 
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.model_selection import train_test_split



## leer todos los archivos y armar la base de datos
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
    #frame = frame.resize((100,100))   
    frame = np.array(frame).ravel()
    clase = file_name.split('\\')[1]
    #clase = '-'.join(clase)
    label = np.where(clases==clase)[0][0]
    labels.append(label)
    frames.append(frame)
    dic_clases[label]=clase
#convertir a array
names = np.array(labels)
print('Done loading')   
X=np.vstack(frames)/255

x_train, x_test, y_train, y_test = train_test_split(X,names,test_size = 0.3)


# Build the feedforward neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')
### guardar
print("[INFO] guardar el modelo...")
model.save('model_Hand.h5', save_format="h5")