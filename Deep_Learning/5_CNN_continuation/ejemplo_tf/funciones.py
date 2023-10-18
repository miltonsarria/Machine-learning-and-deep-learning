from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

import cv2
from glob import glob
import numpy as np

#https://towardsdatascience.com/gru-recurrent-neural-networks-a-smart-way-to-predict-sequences-in-python-80864e4fe9f6

#### funciones

###########################################
###########################################


def loadImages(data_dir, clases,in_size):
    #data_dir carpeta principal
    #clases las subcarpetas
    #input_size  las dimensiones para todas (input_size, input_size)
    X = []
    imagePaths = glob(data_dir+"/**/*.jpg",recursive=True) 
    labels     = np.zeros(len(imagePaths))    
    for i, entry in enumerate(imagePaths):            
            data = cv2.imread(entry)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB) #cambiar de BGR a RGB
            data = cv2.resize(data, (in_size, in_size))
            X.append(data)
            
            for j, clase in enumerate(clases):
                if clase in entry:
                    labels[i]=j
                    
    X=np.array(X)/255
    
    return X, labels

###########################################
###########################################
def CrearModelo(ModelName,in_size, n_classes, num_chans=3,show_summary=False):

    #1. cargar el modelo pre-entrenado
    if ModelName=='vgg16':
        # load the VGG16 network, ensuring the head FC layer sets are left off
        baseModel   = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(in_size, in_size,num_chans)))
    if ModelName=='resnet50':
        # load the VGG16 network, ensuring the head FC layer sets are left off
        baseModel   = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(in_size, in_size,num_chans)))

    #2. congelar (deshabilitar) el entrenamiento de todas las capas
    for layer in baseModel.layers[:]:
        layer.trainable = False

    #3. crear un nuevo modelo
    # Create the model
    model = Sequential()
    # agregar el modelo base
    model.add(baseModel)
    # agregar nuevas capas
    model.add(AveragePooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    # Show a summary of the model. Check the number of trainable parameters
    if show_summary:
        print(model.summary())

    
    return model
###########################################
#funcion para entrenar el modelo
###########################################
def EntrenarModelo(model,batch_size,num_epochs,X_train,y_train):
    #definir el generador de imagenes
    trainAug = ImageDataGenerator(
                             rotation_range=10,
                             fill_mode="nearest")
    INIT_LR = 1e-3  #learning rate
    EPOCHS = num_epochs
    BS = batch_size
    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
    # train the head of the network
    print("[INFO] training head...")
    H = model.fit(trainAug.flow(X_train, y_train, batch_size=BS),                              
                  steps_per_epoch=len(X_train) // BS,              
                  epochs=EPOCHS)
    print("[INFO] Done")
    
    return model

###########################################
#clase para crear una red convolucional tipo vgg
###########################################
class miniVGGnet:
    @staticmethod
    def build(width, height, depth, n_clases):
        # el modelo tiene una entrada en 3D, el canal es el ultimo
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        #definimos la arquitectura de la red
        # CONV => RELU => POOL
        model.add(SeparableConv2D(32, (3, 3), padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # (CONV => RELU => POOL) * 2
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # (CONV => RELU => POOL) * 3
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(SeparableConv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))  
        
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        # softmax classifier
        model.add(Dense(n_clases))
        model.add(Activation("softmax"))
        # return the constructed network architecture
        return model
        

######### evaluar el modelo con una sola imagen
def one_test(x,model):
    #x should be a (in_size,in_size,3) array, i.e., a rgb image    
    x      = np.array([x])
    pred   = model.predict(x,batch_size=1)
    y_pred = np.argmax(pred)
    return y_pred
    
    
            
