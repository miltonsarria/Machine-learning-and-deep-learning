#USE our VGG model

#pip install --upgrade tensorflow
#pip install --upgrade keras

#https://pyserial.readthedocs.io/en/latest/pyserial.html
# import the necessary packages
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import cv2
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from tools import *
##################
def one_test(x,model):
    #x should be a (h,w,3) array, i.e., a rgb image    
    x=x/255
    x = cv2.resize(x, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    
    pred   = model.predict(x)[0]
    y_pred = np.argmax(pred)
    return y_pred


#######################################   
#######################################   
#######################################   

#definir parametros importantes
source      = 0 #camara
new_size    = (224, 224) #dimensiones para las imagenes que se capturan 
IMAGE_DIMS  = (224, 224, 3)#modelo para las imagenes que se alimentan al modelo
#crear aobjeto que captura las imagenes
readObj = capture(source=source,new_size=new_size)
readObj.save_file=False

#cargar el modelo y los labels
file_name_mod =  'models/May_24_vgg_small.h5'
model_test = load_model(file_name_mod)
file_name_lb =  'models/hand_label_bin.p'
lb = pickle.loads(open(file_name_lb, "rb").read())

##### iniciar la ventana donde graficaremos
plt.show() 
axes = plt.gca() 

continuar = True
#realizar un proceso de forma indefinida
while continuar: 
    readObj.get_image()
    img=readObj.img_g
    axes.clear()
    axes.imshow(np.fliplr(img),cmap='gray')
    #plt.draw() 
    plt.pause(1e-17) 
    #determinar si la ventana continua abierta
    existe_plot = plt.get_fignums()
    if len(existe_plot)==0:
        continuar = False
        
    ####
    img = readObj.img_rgb
    y1 = one_test(img,model_test)
    label = lb.classes_[y1]
    
    print('Respuesta : ', label)

readObj.stop()

