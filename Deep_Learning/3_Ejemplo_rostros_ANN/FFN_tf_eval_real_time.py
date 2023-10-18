from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image

######### funcion para evaluar el modelo con una sola imagen
def one_test(x,model):
    #x should be a (in_size,in_size,3) array, i.e., a rgb image    
    x      = np.array([x])
    pred   = model.predict(x,batch_size=1,verbose = 0)
    #y_pred = np.argmax(pred)
    y_pred = np.round(pred[0][0]).astype(int)
    return y_pred
    
###########
device = 'cpu'
print('Running on device: {}'.format(device))
source = 0
cam = cv2.VideoCapture(source)
new_dim = (50,50)
    
    
print("[INFO] cargar el modelo...")
#load the model
model = load_model('model_Hand.h5')
clases=["Botella","Celular"]
continuar = True
#realizar un proceso de forma indefinida
while continuar: 
    retval, frame = cam.read()
    #mostrar toma
    cv2.imshow('frame',frame)
    key =  cv2.waitKey(1)
    if key == 27 :
         break
    #procesar imagen 
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))    
    frame = frame.resize(new_dim)
    frame = np.array(frame).ravel()/255
   
    
    clase = one_test(frame,model)
    print(clases[clase])
    
cv2.destroyAllWindows()  
    
    
    