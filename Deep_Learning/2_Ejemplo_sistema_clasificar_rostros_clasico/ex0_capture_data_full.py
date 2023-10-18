'''
Milton Orlando Sarria
USC
Ejemplo que permite capturar imagenes y guardar en una carpeta
dedicada para cada clase.
Ejemplo:
clases|
      |-Goku
      |-Jack
'''
import torch
import numpy as np
import  cv2
from PIL import Image
import os

device = 'cpu'

print('Running on device: {}'.format(device))

#mtcnn = MTCNN(keep_all=True, device=device)
#habilitar camara
source = 0
cam = cv2.VideoCapture(source)

#carpeta donde se guardaran las imagenes capturadas
destino='clases/';
new_dim = (50,50)
continuar = True
fotos_por_cliente = int(input("Numero de fotos por clase?: "))
while continuar: 
    k=0
    cliente = input("Nombre de la clase: ")
    path_cliente = destino + cliente
    if not os.path.exists(path_cliente):
        print("Creandoel directorio para: '{}' ".format(path_cliente))
        os.makedirs(path_cliente)
    while(k<fotos_por_cliente):    
        retval, frame = cam.read()
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # guardar el frame
        frame_pil  = frame_pil.resize(new_dim)
        new_name  = path_cliente + '/img_' + str(k) + '.png'
        frame_pil.save(new_name)
        k=k+1
                    
        #mostrar toma
        cv2.imshow('frame',frame)
        key =  cv2.waitKey(1)
        if key == 27 :
            break
    cv2.destroyAllWindows()   
    
    cont = input("Desea registrar otra clase? (S/N): ")
    if cont.upper()=='N':
        continuar = False
       
        
print('\nDone')









