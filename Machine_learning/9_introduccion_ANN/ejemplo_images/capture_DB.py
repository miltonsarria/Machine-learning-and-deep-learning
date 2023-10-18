"""
Milton Orlando Sarria Paja
USC
capturar las imagenes de ejemplo
ejecutarlo una vez cada que quiera capturar una nueva clase de ejemplo
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from tools import *

##########################################################
##########################################################   
#definir archivo 
file_name=input('Nombre del archivo para guardar imagenes:')

folder = 'data/'
file_name = folder+file_name

source=0 #camara
dims=(224, 224)#dimensiones para las nuevas imagenes
readObj= capture(source=source,new_size=dims,file_name=file_name)

##### iniciar la ventana donde graficaremos
plt.show() 
axes = plt.gca() 
axes.axis("off")
continuar = True
#realizar un proceso de forma indefinida

while continuar: 
    readObj.get_image()
    img=readObj.img_rgb
    axes.clear()
    axes.imshow(np.fliplr(img))
    #plt.draw() 
    plt.pause(1e-17) 
    #determinar si la ventana continua abierta
    existe_plot = plt.get_fignums()
    if len(existe_plot)==0:
        continuar = False      
    ####
    
readObj.stop()



