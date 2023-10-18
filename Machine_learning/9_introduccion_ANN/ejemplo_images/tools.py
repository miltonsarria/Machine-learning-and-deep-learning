"""
Milton Orlando Sarria Paja
USC
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle

### tools

###################
class capture():
    def __init__(self,source=0,tRead=50,new_size=(32, 32),file_name='img.txt'):
      # Create the VideoCapture object and define default values
      self.cam    = cv2.VideoCapture(source)
      self.img_rgb = None
      self.img_g   = None
      self.img_hsv = None
      self.new_size=new_size
      self.file_name=file_name
      self.save_file=True
      self.matrix = np.array([])
      if not self.cam.isOpened():
        print("Video device or file couldn't be opened")
        exit()
      retval, img_rgb = self.cam.read()      
      self.img_rgb = img_rgb
      self.img_g   = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
      return
#############################3        
    def get_image(self):
      #read from camera
      _, img_rgb = self.cam.read()
      #escalar la imagen
      dim = self.new_size
      # resize image
      img_rgb = cv2.resize(img_rgb, dim, interpolation = cv2.INTER_AREA) 
      #se tienen tres tipos de imagenes: rgb, hsv y escala de gris
      self.img_rgb = img_rgb
      self.img_g   = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
      #por ahora solo guarda la que esta en escala de gris
      if self.save_file:
         if self.matrix.size==0:
            self.matrix = self.img_rgb.ravel()
         else:
            self.matrix = np.vstack((self.matrix, self.img_rgb.ravel()))
    
    ########### si se detiene el proceso, guardar ants de salir
    def save(self):
    #guardar en un archivo txt todo lo que se haya almacenado en matrix        
        pickle.dump(self.matrix, open(self.file_name+'.p', "wb" ) )
        print(f'[INFO] archivo guardado, en total {self.matrix.shape[0]} imagenes')
        self.matrix = None
        return
    def stop(self):
       self.cam.release()
       self.save()
       return
