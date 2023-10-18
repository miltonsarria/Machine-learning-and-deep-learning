import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageOps

file1 = "IMG_20230816_133538.jpg"
file2=  "IMG_20230816_133550.jpg"

img1 = Image.open(file1)
img2 = Image.open(file2)
r1,g1,b1 = img1.split()
r2,g2,b2 = img2.split()

r1,g1,b1 = np.asarray(r1),np.asarray(g1),np.asarray(b1)
r2,g2,b2 = np.asarray(r2),np.asarray(g2),np.asarray(b2)

img1.show()
img2.show()


plt.figure(figsize=(20,5))
##
plt.subplot(231)
b=plt.hist(r1.ravel(),100,density=True)
plt.title("canal rojo 1")
plt.axis([0,256,0,0.04])
##
plt.subplot(232)
b=plt.hist(b1.ravel(),100,density=True)
plt.title("canal azul 1")
plt.axis([0,256,0,0.04])
## 
plt.subplot(233)
b=plt.hist(g1.ravel(),100,density=True)
plt.title("canal verde 1")
plt.axis([0,256,0,0.04])
## 

##
plt.subplot(234)
b=plt.hist(r2.ravel(),100,density=True)
plt.title("canal rojo 2")
plt.axis([0,256,0,0.04])
##
plt.subplot(235)
b=plt.hist(b2.ravel(),100,density=True)
plt.title("canal azul 2")
plt.axis([0,256,0,0.04])
## 
plt.subplot(236)
b=plt.hist(g2.ravel(),100,density=True)
plt.title("canal verde 2")
plt.axis([0,256,0,0.04])
## 

plt.show()

im1 = ImageOps.grayscale(img1)
im2 = ImageOps.grayscale(img2)


im1.show()
im2.show()

im1=np.asarray(im1)
im2=np.asarray(im2)

plt.figure(figsize=(20,5))
##
plt.subplot(211)
b=plt.hist(im1.ravel(),100,density=True)
plt.title("imagen en gris 1")
plt.axis([0,256,0,0.04])
##
plt.subplot(212)
b=plt.hist(im2.ravel(),100,density=True)
plt.title("imagen en gris 2")
plt.axis([0,256,0,0.04])

plt.show()