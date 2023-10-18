from PIL import Image                    #Importamos el modulo
img = Image.open('img1.png')             #Abrimos la imagen

print(img.format, img.size, img.mode)       #caracteristicas de la imagen
                                         #formato, tamanio, orden de canales

nuevaimg = img.rotate(25)                #Rotamos 25 grados
nuevaimg.show()                          #Mostramos


r,g,b = img.split()                      #Obtenemos los canales RGB de la imagen
nuevaimg = Image.merge("RGB", (b, r, g)) #Cambiamos el orden de los canales
nuevaimg.show()


from PIL import ImageFilter             #Ahora importamos 'ImageFilter" para los filtros

nuevaimg = img.filter(ImageFilter.BLUR) #Filtro de Blur
nuevaimg.show()

nuevaimg = img.filter(ImageFilter.BoxBlur(50))  #Filtro de BoxBlur(50)
nuevaimg.show()

nuevaimg = img.filter(ImageFilter.CONTOUR)  #Filtro de CONTOUR
nuevaimg.show()

nuevaimg = img.filter(ImageFilter.EDGE_ENHANCE)  #Filtro de EDGE_ENHANCE
nuevaimg.show()

nuevaimg = img.filter(ImageFilter.EMBOSS)  #Filtro de EMBOSS
nuevaimg.show()

nuevaimg = img.filter(ImageFilter.FIND_EDGES)  #Filtro de FIND_EDGES
nuevaimg.show()


#######
from PIL import Image,ImageFilter
img = Image.open('img1.png')
nuevaimg = img.filter(ImageFilter.FIND_EDGES)  #Filtro de FIND_EDGES
nuevaimg.save('img_result.png') #Guardado en un directorio especifico





