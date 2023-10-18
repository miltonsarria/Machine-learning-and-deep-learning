from PIL import Image, ImageFilter                    #Importamos el modulo
img1 = Image.open('img1.png')             #Abrimos la imagen 1
print(img1.format, img1.size, img1.mode)   

img2 = Image.open('img2.jpeg')             #Abrimos la imagen 2
print(img2.format, img2.size, img2.mode)   


box1 = (100, 100, 350, 350)   #definimos una region de la img1
region1 = img1.crop(box1)     #se copia la region de la imagen en una nueva variable
region1.show()

box2 = (200, 200, 500, 500) #definimos una region de la img2
region2 = img2.crop(box2)     #se copia la region de la imagen en una nueva variable
region2.show()


region1 = region1.filter(ImageFilter.FIND_EDGES)    #aplicar un filtro
img1.paste(region1, box1)                              #pegar en la imagen original
img1.show()								#mostrar resultado

img2.paste(region1, box1)                            
img2.show()	
