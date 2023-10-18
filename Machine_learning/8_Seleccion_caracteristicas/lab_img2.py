from PIL import Image, ImageFilter                    #Importamos el modulo
img1 = Image.open('img1.png')  

new_dim = (500,500) #dar las nuevas dimensiones (alto, ancho)
im_resized = img1.resize(new_dim)
im_resized.save('img1_new_size.png') 

#reducir a la mitad, podria ser un tercio un cuarto.....
new_dim = (img1.width // 2, img1.height // 2)
im_resized = img1.resize(new_dim)
im_resized.save('img1_half.png') 
