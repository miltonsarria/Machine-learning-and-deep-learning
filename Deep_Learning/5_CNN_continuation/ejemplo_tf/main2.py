#pip install opencv-python
#pip install tqdm
#pip install sklearn
#pip install tensorflow

from tensorflow.keras.utils import to_categorical

from funciones import *



########  DATA 
data_dir = "images"
in_size = 100
clases = ['cat','dog','person']
n_clases = len(clases)
X, labels = loadImages(data_dir, clases,in_size)       

#organizar los labels: one hot encoding
labels = to_categorical(labels)


# Batch size for training (change depending on how much memory you have)
batch_size = 8
# Number of epochs to train for (pueden ser menos si nota que no mejora)
num_epochs = 50

#separar en entrenamiento y prueba
(X_train, X_test, y_train, y_test) = train_test_split(X, labels,test_size=0.20)

#crear modelo
model = CrearModelo('vgg16',in_size, n_clases, show_summary=True)

#entrenar modelo
model = EntrenarModelo(model,batch_size,num_epochs,X_train,y_train)


## evaluar modelo
# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(X_test, batch_size=batch_size)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
y_pred = np.argmax(predIdxs, axis=1)
#
acc1= np.sum(y_pred==y_test.argmax(axis=1))/len(y_pred)*100
        

print(" total accuracy = %f"% (acc1))



print("[INFO] saving cat-dog-person class model...")
model.save('model_CatDogPerson.h5', save_format="h5")



#### como evaluar con una sola imagen??

from tensorflow.keras.models import load_model

lista_img=[\
'images/cat/cat_0458.jpg',
'images/cat/cat_0459.jpg',
'images/cat/cat_0460.jpg',
'images/dog/dog_0434.jpg',
'images/dog/dog_0435.jpg',
'images/dog/dog_0436.jpg',
'images/person/person_0547.jpg',
'images/person/person_0571.jpg',
'images/person/person_0591.jpg']
in_size = 100
clases = ['cat','dog','person']

#load the model
model_test = load_model('model_CatDogPerson.h5')

for img in lista_img:

    x1 = cv2.imread(img)
    x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2RGB)
    x1 = cv2.resize(x1, (in_size, in_size))
    x1=x1/255
    
    
    y1 = one_test(x1,model_test)
     
    print(f'Para imagen {img}, la pred del model es: {clases[y1]}')
    


