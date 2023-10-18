'''
Milton Orlando Sarria
USC
train our VGG model
import the necessary packages
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from smallVGG import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
 
#########
# initialize the initial learning rate, number of epochs to train for,
# and batch size
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions

#cambiar en colab
model_dir = 'models/'
data_dir = 'data/'

EPOCHS = 100
INIT_LR = 1e-3
BS = 32
# initialize the data and labels
data = []
labels = []
#load images
print("[INFO] loading images...")

data_file = ['g1.p', 'g2.p', 'nada.p']
clases   = ['uno','dos','nada']

size = (224, 224,3)



# loop over the files
for fh,label in zip(data_file,clases):
    fh =  data_dir + fh
    matrix = pickle.load( open( fh , "rb" ) )
    for row in matrix:
        image = row.reshape(224,224,3)  
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (size[1], size[0]))
        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)
        
    matrix=None


data = np.array(data)/255.0
labels = np.array(labels)
ind=np.random.permutation(labels.size)
X=data[ind,:]
labels=labels[ind]

print("[INFO] data matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))
print(data.shape)

#plt.figure(figsize = (5,5))
#img = data[30] 
#plt.imshow(img);    # plot the image
#plt.axis('off');
#plt.show()

# perform one-hot encoding on the labels
lb = LabelBinarizer()
lb.fit(clases)
LB = lb.transform(labels)


#LB = lb.fit_transform(labels)
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	LB, test_size=0.2, random_state=42)


# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = SmallerVGGNet.build(width=size[1], height=size[0],
	depth=size[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the network
print("[INFO] training network...")
H = model.fit(
	x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the model to disk
print("[INFO] serializing network...")

file_name = model_dir + 'hand_vgg_small.h5'

model.save(file_name, save_format="h5")

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
file_name = model_dir + 'hand_label_bin.h5'

f = open(file_name, "wb")
f.write(pickle.dumps(lb))
f.close()


