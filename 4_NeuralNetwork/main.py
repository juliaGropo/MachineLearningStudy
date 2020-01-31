from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Librariesauxiliares
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os, sys
import scipy
from skimage.io import imread
from random import choice

path = "/home/julia/Documentos/Mestrado/TestDataset/test/"
dirs = os.listdir( path )
def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((200,200), Image.ANTIALIAS)
            imResize.save(f + '_resized.png', 'PNG', quality=90)

resize()

def create_batches():
    train_images = []
    train_labels = []

    path_train = "/home/julia/Documentos/Mestrado/TrainDataset/train/"
    
    dirs_train = os.listdir(path_train)
    
    for folder in dirs_train:
        list_train_images = os.listdir(path_train + folder)
        for img in list_train_images:
            img_map = imread(path_train + folder + "/" + img)
            img_map = np.asarray(img_map, dtype="int32" )
            if (img_map.shape[2] == 3):
                train_images.append(img_map)
                train_labels.append(dirs_train.index(folder))
        print("OK - " + folder + " - Index: " + str(dirs_train.index(folder)))
    train_labels = np.asarray(train_labels)
    
    return [train_images, train_labels]

train_set = create_batches()

plt.figure()
plt.imshow(train_set[0][0])
plt.colorbar()
plt.grid(False)
plt.show()

train_set[0] = np.asarray(train_set[0])
print(train_set[0].shape)

class_names = ['Common wheat', 'Common Chickweed', 'Black-grass', 'Cleavers', 'Charlock']
sequence = [i for i in range(2573)]

plt.figure(figsize=(10,10))
for i in range(25):
    selection = choice(sequence)
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_set[0][selection], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_set[1][selection]])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(200, 200, 3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

images = train_set[0]
labels = train_set[1]
model.fit(images, labels, epochs=5)

def create_test_batches():    
    test_images = []
    path_test = "/home/julia/Documentos/Mestrado/TestDataset/test/"
    dirs_test = os.listdir(path_test)
    
    for img in dirs_test:
        print(img + " - " + str(dirs_test.index(img)))
        img_map = imread(path_test + img)
        img_map = np.asarray(img_map, dtype="int32" )
        if (img_map.shape[2] == 3):
            test_images.append(img_map)
    
    return test_images

test_images = create_test_batches()
test_images = np.asarray(test_images)
print(test_images.shape)

predictions = model.predict(test_images)

def plot_image(i, predictions_array, img):
    predictions_array, img = predictions_array[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue'

    plt.xlabel(class_names[predicted_label] + " - " + str(100*np.max(predictions_array)),color='blue')

def plot_value_array(i, predictions_array):
    predictions_array = predictions_array[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(5), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('blue')

i = 362
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions)
plt.show()
