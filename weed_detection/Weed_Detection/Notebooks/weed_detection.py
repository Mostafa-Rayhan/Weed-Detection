# -*- coding: utf-8 -*-
"""Weed Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17gJXJai5tt2h2sojs5fQkLuiVr4nYSPR
"""

import tensorflow as tf
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 256
BATCH_SIZE = 42
CHANNELS = 3

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/drive/MyDrive/Thesis/Data",
    seed=123,
    shuffle=True,
    image_size =(IMAGE_SIZE,IMAGE_SIZE),
    batch_size =BATCH_SIZE
)

class_names =dataset.class_names
class_names

len(dataset)

for image_batch, labels_batch in dataset.take(1):
    print(image_batch.shape)
    print(labels_batch.numpy())

plt.figure(figsize=(10,10))
for image_batch, labels_batch in dataset.take(1):
    for i in range(4):
        ax = plt.subplot(2,2,i+1)

        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[labels_batch[i]])
        plt.axis("off")

def get_dataset(ds ,train_split=0.8,val_split=0.1 , test_split=0.1,shuffle=True, shuffle_size =1000):
  assert(train_split+val_split+test_split)==1
  ds_size = len(ds)
  if shuffle:
    ds = ds.shuffle(shuffle_size,seed=12)

  train_size = int(train_split*ds_size)
  val_size = int (val_split*ds_size)

  train_ds = ds.take(train_size)
  val_ds = ds.skip(train_size).take(val_size)
  test_ds = ds.skip(train_size).skip(val_size)

  return train_ds, val_ds, test_ds
  train_ds, val_ds, test_ds = get_dataset(dataset)

train_ds, val_ds, test_ds = get_dataset(dataset)

len(train_ds)

len(val_ds)

len(test_ds)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size =tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size =tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size =tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1./255)
])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])

train_ds = train_ds.map(
    lambda x,y: (data_augmentation(x,training=True),y)
).prefetch(buffer_size = tf.data.AUTOTUNE)

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

n_classes =2

model = models.Sequential([
resize_and_rescale,
layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, kernel_size = (3,3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64,(3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu'),
layers.MaxPooling2D((2, 2)),
layers.Flatten(),
layers.Dense (64, activation='relu'),
layers.Dense (n_classes, activation='softmax'),

])

model.build(input_shape = input_shape)

"""**ResNet50**"""

from keras.applications import ResNet50

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator

res = Sequential()
res.add(ResNet50(include_top=False, pooling='avg', weights="imagenet"))
res.add(Dense(n_classes, activation='softmax'))
res.layers[0].trainable = False

res.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

res.summary()

history4 = res.fit(
    train_ds,
    batch_size =BATCH_SIZE,
    validation_data=val_ds,
    epochs=50
)

scores =res.evaluate(test_ds)

plt.figure(figsize=(10,10))
plt.plot(history4.history['loss'], label='train loss')
plt.plot(history4.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.figure(figsize=(10,10))
plt.plot(history4.history['accuracy'], label='train acc')
plt.plot(history4.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()

from google.colab import drive
drive.mount('/content/drive')

def predict (res, img):
    img_array2 = tf.keras.preprocessing.image.img_to_array(images2[i].numpy())
    img_array2 = tf.expand_dims (img_array2, 0)
    predictions2 = res.predict(img_array2)
    predicted_class2 = class_names[np.argmax (predictions2[0])]
    confidence2 = round(100* (np.max(predictions2[0])), 2)
    return predicted_class2, confidence2

plt.figure(figsize=(15, 15))
for images2, labels2 in test_ds.take(1):

  for i in range(4):

    ax = plt.subplot(2, 2, i + 1)
    plt.imshow(images2[i].numpy().astype("uint8"))

    predicted_class2, confidence2= predict(res, images2[i].numpy())
    actual_class2 = class_names[labels2[i]]
    plt.title(f"Actual: {actual_class2},\n Predicted: {predicted_class2}.\n Confidence: {confidence2}%")
    plt.axis("off")