
from joblib import Parallel, delayed
import joblib
import tensorflow as tf
# from tensorflow.keras import models,layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


# from sklearn.model_selection import train_test_split
# from keras.applications import ResNet50

# from keras.models import Sequential

# from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator


# IMAGE_SIZE = 256
# BATCH_SIZE = 42
# CHANNELS = 3

# dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     "C:\\Users\\DCL\\Desktop\\polash\\Weed-Detection\\\weed_detection\\Weed_Detection\\Notebooks\\savedModels",
#     seed=123,
#     shuffle=True,
#     image_size =(IMAGE_SIZE,IMAGE_SIZE),
#     batch_size =BATCH_SIZE
# )

# class_names =dataset.class_names
# class_names

# def get_dataset2(ds ,train_split=0.0,val_split=0.0 , test_split=1.0,shuffle=True, shuffle_size =1000):
#   assert(train_split+val_split+test_split)==1
#   ds_size = len(ds)
#   if shuffle:
#     ds = ds.shuffle(shuffle_size,seed=12)

#   train_size = int(train_split*ds_size)
#   val_size = int (val_split*ds_size)

#   train_ds = ds.take(train_size)
#   val_ds = ds.skip(train_size).take(val_size)
#   test_ds = ds.skip(train_size).skip(val_size)

#   return test_ds


# dataset_given = tf.keras.preprocessing.image_dataset_from_directory(
#     "C:\\Users\\DCL\\Desktop\\polash\\Weed-Detection\\weed_detection\\Weed_Detection\\given",
#     seed=1,
#     shuffle=True,
#     image_size =(IMAGE_SIZE,IMAGE_SIZE),
#     batch_size =BATCH_SIZE
#   )
  
# def data_prepro():
  
#   dataset_given = tf.keras.preprocessing.image_dataset_from_directory(
#     "C:\\Users\\DCL\\Desktop\\polash\\Weed-Detection\\weed_detection\\Weed_Detection\\given",
#     seed=1,
#     shuffle=True,
#     image_size =(IMAGE_SIZE,IMAGE_SIZE),
#     batch_size =BATCH_SIZE
#     )
  
#   test_ds = get_dataset2(dataset_given)

#   return test_ds

# test_ds = data_prepro()

# test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size =tf.data.AUTOTUNE)

# class_names2 =dataset_given.class_names
# class_names2


# Load the model from the file
# knn_from_joblib = joblib.load("C:\\Users\\DCL\\Desktop\\polash\\Weed-Detection\\weed_detection\\res_model.pkl")

  
# Use the loaded model to make predictions


def predict (res, img,class_names):
    img_array2 = tf.keras.preprocessing.image.img_to_array(img)
    img_array2 = tf.expand_dims (img_array2, 0)
    predictions2 = res.predict(img_array2)
    predicted_class2 = class_names[np.argmax (predictions2[0])]
    confidence2 = round(100* (np.max(predictions2[0])), 2)
    return predicted_class2, confidence2

def output(test_ds,class_names2):
    
    IMAGE_SIZE = 256
    BATCH_SIZE = 42
    CHANNELS = 3

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\DCL\\Desktop\\polash\\Weed-Detection\\\weed_detection\\Weed_Detection\\Notebooks\\savedModels",
    seed=123,
    shuffle=True,
    image_size =(IMAGE_SIZE,IMAGE_SIZE),
    batch_size =BATCH_SIZE
    )

    class_names =dataset.class_names
    class_names
    print(class_names)
    
    loaded_res_model = load_model("C:\\Users\\DCL\\Desktop\\polash\\Weed-Detection\\weed_detection\\res_model.h5")
    
    loaded_res_model.evaluate(test_ds)
   
    # plt.figure(figsize=(15, 15))
    for images2, labels2 in test_ds.take(1):

        for i in range(1):

            # ax = plt.subplot(1, 1, i + 1)
            # plt.imshow(images2[i].numpy().astype("uint8"))

            predicted_class2, confidence2= predict(loaded_res_model, images2[i].numpy(),class_names)
            actual_class2 = class_names2[labels2[i]]
            print(class_names2)
            print(actual_class2)
            print(predicted_class2)
            print(confidence2)

            return actual_class2,predicted_class2,confidence2
    
# output(test_ds)
    # plt.title(f"Actual: {actual_class2},\n Predicted: {predicted_class2}.\n Confidence: {confidence2}%")
    # plt.axis("off")