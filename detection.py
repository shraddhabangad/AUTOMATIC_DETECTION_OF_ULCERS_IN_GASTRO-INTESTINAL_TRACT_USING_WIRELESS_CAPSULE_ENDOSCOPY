#!/usr/bin/env python

import cv2
import tensorflow as tf

CATEGORIES = ["dyed-lifted-polyps","esophagitis","dyed-resection-margins","normal-cecum","normal-pylorus","normal-z-line", "polyps","ulcerative-colitis"]

def prepare(filepath):
    IMG_SIZE = 200
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE,IMG_SIZE, 1)


model = tf.keras.models.load_model("ulcer-CNN.model")
#print(model.summary())


prediction = model.predict([prepare('pol2.jpg')]) #image should be in the main python folder
print(prediction)
#print(CATEGORIES[int(prediction[0][0])])