# %%
#importing necessary libraries
import numpy as np
import cv2
import glob
import os
import sys
from random import randint
from math import ceil, sqrt
import natsort
import random
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2

#%%
tf.keras.backend.set_image_data_format('channels_last')
def create_model():

    inputs = tf.keras.layers.Input([270, 480, 3])
    #x = tf.keras.layers.experimental.preprocessing.Normalization(inputs)
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Conv2D(32, (7,7), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(32, (7,7), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Conv2D(64, (5,5), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128  , (5,5), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr=0.001/5),
        metrics=[tf.keras.metrics.RecallAtPrecision(precision=0.9, name='recallAtPrecision'), 
        tf.keras.metrics.BinaryAccuracy(threshold=0.6, name='binaryAccuracy')])

    return model

model = create_model()
model.summary()
model.load_weights("/home/ubuntu/checkpoints/training_temp/cp.ckpt")

#%%

img = cv2.imread("/home/ubuntu/Data/Frames/video33/frame60.jpg")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (480,270))
print(img.shape)

test_input = np.array([img])
print(test_input.shape)

print(model.predict(test_input))

#%%

model.save('/home/ubuntu/savedmodels/training_temp')

loaded = tf.keras.models.load_model('/home/ubuntu/savedmodels/training_temp')
print(loaded.predict(test_input))

#%%
from tensorflow.python.compiler.tensorrt import trt_convert as trt
converter = trt.TrtGraphConverterV2(input_saved_model_dir="/home/ubuntu/savedmodels/training_temp")
converter.convert()
converter.save("/home/ubuntu/tensorrt_models/training_temp")

model = tf.saved_model.load("/home/ubuntu/tensorrt_models/training_temp")
func = model.signatures['serving_default']

#%%
x = tf.convert_to_tensor(test_input, dtype=tf.float32)
print(func(x))