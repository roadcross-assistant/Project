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
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(270, 480, 3),
    include_top=False,
)

# Freeze the base_model
base_model.trainable = True

inputs = tf.keras.layers.Input([270, 480, 3])
inputs_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

x = base_model(inputs_preprocessed, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(1e-5),
        metrics=[tf.keras.metrics.RecallAtPrecision(precision=0.9, name='acc')])

model.load_weights("/home/ubuntu/checkpoints/training_4/cp.ckpt")

#%%

img = cv2.imread("/home/ubuntu/Data/Frames/video2/frame100.jpg")

print(img.shape)
img=cv2.resize(img, (480,270))
print(img.shape)
inp = img.reshape((1, 270, 480,3))
print(inp.shape)

print(model.predict(inp))

#%%

model.save('/home/ubuntu/savedmodel_training4')

loaded = tf.keras.models.load_model('/home/ubuntu/savedmodel_training4')
print(loaded.predict(inp))

#%%
from tensorflow.python.compiler.tensorrt import trt_convert as trt
converter = trt.TrtGraphConverterV2(input_saved_model_dir="/home/ubuntu/savedmodel_training4")
converter.convert()
converter.save("/home/ubuntu/tensorrt_training4")

model = tf.saved_model.load("/home/ubuntu/tensorrt_training4")
func = model.signatures['serving_default']

#%%

img = cv2.imread("/home/ubuntu/Data/Frames/video30/frame50.jpg")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (480,270))
print(img.shape)

test_input = np.array([img])
print(test_input.shape)
x = tf.convert_to_tensor(test_input, dtype=tf.float32)
print(func(x))
# %%
