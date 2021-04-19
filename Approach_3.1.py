# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#importing necessary libraries
from __future__ import print_function
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import os
import sys
from random import randint
from math import ceil, sqrt
import natsort
import pandas as pd
import random
import pickle
from DataLoader import  DataGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import MobileNet

# %%
user = 'aws'

if user == 'siddhi':
    path_videos = 'C:/RoadCrossingAssistant/Data/Videos/'
    path_labels_csv = 'C:/RoadCrossingAssistant/Data/labels_framewise_csv.csv'
    path_labels_list = 'C:/RoadCrossingAssistant/Data/labels_framewise_list.pkl'
    path_frames = 'C:/RoadCrossingAssistant/Data/Frames/'

elif user == 'yagnesh':
    path_videos = '/home/yagnesh/Study/Machine Learning/ML projects/RoadCrossingAssistant_Arrays/videos/'
    path_labels_csv = '/home/yagnesh/Study/Machine Learning/ML projects/RoadCrossingAssistant_Arrays/labels_framewise.csv'
    path_labels_list = '/home/yagnesh/Study/Machine Learning/ML projects/RoadCrossingAssistant_Arrays/labels_framewise.pkl'

elif user == 'aws':
    path_videos = '/home/ubuntu/Data/Videos/'
    path_labels_csv = '/home/ubuntu/Data/labels_framewise_csv.csv'
    path_labels_list = '/home/ubuntu/Data/labels_framewise_list.pkl'
    path_frames = '/home/ubuntu/Data/Frames/'

# frame-wise labels array
open_file = open(path_labels_list, "rb")
labels_list = pickle.load(open_file)
open_file.close()

# %%
#Perform train-test-validation split(62-24-18)

x = np.arange(1, 105)
np.random.shuffle(x)
#np.random.seed(42)
videos_validation = x[:18]
videos_test = x[18: 18+24]
videos_train = x[18+24: ]

# videos_train = [videos[ind] for ind in indices_train]
# videos_test = [videos[ind] for ind in indices_test]

# labels_train_loaded = [labels_list[ind] for ind in indices_train]
# labels_test_loaded = [labels_list[ind] for ind in indices_test]

# print('len of videos_train: ', len(videos_train))
# print('len of videos_test: ', len(videos_test))
# print('len of labels_train_loaded: ', len(labels_train_loaded))
# print('len of labels_test_loaded: ', len(labels_test_loaded))

print(videos_train, len(videos_train))
print(videos_test, len(videos_test))
print(videos_validation, len(videos_validation))

# %%
filenames_train = []
labels_train = []
filenames_validation = []
labels_validation = []
filenames_test = []
labels_test = []

# videos = [1,2]
# filenames = []
# labels = []

for vid in videos_train:
    folder = path_frames + "video{}/".format(vid)
    frames = glob.glob(folder + 'frame*.jpg')
    frames = natsort.natsorted(frames)
    filenames_train.extend(frames)
    labels_path = path_frames + "video{}/".format(vid) + "labels{}.npy".format(vid)
    labels_array = np.load(labels_path)
    labels_list = list(labels_array)
    labels_train.extend(labels_list)

for vid in videos_test:
    folder = path_frames + "video{}/".format(vid)
    frames = glob.glob(folder + 'frame*.jpg')
    frames = natsort.natsorted(frames)
    filenames_test.extend(frames)
    labels_path = path_frames + "video{}/".format(vid) + "labels{}.npy".format(vid)
    labels_array = np.load(labels_path)
    labels_list = list(labels_array)
    labels_test.extend(labels_list)

for vid in videos_validation:
    folder = path_frames + "video{}/".format(vid)
    frames = glob.glob(folder + 'frame*.jpg')
    frames = natsort.natsorted(frames)
    filenames_validation.extend(frames)
    labels_path = path_frames + "video{}/".format(vid) + "labels{}.npy".format(vid)
    labels_array = np.load(labels_path)
    labels_list = list(labels_array)
    labels_validation.extend(labels_list)

print(len(filenames_train), len(filenames_validation), len(filenames_test))
print(len(labels_train), len(labels_validation), len(labels_test))


# %%
# Generators
def parse_function(filename, label):

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [270, 480], method=tf.image.ResizeMethod.AREA, 
                            preserve_aspect_ratio=True)
    
    return image, label


def train_preprocess(image, label):

    image = tf.image.random_brightness(image, 0.15)

    return image, label

dataset_train = tf.data.Dataset.from_tensor_slices((filenames_train,labels_train))
dataset_train = dataset_train.map(parse_function, num_parallel_calls=4)
dataset_train = dataset_train.map(train_preprocess, num_parallel_calls=4)
#d = d.window(2)
#d = d.shuffle(3)
#d = d.flat_map(lambda a,b:tf.data.Dataset.zip((a,b)).batch(2))
#d = d.map(lambda a,b : (a,b[-1]))
dataset_train = dataset_train.batch(32)

dataset_test = tf.data.Dataset.from_tensor_slices((filenames_train,labels_train))
dataset_test = dataset_test.map(parse_function, num_parallel_calls=4)
dataset_test = dataset_test.map(train_preprocess, num_parallel_calls=4)
#d = d.window(2)
#d = d.shuffle(3)
#d = d.flat_map(lambda a,b:tf.data.Dataset.zip((a,b)).batch(2))
#d = d.map(lambda a,b : (a,b[-1]))
dataset_test = dataset_test.batch(32)

dataset_val = tf.data.Dataset.from_tensor_slices((filenames_train,labels_train))
dataset_val = dataset_val.map(parse_function, num_parallel_calls=4)
dataset_val = dataset_val.map(train_preprocess, num_parallel_calls=4)
#d = d.window(2)
#d = d.shuffle(3)
#d = d.flat_map(lambda a,b:tf.data.Dataset.zip((a,b)).batch(2))
#d = d.map(lambda a,b : (a,b[-1]))
dataset_val = dataset_val.batch(32)

# %%
def create_model():


    base_model = MobileNet(input_shape = (270, 480, 3), include_top = False, weights = 'imagenet')

    # for layer in base_model.layers:
    #     layer.trainable = False

    x = tf.keras.layers.Flatten()(base_model.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(base_model.input, x)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])

    return model

model = create_model()
model.summary()

# %%
#history = model.fit_generator(train_generator, shuffle='true', epochs=1, verbose=1, batch_size=16)
checkpoint_path = "/home/ubuntu/Project/checkpoints/training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True, monitor='val_binary_accuracy', verbose=1, 
                                                 save_best_only=True, mode='max')

history = model.fit(x=dataset_train, validation_data=dataset_val, epochs=50, 
                                verbose=1, callbacks = [cp_callback], class_weight = {0: 1 , 1:1.92})

# %%
print("Evaluate on test data")
results = model.evaluate(dataset_test)
print("test loss, test acc:", results)

print("Evaluate on train data")
results = model.evaluate(dataset_train)
print("train loss, trai acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
# print("Generate predictions")
# predictions = model.predict(frames_test)
# print("predictions shape:", predictions.shape)
# print(predictions[:10])



# %%
# from sklearn.metrics import average_precision_score
# average_precision = average_precision_score(np.argmax(y_test, axis = 1), np.argmax(predictions, axis = 1))

# print('Average precision-recall score: {0:0.2f}'.format(
#       average_precision))

