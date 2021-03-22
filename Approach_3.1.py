# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#importing necessary libraries
from __future__ import print_function
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
#from imageai.Detection import VideoObjectDetection
import os
import sys
from random import randint
from math import ceil, sqrt
import natsort
import pandas as pd
import random
from keras.applications.vgg16 import VGG16
import pickle
from DataLoader import  DataGenerator


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
list_IDs_train = {}
list_IDs_test = {}
list_IDs_validation = {}

for vid in videos_train:

    video = path_videos + "video" + str(vid) + ".MOV"
    #print("starting " + video)
    cap = cv2.VideoCapture(video)
    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cv2.destroyAllWindows() 
    list_IDs_train[vid] = no_frames

for vid in videos_test:

    video = path_videos + "video" + str(vid) + ".MOV"
    #print("starting " + video)
    cap = cv2.VideoCapture(video)
    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cv2.destroyAllWindows() 
    list_IDs_test[vid] = no_frames

for vid in videos_validation:

    video = path_videos + "video" + str(vid) + ".MOV"
    #print("starting " + video)
    cap = cv2.VideoCapture(video)
    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cv2.destroyAllWindows() 
    list_IDs_validation[vid] = no_frames

print(list_IDs_train)
print(len(list_IDs_train))

print(list_IDs_test)
print(len(list_IDs_test))

print(list_IDs_validation)
print(len(list_IDs_validation))


# %%
# Generators
training_generator = DataGenerator(list_IDs = list_IDs_train, folder_path = path_frames)
validation_generator = DataGenerator(list_IDs = list_IDs_validation, folder_path = path_frames)
testing_generator = DataGenerator(list_IDs = list_IDs_test, folder_path = path_frames)


# %%
from tensorflow.keras.layers import Flatten, Dense
import keras
import tensorflow
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers 
from tensorflow.keras.applications import MobileNet

base_model = MobileNet(input_shape = (360, 640, 3), include_top = False, weights = 'imagenet')

# for layer in base_model.layers:
#     layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(2, activation='softmax')(x)

model = tensorflow.keras.models.Model(base_model.input, x)
model.compile(
    loss='binary_crossentropy',
    optimizer=tensorflow.keras.optimizers.Adam(),
    metrics=['accuracy'])

model.summary()


# %%
#history = model.fit_generator(train_generator, shuffle='true', epochs=1, verbose=1, batch_size=16)

history = model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=50, verbose=1)


# %%
# import random
# from PIL import Image 

# dim = (1080, 1920)
# X = np.empty((4, *dim, 3))
# y = np.empty((4), dtype=int)

# video_ids = [2,3]

# c = 0

# for vid in video_ids:

#     frame_ids = random.sample(list(range(list_IDs_train[vid])), 2) # generate "no_frames_per_video" random frames
#             #max value of frame given by self.list_ids[vid]
#     labels_temp = np.load(path_frames + "video" + str(vid) + "/labels" + str(vid) + ".npy")

#     for fid in frame_ids:
#         im = Image.open(path_frames + "video" + str(vid) + "/frame" + str(fid) + ".jpg")

#         #X[c, ] = np. asarray(im)

#         d = np.asarray(im)
#         frame_resized = cv2.resize(d, (640, 360), interpolation = cv2.INTER_AREA)

#         print(d.shape, frame_resized.shape)
                
#         y[c] = labels_temp[fid]

#         c = c + 1

# print(c)

# print(X.shape, y.shape)

# print(y)

# print(keras.utils.to_categorical(y, num_classes=2))


# %%
print("Evaluate on test data")
results = model.evaluate(testing_generator)
print("test loss, test acc:", results)

print("Evaluate on train data")
results = model.evaluate(training_generator)
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


# %%



