# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
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
    checkpoint_path = "/home/ubuntu/checkpoints/training_4/cp.ckpt"


# %%
#Perform train-test-validation split(66-22-16)

x = np.arange(1, 105)
np.random.shuffle(x)
#np.random.seed(42)
videos_validation = x[:16]
videos_test = x[16: 16+22]
videos_train = x[16+22: ]

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

# videos_train = [1,2]
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

filenames_train = np.array(filenames_train)
labels_train = np.array(labels_train)

for vid in videos_test:
    folder = path_frames + "video{}/".format(vid)
    frames = glob.glob(folder + 'frame*.jpg')
    frames = natsort.natsorted(frames)
    filenames_test.extend(frames)
    labels_path = path_frames + "video{}/".format(vid) + "labels{}.npy".format(vid)
    labels_array = np.load(labels_path)
    labels_list = list(labels_array)
    labels_test.extend(labels_list)

filenames_test = np.array(filenames_test)
labels_test = np.array(labels_test)

for vid in videos_validation:
    folder = path_frames + "video{}/".format(vid)
    frames = glob.glob(folder + 'frame*.jpg')
    frames = natsort.natsorted(frames)
    filenames_validation.extend(frames)
    labels_path = path_frames + "video{}/".format(vid) + "labels{}.npy".format(vid)
    labels_array = np.load(labels_path)
    labels_list = list(labels_array)
    labels_validation.extend(labels_list)

filenames_validation = np.array(filenames_validation)
labels_validation = np.array(labels_validation)

print(filenames_train.shape, filenames_validation.shape, filenames_test.shape)
print(labels_train.shape, labels_validation.shape, labels_test.shape)


#%%

ind0 = np.where(labels_train==0)[0]
ind1 = np.where(labels_train==1)[0]
random.shuffle(ind0)
random.shuffle(ind1)

if (ind0.shape[0]/ind1.shape[0] > 1.4):
    print('reducing the number of unsafe frames in dataframe\n\n')
    len_ind0 = int(ind1.shape[0]*1.4)
    ind0 = ind0[:len_ind0]

    indices_required = np.concatenate((ind0, ind1))

filenames_train_reduced = filenames_train[indices_required]
labels_train_reduced = labels_train[indices_required]

print(filenames_train_reduced.shape, labels_train_reduced.shape)

print(ind0.shape, ind1.shape)


# %%
# Generators
def parse_function(filename, label):

    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [540, 960], method=tf.image.ResizeMethod.AREA, 
                            preserve_aspect_ratio=True)
    
    return image, label


def train_preprocess(image, label):

    image = tf.image.random_brightness(image, 0.15)

    return image, label

dataset_train = tf.data.Dataset.from_tensor_slices((filenames_train_reduced,labels_train_reduced))
dataset_train = dataset_train.shuffle(len(filenames_train_reduced))
dataset_train = dataset_train.map(parse_function, num_parallel_calls=4)
#dataset_train = dataset_train.map(train_preprocess, num_parallel_calls=4)
#d = d.window(2)
#dataset_train = dataset_train.shuffle(len(filenames_train))
#d = d.flat_map(lambda a,b:tf.data.Dataset.zip((a,b)).batch(2))
#d = d.map(lambda a,b : (a,b[-1]))
dataset_train = dataset_train.batch(8)
dataset_train = dataset_train.prefetch(1)

dataset_test = tf.data.Dataset.from_tensor_slices((filenames_test,labels_test))
dataset_test = dataset_test.shuffle(len(filenames_test))
dataset_test = dataset_test.map(parse_function, num_parallel_calls=4)
#dataset_test = dataset_test.map(train_preprocess, num_parallel_calls=4)
#d = d.window(2)
#dataset_test = dataset_test.shuffle(len(filenames_test))
#d = d.flat_map(lambda a,b:tf.data.Dataset.zip((a,b)).batch(2))
#d = d.map(lambda a,b : (a,b[-1]))
dataset_test = dataset_test.batch(8)
dataset_test = dataset_test.prefetch(1)

dataset_val = tf.data.Dataset.from_tensor_slices((filenames_validation,labels_validation))
dataset_val = dataset_val.shuffle(len(filenames_validation))
dataset_val = dataset_val.map(parse_function, num_parallel_calls=4)
#dataset_val = dataset_val.map(train_preprocess, num_parallel_calls=4)
#d = d.window(2)
#dataset_val = dataset_val.shuffle(len(filenames_validation))
#d = d.flat_map(lambda a,b:tf.data.Dataset.zip((a,b)).batch(2))
#d = d.map(lambda a,b : (a,b[-1]))
dataset_val = dataset_val.batch(8)
dataset_val = dataset_val.prefetch(1)

# %%
tf.keras.backend.set_image_data_format('channels_last')

def create_model():

    inputs = tf.keras.layers.Input([270, 480, 3])
    inputs_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)


    base_model = MobileNetV2(include_top = False, weights = 'imagenet')(inputs_preprocessed, training =False)
    # for layer in base_model.layers:
    #     layer.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model)
    #x = tf.keras.layers.Flatten()(base_model.output)
    #x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    base_learning_rate = 0.0001

    model = tf.keras.models.Model(base_model.input, x)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate/5),
        metrics=[tf.keras.metrics.RecallAtPrecision(precision=0.9, name='acc')])

    return model

model = create_model()
model.summary()

# %%
#history = model.fit_generator(train_generator, shuffle='true', epochs=1, verbose=1, batch_size=16)
#checkpoint_path = "/home/ubuntu/checkpoints/training_3/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True, monitor='val_acc', verbose=1, 
                                                 save_best_only=True, mode='max')

history = model.fit(x=dataset_train, validation_data=dataset_val, epochs=200, 
                                verbose=1, callbacks = [cp_callback], class_weight = {0: 1 , 1:2})

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

