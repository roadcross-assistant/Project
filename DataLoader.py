import numpy as np
import keras
import random
import tensorflow as tf
from PIL import Image 
import cv2


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, folder_path, history=0, no_videos = 8, batch_size=32, dim=(360,640),
     n_channels=3, n_classes=2, shuffle=True):
        'Initialization'

        self.list_IDs = list_IDs
        #self.labels = labels
        self.folder_path = folder_path
        self.history = history
        self.no_videos = no_videos
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        #self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(sum(self.list_IDs.values()) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        #index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        #get 8 random video ids from list_ids
        video_ids = random.sample(list(self.list_IDs.keys()), self.no_videos)    

        no_frames_per_video = int(self.batch_size / self.no_videos)

        # Generate data
        X, y = self.__data_generation(video_ids, no_frames_per_video)

        return X, y

    # def on_epoch_end(self):
    #     'Updates indexes after each epoch'
    #     self.indexes = np.arange(len(self.list_IDs))
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)

    def __data_generation(self, video_ids, no_frames_per_video):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        ctr = 0
        # Generate data
        for vid in video_ids:

            frame_ids = random.sample(list(range(self.list_IDs[vid])), no_frames_per_video) # generate "no_frames_per_video" random frames
            #max value of frame given by self.list_ids[vid]
            labels_temp = np.load(self.folder_path + "video" + str(vid) + "/labels" + str(vid) + ".npy")

            for fid in frame_ids:

                im = Image.open(self.folder_path + "video" + str(vid) + "/frame" + str(fid) + ".jpg" )
                frame = np.asarray(im)
                frame_resized = cv2.resize(frame, (640, 360), interpolation = cv2.INTER_AREA)
                X[ctr, ] = np.asarray(frame_resized)
                y[ctr] = labels_temp[fid]
                ctr = ctr + 1

        #print(y)
        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)