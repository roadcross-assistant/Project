import numpy as np
import keras
import random

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, history=0, no_videos = 8, folder_path = None, batch_size=32, dim=(360,640),
     n_channels=3, n_classes=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.history = history
        self.no_videos = no_videos
        self.videos_path = videos_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(sum(self.list_IDs.values()) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        #get 8 random video ids from list_ids
        video_ids = random.sample(list(list_IDs.keys()), self.no_videos)    

        no_frames_per_video = self.batch_size / self.no_videos

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

        # Generate data
        for i, vid in enumerate(video_ids):

            frame_ids = random.sample(list(range(self.list_IDs[vid])), no_frames_per_video) # generate "no_frames_per_video" random frames
            #max value of frame given by self.list_ids[vid]

            for fid in range(frame_ids):
                X[i, ] = np.load(self.folder_path + "/video" + str(vid) + "/frame" + str(fid) + ".jpg" )
                labels_temp = np.load(self.folder_path + "/video" + str(vid) + "/labels" + str(fid) + ".npy")
                y[i] = labels_temp[fid]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)