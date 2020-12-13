#%%
#Python script to generate arrays containing detected vehicles of each frame from a video so that we
#do not need to process videos everytime.


#importing required libraries
import glob
from imageai.Detection import VideoObjectDetection
import natsort
import numpy as np
import os
import sys


#checkout https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/VIDEO.md 
#in case of any confusion in the following function's logic 

def detection_of_vehicles_from_video(folder1,folder2,findex):

    '''
    Detects and saves the arrays containing bounding boxes of detected
    vehicles from videos of a given folder

    Parameters:
    folder1 : path of the folder containing videos
    folder2 : path of the folder in which arrays are required to be stored
    findex : index number of the first video in folder1 
    '''

    #modifying forFrame function of ImageAI to make a list 
    #of bounding box coordinates for vehichles detected in a 
    #particular frame 
    def forFrame(frame_number, output_array, output_count):
            
            bboxes = []
            
            for i in range(len(output_array)):
                bboxes.append(list(output_array[i]['box_points']))
                
            B.append(bboxes)
    
    #reading and sorting the filenames of folder1
    videos = glob.glob(folder1+'/video*.MOV')
    videos = natsort.natsorted(videos)

    #set and load ResNet Model for detection of vehicles
    execution_path = os.getcwd()
    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet() 
    #use detector.setModelTypeAsYOLOv3() to use YOLOv3 instead of RetinaNet
    detector.setModelPath(os.path.join(execution_path,"/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/resnet50_coco_best_v2.0.1.h5"))
    #use model path of yolo.h5 if to use YOLOv3 instead of RetinaNet
    detector.loadModel()
    custom_objects = detector.CustomObjects(bicycle=True, motorcycle=True,car=True,truck=True)


    for video in videos:
        print('processing' + video )
        B = []
        detector.detectCustomObjectsFromVideo(
            save_detected_video=False,
            custom_objects = custom_objects,
            input_file_path=os.path.join(execution_path, video),
            frames_per_second=30,
            per_frame_function=forFrame,
            minimum_percentage_probability=40)
        B = np.array(B)
        print('saving array for video' + video + '\n shape of array: ' + str(B.shape))
        np.save(folder2+'/array'+str(findex),B)
        findex = findex + 1

#%%
# detection_of_vehicles_from_video('/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/videos_train',
# '/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/arrays_train_v2',1)

# print('saved arrays for videos_train')

# detection_of_vehicles_from_video('/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/videos_test',
# '/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/arrays_test_v2',61)

# print('saved arrays for videos_test')

detection_of_vehicles_from_video('/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/videos_new',
'/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/arrays_new_v2',77)

print('saved arrays for videos_new')

# %%
