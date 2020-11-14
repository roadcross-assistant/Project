#%%
#importing required libraries
import glob
from imageai.Detection import VideoObjectDetection
import natsort
import numpy as np
import os
import sys
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
%matplotlib inline  
from imageai.Detection import VideoObjectDetection
import os
import sys
from random import randint
from math import ceil, sqrt
import natsort
import pandas as pd
import random
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import GridSearchCV
import sys
import cv2
from random import randint

#%%
trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTrackerByName(trackerType):
    
  # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
  
    elif trackerType == trackerTypes[1]: 
        tracker = cv2.TrackerMIL_create()
   
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
  
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
   
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
            
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
        
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
        
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
     
    return tracker


def get_direction(videoPath, trackerType, boxes_list, frame_no):

    '''
    Detects direction of each vehicle in the video and returns and a list 
    containing these directions

    Parameters:
    videoPath(string) : path of the video
    trackerType(string) : the openCV tracker type to be used  
    boxes_list(list) : list containing bounding boxes for all detected vehicles 
    frame_no : frame no of the video for which you require directions

    '''
    k = 0 #iterator for frame no
    multiTracker_back = cv2.MultiTracker_create()
    #multiTracker = cv2.MultiTracker_create()
    n = len(boxes_list)
    directions = [0]*n
    x1 = [0]*n  #x coordinates of vehicles in frame_no
    x2 = [0]*n  #x coordinates of vehicles in frame_no - 5

    #reading the video
    cap = cv2.VideoCapture(videoPath)
    success, frame = cap.read()
    if not success:
        print('Failed to read video')
        sys.exit(1)

    #the direction detection logic starts here
    while cap.isOpened() :
        success, frame = cap.read()
        if not success:
            break
        
        #saving past few frames as we will apply traking in backward direction
        #form the frame_no
        # if( k == frame_no -10 ):
        #     frame_b10 = frame
        # if ( k == frame_no - 9):
        #     frame_b9 = frame
        # if ( k == frame_no - 8):
        #     frame_b8 = frame
        # if ( k == frame_no - 7):
        #     frame_b7 = frame
        # if ( k == frame_no - 6):
        #     frame_b6 = frame
        if ( k == frame_no - 5):
            frame_b5 = frame
        if ( k == frame_no - 4):
            frame_b4 = frame
        if ( k == frame_no - 3):
            frame_b3 = frame
        if ( k == frame_no - 2):
            frame_b2 = frame
        if ( k == frame_no - 1):
            frame_b1 = frame


        if k == frame_no:
            #boxes_list = [tuple(l) for l in boxes_list]
            for i in range(n):
                box = boxes_list[i]
                x1[i] = (box[0]+2*box[2])/2
                multiTracker_back.add(createTrackerByName(trackerType), frame, box)
            
            #all the vehicles from boxes_list added in the traker instance
            #applying vehicle tracking in backward direction
                
            (success_back, boxes_back) = multiTracker_back.update(frame_b1)
            (success_back, boxes_back) = multiTracker_back.update(frame_b2)
            (success_back, boxes_back) = multiTracker_back.update(frame_b3)
            (success_back, boxes_back) = multiTracker_back.update(frame_b4)
            (success_back, boxes_back) = multiTracker_back.update(frame_b5)
            #(success_back, boxes_back) = multiTracker_back.update(frame_b6)
            #(success_back, boxes_back) = multiTracker_back.update(frame_b7)
            #(success_back, boxes_back) = multiTracker_back.update(frame_b8)
            #(success_back, boxes_back) = multiTracker_back.update(frame_b9)
            #(success_back, boxes_back) = multiTracker_back.update(frame_b10)


            for i in range(n):
                    b = boxes_back[i]
                    x2[i] = (b[0]+2*b[2])/2
                    #print('x1',i,x1[i])


            for i in range(n):
                
                x_d =  x2[i] - x1[i]
                #print(x_d)
                if x_d > 0:
                    directions[i] = 1
                elif x_d < 0:
                    directions[i] = -1

            for i in range(n):
                box = boxes_list[i]
                (x,y,w,h) = [int(v) for v in box]
                if(directions[i] == 0): # not moving
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                elif(directions[i] == 1): # direction of interest
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                elif(directions[i] == -1): # opposite to direction of interest
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

            #cv2.imshow('frame', frame)
            cv2.waitKey(0)
            break
        k = k+1
        
    cv2.destroyAllWindows()
    return np.array(directions)
        

#%%

def get_directions_from_videos(videos_folder, arrays_folder, target_folder, findex, trackerType = "KCF"):

    arrays = glob.glob(arrays_folder+'/array*.npy')
    arrays = natsort.natsorted(arrays)
    videos = glob.glob(videos_folder+'/video*.MOV')
    videos = natsort.natsorted(videos)

    for (fname, vname) in zip(arrays, videos):
        D = [[],[],[],[],[]]
        print("processing ",vname,":")
        bounding_boxes = np.load(fname, allow_pickle=True) #loading the numpy array containing all detected vehicles
        no_frames = bounding_boxes.shape[0]

        for frame_no in range(5, no_frames):

            temp = bounding_boxes[frame_no]
            boxes_list = [(b[0],b[1],b[2]-b[0],b[3]-b[1]) for b in temp]

            directions = get_direction(vname, trackerType, boxes_list, frame_no)
            D.append(directions)
        
        D = np.array(D)
        print('saving directions for video' + vname)
        np.save(target_folder+'/directions'+str(findex),D)
        findex = findex + 1

#%%
get_directions_from_videos('/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/videos_train', 
                            '/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/arrays_train',
                            '/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/directions_train', 1)

get_directions_from_videos('/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/videos_test', 
                            '/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/arrays_test',
                            '/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/directions_test', 61)



# %%
