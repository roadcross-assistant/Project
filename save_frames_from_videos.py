#%%
import cv2
import os
import pickle
import numpy as np

user = 'aws'

if user == 'siddhi':
    path_videos = 'C:/RoadCrossingAssistant/Data/Videos/'
    path_frames = 'C:/RoadCrossingAssistant/Data/Frames/'
    path_labels_csv = 'C:/RoadCrossingAssistant/Data/labels_framewise_csv.csv'
    path_labels_list = 'C:/RoadCrossingAssistant/Data/labels_framewise_list.pkl'

elif user == 'yagnesh':
    path_videos = '/home/yagnesh/Study/Machine Learning/ML projects/RoadCrossingAssistant_Arrays/videos/'
    path_labels_csv = '/home/yagnesh/Study/Machine Learning/ML projects/RoadCrossingAssistant_Arrays/labels_framewise.csv'
    path_labels_list = '/home/yagnesh/Study/Machine Learning/ML projects/RoadCrossingAssistant_Arrays/labels_framewise.pkl'

elif user == 'aws':
    path_videos = '/home/ubuntu/Data/Videos/'
    path_labels_csv = '/home/ubuntu/Data/labels_framewise_csv.csv'
    path_labels_list = '/home/ubuntu/Data/labels_framewise_list.pkl'
    path_frames = '/home/ubuntu/Data/Frames/'
#%%

video_ids = list(range(1,105))

for id in video_ids:

    cam = cv2.VideoCapture(path_videos + "video" + str(id) + ".MOV")

    try: 
      
        # creating a folder named data 
        if not os.path.exists(path_frames + "/video" + str(id)): 
            os.makedirs(path_frames + "/video" + str(id)) 
    
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data')

    currentframe = 0
    print("starting " + path_videos + "video" + str(id) + ".MOV")
    while(True): 
        
        # reading from frame 
        ret,frame = cam.read() 
    
        if ret: 
            # if video is still left continue creating images 
            name = path_frames + "/video" + str(id) + "/frame" + str(currentframe) + '.jpg'
            #print ('Creating...' + name) 
    
            # writing the extracted images 
            cv2.imwrite(name, frame) 
    
            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 1
        else: 
            break
    
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 



#%%

def get_labels_from_video(no_frames, safe_duration_list):
    
    '''
    Get labels for a particular video 

    Parameters:
    no_frames(int) : no of frames in the given video
    safe_duration_list(list) : a list of the type [safe_start1, safe_end1, safe_start2, safe_end2,......]

    Returns:
    list : list with len = no of frames and the value at each index represents safe/unsafe at that frame_no (frame_no starting at 0)
    int : -1 if there is no safe duration in video, 1 otherwise
    '''

    labels = [0]*no_frames
    no_safe_durations = int(len(safe_duration_list)/2)
    if(no_safe_durations == 0):
        return labels,-1 # there is no safe duration in the given video so all labels marked 0
    else:

        for i in range(no_safe_durations):
            safe_start = max(safe_duration_list[i*2] - 1, 0)
            safe_end = min(safe_duration_list[i*2 +1] - 1, no_frames-1)
            labels[safe_start:safe_end+1] = [1]*(safe_end-safe_start+1) # marking the value b/w safe_start and safe_end with 1

    if len(labels) > no_frames: #len of labels cannot be greater than no_frames in video
        raise Exception('Check the labels assigned in CSV file!')
    return labels,1


open_file = open(path_labels_list, "rb")
labels_list = pickle.load(open_file)
open_file.close()
print(len(labels_list))
video_ids = list(range(1, 105))

#%%

for id in video_ids:

    video = path_videos + "video" + str(id) + ".MOV"

    print("starting " + video)
    cap = cv2.VideoCapture(video)
    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    l, f = get_labels_from_video(no_frames, labels_list[id-1])
    print(len(l))

    labels = np.array(l)
    name = path_frames + "video" + str(id) + "/labels" + str(id) + ".npy"

    np.save(name, labels)

# %%
