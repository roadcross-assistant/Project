import cv2
import pickle
import pyttsx3
import random
import numpy as np
import tensorflow as tf
import time

text_to_speech_engine = pyttsx3.init()

global WELCOME_COMMAND
WELCOME_COMMAND = "Welcome to Roadcrossing Assistant!"
global SAFE_COMMAND
SAFE_COMMAND = "GO!!"
global UNSAFE_COMMAND
UNSAFE_COMMAND = "STOP!!"
global CLOSING_COMMAND
CLOSING_COMMAND = "Thanks for using RoadCrossing Assistant."

def load_model_from_path():

    model = tf.keras.models.load_model("C:/Users/Siddhi/Desktop/GCET/Project/savedmodels/training_deploy")
    return model

model = load_model_from_path()


def speak_command(command_text):

    text_to_speech_engine.say(command_text)
    text_to_speech_engine.runAndWait()


# def initialize_commands():
#     global WELCOME_COMMAND
#     WELCOME_COMMAND = "Welcome to Roadcrossing Assistant!"
#     global SAFE_COMMAND
#     SAFE_COMMAND = "GO!!"
#     global UNSAFE_COMMAND
#     UNSAFE_COMMAND = "STOP!!"
#     global CLOSING_COMMAND
#     CLOSING_COMMAND = "Thanks for using RoadCrossing Assistant."

def cross_roads_main_func(video):

    cap = cv2.VideoCapture(video)  # static video input
    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow(video, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(video, height=650, width=1156)

    result = cv2.VideoWriter('C:/RoadCrossingAssistant/Project/filename.mp4', -1 ,30, (1920,1080))

    print("showing predictions for {} with total frames {} ".format(video, no_frames))
    
    frame_count = -1
    safe_frame_count = 0
    unsafe_frame_count = 0
    safe_speak_flag = False
    unsafe_speak_flag = False
    welcome_speak_flag = False

    while cap.isOpened():

        success, frame = cap.read()
        
        #print(success, frame_count)
        
        if success:

            if frame_count == -1:
                
                while True:
                    key_init = cv2.waitKey(25)
                    cv2.imshow(video, frame)

                    if key_init == ord("a"):
                        break

            frame_count = frame_count + 1

            key_init_stop = cv2.waitKey(15)
            if key_init_stop == ord("q"):
                break

            if not welcome_speak_flag:
                speak_command(WELCOME_COMMAND)
                welcome_speak_flag = True

            frame_save = frame

            #if (frame_count % 2 == 0):

            start = time.time()
            frame_input = tf.image.convert_image_dtype(frame, tf.float32)
            frame_input = tf.image.resize(frame_input, [270, 480], method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True)
            #print(frame_input.shape)
            frame_input = np.expand_dims(frame_input, axis = 0)
            output = model.predict(frame_input)
            end = time.time()
            print("frame ", frame_count, "-----> ", output, "time: ", end - start)
            predicted_label = output[0][0]

            #assuming that we are getting the predictions per frame
            if predicted_label > 0.85:  # safe frame
                #cv2.rectangle(frame, (1576,3), (1890,95), (0,200,0), thickness=-1)
                #frame_save = cv2.putText(frame, 'SAFE', (1580,90), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 6, cv2.LINE_AA)
                # result.write(frame_save)
                # cv2.imshow(video, frame_save)
                # cv2.waitKey(34)

                unsafe_frame_count = 0
                safe_frame_count = safe_frame_count + 1

                if safe_frame_count > 3 and not safe_speak_flag:
                    speak_command(SAFE_COMMAND)
                    safe_speak_flag = True
                    unsafe_speak_flag = False

            else:
                #cv2.rectangle(frame, (1396,3), (1890,95), (0,0,200), thickness=-1)
                #frame_save = cv2.putText(frame, 'UNSAFE', (1400,90), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 6, cv2.LINE_AA)
                # result.write(frame_save)
                # cv2.imshow(video, frame_save)
                # cv2.waitKey(34)

                safe_frame_count = 0
                unsafe_frame_count = unsafe_frame_count + 1

                if unsafe_frame_count > 2 and not unsafe_speak_flag:
                    speak_command(UNSAFE_COMMAND)
                    unsafe_speak_flag = True
                    safe_speak_flag = False
                

            if safe_speak_flag:
                    cv2.rectangle(frame, (1576,3), (1890,95), (0,200,0), thickness=-1)
                    frame_save = cv2.putText(frame, 'GO!!', (1580,90), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 6, cv2.LINE_AA)

            if unsafe_speak_flag:
                    cv2.rectangle(frame, (1396,3), (1890,95), (0,0,200), thickness=-1)
                    frame_save = cv2.putText(frame, 'WAIT', (1400,90), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 6, cv2.LINE_AA)

            result.write(frame_save)
            cv2.imshow(video, frame_save)
            cv2.waitKey(15)

        else:
            
            break

    cap.release()
    result.release()
    cv2.destroyAllWindows()
    speak_command(CLOSING_COMMAND)


if __name__ == "__main__":

    cross_roads_main_func(
        "C:/RoadCrossingAssistant/Data/Videos/video19.MOV"
    )
