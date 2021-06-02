import cv2
import pickle
import pyttsx3
import random
import numpy as np
import tensorflow as tf
import time

text_to_speech_engine = pyttsx3.init()


def load_model_from_path():

    model = tf.keras.models.load_model("C:/Users/Siddhi/Desktop/GCET/Project/savedmodels/training_deploy")
    return model

model = load_model_from_path()


def speak_command(command_text):

    text_to_speech_engine.say(command_text)
    text_to_speech_engine.runAndWait()


def initialize_commands():
    global WELCOME_COMMAND
    WELCOME_COMMAND = "Welcome to Roadcrossing Assistant! Please wait while we are loading the camera and prediction Model"
    global SAFE_COMMAND
    SAFE_COMMAND = "It is Safe to cross roads now. You are Good to go!"
    global UNSAFE_COMMAND
    UNSAFE_COMMAND = (
        "Please don't go futher, It is not safe to cross the roads."
    )
    global CLOSING_COMMAND
    CLOSING_COMMAND = "Thanks for using RoadCrossing Assistant. We hope it's been useful to you."


def generate_random_label():
    time.sleep(1)
    return random.randint(0, 1)


def cross_roads_main_func(video):

    # loaded_model = load_model_from_path()
    cap = cv2.VideoCapture(video)  # static video input
   
    # for camera input
    #cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    #no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # labels = np.load(
    #     "C:/Users/yagnesh.patil/Documents/Personal/RoadCrossingAssistant_Data/Trained Models/temp labels/labels96.npy"
    # )
    frame_count = -1
    safe_frame_count = 0
    unsafe_frame_count = 0
    safe_speak_flag = False
    unsafe_speak_flag = False
    welcome_speak_flag = False

    #model = load_model_from_path()

    #start = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        print(success, frame_count)
        cv2.namedWindow(video, cv2.WINDOW_NORMAL)
        if success:
            if frame_count == -1:
                
                while True:
                    key_init = cv2.waitKey(25)
                    cv2.imshow(video, frame)
                    if key_init == ord("a"):
                        break

            if not welcome_speak_flag:
                speak_command(WELCOME_COMMAND)
                welcome_speak_flag = True

            frame_count = frame_count + 1

            # inp = img.reshape((1, 270, 480, 3))
            # oup = loaded_model.predict(inp)[0][0]

            # oup = generate_random_label()
            image = tf.image.convert_image_dtype(frame, tf.float32)
            image = tf.image.resize(image, [270, 480], method=tf.image.ResizeMethod.AREA, 
                            preserve_aspect_ratio=True)
            print(image.shape)
            image = np.expand_dims(image, axis = 0)

            start = time.time()
            output = model.predict(image)
            end = time.time()
            print("frame ", frame_count, "-----> ", output, "time: ", end - start)
            #print(frame_count, generate_random_label())

            # if frame_count == 20:
            #     end= time.time()

            #     break

            #frame = cv2.resize(frame, cv2.WINDOW_NORMAL)

            predicted_label = output[0][0]

            #assuming that we are getting the predictions per frame
            if predicted_label > 0.6:  # safe frame
                cv2.rectangle(frame, (1576,3), (1890,95), (0,200,0), thickness=-1)
                frame_save = cv2.putText(frame, 'SAFE', (1580,90), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 6, cv2.LINE_AA)
                cv2.imshow(video, frame_save)
                cv2.waitKey(15)

                unsafe_frame_count = 0
                safe_frame_count = safe_frame_count + 1

                if safe_frame_count > 3 and not safe_speak_flag:
                    speak_command(SAFE_COMMAND)
                    safe_speak_flag = True
                    unsafe_speak_flag = False
            else:
                cv2.rectangle(frame, (1396,3), (1890,95), (0,0,200), thickness=-1)
                frame_save = cv2.putText(frame, 'UNSAFE', (1400,90), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 6, cv2.LINE_AA)
                cv2.imshow(video, frame_save)
                cv2.waitKey(15)

                safe_frame_count = 0
                unsafe_frame_count = unsafe_frame_count + 1

                if unsafe_frame_count > 3 and not unsafe_speak_flag:
                    speak_command(UNSAFE_COMMAND)
                    unsafe_speak_flag = True
                    safe_speak_flag = False

        else:
            
            break
        # print(labels)
    cap.release()
    cv2.destroyAllWindows()
    # speak_command(CLOSING_COMMAND)
    speak_command(CLOSING_COMMAND)

    #print(f"Runtime of the program is {end - start}")



if __name__ == "__main__":

    initialize_commands()
    cross_roads_main_func(
        "C:/RoadCrossingAssistant/Data/Videos/video10.MOV"
    )
