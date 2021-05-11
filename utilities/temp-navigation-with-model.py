import cv2
import pickle
import pyttsx3
import random
import numpy as np
import tensorflow as tf

""" For Camera Input
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
"""

text_to_speech_engine = pyttsx3.init()


def load_model_from_path():

    model = tf.saved_model.load(
        "C:/Users/yagnesh.patil/Documents/Personal/RoadCrossingAssistant_Data/Trained Models/training_acc-77%"
    )
    func = model.signatures["serving_default"]

    return func


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
    return random.randint(0, 1)


def cross_roads_main_func(video):

    # loaded_model = load_model_from_path()
    cap = cv2.VideoCapture(video)  # static video input

    # for camera input
    # cap = cv2.VideoCapture(gstreamer_pipline(flip_method=0), cv2.CAP_GSTREAMER)

    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # labels = np.load(
    #     "C:/Users/yagnesh.patil/Documents/Personal/RoadCrossingAssistant_Data/Trained Models/temp labels/labels96.npy"
    # )
    frame_count = -1
    safe_frame_count = 0
    unsafe_frame_count = 0
    safe_speak_flag = False
    unsafe_speak_flag = False
    welcome_speak_flag = False

    predict = load_model_from_path()

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            if frame_count == -1:
                frame = cv2.resize(frame, (1080, 720))
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

            test_frame = cv2.resize(frame, (480, 270))
            test_inp = np.array([test_frame])
            x = tf.convert_to_tensor(test_inp, dtype=tf.float32)
            output = predict(x)

            frame = cv2.resize(frame, (1366, 760))

            predicted_label = tf.keras.backend.get_value(output["dense_1"])[0]

            # assuming that we are getting the predictions per frame
            if predicted_label > 0.9:  # safe frame
                cv2.rectangle(
                    frame, (1100, 3), (1360, 95), (0, 200, 0), thickness=-1
                )
                frame_save = cv2.putText(
                    frame,
                    "SAFE",
                    (1130, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (255, 255, 255),
                    6,
                    cv2.LINE_AA,
                )
                cv2.imshow(video, frame_save)
                cv2.waitKey(34)

                unsafe_frame_count = 0
                safe_frame_count = safe_frame_count + 1

                if safe_frame_count > 3 and not safe_speak_flag:
                    speak_command(SAFE_COMMAND)
                    safe_speak_flag = True
                    unsafe_speak_flag = False
            else:
                cv2.rectangle(
                    frame, (1100, 3), (1360, 95), (0, 200, 200), thickness=-1
                )
                frame_save = cv2.putText(
                    frame,
                    "UNSAFE",
                    (1130, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    4,
                    (255, 255, 255),
                    6,
                    cv2.LINE_AA,
                )
                cv2.imshow(video, frame_save)
                cv2.waitKey(34)

                safe_frame_count = 0
                unsafe_frame_count = unsafe_frame_count + 1

                if unsafe_frame_count > 3 and not unsafe_speak_flag:
                    speak_command(UNSAFE_COMMAND)
                    unsafe_speak_flag = True
                    safe_speak_flag = False

        else:
            speak_command(CLOSING_COMMAND)
            break
        # print(labels)
    cap.release()
    cv2.destroyAllWindows()
    # speak_command(CLOSING_COMMAND)


if __name__ == "__main__":

    initialize_commands()
    cross_roads_main_func(
        "C:/Users/yagnesh.patil/Documents/Personal/RoadCrossingAssistant_Data/Videos/video96.MOV"
    )
