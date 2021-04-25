import cv2
import pickle
import pyttsx3

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

    model_path = "C:/Users/yagnesh.patil/Documents/Personal/RoadCrossingAssistant_Data/Trained Models/approach-2.2-classifier.pkl"
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    return model


def speak_command(command_text):

    text_to_speech_engine.say(command_text)
    text_to_speech_engine.runAndWait()


def initialize_commands():
    global WELCOME_COMMAND
    WELCOME_COMMAND = "Welcome to Roadcross Assistant! Please wait while we are loading the camera and prediction Model"
    global SAFE_COMMAND
    SAFE_COMMAND = "It is Safe to cross roads now. You are Good to go!"
    global UNSAFE_COMMAND
    UNSAFE_COMMAND = (
        "Please don't go futher, It is not safe to cross the roads."
    )
    global CLOSING_COMMAND
    CLOSING_COMMAND = (
        "Thanks for RoadCrossing Assistant. We hope it's been useful to you."
    )


def cross_roads_main_func(video):

    loaded_model = load_model_from_path()
    cap = cv2.VideoCapture(video)  # static video input

    # for camera input
    # cap = cv2.VideoCapture(gstreamer_pipline(flip_method=0), cv2.CAP_GSTREAMER)

    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    labels = [0] * no_frames
    frame_count = -1
    safe_frame_count = 0
    unsafe_frame_count = 0
    safe_speak_flag = False
    unsafe_speak_flag = False

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frame_count = frame_count + 1

            img = cv2.resize(frame, (480, 270))
            inp = img.reshape((1, 270, 480, 3))
            oup = loaded_model.predict(inp)[0][0]

            # assuming that we are getting the predictions per frame
            labels[frame_count] = oup

            if oup == 1:  # safe frame
                unsafe_frame_count = 0
                safe_frame_count = safe_frame_count + 1

                if safe_frame_count > 3 and not safe_speak_flag:
                    speak_command(SAFE_COMMAND)
                    safe_speak_flag = True
                    unsafe_speak_flag = False
            else:
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


if __name__ == "__main__":

    initialize_commands()
    cross_roads_main_func(
        "/home1/RoadCrossingAssistant_FY_Project_Data/videos/video2.MOV"
    )
