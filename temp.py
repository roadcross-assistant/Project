from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(bicycle=True, motorcycle=True,car=True,truck=True)

# video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/videos_train/video11.MOV"),
#                                 output_file_path=os.path.join(execution_path, "traffic_detected")
#                                 , frames_per_second=30, log_progress=True, minimum_percentage_probability=40)

video_path = detector.detectCustomObjectsFromVideo(
            input_file_path=os.path.join(execution_path, "/home/siddhi/Desktop/RoadCrossingAssistant_FY_Project_Data/videos_train/video11.MOV"),
                                output_file_path=os.path.join(execution_path, "traffic_detected")
                                , frames_per_second=30, log_progress=True, minimum_percentage_probability=40, custom_objects=custom_objects)
                                
print(video_path)
