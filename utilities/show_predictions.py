import tensorflow as tf
import cv2
import numpy as np

img = cv2.imread("C:/RoadCrossingAssistant/Data/Frames/video79/frame189.jpg")
cv2.imwrite("t.jpg",img)
model = tf.keras.models.load_model('C:/Users/Siddhi/Desktop/GCET/Project/savedmodels/approach_3_2')


frame_input = tf.image.convert_image_dtype(img, tf.float32)
frame_input = tf.image.resize(frame_input, [270, 480], method=tf.image.ResizeMethod.AREA, preserve_aspect_ratio=True)
frame_input = np.expand_dims(frame_input, axis = 0)
print("input shape: ",frame_input.shape)
print("op: ", model.predict(frame_input))