import cv2
import numpy as np
import tensorflow as tf

loaded = tf.keras.models.load_model('C:/Users/Siddhi/Desktop/GCET/Project/savedmodels/training_deploy')
print("loaded saved model")


# img = cv2.imread("C:/RoadCrossingAssistant/Data/Frames/video2/frame100.jpg")
# print(img.shape)
# img=cv2.resize(img, (480,270))
# print(img.shape)
# inp = img.reshape((1, 270, 480,3))
# print(inp.shape)

img = cv2.imread("C:/RoadCrossingAssistant/Data/Frames/video2/frame100.jpg")
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (480,270))
print(img.shape)

test_input = np.array([img])
print(test_input.shape)
batched_input = np.zeros((1, 270, 480, 3), dtype=np.float32)

image = tf.io.read_file("C:/RoadCrossingAssistant/Data/Frames/video2/frame100.jpg")
image = tf.image.decode_jpeg(image)
image = tf.image.convert_image_dtype(image, tf.float32)
image = tf.image.resize(image, [270, 480], method=tf.image.ResizeMethod.AREA, 
                            preserve_aspect_ratio=True)
print(image.shape)
image = np.expand_dims(image, axis = 0)
print(image.shape)

batched_input[0, :] = image

o = loaded.predict(image)
o1 = loaded.predict(test_input)


print(o, o1)