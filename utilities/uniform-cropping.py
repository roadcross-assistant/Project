import cv2
import math

x = y = 0
crop_w = crop_h = 270
img_h = 270
img_w = 480
n = 5  # number of crops to be performed

remainig_width = img_w - crop_w
equal_part_w = math.floor(remainig_width / (n - 1))
print("Remainig Width => ", remainig_width)
print("Equal Part width => ", equal_part_w)


img = cv2.imread(
    "C:/Users/yagnesh.patil/Documents/Personal/css-js-revisions/images/earth_map.jpg"
)

img = cv2.resize(img, (480, 270))
cv2.imwrite("C:/Users/yagnesh.patil/Desktop/map.png", img)

for i in range(n):
    crop_img = img[y : y + crop_h, x : x + crop_w].copy()
    x = x + equal_part_w
    cv2.imshow("Cropped Image", crop_img)
    cv2.imwrite(
        "C:/Users/yagnesh.patil/Desktop/" + str(i + 1) + ".png", crop_img
    )
    cv2.waitKey(0)
