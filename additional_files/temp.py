import cv2
cap = cv2.VideoCapture("rtsp://192.168.1.254/sjcam.mov")
i = 0
while(cap.isOpened()):
    i = i+1
    print(i)
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(20)
    if i%200 == 0:
        cap.release()
        cap = cv2.VideoCapture("rtsp://192.168.1.254/sjcam.mov")
    if i==2000:
        break
cap.release()
cv2.destroyAllWindows()