import cv2
import time
from realsensecv import RealsenseCapture

video_path = 0
cap = cv2.VideoCapture(video_path)
# cap = VideoCapture()

num = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imwrite("./capture_1/graycode_{:0=2}".format(num)+".png",frame)
        print("save picture{:0=3}".format(num)+".png")
        num += 1
    else:
        break
    time.sleep(1)

cap.release()