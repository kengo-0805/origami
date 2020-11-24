# cannyの値をいい感じに調整する方法
# https://qiita.com/Takarasawa_/items/1556bf8e0513dca34a19

import cv2
import datetime 
import numpy as np

# VideoCapture オブジェクトを取得
capture = cv2.VideoCapture(0)

# try:
while(True):
    # フレームの読み取り
    ret,frame=capture.read()

    # グレースケール変換
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # エッジ検出
    gray=cv2.Canny(gray,200,200)
    cv2.imshow("frame_canny",gray)

    # ハリスコーナーをここに書きたい
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    frame[dst>0.01*dst.max()]=[0,0,255]
    cv2.imshow('dst',frame)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


    # print(座標)
    cv2.waitKey(100)

# except:
#     cv2.destroyWindow("Window")

