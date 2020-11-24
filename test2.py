import cv2
import numpy as np
from numpy.lib.type_check import imag
from realsensecv import RealsenseCapture

def VideoCapture():
  cap = RealsenseCapture()
  # プロパティの設定
  cap.WIDTH = 640
  cap.HEIGHT = 480
  cap.FPS = 30
  # cv2.VideoCapture()と違ってcap.start()を忘れずに
  cap.start()

  while True:
      ret, frames = cap.read()  # frames[0]にRGB、frames[1]にDepthの画像がndarrayが入っている
      color_frame = frames[0]
      depth_frame = frames[1]
      # ヒートマップに変換
      depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
          depth_frame, alpha=0.08), cv2.COLORMAP_JET)
      # レンダリング
      images = np.hstack((color_frame, depth_colormap))  # RGBとDepthを横に並べて表示
      images = color_frame
      cv2.imshow('RealSense', images)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  # ストリーミング停止
  cap.release()
  cv2.destroyAllWindows()



capture = VideoCapture()

