import cv2 
import numpy as np
from matplotlib import pyplot as plt
# from numpy.lib.type_check import imag
# from PIL import Image

# 画像の読み込み
image = cv2.imread("fig/square_risize.png")
# グレイスケール
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# エッジ処理
edge = cv2.Canny(gray,400,20)
# エッジ処理後の画像を書き込む
cv2.imwrite("fig/edge.png", edge)
# 32bit化？
edge = np.float32(edge)
# コーナー検出
dst = cv2.cornerHarris(edge,2,3,0.11)
# 膨張処理
dst = cv2.dilate(dst,None)
# 赤い点をつける
image[dst>0.01*dst.max()] = [0,0,255]
# 表示
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.title('cornerHarris image')
plt.show()



# cv2.imwrite("fig/corner.png",dst)
