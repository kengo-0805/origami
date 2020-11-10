import cv2 
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("fig/square.png")
# グレイスケール
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# エッジ処理
edge = cv2.Canny(gray,250,400)
# エッジ処理後の画像を書き込む
cv2.imwrite("fig/edge.png", edge)

# 32bit化？
edge = np.float32(edge)
# コーナー検出
dst = cv2.cornerHarris(edge,2,3,0.04)
# 膨張処理
dst = cv2.dilate(dst,None)
# 赤い点をつける
image[dst>0.01*dst.max()] = [255,0,0]
# 表示
plt.imshow(image)
plt.title('cornerHarris image')
plt.show()



# cv2.imwrite("fig/corner.png",dst)
