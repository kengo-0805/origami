# edge画像でコーナーを見つけるプログラム
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('fig/edge.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)

img[dst>0.01*dst.max()]=[255,0,0]

# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()

plt.imshow(img)
plt.title('cornerHarris image')
plt.show()