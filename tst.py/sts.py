import cv2
import numpy as np
import copy

img_zeros = np.zeros((240, 320, 3))
cv2.imwrite("./hoge.png", img_zeros)

img_line = copy.deepcopy(img_zeros)

img_line[0:240:5, :, 0:3] = 255
print(img_line[25, 30])
cv2.imwrite("./fuga.png", img_line) 

img = cv2.imread("./hoge.png")
print(img.shape)