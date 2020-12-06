# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys

image = cv2.imread("fig/ipad.png", 1)

if image is None:
    # If an image was not loaded, it may cause tuple error.
    sys.exit("File not found.")

image = 255 - image
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray= np.float32(gray)

# Detect corner
dst = cv2.cornerHarris(gray,2,9,0.16)

# Expand corner
# dst = cv2.dilate(dst, None) 

# Draw red points
image[dst>0.1*dst.max()]= [0,0,255]

# Detect red pixel
coord = np.where(np.all(image == (0, 0, 255), axis=-1))

# Print coordinate
for i in range(len(coord[0])):
    print("X:%s Y:%s"%(coord[1][i],coord[0][i]))

cv2.imshow('imageWithCorner',image)
cv2.waitKey(0)