from matplotlib import pyplot as plt
import cv2

image = cv2.imread("fig/image_corner_resizeのコピー.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.title('cornerHarris image')
plt.show()