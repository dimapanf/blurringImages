import numpy as np
import cv2
from blur import gauss2D, filterImageSobel, read_image
std = 0.2
kernel = gauss2D(std)
print(kernel)
print(np.shape(kernel))
print(np.sum(kernel))
image = read_image('image.png')
img = filterImageSobel(image, kernel)