import numpy as np
import cv2
from blur import gauss2D, filterImageSobel, read_image, gaussFilter
from matplotlib import pyplot as plt
std = 3
kernel = gauss2D(std)
# print(kernel)
print(np.shape(kernel))
print(np.sum(kernel))
image = read_image('image.png')

#print original greyscale image
plt.imshow(image)
plt.show()
print('Image shape', image.shape)
plt.draw()
plt.savefig('1.png', dpi=100)
#test print cv2.filter2d
# gaussian_filter_img = cv2.filter2D(image,-1,kernel)
# plt.imshow(gaussian_filter_img)
# plt.show()
# # plt.savefig('2.png')
# print('Image shape', gaussian_filter_img.shape)
#print the gaussian blur image
gaussian_filter_img = gaussFilter(image,kernel)
plt.imshow(gaussian_filter_img)
plt.show()
# plt.savefig('3.pn5g')
print('Image shape', gaussian_filter_img.shape)
# img = filterImageSobel(image, kernel)