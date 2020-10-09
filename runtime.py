import numpy as np
import cv2
from blur import gauss2D, sobelFilter, read_image, filterImg, thresholdCalc, connected_labeling
from matplotlib import pyplot as plt
std = 3
kernel = gauss2D(std)
print("Step 1: \nThe shape of the Gaussian matrix is ", kernel.shape)
image = read_image('helper.jpeg')

#print original greyscale image
print("This is the image we are working without any filters")
plt.imshow(image, cmap='gray')
plt.show()
print('Image shape', image.shape)

gaussian_filter_img = filterImg(image,kernel)
print("Step 2: \nThis is the image with a gaussian blur applied to it")
plt.imshow(gaussian_filter_img, cmap='gray')
plt.show()
print('The image dimensions are: ', gaussian_filter_img.shape)

sobel_filter_img = sobelFilter(gaussian_filter_img)
print("This is the blurred image with the sobel filter applied")
plt.imshow(sobel_filter_img, cmap='gray')
plt.show()
print('The image dimensions are: ', sobel_filter_img.shape)

thresh_image = thresholdCalc(sobel_filter_img)
print("Step 3: \nThis is the image with the threshold algorithm applied")
plt.imshow(thresh_image, cmap='gray')
plt.show()
print('The image dimensions are:', thresh_image.shape)

# Please discuss how the algorithm works for these examples and highlight its strengths and/or its weaknesses.

''' The algorithms work by first applying a gaussian kernel which is a weighted matrix of size thats determined by a chosen
standard deviation. this kernel prioritizes pixels closer to the center of the kernel thus emphasizes the 
chosen pixel most, and the further the pixel, the less effect it has on the chosen pixel. After it is applied on the 
image, the sobel filter is applied by computing once again a weighted image through its own kernels and then combining
both the vertical and horizontal filtered images to make a more obvious edges image. From this edges image, we use
the threshold algorithm to take all pixels above a given threshold and make them more obvious by making them white 
and taking very dim pixels and setting them to black. This way the obvious edges are more shown.
Strengths: it is good at finding images with high contrast edges.
Weaknessses: when there are shadows in an image, it may make the more highlited areas shown but that may turn the image
hard to understand for an average user who just sees the image after the threshold algorithm is applied. '''

image2 = read_image('Q6.png')
thresh_image = thresholdCalc(image2)
print("Q6: \nThis is the image with the threshold algorithm applied")
plt.imshow(thresh_image, cmap='gray')
plt.show()
print('The image dimensions are:', thresh_image.shape)
ans = connected_labeling(thresh_image)
print("The amount of cells in the image provided is: ", ans)

# Q6: Discuss your results and explain how one can improve the estimation.

''' The results show how many fully independent objects there are in the picture. The problem with this method of
representation is that if there is a single pixel combining 2 elements and otherwise the elements are completely, 
obviously detached, the algorithm will consider them as 1 object since it has found a related pixel connecting the 
objects. 
'''