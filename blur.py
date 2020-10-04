import numpy as np
import cv2

# this function was copied
def read_image(path, as_float=True):
    
    # CV2 reads in BGR by default, ::-1 will reverse the channel dimensions to RGB.
    image = cv2.imread(path)[..., ::-1]
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if as_float:
        # Typically easier to work with when range is [0.0, 1.0]
        return image / 255.0
    
    return image

# Build a 2D gaussian matrix 
def gauss2D(std):
  krad = int(3 * std)
  ksize = 2 * krad + 1
  kernel = np.zeros((ksize, ksize))
  for i in range(ksize):
      for j in range(ksize):
          x = i - krad
          y = j - krad
          kernel[i][j] = 1.0 / (2 * np.pi * std ** 2) * np.exp(-(x ** 2 + y ** 2) / (2.0 * std ** 2))
  return kernel

# filter a given image using a Sobel operator
def filterImageSobel(img, kernel):
  Mx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
  My = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
  gradient = np.zeros((img.size[0], img.size[1]))

  for i in range(img.size[0]):
    for j in range(img.size[1]):
      print(Mx)
      print(My)
      print(img[i:i+2, j:j+2])
      print(Mx*img[i:i+2, j:j+2])
      Gx = sum(Mx*img[i:i+2, j:j+2])
      Gy = sum(My*img[i:i+2, j:j+2])

      gradient[i+1, j+1] = sqrt(Gx^2 + Gy^2)
  return gradient

