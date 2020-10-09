import numpy as np
import cv2

# this function was copied
def read_image(path, as_float=True):
    
    # CV2 reads in BGR by default, ::-1 will reverse the channel dimensions to RGB.
    image = cv2.imread(path)[..., ::-1]
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if as_float:
        # Typically easier to work with when range is [0.0, 1.0]
        return img_gray / 255.0
    
    return img_gray

# Build a 2D gaussian matrix 
def gauss2D(std):
  krad = int(1 * std)
  ksize = 2 * krad + 1
  kernel = np.zeros((ksize, ksize))
  for i in range(ksize):
      for j in range(ksize):
          x = i - krad
          y = j - krad
          kernel[i][j] = 1.0 / (2 * np.pi * std ** 2) * np.exp(-(x ** 2 + y ** 2) / (2.0 * std ** 2))
  return kernel

# gaussian filter
def filterImg(img, kernel):
  klength = kernel.shape[0]
  kheight = kernel.shape[1]

  # pad the image with 0s so we can perform a full blur
  img = np.pad(img, (klength-1,kheight-1), mode='constant')

  ilength = img.shape[0]
  iheight = img.shape[1]

  # we cannot start in the top right corner as our filter only begins where the kernel can fit
  i = klength // 2
  j = kheight // 2

  # Result image
  result = np.zeros([ilength, iheight])

  #no need to process pixels that will have purely 0s
  while i < ilength - klength // 2:
    while j < iheight - kheight // 2:
      
      # Build up the same dimensions of image pixels so we can perform a kernel img multiply
      toFilter = img[i-klength//2 : i + 1+klength//2, j - kheight//2 : j + 1+kheight //2]
      pixel = np.multiply(toFilter, kernel)

      # WARNING: POTENTIALLY NEED TO DIVIDE???
      result[i][j] = np.sum(pixel)
      j+=1
    i+=1

    # Must reset the j with every incrementation of i
    j = kheight // 2
  
  # remove the borders of image
  return result[klength-1:img.shape[0]-klength+1, kheight-1:img.shape[1]-kheight+1]

# filter a given image using a Sobel operator
def sobelFilter(img):
  Mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  My = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
  imageMx = filterImg(img, Mx)
  imageMy = filterImg(img, My)

  Gx = np.power(imageMx, 2)
  Gy = np.power(imageMy, 2)

  return np.sqrt(Gx + Gy)

def initial_threshold(grad_img):
  numerator = np.sum(grad_img)
  denominator = grad_img.shape[0]*grad_img.shape[1]
  return numerator/denominator

def temp(grad_img, tiLast):
  lowerClass = []
  upperClass = []
  for i in range(grad_img.shape[0]):
    for j in range(grad_img.shape[1]):
      p = grad_img[i][j]
      if p < tiLast:
        lowerClass.append(p)
      else:
        upperClass.append(p)
  mL = np.sum(lowerClass)/len(lowerClass)
  mH = np.sum(upperClass)/len(upperClass)

  # return ti
  return (mL + mH)/2

def thresholdCalc(grad_img):

  # chosen epsilon
  epsilon = 0.0001
  output = grad_img.copy()
  tiLast = initial_threshold(grad_img)
  ti = temp(grad_img, tiLast)
  
  while abs(ti - tiLast) > epsilon:
    tiLast = ti
    ti = temp(grad_img, tiLast)

  for i in range(grad_img.shape[0]):
    for j in range(grad_img.shape[1]):
      p = grad_img[i][j]
      if p < ti:
        output[i][j] = 0
      else:
        output[i][j] = 255
  return output

def connected_labeling(img):
  labeled_img = np.zeros((img.shape[0], img.shape[1]))
  queue = []
  curr_label = 1

  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if img[i][j] == 255 and labeled_img[i][j] < 1:
        labeled_img[i][j] = curr_label
        queue.append((i, j))

        while len(queue) > 0:
          popped = queue[0]
          queue = queue[1:]

          for k in range(-1,2):
            for l in range(-1,2):
              if -1 < popped[0]+k < img.shape[0] and -1 < popped[1]+l < img.shape[1]:
                if img[popped[0] + k][popped[1] + l] == 255 and labeled_img[popped[0] + k][popped[1] + l] < 1:
                  labeled_img[popped[0] + k][popped[1] + l] = curr_label
                  queue.append((popped[0]+k, popped[1]+l))
        curr_label += 1
  return curr_label
