import numpy as np

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

