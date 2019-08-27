import cv2
import numpy as np


class simplemaxpool():
  def __init__(self,image,stride):
      self.image = cv2.imread(image)
      self.stride = stride

  def isInBounds(self,x,y,arr):
      try:
          a = arr[x][y]
      except:
          return False
      return True

  def newdim(self,n):
      if n % self.stride == 0:
          return int (n / self.stride)
      else:
          return int(n // self.stride + 1)

  def max_pool_area(self,x,y,arr):
      q = np.array(1)
      for i in range(self.stride):
        for j in range(self.stride):
          if self.isInBounds(i+x,j+y,arr):
              q = np.append(q,arr[i+x][j+y])
          else:
              continue
      return np.amax(q)        

  def MaxPool2D(self):      
    height = self.image.shape[0]
    width = self.image.shape[1]

    red = np.zeros((height,width))
    green = np.zeros((height,width))
    blue = np.zeros((height,width))

    for i in range(height):
        for j in range(width):
            red[i][j] = self.image[i][j][0]
            green[i][j] = self.image[i][j][1]
            blue[i][j] = self.image[i][j][2]

    newHeight = self.newdim(height)
    newWidth = self.newdim(width)

    maximg = np.zeros((newHeight,newWidth,3),dtype=np.uint8)
    for k in range(0,height,self.stride):
        for l in range(0,width,self.stride):
            newk = int(k / self.stride)
            newl = int(l / self.stride)
            maximg[newk][newl][0] = self.max_pool_area(k,l,red)
            maximg[newk][newl][1] = self.max_pool_area(k,l,green)
            maximg[newk][newl][2] = self.max_pool_area(k,l,blue)

    return maximg


        
    
