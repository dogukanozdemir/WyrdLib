import cv2
import numpy as np
import math

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
    
  def max_pool_area(self,x,y,arr):
      q = []
      for i in range(self.stride):
        for j in range(self.stride):
          if self.isInBounds(i+x,j+y,arr):
              q.append(arr[i+x][j+y])
          else:
              continue
      return np.amax(q)        

  def MaxPool2D(self):      
    height = self.image.shape[0]
    width = self.image.shape[1]
    
    red = self.image[:,:,0]
    green = self.image[:,:,1]
    blue = self.image[:,:,2]
    
    newHeight = math.ceil(height / self.stride)
    newWidth = math.ceil(width / self.stride)

    maximg = np.zeros((newHeight,newWidth,3),dtype=np.uint8)
    for k in range(0,height,self.stride):
        for l in range(0,width,self.stride):
            newk = int(k / self.stride)
            newl = int(l / self.stride)
            maximg[newk][newl][0] = self.max_pool_area(k,l,red)
            maximg[newk][newl][1] = self.max_pool_area(k,l,green)
            maximg[newk][newl][2] = self.max_pool_area(k,l,blue)

    return maximg
