import cv2
import numpy as np
import math
import time
import datetime

class simplemaxpool():
  
  def __init__(self,image,verbose=1):
    self.image = cv2.imread(image)
    self.verbose = verbose

  def isInBounds(self,x,y,arr):
    try:
        a = arr[x][y]
    except:
        return False
    return True
  def max_pool_area(self,x,y,arr):
    q = []
    for i in range(self.strideX):
      for j in range(self.strideY):
        if self.isInBounds(i+x,j+y,arr):
          q.append(arr[i+x][j+y])
        else:
          continue
    return np.amax(q)        

  def MaxPool2D(self,strides=(3,3)):
    start = time.time()
    self.strideX = strides[0]
    self.strideY = strides[1]
    height = self.image.shape[0]
    width = self.image.shape[1]
    
    red = self.image[:,:,0]
    green = self.image[:,:,1]
    blue = self.image[:,:,2]
    
    newHeight = math.ceil(height / self.strideX)
    newWidth = math.ceil(width / self.strideY)

    maximg = np.zeros((newHeight,newWidth,3),dtype=np.uint8)
    for k in range(0,height,self.strideX):
      for l in range(0,width,self.strideY):
        newk = int(k / self.strideX)
        newl = int(l / self.strideY)
        maximg[newk][newl][0] = self.max_pool_area(k,l,red)
        maximg[newk][newl][1] = self.max_pool_area(k,l,green)
        maximg[newk][newl][2] = self.max_pool_area(k,l,blue)
    end = time.time()
    
    if self.verbose == 1:
      print("It took {0} for a stride of {1} to complete maxpooling.".
            format(datetime.timedelta(seconds=end-start),
                   strides))
    return maximg
