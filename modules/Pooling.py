import cv2
import numpy as np
import math
import time
import datetime

class simplepooling():
  
  def __init__(self,verbose=1):
    self.verbose = verbose

  def isInBounds(self,x,y,arr):
    try:
        a = arr[x][y]
    except:
        return False
    return True
  def max_pool_area(self,x,y,arr):
    if self.pooling_type == 'max':
      return np.amax(arr[x:x+self.strideX,y:y+self.strideY]) 
    elif self.pooling_type == 'average':
      return np.average(arr[x:x+self.strideX,y:y+self.strideY])
    else:
      raise TypeError("The pooling type '" + str(self.pooling_type) + "' does not exist.\nEither use 'max' or 'average'")
      
  def MaxPool2D(self,image,pooling_type,kernel_size=(3,3)):
    start = time.time()
    self.pooling_type = pooling_type
    
    self.strideX = kernel_size[0]
    self.strideY = kernel_size[1]
    
    height = image.shape[0]
    width = image.shape[1]
    
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]
    
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
      print("It took {0} for a kernel of {1} to complete {2}-pooling.".
            format(datetime.timedelta(seconds=end-start),
                   kernel_size,self.pooling_type))
    return maximg
