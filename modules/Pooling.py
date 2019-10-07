import cv2
import numpy as np
import math
import time
import datetime

class Pooling():
  
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
      return np.amax(arr[x:x+self.kernelX,y:y+self.kernelY]) 
    elif self.pooling_type == 'average':
      return np.average(arr[x:x+self.kernelX,y:y+self.kernelY])
    else:
      raise TypeError("The pooling type '" + str(self.pooling_type) + "' does not exist.\nEither use 'max' or 'average'")
      
  def Pool2D(self,image,pooling_type,kernel_size=(3,3),stride=(3,3)):
    start = time.time()
    self.pooling_type = pooling_type
    
    if any(x <= 0 for x in kernel_size) or any(x <= 0 for x in stride):
      raise TypeError('Neither kernel size nor stride can have a value less than 1')
      
    self.strideX = stride[0]
    self.strideY = stride[1]
    
    self.kernelX = kernel_size[0]
    self.kernelY = kernel_size[1]
    
    height = image.shape[0]
    width = image.shape[1]
    
    try:
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
        print("It took {0} for a kernel of {1} with strides {2} to complete {3}-pooling.".
              format(datetime.timedelta(seconds=end-start),
                     kernel_size,
                     stride,
                     self.pooling_type))
      return maximg
    
    except IndexError as e:
      print("Index Error: ", e)
      print("Taking Image as one channel image")
      channel = image[:,:]
      
      newHeight = math.ceil(height / self.strideX)
      newWidth = math.ceil(width / self.strideY)

      maximg = np.zeros((newHeight,newWidth),dtype=np.uint8)
      for k in range(0,height,self.strideX):
        for l in range(0,width,self.strideY):
          newk = int(k / self.strideX)
          newl = int(l / self.strideY)
          maximg[newk][newl] = self.max_pool_area(k,l,channel)
      end = time.time()

      if self.verbose == 1:
        print("It took {0} for a kernel of {1} with strides {2} to complete {3}-pooling.".
              format(datetime.timedelta(seconds=end-start),
                     kernel_size,
                     stride,
                     self.pooling_type))
      return maximg
      

