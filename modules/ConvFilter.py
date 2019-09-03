import numpy as np
import cv2
import time
import datetime

class ConvFilter():
  
  edge1 = np.array([[1,0,-1],
                    [0,0,0],
                    [-1,0,1]])
  
  edge2 = np.array([[0,1,0],
                    [1,-4,1],
                    [0,1,0]])
  
  laplacian = np.array([[-1,-1,-1],
                        [-1,8,-1],
                        [-1,-1,-1]])
  
  sharpen = np.array([[0 ,-1,0],
                      [-1, 5,-1],
                      [0 ,-1,0]])
  
  gblur= np.array([[0.0625,0.125,0.0625],
                   [0.125, 0.25, 0.125],
                   [0.0625,0.125,0.0625]])
  
  bblur = np.array([[1,1,1],
                    [1,1,1],
                    [1,1,1]]) / 9
  
  bsobel = np.array([[-1,-2,-1],
                     [0,0,0],
                     [1,2,1]])
  
  lsobel = np.array([[1,0,-1],
                     [2,0,-2],
                     [1,0,-1]])
  
  rsobel = np.array([[-1,0,1],
                     [-2,0,2],
                     [-1,0,1]])
  
  tsobel = np.array([[1,2,1],
                     [0,0,0],
                     [-1,-2,-1]])
  
  emboss = np.array([[-2,-1,0],
                     [-1,1,1],
                     [0,1,2]])
  
  filter_dict = {'edge1' : edge1,
                'edge2' : edge2,
                'lapl' : laplacian,
                'sharpen' : sharpen,
                'gblur' : gblur,
                'bblur' : bblur,
                'bsobel' : bsobel,
                'lsobel' : lsobel,
                'rsobel' : rsobel,
                'tsobel' : tsobel,
                'emboss' : emboss}
  
  def __init__(self,verbose=1):
    self.verbose = verbose

  def isInBounds(self,x,y,arr):
    if x >= 0 and y >= 0:
      try:
        t = arr[x][y]
      except:
        return False
      return True
    else:
      return False
    
  def kernel_area(self,x,y,arr):
    startw = x-1
    starth = y-1 
    if startw < 0:
      startw = 0
    if starth < 0:
      starth = 0
    image_kernel = arr[startw:x+self.kernelw-1,starth:y+self.kernelh-1]
    kernel_res = sum(sum(image_kernel * self.kernel_matrix))
    
    if kernel_res < 0:
      kernel_res = 0
    elif kernel_res > 255:
      kernel_res = 255

  def transform(self,image,ktype):
    start = time.time()
    if ktype not in self.__class__.filter_dict:
      raise TypeError("The filter '" + str(ktype) + "' does not exist.")
    else:
      self.kernel_matrix = self.__class__.filter_dict[ktype]
      self.kernelh = len(self.kernel_matrix[0])
      self.kernelw = len(self.kernel_matrix[1])
      
    red = image[:,:,0]
    green = image[:,:,1]
    blue = image[:,:,2]
    
    red = np.pad(red, pad_width=1, mode='constant', constant_values=0)
    green = np.pad(green, pad_width=1, mode='constant', constant_values=0)
    blue = np.pad(blue, pad_width=1, mode='constant', constant_values=0)

    
    height = image.shape[0]
    width = image.shape[1]
    
    for k in range(1,height-1):
      for l in range(1,width-1):
        image[k][l][0] = self.new_kernel(k,l,red)
        image[k][l][1] = self.new_kernel(k,l,green)
        image[k][l][2] = self.new_kernel(k,l,blue)
    end = time.time()
    if self.verbose == 1:
      print("It took {0} for the filter '{1}' to complete transformation"
            .format(datetime.timedelta(seconds=end-start),
                    ktype))           
    return image[1:-1, 1:-1]
