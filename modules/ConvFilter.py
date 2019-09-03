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
  
  def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

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
    kernel_res = 0
    for i in range(self.kernelh):
      for j in range(self.kernelw):
        if self.isInBounds(i+x-1,j+y-1,arr):
          kernel_res += arr[i+x-1][j+y-1] * self.kernel_matrix[i][j]
        else:
          continue
    if kernel_res < 0:
      kernel_res = 0
    elif kernel_res > 255:
      kernel_res = 255
      
    return kernel_res
  

  def new_kernel(self,x,y,arr):
    startw = x-1
    starth = y-1 
    if startw < 0:
      startw = 0
    if starth < 0:
      starth = 0

    image_kernel = arr[startw:x+self.kernelw-1,starth:y+self.kernelh-1]
    image_kernel = np.pad(image_kernel,self.kernel_matrix.shape[0] , self.pad_with)
    print(image_kernel)
    return sum(sum(image_kernel * self.kernel_matrix))
  
  
  def transform(self,image,ktype):
    """padding = 'same' or 'valid' """
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
    
    height = image.shape[0]
    width = image.shape[1]
    
    for k in range(height):
      for l in range(width):
        image[k][l][0] = self.new_kernel(k,l,red)
        image[k][l][1] = self.new_kernel(k,l,green)
        image[k][l][2] = self.new_kernel(k,l,blue)
    end = time.time()
    if self.verbose == 1:
      print("It took {0} for the filter '{1}' to complete transformation"
            .format(datetime.timedelta(seconds=end-start),
                    ktype))           
    return image
