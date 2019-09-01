import numpy as np
import cv2
import time
import datetime
class ConvFilter:
  
  edge1 = np.array([[1,0,-1],
                    [0,0,0],
                    [-1,0,1]])
  
  edge2 = np.array([[0,1,0],
                    [1,-4,1],
                    [0,1,0]])
  
  laplacian = np.array([[-1,-1,-1],
                        [-1,8,-1],
                        [-1,-1,-1]])
  
  sharpen = np.array([[0,-1,0],
                      [-1,5,-1],
                      [0,-1,0]])
  
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
  
  def __init__(self,image,verbose=1):
    self.image = cv2.imread(image)
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
  
  def transfrom(self,ktype):
    start = time.time()
    if ktype not in self.__class__.filter_dict:
      raise TypeError("The filter '" + str(ktype) + "' does not exist.")
    else:
      self.kernel_matrix = self.__class__.filter_dict[ktype]
      self.kernelh = len(self.kernel_matrix[0])
      self.kernelw = len(self.kernel_matrix[1])
      
    red = self.image[:,:,0]
    green = self.image[:,:,1]
    blue = self.image[:,:,2]
    
    height = self.image.shape[0]
    width = self.image.shape[1]
    
    filtered_image = np.zeros((height,width,3),dtype=np.uint8)
    for k in range(height):
      for l in range(width):
        filtered_image[k][l][0] = self.kernel_area(k,l,red)
        filtered_image[k][l][1] = self.kernel_area(k,l,green)
        filtered_image[k][l][2] = self.kernel_area(k,l,blue)
    end = time.time()
    if self.verbose == 1:
      print("It took {0} for the filter '{1}' to complete transformation"
            .format(datetime.timedelta(seconds=end-start),
                    ktype))
            
    return filtered_image