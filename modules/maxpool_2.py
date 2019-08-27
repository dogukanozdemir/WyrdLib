#BAŞARILI!
import numpy as np

class Image :
  def __init__ (self,image,split,color):
    self.image=image
    self.split=split
    self.color=color
    
    
  def max_pooling (image,split,color):
    import cv2
    import matplotlib.pyplot as plt
    image= cv2.imread(image) 
    
    
    red = np.zeros((len(image),len(image[0])),dtype=int)
    green = np.zeros((len(image),len(image[0])),dtype=int)
    blue = np.zeros((len(image),len(image[0])),dtype=int)
    white= np.zeros((len(image),len(image[0])))
    black= np.zeros((len(image),len(image[0])))
    colors=np.array((red,green,blue),dtype=int)
    wb=np.array((red,green,blue),dtype=int)
    
    for t in range(len(colors)):
      for i in range (image.shape[0]):
        for j in range (image.shape[1]):
          colors[t][i][j]=int(image[i][j][t])
          wb[t][i][j]=int((int(image[i][j][0])+int(image[i][j][1])+int(image[i][j][2]))//3)
    
    list_=[wb,colors]
    workspace=list_[color]
    b=np.zeros((split,split),dtype=int)
    
    k=(len(image[0])//len(b[0]))
    p=(len(image)//len(b[0]))
    
    max_val_b=np.zeros((1,k),dtype=int)
    max_val_u=np.zeros((p,k),dtype=int)
    max_val_tot=np.zeros((3,p,k),dtype=int)
    u=np.zeros((k,split,split))
  
    reddy=np.zeros((p,k,split,split),dtype=int)
    greeny=np.zeros((p,k,split,split),dtype=int)
    bluey=np.zeros((p,k,split,split),dtype=int)
    colorsy=[reddy,greeny,bluey]
        
    x=0
    y=0

    total=np.zeros((3,p,k,split,split),dtype=int)
    
    for tzu in range (len(workspace)):
      a=workspace[tzu]
      l=colorsy[tzu]

      s=len(a)
      d=len(a[0])

      for gu in (0,split):
        if (s%split) != 0:
          a=np.delete(a,[s-1],axis=0)
          s-=1
        else:
          pass
        if (d%split) != 0:
          a=np.delete(a,[d-1],axis=1)
          d-=1
        else:
          break
    
      for ı in range (0,p):    
        for t in range (0,k):
          for i in range (0,len(b)):
            for j in range (0,len(b[0])):
              m=(x%split)
              n=(y%split)
              b[m][n]=a[x][y]
              y+=1        
            x+=1
            y-=(len(b))
        
          max_val=np.amax(b)
          max_val_b[0][t]= max_val
          u[t]=b            
          x-=len(b[0])
          y+=len(b)
    
        max_val_u[ı]=max_val_b
        l[ı]=u        
        x+=len(b)
        y-=((len(b[0]))*k)
    
      x=0
      y=0
    
      max_val_tot[tzu]=max_val_u
    
      total[tzu]=l
  
  
    max_red=max_val_tot[0]  
    max_green=max_val_tot[1]  
    max_blue=max_val_tot[2]
    rgb=np.dstack((max_red,max_green,max_blue))
    
    plt.imshow(rgb)

