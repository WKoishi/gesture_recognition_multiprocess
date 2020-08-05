import cv2
import numpy as np

skinCrCbHist=np.zeros((256,256),dtype=np.uint8)
cv2.ellipse(skinCrCbHist,(113,156),(24,16),43,0,360,255,-1)  #肤色CrCb椭圆模型

def Ellipse_Skin_Detect(frame):

    frame=cv2.blur(frame,(5,5))
    ycrcb=cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    cr=ycrcb[:,:,1]
    cb=ycrcb[:,:,2]

    result = skinCrCbHist[cr,cb]  #《这是神句》
    #result = np.zeros(cr.shape, dtype=np.uint8)
    #for i in range(cr.shape[0]):
    #    for j in range(cr.shape[1]):
    #        result[i,j] = skinCrCbHist[cr[i,j], cb[i,j]]
                
    kernel_1 = np.ones((3,3),np.uint8)
    kernel_2 = np.ones((5,5),np.uint8)
    result= cv2.erode(result,kernel_1,iterations=1)
    result= cv2.dilate(result,kernel_2,iterations=1)
    #result=cv2.medianBlur(result,3)
    
    return result