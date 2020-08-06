import cv2
import numpy as np

BG_model = cv2.createBackgroundSubtractorKNN(history=300,detectShadows=False)  #背景移除KNN模型

#肤色CrCb椭圆模型
skinCrCbHist = np.zeros((256,256),dtype=np.uint8)
cv2.ellipse(skinCrCbHist,(113,156),(24,16),43,0,360,255,-1)


#背景移除
def Remove_Background(frame):
    fgmask = BG_model.apply(frame)
#    kernel =cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
#    kernel_2 =cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel_3 =cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    fgmask =cv2.dilate(fgmask,kernel_3,iterations=1)  #膨胀5*5
#    fgmask =cv2.erode(fgmask,kernel,iterations=1)  #腐蚀3*3
#    fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_CLOSE,kernel_2)
#    fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
#    fgmask=cv2.GaussianBlur(fgmask,(3,3),0)
    res =cv2.bitwise_and(frame,frame,mask=fgmask)
    return res


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


# def CrCb_Skin_Detect(frame):
#     #frame = Remove_Background(frame)
#     #frame=cv2.blur(frame,(5,5))
#     ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
#     cr = ycrcb[:,:,1]
#     cb = ycrcb[:,:,2]

#     cr = cv2.blur(cr, (5,5))
#     cb = cv2.blur(cb, (5,5))

#     _, cr_mask_1 = cv2.threshold(cr, 133, 0XFF, cv2.THRESH_BINARY)
#     _, cr_mask_2 = cv2.threshold(cr, 173, 0XFF, cv2.THRESH_BINARY_INV)
#     cr_mask = cv2.bitwise_and(cr_mask_1, cr_mask_2)

#     _, cb_mask_1 = cv2.threshold(cb, 77, 0XFF, cv2.THRESH_BINARY)
#     _, cb_mask_2 = cv2.threshold(cb, 127, 0XFF, cv2.THRESH_BINARY_INV)
#     cb_mask = cv2.bitwise_and(cb_mask_1, cb_mask_2)

#     result = cv2.bitwise_and(cr_mask, cb_mask)

#     return result