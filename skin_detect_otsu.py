#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :skin_detect_otsu.py
@说明        :OTSU算法及肤色识别
@时间        :2020/04/19
@作者        :phi
@第三方库     : numpy 1.18.1; opencv-python 4.2.0
'''

import cv2
import numpy as np

BG_model = cv2.createBackgroundSubtractorKNN(history=300,detectShadows=False)  #背景移除KNN模型

'''背景移除'''
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


'''利用OTSU算法计算直方图中设定区间[sta:fin]的最佳阈值'''
def OTSU_AdaptiveThreshold(img,sta,end):

    hist = cv2.calcHist([img],[0],None,[end-sta],[sta,end])
    hist = np.squeeze(hist)
    total = np.sum(hist)
    current_max, threshold = 0, 0
    sumF, sumB = 0, 0
    sumT = np.inner(hist,np.arange(0,end-sta))
    weightB, weightF = 0, 0
    varBetween, meanB, meanF = 0, 0, 0

    for i in range(0,end-sta):
        weightB += hist[i]
        weightF = total - weightB
        if weightF == 0:
            break
        sumB += i*hist[i]
        sumF = sumT - sumB
        meanB = sumB/weightB
        meanF = sumF/weightF
        varBetween = weightB * weightF
        varBetween *= (meanB-meanF)*(meanB-meanF)
        if varBetween > current_max:
            current_max = varBetween
            threshold = i 
    threshold += sta+1

    return threshold


'''肤色识别'''
def Bodyskin_Detect_Otsu(frame):
    frame = Remove_Background(frame)
    ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:,:,1]
    
    cr = cv2.GaussianBlur(cr,(7,7),0)
    thresh = OTSU_AdaptiveThreshold(cr,127,168)  #在给定的范围内计算阈值
    _,skin = cv2.threshold(cr,thresh,255,cv2.THRESH_BINARY)
    _,skin2 = cv2.threshold(cr,168,255,cv2.THRESH_BINARY_INV)  #上限
    skin = cv2.bitwise_and(skin,skin2)
    
    kernel = np.ones((5,5),np.uint8)
    skin = cv2.morphologyEx(skin,cv2.MORPH_OPEN,kernel)
    skin = cv2.morphologyEx(skin,cv2.MORPH_CLOSE,kernel)
    
    return skin
