#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :main.py
@说明        :简单的手势识别，适合的摄像头输出帧的大小：640*480
@时间        :2020/04/19
@作者        :phi
@第三方库     : numpy 1.18.1; opencv-python 4.2.0
'''

from phi_contour import Find_Contour, Eucledian_Distance
from skin_detect_otsu import Bodyskin_Detect_Otsu
import cv2
import numpy as np
from time import perf_counter, sleep


length=30  #滤波长度
ges_num_buf=[0]*length  #滤波器的缓存列表
err_times=0
time_add,nor_times=0,0

def Gestures_Recognize(frame):

    skin = Bodyskin_Detect_Otsu(frame)

    ndefects,right_cont=Find_Contour(skin)

    if ndefects==0:
        ndefects=11  #返回contours为空的信息，只作调试用
        center=tuple([a//2 for a in reversed(skin.shape)])  #返回图像的中心坐标
    else:
        '''
        black2 = np.ones(skin.shape, np.uint8) #创建黑色幕布
        cv2.drawContours(black2,right_cont,-1,(255,255,255),2) #绘制白色轮廓
        cv2.imshow('right_cont',black2)
        '''
        M=cv2.moments(right_cont)
        center=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))  #手部的质心坐标

        x,y,w,h = cv2.boundingRect(right_cont)
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow('skin',skin)
    cv2.imshow('origin',frame)

    return ndefects

    #当识别不出来时打印'11'或'13'
    #print('第{}次帧处理: {:.5f}  手势表示的数字: {}  Time:{:.6f}'.format(nor_times,ndefects, real_ges_num,finish-start))




