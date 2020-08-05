from os import getpid
import cv2
import numpy as np
from multiprocessing import Process, Manager, Lock
from gestures_recognition import Gestures_Recognize
from time import perf_counter


# 向共享缓冲栈中写入数据:
def Cam_Write(stack, top: int, lock, STOP) -> None:
    """
    :param stack: Manager.list对象
    :param top: 缓冲栈容量
    :return: None
    """
    print('Process to write: %s' % getpid())
    cap = cv2.VideoCapture(0)
    while 1:
        ret, frame = cap.read()
        if ret:
            lock.acquire()
            stack.append(frame)
            # 每到一定容量删除栈尾元素
            # 注意防止内存溢出
            if len(stack) >= top:
                stack.pop(0)
            lock.release()
        
        if STOP.value == 1:
            break
    
    cap.release()


# 在缓冲栈中读取数据并进行识别:
def Recognize(cap_stack, result_list, lock, STOP, count) -> None:
    print('Process to read: %s' % getpid())
    while 1:
        lock.acquire()
        if len(cap_stack) > 0:
            frame = cap_stack.pop()  #当栈不为空时，从栈中拿出图片
            lock.release()
            
            gesture = Gestures_Recognize(frame)  #进行手势识别

            lock.acquire()
            result_list.append(gesture)  #将本次识别的结果放入列表，在其他进程中进行处理
            lock.release()
            
            #测试用###############################
            with lock:
                count.value += 1
                print(count.value)
                if count.value >= 200:
                    STOP.value = 1
                    break
            #####################################
        else:
            lock.release()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            STOP.value = 1
            break
        if STOP.value == 1:
            break



#输出处理
def Result_Process(result_list, lock, STOP) -> None:
    print('Process to result process: %s' % getpid())
    count = 0
    filter_len = 30
    filter_buff = np.zeros(filter_len, dtype=np.uint8)  #储存过去30次的识别结果
    while 1:
        # 在新获得4个及以上的手势识别结果时进行处理
        if filter_len > len(result_list) >= 4: 
            lock.acquire()
            length = len(result_list)
            filter_buff = np.roll(filter_buff, length)  #将滤波数组的元素向后滚动length个单位
            for i in range(length):
                filter_buff[i] = result_list[length-1-i]  #用新获得的数据替换掉滚动后数组前部的旧数据
            del(result_list[:])  #清空列表
            lock.release()

            hist = np.bincount(filter_buff)  #统计数组中各个手势的出现次数
            max_count = np.max(hist)
            #当某一手势出现的次数大于数组长度的一半时，则认为该手势是可信的
            if max_count >= int(filter_len/2):
                result_val = np.argmax(hist)
                #print(result_val)

        if STOP.value == 1:
            break



if __name__ == '__main__':
    lock = Lock()  #进程锁
    # 父进程创建缓冲栈，并传给各个子进程：
    cap_stack = Manager().list()  #摄像头图片栈区
    result_list = Manager().list()  #手势判断结果输出列表
    STOP = Manager().Value('d',0)  #总停止标志
    Count = Manager().Value('d',0)  #测试用

    #摄像头读取图片进程
    pw = Process(target=Cam_Write, args=(cap_stack, 10, lock, STOP))

    #手势识别进程
    pr1 = Process(target=Recognize, args=(cap_stack, result_list, lock, STOP, Count))
    pr2 = Process(target=Recognize, args=(cap_stack, result_list, lock, STOP, Count))

    #手势识别结果的处理进程
    pRp = Process(target=Result_Process, args=(result_list, lock, STOP))

    pw.start()

    start = perf_counter()
    pr1.start()
    pr2.start()
    pRp.start()

    pr1.join()
    pr2.join()
    finish = perf_counter()

    pRp.join()
    pw.join()

    print(finish-start)

