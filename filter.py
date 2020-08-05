
# 一阶低通滤波器
class FirstOrderFilter:
    
    __last_baro = 0

    def __init__(self, kparam):
        self.kparam = kparam

    def Get_Filter_Res(self, value):
        baro = self.kparam * value + (1-self.kparam) * self.__last_baro
        self.__last_baro = baro
        return baro
