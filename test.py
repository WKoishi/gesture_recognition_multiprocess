from filter import FirstOrderFilter
import numpy as np
import matplotlib.pyplot as plt

"""
filter_1 = FirstOrderFilter(0.1)
f1 = 1
T = 0.01
length = 400

t = np.arange(1, length+1) * T
#t = np.expand_dims(t, axis=0)
origin = np.zeros(length)
for i in range(length):
    origin[i] = np.sin(t[i]*2*np.pi*f1)

out = np.zeros(length)
for i in range(length):
    out[i] = filter_1.Get_Filter_Res(origin[i])

plt.plot(t, origin)
plt.plot(t, out)
plt.show()
"""

koi = np.array([2,3,3,4,4,4,4,4,4,5,5,5,5,5,5,1,1,13,1,13])
sat = np.bincount(koi, minlength=6)
test = np.zeros(30, dtype=np.uint8)
print(sat)

ls = [1,2]
an = 45
ls.append(val(an))
print(id(an))
print(id(ls[2]))




