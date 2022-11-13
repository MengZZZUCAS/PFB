import numpy as np
import time
import matplotlib.pyplot as plt
import data
import cspfb
import opfb
import psr

# from .cspfb

flag = True

blocks = pow(2,22)

filename = "../data/J0437-4715.dada"

# load data, and skip header(4096)
ptr,file_size = data.load_data(filename)

# the num of blocks
nblock = (file_size - 4096) // (2 * blocks)

# nblock = 10

if flag:
    print("the nblock is %d\n" % nblock)

psize = psr.get_period_size(400.0)
location = 0
pdata = np.zeros((psize))
pnum = np.zeros((psize))

start = time.time()
for i in range(nblock):
    print("the %d block(s)" % (i+1))
    pol1,pol2 = data.read_data(ptr,blocks)
    bdata = psr.coherent_dedispersion(pol1,pol2,blocks)
    location = psr.fold_data(bdata,blocks//2,psize,pdata,pnum,location)

idata = psr.integral_data(pdata,psize)

end = time.time()
print('程序运行时间:%s秒' % ((end - start)))

plt.figure(figsize=(10,5),dpi=100)
plt.ylabel("Magnitude(dB)")
plt.xlabel("Phase")
plt.plot(np.abs(idata))
plt.show()