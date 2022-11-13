import numpy as np
import time
import matplotlib.pyplot as plt
import data
import cspfb
import opfb
import psr

flag = True
pfb_type = "cspfb"

blocks = pow(2,22)

filename = "../data/J0437-4715.dada"

# load data, and skip header(4096)
ptr,file_size = data.load_data(filename)

# the num of blocks
nblock = (file_size - 4096) // (2 * blocks)

nblock = 10

if flag:
    print("the nblock is %d\n" % nblock)

psize = psr.get_period_size(400.0)

# psize = 2302919

print(psize)
ntaps = 64
nchannels = 16
num = nchannels // 2 + 1
bw = 400 / (num-1)
location = np.zeros((num),dtype=int)


pdata = np.zeros((psize))
pnum = np.zeros((psize))

start = time.time()

if pfb_type == "opfb":
    ocoeff = opfb.gen_filter_coeffs(ntaps,nchannels)
else:
    coeff = cspfb.gen_filter_coeffs(ntaps,nchannels)

for i in range(nblock):
    print("the %d block(s)" % (i+1))
    pol1,pol2 = data.read_data(ptr,blocks)
    if pfb_type == "opfb":
        p1,p2,subfreq1,subfreq2 = opfb.oversample_pfb(pol1,pol2,ocoeff,nchannels)
        psr.coherent_dedispersion_opfb(subfreq1,subfreq2,nchannels,pdata,pnum,location)
    else:
        p1,p2,subfreq1,subfreq2 = cspfb.criticalsample_pfb(pol1,pol2,coeff,nchannels)
        psr.coherent_dedispersion_cspfb(subfreq1,subfreq2,nchannels,pdata,pnum,location)
        
if pfb_type == "opfb":
    idata = psr.integral_data_opfb(pdata,psize,nchannels)
else:
    idata = psr.integral_data_cspfb(pdata,psize,nchannels)     


end = time.time()
print('程序运行时间:%s秒' % ((end - start)))

if pfb_type == "opfb":
    np.savetxt("opfb.txt",idata)
else:
    np.savetxt("cspfb.txt",idata)


plt.figure(figsize=(10,5),dpi=100)
plt.ylabel("Magnitude(dB)")
plt.xlabel("Phase")
plt.plot(np.abs(idata))
plt.show()