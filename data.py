import numpy as np
import struct
import os

blocks = pow(2,22)

def load_data(filename):
    ptr = open(filename,"rb")
    file_size = os.path.getsize(filename)
    ptr.seek(4096,0);
    return ptr,file_size

def read_data(ptr,num = blocks):
    # read two pol
    raw_data = ptr.read(2 * num)
    data = struct.unpack('<'+str(2 * num)+'b',raw_data)
    pol1 = np.zeros((num))
    pol2 = np.zeros((num))
    for i in range(num // 4):
        pol1[4*i:4*(i+1)] = data[8*i:8*i+4]
        pol2[4*i:4*(i+1)] = data[8*i+4:8*(i+1)]
    return pol1,pol2


## test

# filename = "../data/J0437-4715.dada"
# file_size = os.path.getsize(filename)
# print(file_size)
# ptr = load_data(filename)
# pol1,pol2 = read_data(ptr)
# print(pol1[:10])
# print(pol2[:10])
# print(ptr.tell())