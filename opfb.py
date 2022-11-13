import numpy as np
import scipy

def realignment_coeffs(data):
    """
    对系数进行隔列插零
    """
    z = np.zeros(data.shape[0])
    for i in range(data.shape[1]-1,0,-1):
        data = np.insert(data,i,z,axis=1)
    return data

def realignment_data(data,channel_num):
    disp_len = int(np.ceil(data.size / channel_num))
    patch_size = int(disp_len * channel_num - data.size)
    patch_data = np.concatenate((data, np.zeros(patch_size)))
    polyphase_data = np.zeros(patch_data.size*2) 
    half = (channel_num // 2)
    for i in range(patch_data.size // half):
        if(i == (patch_data.size // half - 1)):
            polyphase_data[i*channel_num+half:(i+1)*channel_num] = patch_data[i*half:(i+1)*half]
        else:
            polyphase_data[i*channel_num+half:(i+1)*channel_num+half] = list(patch_data[i*half:(i+1)*half])*2
    polyphase_data = polyphase_data.reshape( (channel_num, -1), order='F')
    polyphase_data = np.flip(polyphase_data,0)
    return polyphase_data

def gen_filter_coeffs(numtaps, M):
    coeffs = scipy.signal.firwin(numtaps*M, cutoff=2.0/M, window="hamming")
    coeffs = np.reshape(coeffs, (M, -1), order='F')
    coeffs = realignment_coeffs(coeffs)
    return coeffs

def polyphase_filter(data,filter_coeffs,channel_num):   
     
    polyphase_data = realignment_data(data, channel_num)
    polyphase_data = polyphase_data.reshape( (channel_num, -1), order='F')
    
    # print(polyphase_data)
    
    filt_data = np.zeros(polyphase_data.shape)
    for k in range(channel_num):
        filt_data[k] = scipy.signal.lfilter(filter_coeffs[k], 1, polyphase_data[k])
        # filt_data[k] = scipy.signal.rfilter(filter_coeffs[k], 1, filt_data[k])

    dispatch_data = scipy.fft.ifft(filt_data, axis=0)
    return dispatch_data