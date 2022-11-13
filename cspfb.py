import numpy as np
import scipy

def realignment_data(data,channel_num):
    # 数据补零，使得data个数是channel_num的整数倍
    disp_len = np.ceil(data.size / channel_num)
    patch_size = int(disp_len * channel_num - data.size)
    patch_data = np.concatenate((data, np.zeros(patch_size)))#后面补零
    
    # 多相分解与数据重排
    reshape_data = np.reshape(patch_data, (channel_num, -1), order='F')
    # filter_coeffs = np.reshape(filter_coeffs, (channel_num, -1), order='F') 
    polyphase_data = np.flipud(reshape_data)  # 输入数据上下翻转
    return polyphase_data

def gen_filter_coeffs(numtaps, M):
    coeffs = scipy.signal.firwin(numtaps*M, cutoff=1.0/M, window="hamming")
    coeffs = np.reshape(coeffs, (M, -1), order='F')
    return coeffs

def polyphase_filter(data,filter_coeffs,channel_num):   
     
    polyphase_data = realignment_data(data, channel_num)
    polyphase_data = polyphase_data.reshape( (channel_num, -1), order='F')
    
    filt_data = np.zeros(polyphase_data.shape)
    for k in range(channel_num):
        filt_data[k] = scipy.signal.lfilter(filter_coeffs[k], 1, polyphase_data[k])

    dispatch_data = scipy.fft.ifft(filt_data, axis=0)
    return dispatch_data