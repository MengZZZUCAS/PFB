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

def kernel(data,coeffs,nchannels): 
    freq = polyphase_filter(data,coeffs,nchannels)
    freq_size = freq.shape[1]
    N = int(freq_size * nchannels // 2)
    myfreq = np.zeros((N), dtype=complex)
    start = 0
    end = 0
    mystart = 0
    myend = 0
    size = 0
    for i in range(nchannels // 2 + 1):
        mystart += size
        if i == 0:
            start = freq_size // 2
            end = freq_size
        elif i == nchannels // 2:
            start = 0
            end = freq_size // 2 
        else:
            start = 0
            end = freq_size
        size = end - start
        myend = mystart + size
        myfreq[mystart:myend] = scipy.fft.fftshift(scipy.fft.fft(freq[i]))[start:end]
        
    return myfreq,freq

def criticalsample_pfb(pol1,pol2,coeffs,nchannels):
    cspfb1_f,sunfreq1 = kernel(pol1,coeffs,nchannels)
    cspfb2_f,sunfreq2 = kernel(pol2,coeffs,nchannels)
    
    return cspfb1_f,cspfb2_f,sunfreq1,sunfreq2