import numpy as np
import scipy

def realignment_coeffs(data):
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
    
    filt_data = np.zeros(polyphase_data.shape)
    for k in range(channel_num):
        filt_data[k] = scipy.signal.lfilter(filter_coeffs[k], 1, polyphase_data[k])

    dispatch_data = scipy.fft.ifft(filt_data, axis=0)
    return dispatch_data

def kernel(data,coeffs,nchannels):
       # coeffs = gen_filter_coeffs(ntaps,channel_num)  
    subfreq = polyphase_filter(data,coeffs,nchannels)
    freq_size = subfreq.shape[1]
    N = int(freq_size * nchannels // 4)
    myfreq = np.zeros((N), dtype=complex)
    start = 0
    end = 0
    mystart = 0
    myend = 0
    bw = 0
    for i in range(nchannels // 2 + 1):
        mystart += bw
        if i == 0:
            start = freq_size // 2
            end = freq_size // 2 + freq_size // 4
        elif i == nchannels // 2:
            start = freq_size // 2 - freq_size // 4
            end = freq_size // 2 
        else:
            start = freq_size // 2 - freq_size // 4
            end = freq_size // 2 + freq_size // 4
        bw = end - start
        myend = mystart + bw
        if i % 2 == 0:
            myfreq[mystart:myend] = scipy.fft.fftshift(scipy.fft.fft(subfreq[i]))[start:end]
        else:
            myfreq[mystart:myend] = scipy.fft.fft(subfreq[i])[start:end]
        
    # opfb_t = np.abs(scipy.fft.fft(myfreq))
    
    return myfreq,subfreq

def oversample_pfb(pol1,pol2,coeffs,nchannels):
    opfb1_f,sunfreq1 = kernel(pol1,coeffs,nchannels)
    opfb2_f,sunfreq2 = kernel(pol2,coeffs,nchannels)
    
    return opfb1_f,opfb2_f,sunfreq1,sunfreq2
 

