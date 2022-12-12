import numpy as np
import scipy
from scipy import signal

def realignment_data(data,channel_num):

    disp_len = np.ceil(data.size / channel_num)
    patch_size = int(disp_len * channel_num - data.size)
    patch_data = np.concatenate((data, np.zeros(patch_size)))
    
    reshape_data = np.reshape(patch_data, (channel_num, -1), order='F')
    # filter_coeffs = np.reshape(filter_coeffs, (channel_num, -1), order='F') 
    polyphase_data = np.flipud(reshape_data)  
    return polyphase_data

def gen_filter_coeffs(numtaps, M):
    # coeffs = scipy.signal.firwin(numtaps*M, cutoff=1.0/M, window="hamming")
    win_coeffs = scipy.signal.get_window("hamming",numtaps*M)
    sinc = scipy.signal.firwin(numtaps*M,cutoff=1.0/M,window="hamming")
    coeffs = np.zeros(win_coeffs.shape[0],dtype=complex)
    for i in range(coeffs.shape[0]):
        coeffs[i] = sinc[i] * win_coeffs[i]
    # coeffs = sinc * win_coeffs
    nv = np.arange(numtaps*M)
    for i in range(coeffs.shape[0]):
        coeffs[i] *= np.exp(1j * np.pi * nv[i] / M)

    # coeffs = np.abs(coeffs)
    coeffs = np.reshape(coeffs, (M, -1), order='F')
    return coeffs

def polyphase_filter(data,filter_coeffs,channel_num):  
    
    polyphase_data = realignment_data(data, channel_num)
    # polyphase_data = polyphase_data.reshape( (channel_num, -1), order='F')
    
    filt_data = np.zeros(polyphase_data.shape,dtype=complex)
    for k in range(channel_num):
        filt_data[k] = scipy.signal.lfilter(filter_coeffs[k], 1, polyphase_data[k])

    dispatch_data = scipy.fft.ifft(filt_data, axis=0)
    return dispatch_data

def kernel(data,coeffs,nchannels): 
    freq = polyphase_filter(data,coeffs,nchannels)
    freq_size = freq.shape[1]
    N = int(freq_size * nchannels // 2)
    myfreq = np.zeros((N), dtype=complex)
    mystart = 0
    myend = 0
    for i in range(nchannels // 2):
        myend = mystart + freq_size
        myfreq[mystart:myend] = scipy.fft.fft(freq[i])
        mystart += freq_size
        
    return myfreq,freq

def criticalsample_pfb(pol1,pol2,coeffs,nchannels):
    cspfb1_f,sunfreq1 = kernel(pol1,coeffs,nchannels)
    cspfb2_f,sunfreq2 = kernel(pol2,coeffs,nchannels)
    
    return cspfb1_f,cspfb2_f,sunfreq1,sunfreq2