import numpy as np
import scipy

def apply_chirp(freq,size,bw=400.0,fc=1382.0):
    dm = 2.64476
    dm_dispersion = 2.41e-4
    dispersion_per_MHz = 1e6 * dm / dm_dispersion
    phasors = np.zeros((size),dtype=complex)
    binwidth = -bw / size
    coeff = 2 * np.pi * dispersion_per_MHz / (fc * fc);
    for i in range (size):
        f = i * binwidth + 0.5 * bw
        phasors[i] = np.exp(1j * coeff * f * f / (fc+f));
        if i == 0:
            phasors[i] = 0; 
        freq[i] = phasors[i] * freq[i];

def get_period_size(bw):
    period = 0.00575730363767324
    return int(period * bw * 1e6)

def fold_data(data,blocks,psize,pdata,pnum,location):
    cur = location
    # pdata = np.zeros((psize))
    # pnum = np.zeros((psize))
    for i in range(blocks):
        if(cur >= psize):
            cur = 0
        pnum[cur] = pnum[cur] + 1
        pdata[cur] = (pnum[cur] - 1) * pdata[cur] / pnum[cur] + data[i] / pnum[cur]
        cur = cur + 1
    location = cur
    return location    

def integral_data1(data,size,n=1024):
    data = np.abs(data)
    bins = size // n
    out = np.zeros((1024))
    sum = 0
    j = 0
    z = 0
    for i in range(bins * n):
        if j == bins:
            out[z] = sum / bins
            z += 1
            sum = 0
            j = 0
        j += 1
        sum += data[i]
    return out

def integral_data(data,size,n=1024):
    data = np.abs(data)
    bins = size // n
    out = np.zeros((1024))
    sum = 0
    index = 0
    i = 0
    while(1):
        sum = 0
        j = 0
        while(1):
            if j > bins or i > (bins * (n-1)):
                break
            sum += data[i]
            i += 1
            j += 1
        out[index] = sum / j
        index += 1
        # print(i)
        if i > (bins * (n-1)):
            break
    sum = 0
    while(1):
        if i >= size:
            break
        sum += data[i]
        i += 1
    
    out[n-1] = sum / (size - n * bins + bins);
    return out

def integral_data_opfb(data,psize,nchannels):
    num = nchannels // 2 + 1
    subpsize = psize // (num -1)
    idata = np.zeros((1024 * num))
    start = 0
    end = 0
    size = 0
    for i in range(num):
        start += size
        if i == 0 or i == (num-1):
            size = subpsize // 2
        else:
            size = subpsize
        end = start + size
        idata[i*1024:(i+1)*1024] = integral_data(data[start:end],size)
    return idata

def integral_data_cspfb(data,psize,nchannels):
    num = nchannels // 2
    subpsize = psize // num
    idata = np.zeros((1024 * num))
    start = 0
    end = 0
    size = 0
    for i in range(num):
        start += size
        size = subpsize
        end = start + size
        idata[i*1024:(i+1)*1024] = integral_data(data[start:end],size)
    return idata

def coherent_dedispersion(pol1,pol2,num):
    num = num // 2
    pol1_f = scipy.fft.fft(pol1)[:num]
    pol2_f = scipy.fft.fft(pol2)[:num]
    apply_chirp(pol1_f,num)
    apply_chirp(pol2_f,num)
    pol1_t = scipy.fft.ifft(pol1_f)
    pol2_t = scipy.fft.ifft(pol2_f)
    pol_out = np.sqrt(np.abs(pol1_t)**2 + np.abs(pol2_t)**2)
    return pol_out

def coherent_dedispersion_pfb(pol1,pol2,num):
    num = num // 2
    apply_chirp(pol1,num)
    apply_chirp(pol2,num)
    pol1_t = scipy.fft.ifft(pol1)
    pol2_t = scipy.fft.ifft(pol2)
    pol_out = np.sqrt(np.abs(pol1_t)**2 + np.abs(pol2_t)**2)
    return pol_out

def coherent_dedispersion_opfb(pol1,pol2,nchannels,pdata,pnum,location):
    freq_size = pol1.shape[1]
    num = nchannels // 2 + 1
    start = 0
    end = 0
    bandwidth = 400 / (num - 1)
    bw = 0
    size = 0
    fc = 1582
    fstart = 0
    fend = 0
    psize = 0
    for i in range(num):
        fc -= bw / 2
        if i == 0:
            start = freq_size // 2
            end = freq_size // 2 + freq_size // 4
            bw = bandwidth / 2
        elif i == 8:
            start = freq_size // 2 - freq_size // 4
            end = freq_size // 2 
            bw = bandwidth / 2
        else:
            start = freq_size // 2 - freq_size // 4
            end = freq_size // 2 + freq_size // 4
            bw = bandwidth
        size = end - start
        fc -= bw / 2
        if i % 2 == 0:
            freq1 = scipy.fft.fftshift(scipy.fft.fft(pol1[i]))[start:end]
            freq2 = scipy.fft.fftshift(scipy.fft.fft(pol2[i]))[start:end]
        else:
            freq1 = scipy.fft.fft(pol1[i])[start:end]
            freq2 = scipy.fft.fft(pol2[i])[start:end]
            
        # print(size,bw,fc)
        # myfreq = np.concatenate((myfreq,freq1),axis=0)
        apply_chirp(freq1,size,bw,fc)
        apply_chirp(freq2,size,bw,fc)
        
        p1 = scipy.fft.ifft(freq1)
        p2 = scipy.fft.ifft(freq2)
        
        p = np.sqrt(np.abs(p1)**2+np.abs(p2)**2)
        fstart += psize
        psize = get_period_size(bw)
        fend = fstart + psize
        location[i] = fold_data(p,size,psize,pdata[fstart:fend],pnum[fstart:fend],location[i])
        
def coherent_dedispersion_cspfb(pol1,pol2,nchannels,pdata,pnum,location):
    freq_size = pol1.shape[1]
    num = nchannels // 2
    bw = 400 / num    
    fc = 1582
    fstart = 0
    fend = 0
    psize = 0
    for i in range(num):
        
        fc -= bw / 2
        freq1 = scipy.fft.fft(pol1[i])
        freq2 = scipy.fft.fft(pol2[i])
        
        apply_chirp(freq1,freq_size,bw,fc)
        apply_chirp(freq2,freq_size,bw,fc)
        
        fc -= bw / 2
        p1 = scipy.fft.ifft(freq1)
        p2 = scipy.fft.ifft(freq2)
        
        p = np.sqrt(np.abs(p1)**2+np.abs(p2)**2)
        fstart += psize
        psize = get_period_size(bw)
        fend = fstart + psize
        location[i] = fold_data(p,freq_size,psize,pdata[fstart:fend],pnum[fstart:fend],location[i])