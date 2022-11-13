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

def integral_data(data,size,n=1024):
    data = np.abs(data)
    t = size // n
    d = []
    sum = 0
    j = 0
    for i in range(size):
        if j == t:
            d.append(sum)
            sum = 0
            j = 0
        j = j+1
        sum += data[i]
    return d

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
    