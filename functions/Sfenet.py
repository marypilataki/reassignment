import sys
import os
import numpy as np
import ctypes
import warnings

def calculate_cqt(x, fs, hop, n_bins, bins_per_semitone, f_min, window='hann'): # bins_per_semitone == bins per note
    # Calculation of the Harmonic Kernels
    f_cqt = np.zeros((n_bins))
    for k in range(n_bins):
        f_cqt[k] = f_min*2**(float(k)/(12*bins_per_semitone))
        delta_f_k = (2.**(1./(12.*bins_per_semitone))-1.)*f_cqt[k]
        N_k = np.floor((1./delta_f_k)*fs)
        N_k = int(N_k)  #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if window == 'rect':
            window_k = np.ones(N_k)
        elif window == 'hann':
            window_k = np.hanning(N_k)
        elif window == 'hamming':
            window_k = np.hamming(N_k)
        else:
            raise SystemError('PROBLEM IN THE CODE, unexpected & forbidden'
                                                        'value for window')
        if k == 0:
            frame_len = N_k
            kernel = np.zeros(shape=(n_bins, N_k), dtype=np.complex128)
            kersupp = np.zeros(shape=(n_bins, 2), dtype=np.float64)

        t_red = np.linspace(0, N_k-1, N_k) ## window length times (in samples)
        harmonic_func = np.exp(-2j*np.pi*f_cqt[k]/fs*t_red)
        kernel[k] = np.pad(window_k*harmonic_func,
                           (int(np.floor((frame_len - N_k)/2)), int(frame_len - (N_k+np.floor((frame_len - N_k)/2)))),
                           'constant')/N_k
        kersupp[k] = [(np.floor((frame_len - N_k)/2)),
                         (np.floor((frame_len - N_k)/2)+N_k-1)]   ## N_k changes in every iteration
                         
    # Allocating memory *******************************************************
    CQgram_len = np.ceil((len(x)-frame_len+1)/np.floor(hop*fs)) ## number of frames??
    CQgram_len = int(CQgram_len)
    CQgram = np.zeros([n_bins, CQgram_len]) 
    t_cqt = (frame_len/2.+np.arange(0, len(x)-frame_len+1, np.floor(hop*fs)))/fs
    
    # Calculating CQgram ******************************************************
    j = 0.
    j_red = 0
    j = int(j)
    j_red = int(j_red)

    while j+frame_len-1 < len(x):
        signal = x[j:j+frame_len]
        for i in range(n_bins):
            i = int(i)
            CQgram[i, j_red] = abs(sum(signal[int(kersupp[i,0]):int(kersupp[i,1]+1)]*kernel[i,int(kersupp[i,0]):int(kersupp[i,1]+1)]))**2
        j += int(np.floor(hop*fs))
        j_red += int(1)
        
    return CQgram, t_cqt, frame_len
    

def calculate_reassigned_cqt(x, fs, hop, n_bins, bins_per_semitone, f_min, filt_ord=500, window='hann'): # bins_per_semitone == bins per note

    # Calculation of the Harmonic Kernels
    f_cqt = np.zeros(n_bins)
    for k in range(n_bins):
        f_cqt[k] = f_min*2**(float(k)/(12*bins_per_semitone))
        delta_f_k = (2.**(1./(12.*bins_per_semitone))-1.) * f_cqt[k]
        N_k = np.floor((1./delta_f_k)*fs)
        N_k = int(N_k)
        if window == 'rect':
            window_k = np.ones(N_k)
        elif window == 'hann':
            window_k = np.hanning(N_k)
        elif window == 'hamming':
            window_k = np.hamming(N_k)
        else:
            raise SystemError('PROBLEM IN THE CODE, unexpected & forbidden value for window')
        
        ## prepare ideal differentiator filter
        u = np.arange(-filt_ord, filt_ord+1, 1, dtype=np.double)
        
        g = []
        for value in u:
            if value==0:
                g.append(0)
            else:
                g.append((-1)**value/value)
        g = np.asarray(g)

        # g[filt_ord] = 0
        window_k_diff = np.convolve(window_k, g)

        N_prim_k = len(window_k_diff)
        supp_size_k = N_prim_k  # /!\
        window_k = np.pad(window_k, (int(np.floor((supp_size_k - N_k)/2)),
                                    int(supp_size_k - N_k - np.floor((supp_size_k-N_k)/2))),'constant')
        window_k_diff = np.pad(window_k_diff, (int(np.floor((supp_size_k - N_prim_k)/2)),
                                    supp_size_k - N_prim_k - int(np.floor((supp_size_k-N_prim_k)/2))),'constant')
        if k == 0:
            frame_len = supp_size_k ## size of derivative of window at the first bin
            kernel = np.zeros([n_bins, frame_len], dtype=np.complex128)
            dkernel = np.zeros([n_bins, frame_len], dtype=np.complex128)
            ikernel = np.zeros([n_bins, frame_len], dtype=np.complex128)                
            kersupp = np.zeros([n_bins, 2], dtype=np.float64)

        
        t_centered = np.linspace(-np.floor(supp_size_k/2), -np.floor(supp_size_k/2)+supp_size_k-1, supp_size_k)
        
        harmonic_func = np.exp(-2j*np.pi*f_cqt[k]/fs*t_centered)
        a = np.floor((frame_len - supp_size_k)/2)
        b = frame_len - (supp_size_k+np.floor((frame_len - supp_size_k)/2))
        a = int(a)
        b = int(b)
        
        kernel[k] = np.pad(window_k*harmonic_func, (a, b), 'constant')/N_k
        dkernel[k] = np.pad(window_k_diff*harmonic_func, (a, b), 'constant')/N_k
        ikernel[k] = np.pad(t_centered/fs*window_k*harmonic_func, (a, b), 'constant')/N_k
        kersupp[k] = [np.floor((frame_len - supp_size_k)/2), np.floor((frame_len - supp_size_k)/2)+N_prim_k-1]
        
    ## kersupp serves as index, type should be int
    kersupp = kersupp.astype('int32') 
    
    # Allocating memory *******************************************************
    ReCQgram_len = np.ceil((len(x)-frame_len+1)/np.floor(hop*fs))
    ReCQgram_len = int(ReCQgram_len)
    ReCQgram = np.zeros([n_bins, ReCQgram_len])
    t_cqt = (frame_len/2.+np.arange(0, len(x)-frame_len+1, np.floor(hop*fs)))/fs
    
    j = 0
    j_red = 0

    while (j+frame_len-1) < len(x):
        signal = x[j:j+frame_len]
        for i in range(n_bins):
            CQcomplex_i_jred = sum(signal[kersupp[i,0]:kersupp[i,1]+1]*kernel[i,kersupp[i,0]:kersupp[i,1]+1])

            delta_t = np.real(sum(ikernel[i,kersupp[i,0]:kersupp[i,1]+1]         
                            * signal[kersupp[i,0]:kersupp[i,1]+1]) / CQcomplex_i_jred )                
            delta_f = fs/(2*np.pi)*np.imag(sum(dkernel[i,kersupp[i,0]:kersupp[i,1]+1]
                            *signal[kersupp[i,0]:kersupp[i,1]+1]) / CQcomplex_i_jred )            

            if(abs(delta_t) >frame_len/(2*fs)).all():
                delta_t = 0
                delta_f = 0

            t_rea = t_cqt[j_red]+delta_t
            j_rea = np.round((t_rea-t_cqt[0]) / (t_cqt[1]-t_cqt[0]))
            f_rea = f_cqt[i]-delta_f
            i_rea = np.round(np.log(np.abs(f_rea/f_cqt[0])) / np.log(f_cqt[1]/f_cqt[0]))
            
            if (i_rea >= 0).all() and (i_rea < ReCQgram.shape[0]).all() and (j_rea >= 0).all() and (j_rea < ReCQgram.shape[1]).all():
                ReCQgram[int(i_rea), int(j_rea)] += CQcomplex_i_jred*CQcomplex_i_jred.conjugate()

        j += np.floor(hop*fs)
        j = int(j)
        j_red += 1
        
    return ReCQgram, t_cqt, frame_len
        