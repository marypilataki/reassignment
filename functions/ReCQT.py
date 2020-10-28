import sys
import os
sys.path.append('C:\\Users\\maryp\\Documents\\envs\\representations\\Lib\\site-packages')
import librosa
from librosa import display
from librosa.filters import get_window
import numpy as np
import matplotlib.pyplot as plt

## Signal Processing parameters
hop_length = 512
n_octaves = 7
bins_per_semitone = 1
bins_per_octave = 12 * bins_per_semitone
n_bins = n_octaves * bins_per_octave


def cqt_response(y, n_fft, hop_length, fft_basis, mode, fs):
    '''Compute the filter response with a target STFT hop.'''

    # Compute the STFT matrix
    D = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length,
             window='ones',
             pad_mode=mode)
    
    f, t, m = librosa.core.reassigned_spectrogram(y, sr=fs, S=D, n_fft=n_fft, hop_length=hop_length, 
    window='ones', center=True, reassign_frequencies=True, reassign_times=True, ref_power=1e-06, pad_mode=mode)
    
    re_D, _, _ = re_matrix(t, f, m, fs, hop_length, win_length=n_fft)

    # And filter response energy
    return fft_basis.dot(re_D)
    
'''
Converts times and frequencies to time frames and frequency bins and creates the reassigned spectrogram matrix.
Input parameters: 
t: reassigned times in seconds for each time-frequency point
f: reassigned frequencies in hz for each time-frequency point
m: stft magnitudes
fs: sampling rate in hz
hop_length: hop length in samples
win_length: window length in sample

Returns:
re_stft: the reassigned spectrogram
re_time_frames: reassigned time frame number for each time-frequency point
re_frequency_bins: reassigned frequency bin number for each time-frequency point
'''
def re_matrix(t, f, m, fs, hop_length, win_length):
    ## convert from seconds to time frames
    re_time_frames = np.zeros(t.shape)
    re_time_frames = np.where(np.isnan(t), np.nan, np.ceil(t * fs / hop_length))
    
    ## convert from frequency (hz) to frequency bins
    re_frequency_bins = np.zeros(f.shape)
    re_frequency_bins = np.where(np.isnan(f), np.nan, np.ceil(f * win_length / fs))
    
    ## create the new reassigned stft matrix
    re_stft = np.zeros(m.shape)
    for i in range(f.shape[0]):     # freq bins
        for j in range(f.shape[1]): # time frames 
            if (np.isnan(f[i,j])) or (np.isnan(t[i,j])):
                re_stft[i,j] += m[i,j]
            else:
                f_b = int(re_frequency_bins[i, j]-1)
                t_f = int(re_time_frames[i, j]-1)
                mag = m[i, j]
                re_stft[f_b, t_f] += mag
            
    return re_stft, re_time_frames, re_frequency_bins
    
def reassigned_cqt(y, fs, n_bins, n_octaves, bins_per_octave, hop_length, window_name='hann', pad_mode='reflect'):
    n_filters = min(bins_per_octave, n_bins)
    fmin = librosa.note_to_hz('C1')

    n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))
    win_length = n_fft
    window_name = window_name

    #hann_bandwidth = 1.50018310546875
    len_orig = len(y)

    freqs_topoctave = librosa.cqt_frequencies(n_bins, fmin, bins_per_octave=bins_per_octave)[-bins_per_octave:]

    # freqs: frequencies of the top octave
    fmin_t = np.min(freqs_topoctave)
    fmax_t = np.max(freqs_topoctave)

    # Determine required resampling quality
    filter_scale = 1
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)

    filter_cutoff = fmax_t * (1 + 0.5 * librosa.filters.window_bandwidth(window_name) / Q)
    #filter_cutoff = fmax_t * (1 + 0.5 * hann_bandwidth / Q)

    nyquist = fs / 2.0

    auto_resample = False
    res_type = None
    
    if not res_type:
        auto_resample = True
        if filter_cutoff < librosa.audio.BW_FASTEST * nyquist:
            res_type = 'kaiser_fast'
        else:
            res_type = 'kaiser_best'

    scale = True
    y, fs, hop_length = librosa.constantq.__early_downsample(y, fs, hop_length,
                                           res_type,
                                           n_octaves,
                                           nyquist, filter_cutoff, scale)
                                           
    cqt_resp = []

    if auto_resample and res_type != 'kaiser_fast':

        # Do the top octave before resampling to allow for fast resampling
        fft_basis, n_fft, _ = librosa.constantq.__cqt_filter_fft(fs, fmin_t,
                                               n_filters,
                                               bins_per_octave,
                                               filter_scale,
                                               norm=1,
                                               sparsity=0.01,
                                               window=window_name)

        # Compute the CQT filter response and append it to the stack
        cqt_resp.append(cqt_response(y, n_fft, hop_length, fft_basis, pad_mode, fs))

        fmin_t /= 2
        fmax_t /= 2
        n_octaves -= 1

        filter_cutoff = fmax_t * (1 + 0.5 * librosa.filters.window_bandwidth(window_name) / Q)

        res_type = 'kaiser_fast'

    # Make sure our hop is long enough to support the bottom octave
    num_twos = librosa.constantq.__num_two_factors(hop_length)
    
    if num_twos < n_octaves - 1:
        raise ParameterError('hop_length must be a positive integer '
                             'multiple of 2^{0:d} for {1:d}-octave CQT'
                             .format(n_octaves - 1, n_octaves))
    
    # Now do the recursive bit
    fft_basis, n_fft, _ = librosa.constantq.__cqt_filter_fft(fs, fmin_t,
                                           n_filters,
                                           bins_per_octave,
                                           filter_scale,
                                           norm=1,
                                           sparsity=0.01,
                                           window=window_name)

    my_y, my_sr, my_hop = y, fs, hop_length

    # Iterate down the octaves
    for i in range(n_octaves):

        # Resample (except first time)
        if i > 0:
            if len(my_y) < 2:
                raise ParameterError('Input signal length={} is too short for '
                                     '{:d}-octave CQT'.format(len_orig,
                                                              n_octaves))

            my_y = librosa.audio.resample(my_y, 2, 1,
                                  res_type=res_type,
                                  scale=True)
            # The re-scale the filters to compensate for downsampling
            fft_basis[:] *= np.sqrt(2)

            my_sr /= 2.0
            my_hop //= 2

        # Compute the cqt filter response and append to the stack
        cqt_resp.append(cqt_response(my_y, n_fft, my_hop, fft_basis, mode=pad_mode, fs=fs))

    C = librosa.core.constantq.__trim_stack(cqt_resp, n_bins, dtype = librosa.util.dtype_r2c(y.dtype))

    if scale:
        lengths = librosa.filters.constant_q_lengths(fs, fmin,
                                             n_bins=n_bins,
                                             bins_per_octave=bins_per_octave,
                                             window=window_name,
                                             filter_scale=filter_scale)
        C /= np.sqrt(lengths[:, np.newaxis])
        
    return C
