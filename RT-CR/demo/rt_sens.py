import numpy as np
from scipy import signal

from settings import *

def sens_spectrum(rx2sens_out, plot2sens_out):
    while True: 
        sdr_data = rx2sens_out.recv()
        IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, _, _ = sdr_data
        freqs, IQ_ps_1  = signal.welch(IQ_t_1, fs=sample_rate, window="hanning", nperseg=None, noverlap=SWIN_OVERLAP_LEN, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", axis=-1, average="mean")
        IQ_ps_1        = 10 * np.log10(IQ_ps_1 + 1e-16)
        freqs          = np.fft.fftshift(freqs)
        IQ_ps_1        = np.fft.fftshift(IQ_ps_1)
        max_ind, _ = signal.find_peaks(IQ_ps_1, height=np.mean(IQ_ps_1), threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)
        min_ind, _ = signal.find_peaks(-IQ_ps_1, height=np.mean(-IQ_ps_1), threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)
        maxmin_ind = np.append(max_ind, min_ind)
        maxmin_ind.sort()
        noise_clas = np.ones(FFT_LEN, dtype=bool)
        cur_ind = 0
        for curmax_ind in max_ind:
            if cur_ind > curmax_ind:
                continue
            else:
                cur_ind = curmax_ind
            if np.any(min_ind < curmax_ind):
                prevmin_ind = min_ind[np.where(min_ind < curmax_ind)[0][-1]]
            else:
                prevmin_ind = 0
            if np.any(min_ind > curmax_ind):
                nextmin_ind = min_ind[np.where(min_ind > curmax_ind)[0][0]]
            else:
                nextmin_ind = -1
            noise_clas[prevmin_ind+1:nextmin_ind] = False
            if np.any((max_ind > curmax_ind) & (max_ind < nextmin_ind)):
                cur_ind = max_ind[np.where((max_ind > curmax_ind) & (max_ind < nextmin_ind))[0][-1]]
        signal_clas = ~noise_clas
        signal_clas = np.abs(IQ_ps_1.max() - IQ_ps_1.min()) * signal_clas + IQ_ps_1.min()
        plot2sens_out.send([signal_clas, max_ind, min_ind])
