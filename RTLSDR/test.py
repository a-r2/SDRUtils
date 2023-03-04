import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
from scipy import fft
from scipy import signal
import sounddevice as sd
import time
import wave

from rtlsdr import RtlSdr

sdr = RtlSdr()

# PLOT_TYPE PARAMETERS
PLOT_TYPE  = 5
PLOT_SLEEP = 0 

# SDR PARAMETERS
sdr.sample_rate     = int(1.92e6) #Hz
sdr.center_freq     = int(90.300e6) #Hz
sdr.freq_correction = int(60) #PPM
sdr.gain            = "auto"

N_BUF           = 4096
WIN_LEN         = 64
WIN_OVERLAP_LEN = 16

Fs = sdr.sample_rate
Fc = sdr.center_freq
Ts = 1 / Fs
t  = np.linspace(0, (N_BUF - 1) * Ts, N_BUF)

COLORMAP = plt.get_cmap("hsv")

plt.ion()

if PLOT_TYPE == 0: #IQ
    fig, (ax1, ax2) = plt.subplots(2, num="IQ")
    fig.suptitle("IQ")
    while True:
        iq       = sdr.read_samples(N_BUF)
        ax1.plot(t, iq.real)
        ax2.plot(t, iq.imag)
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("I [V]")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Q [V]")
        fig.canvas.flush_events()
        ax1.clear()
        ax2.clear()

if PLOT_TYPE == 1: #CONSTELLATION
    fig, ax = plt.subplots(1, num="IQ")
    fig.suptitle("IQ")
    while True:
        iq = sdr.read_samples(N_BUF)
        ax.plot(iq.real, iq.imag, ".")
        ax.set_xlabel("I [V]")
        ax.set_ylabel("Q [V]")
        fig.canvas.flush_events()
        ax.clear()

elif PLOT_TYPE == 2: #PEAKS DETECTOR
    fig, ax = plt.subplots(1, num="Peaks detector")
    fig.suptitle("Peaks detector")
    while True:
        iq         = sdr.read_samples(N_BUF)
        f, P       = signal.welch(iq, fs=Fs, window="hanning", nperseg=WIN_LEN, noverlap=WIN_OVERLAP_LEN, nfft=N_BUF, detrend="constant", return_onesided=False, scaling="spectrum", axis=-1, average="mean")
        P_dBW      = 10 * np.log10(P)
        f          = fft.fftshift(f)
        P_dBW      = fft.fftshift(P_dBW)
        max_ind, _ = signal.find_peaks(P_dBW, height=np.mean(P_dBW), threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)
        min_ind, _ = signal.find_peaks(-P_dBW, height=np.mean(-P_dBW), threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)
        maxmin_ind = np.append(max_ind, min_ind)
        maxmin_ind.sort()
        noise_clas = np.ones(N_BUF, dtype=bool)
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
        signal_clas = np.abs(P_dBW.max() - P_dBW.min()) * signal_clas + P_dBW.min()
        ax.plot(f, P_dBW, ".-", f[max_ind], P_dBW[max_ind], "o", f[min_ind], P_dBW[min_ind], "o", f, signal_clas)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power [dB]")
        fig.canvas.flush_events()
        ax.clear()
        time.sleep(PLOT_SLEEP)

elif PLOT_TYPE == 3: #POWER SPECTRUM DISTRIBUTION
    fig, ax = plt.subplots(1, num="Histogram")
    fig.suptitle("Probability density (windowed power spectrum)")
    while True:
        iq    = sdr.read_samples(N_BUF)
        f, P  = signal.welch(iq, fs=Fs, window="hanning", nperseg=WIN_LEN, noverlap=WIN_OVERLAP_LEN, nfft=N_BUF, detrend="constant", return_onesided=False, scaling="spectrum", axis=-1, average="mean")
        P_dBW = 10 * np.log10(P)
        f     = fft.fftshift(f)
        P_dBW = fft.fftshift(P_dBW)
        P_dBW = np.append(P_dBW, np.zeros(WIN_LEN - 1))
        if WIN_OVERLAP_LEN > 0:
            P_dBW_win = np.lib.stride_tricks.sliding_window_view(P_dBW, window_shape=WIN_LEN)[::WIN_LEN - WIN_OVERLAP_LEN]
        else:
            P_dBW_win = np.lib.stride_tricks.sliding_window_view(P_dBW, window_shape=WIN_LEN)[::WIN_LEN]
        WIN_NUM = len(P_dBW_win)
        ax.set_prop_cycle("color", [COLORMAP(1.*i/WIN_NUM) for i in range(WIN_NUM)])
        for windw_ind in range(WIN_NUM):
            ax.hist(P_dBW_win[windw_ind], bins="auto", range=None, density=True, weights=None, cumulative=False, bottom=None, histtype="bar", align="mid", orientation="vertical", rwidth=None, log=False, color=None, label=None, stacked=False)
        #ax.legend(["Window " + str(i) for i in range(WIN_NUM)])
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power [dB]")
        fig.canvas.flush_events()
        ax.clear()

elif PLOT_TYPE == 4: #CROSS-CORRELATION DENSITY
    fig, ax = plt.subplots(1, num="Cross-correlation")
    fig.suptitle("Cross-correlation")
    while True:
        iq     = sdr.read_samples(N_BUF)
        iq_mag = np.abs(iq)
        noise  = np.random.random(N_BUF)
        ax.xcorr(iq_mag, iq_mag, normed=True, detrend=mlab.detrend_mean, usevlines=False, marker=".", maxlags=10)
        ax.xcorr(noise, noise, normed=True, detrend=mlab.detrend_mean, usevlines=False, marker=".", maxlags=10)
        ax.set_xlabel("Lag [sample]")
        ax.set_ylabel("Correlation coefficient [-]")
        fig.canvas.flush_events()
        ax.clear()

elif PLOT_TYPE == 5: #CROSS-SPECTRAL DENSITY
    fig, ax = plt.subplots(1, num="Cross-spectral density")
    fig.suptitle("Cross-spectral density")
    known_signal = np.exp(1j * 2 * np.pi * 1 * t)
    while True:
        iq = sdr.read_samples(N_BUF)
        ax.csd(iq, known_signal, NFFT=WIN_LEN, Fs=Fs, Fc=None, detrend="mean", window=np.hanning(WIN_LEN), noverlap=WIN_OVERLAP_LEN, pad_to=None, sides=None, scale_by_freq=None)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Cross-spectral magnitude [dB]")
        fig.canvas.flush_events()
        ax.clear()

elif PLOT_TYPE == 6: #COHERENCE
    fig, ax = plt.subplots(1, num="Coherence")
    fig.suptitle("Coherence")
    known_signal = np.exp(1j * 2 * np.pi * 1 * t)
    while True:
        iq = sdr.read_samples(N_BUF)
        ax.cohere(iq, known_signal, NFFT=WIN_LEN, Fs=Fs, Fc=0, detrend="mean", window=np.hanning(WIN_LEN), noverlap=WIN_OVERLAP_LEN, pad_to=None, sides="default")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Coherence Magnitude [-]")
        fig.canvas.flush_events()
        ax.clear()

elif PLOT_TYPE == 7: #WAVELET TRANSFORM
    fig, ax = plt.subplots(1, num="Wavelet Transform")
    fig.suptitle("Wavelet Transform")
    known_signal = np.exp(1j * 2 * np.pi * 1 * t)
    while True:
        iq  = sdr.read_samples(N_BUF)
        wvl = signal.cwt(iq, signal.ricker, np.arange(1,64))
        ax.pcolormesh(wvl)
        ax.set_xlabel("Translation [sample]")
        ax.set_ylabel("Scale (1/Frequency) [1/Hz]")
        fig.canvas.flush_events()
        ax.clear()

if PLOT_TYPE == 8: #FM DEMODULATOR
    df = 1
    wav_file = wave.open("fm_test.wav", "wb")
    wav_file.setparams((1, 2, 48000, 1, "NONE", "not compressed"))
    lpf = signal.butter(4, 20e3, btype='low', analog=False, output='sos', fs=int(Fs/5))
    while True:
        iq = sdr.read_samples(N_BUF)
        iq = signal.decimate(iq, 5)
        phi = np.angle(iq)
        y = np.diff(np.unwrap(phi)/(2*np.pi*df))
        y = np.append(y[0], y)
        y = signal.sosfilt(lpf, y)
        y = signal.decimate(y, 8)
        scaled = np.int16((y/np.max(np.abs(y))) * 32767)
        wav_file.writeframes(scaled)

if PLOT_TYPE == 9: #SOUNDDEVICE
    df = 1
    lpf = signal.butter(4, 20e3, btype='low', analog=False, output='sos', fs=int(Fs/5))
    while True:
        iq = sdr.read_samples(N_BUF)
        iq = signal.decimate(iq, 5)
        phi = np.angle(iq)
        y = np.diff(np.unwrap(phi)/(2*np.pi*df))
        y = np.append(y[0], y)
        y = signal.sosfilt(lpf, y)
        y = signal.decimate(y, 8)
        scaled = np.int16((y/np.max(np.abs(y))) * 32767)
        sd.play(scaled, 48000)
