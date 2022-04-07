import sys

import numpy as np
from PIL import Image, ImageEnhance

import matplotlib.pyplot as plt
from scipy import fft, signal

B   = int(1e3) #bandwidth (one-sided)
Fs  = int(4 * B) #sampling frequency
Ts  = 1 / Fs #sampling period
N   = int(4 * Fs) #number of samples
Tau = Fs / N #FFT resolution

SYS_ARGV = sys.argv
assert len(SYS_ARGV) < 3, "Only one image at a time"
IMG_PATH = SYS_ARGV[1]
img      = Image.open(IMG_PATH)
img      = img.convert("1")
img_mat  = np.asarray(img)

X_LEN, Y_LEN = img_mat.shape[0], img_mat.shape[1]
FREQ_STEP    = (2 * B) / (Y_LEN - 1)

assert Fs > (2 * B), "Sampling frequency (" + str(Fs) + ") must satisfy Nyquist criterion (Fs > " + str(2 * B) + ")"
assert Tau <= FREQ_STEP, "FFT resolution (" + str(Tau) + ") must be less or equal than the frequency step length (" + str(FREQ_STEP) + ")" 

times      = np.linspace(0, (N - 1) * Ts, N)
freqs      = np.linspace(-B, B, Y_LEN)
freqs_Pxx  = fft.fftshift(fft.fftfreq(N, Ts))
signal_arr = np.zeros((X_LEN,), dtype=object)
for i in range(X_LEN):
    for j in range(Y_LEN):
        if img_mat[i,j]:
            signal_arr[i] += np.exp(1j * 2 * np.pi * freqs[j] * times)

Pxx_img_mat = np.NaN * np.ones((X_LEN, N))
for i in range(X_LEN):
    _, Pxx_row       = signal.welch(signal_arr[i], fs=Fs, window="hann", nfft=N, detrend=False, scaling="spectrum", return_onesided=False)
    Pxx_img_mat[i,:] = fft.fftshift(Pxx_row)

Pxx_img_mat = np.repeat(Pxx_img_mat, int(N/X_LEN), axis=0)
plt.imshow(Pxx_img_mat)
plt.show()
