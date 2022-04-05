import sys

import numpy as np
from PIL import Image, ImageEnhance

import matplotlib.pyplot as plt
from scipy import fft, signal

B  = 10000
Fs = 2 * B
Ts = 1 / Fs

SYS_ARGV = sys.argv
assert len(SYS_ARGV) < 3, "Only one image at a time"
IMG_PATH = SYS_ARGV[1]
img      = Image.open(IMG_PATH)
img      = img.convert("1")
img_mat  = np.asarray(img)

X_LEN, Y_LEN = img_mat.shape[0], img_mat.shape[1]
times        = np.linspace(0, (X_LEN - 1) * Ts, X_LEN)
freqs        = np.linspace(-0.5 * B, 0.5 * B, Y_LEN)

signal_arr = np.zeros((X_LEN,), dtype=object)
for i in range(10):
    for j in range(Y_LEN):
        if img_mat[i,j]:
            signal_arr[i] += np.exp(1j * 2 * np.pi * freqs[j] * times)

print(signal_arr.shape)
print(signal_arr[0].shape)
FFT = fft.fft(signal_arr[0], n=Y_LEN)
FFT = fft.fftshift(FFT)
plt.plot(freqs, img_mat[0], freqs, FFT/np.max(np.abs(FFT)))
plt.show()
exit()

Pxx_img = np.NaN * np.ones((X_LEN,), dtype=object)
for i in range(10):
    _, Pxx     = signal.welch(signal_arr[i], fs=Fs, window="hann", nfft=Y_LEN, detrend=False, scaling="spectrum", return_onesided=False)
    Pxx_img[i] = fft.fftshift(Pxx)

plt.plot(freqs, img_mat[9,:], freqs, Pxx_img[9]/np.max(np.abs(Pxx_img[9])))
plt.show()
