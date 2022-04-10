import sys

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from scipy import fft, signal

SYS_ARGV = sys.argv #command-line arguments

# HELP
if "-h" in SYS_ARGV or "--help" in SYS_ARGV: #bandwidth (one-sided)
    print(
    "Img2RF\n\n\
    Transforms an image to its IQ representation\n\n\
    It saves a 1-D numpy array (IQ representation) of length equal to the input\n\
    image's vertical resolution where every element contains a signal in time domain,\n\
    composed by N complex IQ samples, that represents a row of the input image. By\n\
    transmitting all signals successively, it is possible to see the image in a\n\
    spectrogram (RF image) at a receiver as long as the signals are transmitted and\n\
    received using adequate TX and RX settings (B, Fs, N)\n\n\
    Command-line options:\n\n\
    \t-b\tOne-sided bandwidth (B) enforced to allocate the output IQ samples\n\
    \t-f\tSampling frequency (Fs) of the output IQ samples\n\
    \t-i\tPath to input image\n\
    \t-n\tNumber of output IQ samples per output array element (N)\n\n\
    Considerations:\n\n\
    \t1) High-resolution input images take too long to be transformed and might give\n\
    \t   poor results\n\
    \t2) The resolution of an FFT is Fs/N. Therefore, this ratio at the receiver side\n\
    \t   should be less or equal than the one resulting from the command-line options\n\
    \t   from this script (\"-f\", \"-n\")\n\
    \t3) Prior to transmitting the output signals, it is recommended using the\n\
    \t   \"-t\" option in order to verify that, indeed, the expected RF image\n\
    \t   (spectrogram from output IQ samples) is created. If the RF image is not\n\
    \t   properly received after verification, the issue might be due to incorrect\n\
    \t   TX settings, RX settings and/or channel effects\n\
    \t4) Since the output array contains as many elements as the input image's\n\
    \t   vertical resolution (X_LEN) and every element of the array (signal) is meant\n\
    \t   to use a bandwidth of 2*B (generally, much higher than X_LEN), it is\n\
    \t   recommended to repeat the transmission of every signal 2*B/X_LEN times\n\
    \t   so that the resulting RF image is squared"
    )
    exit()

# COMMAND-LINE OPTIONS
if "-b" in SYS_ARGV: #bandwidth (one-sided)
    arg_ind = SYS_ARGV.index("-b")
    B       = SYS_ARGV[arg_ind+1]
else:
    B = int(5e3)
    print("Using default one-sided bandwidth (B=" + str(B) + ")\n")
if "-f" in SYS_ARGV: #sampling frequency
    arg_ind = SYS_ARGV.index("-f")
    Fs      = SYS_ARGV[arg_ind+1]
else:
    Fs = int(2.1 * B)
    print("Using default sampling frequency (Fs=" + str(Fs) + ")\n")
if "-i" in SYS_ARGV: #path to input image
    arg_ind     = SYS_ARGV.index("-i")
    IN_IMG_PATH = SYS_ARGV[arg_ind+1]
    img         = Image.open(IN_IMG_PATH)
else:
    raise ValueError("Input image path missing. Please, input an image path by using the command-line option \"-i\"")
if "-n" in SYS_ARGV: #number of IQ samples
    arg_ind = SYS_ARGV.index("-n")
    N       = SYS_ARGV[arg_ind+1]
else:
    N = int(Fs)
    print("Using default number of IQ samples (N=" + str(N) + ")\n")
if "-t" not in SYS_ARGV: #test
    if "-o" in SYS_ARGV: #path to output array
        arg_ind      = SYS_ARGV.index("-o")
        OUT_IMG_PATH = SYS_ARGV[arg_ind+1]
    else:
        OUT_IMG_PATH = "./rf_img"
        print("Using default output RF image path (" + OUT_IMG_PATH + ")\n")
else:
    if "-o" in SYS_ARGV: #path to output array
        print("Ignoring \"-o\" argument since in testing mode (\"-t\")\n")

# Convert input image to 1-bit pixels, black-and-white image
img      = img.convert("1")
img_mat  = np.asarray(img)

X_LEN, Y_LEN = img_mat.shape[0], img_mat.shape[1] #image resolution
FREQ_STEP    = (2 * B) / (Y_LEN - 1) #frequency step for allocating image in bandwidth

assert Fs > (2 * B), "Sampling frequency (" + str(Fs) + ") must satisfy Nyquist criterion (Fs > " + str(2 * B) + ")"

# Build IQ representation
Ts         = 1 / Fs #sampling period
times      = np.linspace(0, (N - 1) * Ts, N)
freqs      = np.linspace(-B, B, Y_LEN)
freqs_Pxx  = fft.fftshift(fft.fftfreq(N, Ts))
signal_arr = np.zeros((X_LEN,), dtype=object)
print("Building IQ representation...\n")
for i in range(X_LEN):
    for j in range(Y_LEN):
        if img_mat[i,j]:
            signal_arr[i] += np.exp(1j * 2 * np.pi * freqs[j] * times)

# Save IQ representation
if "-t" not in SYS_ARGV: #test
    np.save(OUT_IMG_PATH, signal_arr)

# Plot spectrogram of IQ representation (RF image)
if "-t" in SYS_ARGV: #test
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, num="Image to RF")
    Tau = Fs / N #FFT resolution
    assert Tau <= FREQ_STEP, "FFT resolution (" + str(Tau) + ") must be less or equal than the frequency step length (" + str(FREQ_STEP) + ")" 
    print("Plotting spectrogram of IQ representation (RF image)...\n")
    Pxx_img_mat = np.NaN * np.ones((X_LEN, N))
    for i in range(X_LEN):
        _, Pxx_row       = signal.welch(signal_arr[i], fs=Fs, window="hann", nfft=N, detrend=False, scaling="spectrum", return_onesided=False)
        Pxx_img_mat[i,:] = fft.fftshift(Pxx_row)
    Pxx_img_mat = np.repeat(Pxx_img_mat, int(2*B/X_LEN), axis=0)
    ax1.imshow(img)
    ax1.set_xlabel("Pixel")
    ax1.set_ylabel("Pixel")
    ax1.set_title("Original image")
    ax2.imshow(img_mat)
    ax2.set_xlabel("Pixel")
    ax2.set_ylabel("Pixel")
    ax2.set_title("1-bit image")
    ax3.imshow(Pxx_img_mat)
    ax3.set_xlabel("Frequency Index from -B to B")
    ax3.set_ylabel("Signal Index")
    ax3.set_title("RF image")
    plt.show()
