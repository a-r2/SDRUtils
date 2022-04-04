import sys

import numpy as np
from PIL import Image, ImageEnhance

import matplotlib.pyplot as plt

B  = 10000
Fs = 1e6
Ts = 1 / Fs

SYS_ARGV = sys.argv
assert len(SYS_ARGV) < 3, "Only one image at a time"
IMG_PATH = SYS_ARGV[1]
img      = Image.open(IMG_PATH)
img      = img.convert("1")
img_arr  = np.asarray(img)

X_LEN, Y_LEN = img_arr.shape[0], img_arr.shape[1]
freqs        = np.linspace(-0.5 * B, 0.5 * B, Y_LEN)
t            = np.linspace(0, (Y_LEN - 1) * Ts, Y_LEN)
exps         = np.exp(1j * 2 * np.pi * freqs * t)
img_exps     = img_arr @ exps

plt.imshow(img_arr)
plt.show()
