import sys

import numpy as np
import imageio 

SYS_ARGV = sys.argv
assert len(SYS_ARGV) < 3, "Only one image at a time"
IMG_PATH = SYS_ARGV[1]
img      = imageio.imread(IMG_PATH)
img_arr  = np.asarray(img)
