import numpy as np
import numpy.matlib as matlib
from scipy import signal

def split_IQ(IQ):
    """
    Split IQ samples

    Splits IQ data into real (in-phase) and imaginary (quadrature) parts
    
    Args:

        IQ: IQ samples (1-D numpy.ndarray)

    Returns:

        I: in-phase component (1-D numpy.ndarray)
        Q: quadrature component (1-D numpy.ndarray)
    """
    I = IQ.real
    Q = IQ.imag
    return I, Q

def sort_channels(channels, indexes):
    """
    Sort channels

    Sorts RX/TX channels based on indexes order of enabled channels

    Args:

        channels: IQ data channels (N-D numpy.ndarray)
        indexes: channels sort (1-D numpy.ndarray)

    Returns sorted channels
    """
    return channels[indexes]

def subwindows(x, win_len, overlap_len):
    X_LEN = len(x)
    assert win_len < X_LEN, "Windows length (" + str(win_len) + ") must be less than the input length(" + str(X_LEN) + ")"
    assert overlap_len < win_len, "Overlap length (" + str(overlap_len) + ") must be less than the windows length(" + str(win_len) + ")"
    x     = np.append(x, np.zeros(win_len - 1))
    if overlap_len > 0:
        y = np.lib.stride_tricks.sliding_window_view(x, window_shape=win_len)[::win_len - overlap_len]
    else:
        y = np.lib.stride_tricks.sliding_window_view(x, window_shape=win_len)[::win_len]
    return y

def array1D_part(array, size, index):
    """
    1-D array partition

    Get the partition from 1-D array corresponding to the given size and partition index

    Args:

        array: array to partitionate (numpy.ndarray)
        size: partition size (int)
        ind: partition index (int)

    Returns:

        part: partition (numpy.ndarray)
    """
    N           = len(array) 
    n           = size
    assert n <= N, "Partition size (n) must be less than or equal to the array size (N)"
    i           = index
    first_index = i * n
    last_index  = (i + 1) * n
    if last_index > N:
        last_index += - (last_index // N) * N
        if first_index > N:
            first_index += - (first_index // N) * N
    if first_index < last_index:
        part = array[first_index:last_index]
    else:
        part = np.concatenate((array[first_index:], array[:last_index]))
    return part.flatten()
