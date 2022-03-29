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

    Returns:

        channels: sorted channels (N-D numpy.ndarray)
    """
    return channels[indexes]

def subwindows(x, win_len, overlap_len):
    """
    Subwindows

    Distribute input among overlapped subwindows

    Args:

        x: input array (1-D numpy.ndarray)
        win_len: subwindows length (int)
        overlap_len: subwindows overlap length (int)

    Returns:

        y: output N-D array (N-D numpy.ndarray)
    """
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

def scale_axes(axis, data, cur_lim, axes="Y", multichan=False, symm=False):
    data_len             = len(data)
    cur_x_lim, cur_y_lim = cur_lim[0], cur_lim[1]
    new_x_lim, new_y_lim = cur_x_lim, cur_y_lim
    if data_len == 2:
        if axes == "XY":
            data_x, data_y = data[0], data[1]
        else:
            data_x, data_y = data, data
    else:
        if axes == "X":
            data_x = data
        elif axes == "Y":
            data_y = data
        elif axes == "XY":
            data_x, data_y = data, data
    if axes == "X" or axes == "XY":
        if symm:
            if multichan:
                data_x_len   = len(data_x)
                max_abs_data = 0
                for i in range(data_x_len):
                    aux_max_abs_data = max(abs(data_x[i]))
                    if aux_max_abs_data > max_abs_data:
                        max_abs_data = aux_max_abs_data
            else:
                max_abs_data = max(abs(data_x))
            if max_abs_data > cur_x_lim[1]:
                new_x_lim = [-max_abs_data, max_abs_data]
            axis.set_xlim(new_x_lim)
        else:
            if multichan:
                data_x_len = len(data_x)
                min_data   = 0
                max_data   = 0
                for i in range(data_x_len):
                    aux_min_data = min(data_x[i])
                    aux_max_data = max(data_x[i])
                    if aux_min_data < min_data:
                        min_data = aux_min_data
                    if aux_max_data > max_data:
                        max_data = aux_max_data
            else:
                min_data = min(data_x)
                max_data = max(data_x)
            if min_data < cur_x_lim[0]:
                new_x_lim[0] = min_data
            if max_data > cur_x_lim[1]:
                new_x_lim[1] = max_data
            axis.set_xlim(new_x_lim)
    if axes == "Y" or axes == "XY":
        if symm:
            if multichan:
                data_y_len   = len(data_y)
                max_abs_data = 0
                for i in range(data_y_len):
                    aux_max_abs_data = max(abs(data_y[i]))
                    if aux_max_abs_data > max_abs_data:
                        max_abs_data = aux_max_abs_data
            else:
                max_abs_data = max(abs(data_y))
            if max_abs_data > cur_y_lim[1]:
                new_y_lim = [-max_abs_data, max_abs_data]
            axis.set_ylim(new_y_lim)
        else:
            if multichan:
                data_y_len = len(data_y)
                min_data   = 0
                max_data   = 0
                for i in range(data_y_len):
                    aux_min_data = min(data_y[i])
                    aux_max_data = max(data_y[i])
                    if aux_min_data < min_data:
                        min_data = aux_min_data
                    if aux_max_data > max_data:
                        max_data = aux_max_data
            else:
                min_data = min(data_y)
                max_data = max(data_y)
            if min_data < cur_y_lim[0]:
                new_y_lim[0] = min_data
            if max_data > cur_y_lim[1]:
                new_y_lim[1] = max_data
            axis.set_ylim(new_y_lim)
    return np.array([new_x_lim, new_y_lim])
