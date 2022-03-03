import numpy as np

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
    assert N > n, "Partition size (n) must be less or equal to the array size (N)"
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
