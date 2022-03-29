import numpy as np

from settings import *

def receive_data(sdr, rx2plot_in, rx2sens_in, rx2demod_in):
    while True:
        sdr_data = get_sdr_data(sdr)
        if PLOT_EN:
            rx2plot_in.send(sdr_data)
        if SENS_EN:
            rx2sens_in.send(sdr_data)
        if DEMOD_EN:
            rx2demod_in.send(sdr_data)

def get_sdr_data(sdr):
    """
    RX IQ data and parameters

    Gets RX IQ data and SDR parameters

    Args:

        sdr: AD9361 object (class)

    Returns:

        IQ_t_1: IQ samples from RX channel 1 (1-D numpy.ndarray)
        IQ_t_2: IQ samples from RX channel 2 (1-D numpy.ndarray)
        sample_rate: sampling frequency (int)
        samples_num: number of samples (int)
        channels: enabled RX channels (list)
        channels_num: number of enabled RX channels (int)
        rx_lo: RX local oscillator frequency (int)
        rx_rf_bw: RX filter bandwidth (int)
    """
    rx_block     = sdr.rx() #one array per channel
    sample_rate  = sdr.sample_rate
    samples_num  = sdr.rx_buffer_size
    rx_lo        = sdr.rx_lo
    rx_rf_bw     = sdr.rx_rf_bandwidth
    channels     = sdr.rx_enabled_channels
    channels_num = len(channels)
    if channels_num == 2:
        IQ_t_1 = rx_block[0]
        IQ_t_2 = rx_block[1]
    else:
        IQ_t_1 = rx_block
        IQ_t_2 = np.array([])
    return IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, rx_lo, rx_rf_bw 
