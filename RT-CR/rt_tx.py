import numpy as np

def transmit_data(sdr, tx_flag):
    """
    Transmit data

    Transmits IQ data in real time

    Args:

        sdr: AD9361 object (class)
        tx_flag: TX activation and deactivation (multiprocessing.Event)
    """
    N  = sdr.rx_buffer_size
    fs = sdr.sample_rate
    ts = 1 / fs
    f  = 500e3
    try:
        i = 0 #tx iteration
        while True:
            if tx_flag.wait():
                t1 = i * N * ts
                t2 = (i + 1) * N * ts
                t  = np.linspace(t1, t2, N)
                i += 1
                IQ  = (np.exp(1j*2 * np.pi * (-f) * t) + np.exp(1j*2 * np.pi * f * t)) / np.sqrt(2)
                IQ *= ADC_SCALE
                sdr.tx(IQ)
                if not tx_flag.is_set():
                    sdr.tx_destroy_buffer()
    except KeyboardInterrupt:
        sdr.tx_destroy_buffer()
        exit()

