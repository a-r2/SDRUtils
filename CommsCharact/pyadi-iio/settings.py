import numpy as np

import adi

# PARAMETERS
IP_SDR       = "169.254.92.202" #SDR IP
ITER_NUM     = 100 #number of iterations to calculate time performance for a given (sample_rate, buffer_size) combination
RESULTS_FILE = "measurements" #.npy file that stores (latency, jitter, throughput) measurements

sample_rates = np.array([586e3, 1e6, 2e6, 3e6, 4e6, 5e6, 10e6, 15e6, 20e6, 25e6, 30e6, 35e6, 40e6, 45e6, 50e6, 55e6, 61.44e6], dtype=int) #sample rates to sweep
buffer_sizes = np.array([256 * 2 ** exp for exp in range(15)], dtype=int) #buffer sizes to sweep

# SDR INIT
sdr = adi.ad9361(uri="ip:" + IP_SDR)

sdr.loopback                = 2 #0 = disabled, 1 = digital, 2 = RF
sdr.sample_rate             = 2e6 #min = 590e3, max = 6.144e7 [Hz]
sdr.rx_buffer_size          = 1024
sdr.rx_output_type          = "raw" #"SI", "raw"
sdr.tx_cyclic_buffer        = False

sdr.gain_control_mode_chan0 = "fast_attack" #"manual", "slow_attack", "fast_attack"
sdr.gain_control_mode_chan1 = "fast_attack" #"manual", "slow_attack", "fast_attack"

sdr.tx_hardwaregain_chan0   = -89 #max = -89, max =0 
sdr.tx_hardwaregain_chan1   = -89 #max = -89, max = 0

sdr.rx_enabled_channels     = [0] #channel 1 = [0], channel 2 = [1], channel 1+2 = [0, 1]
sdr.tx_enabled_channels     = [0] #channel 1 = [0], channel 2 = [1], channel 1+2 = [0, 1]

sdr.rx_hardwaregain_chan0   = 0 #min = -1, max = 73, N/A unless gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan1   = 0 #min = -1, max = 73, N/A unless gain_control_mode_chan1 = "manual"

sdr.rx_lo                   = int(93.9e6) #min = 7e7, max = 6e9 [Hz]
sdr.tx_lo                   = int(433e6) #min = 4.6875011e7, max = 6e9 [Hz]

sdr.rx_rf_bandwidth         = int(2e6) #min = 0, max = 2.147483647e9 [Hz]
sdr.tx_rf_bandwidth         = int(2e6) #min = 0, max = 2.147483647e9 [Hz]

sdr.disable_dds() #Disable Direct Digital Synthesizer (DDS)

