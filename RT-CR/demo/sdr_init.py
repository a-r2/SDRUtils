import adi

from settings import *

sdr = adi.ad9361(uri="ip:" + IP_SDR) #SDR object

sdr.loopback                = int(2) #0 = disabled, 1 = digital, 2 = RF
sdr.sample_rate             = int(1.92e6) #min = 590e3, max = 6.144e7 [Hz]
sdr.rx_annotated            = False
sdr.rx_buffer_size          = int(4096)
sdr.rx_output_type          = "raw" #"SI", "raw"
sdr.tx_cyclic_buffer        = False

sdr.gain_control_mode_chan0 = "fast_attack" #"manual", "slow_attack", "fast_attack"
sdr.gain_control_mode_chan1 = "fast_attack" #"manual", "slow_attack", "fast_attack"

sdr.tx_hardwaregain_chan0   = -89.75 #max = -89.75, max = 0 
sdr.tx_hardwaregain_chan1   = -89.75 #max = -89.75, max = 0

sdr.rx_enabled_channels     = [0, 1] #channel 1 = [0], channel 2 = [1], channel 1+2 = [0, 1]
sdr.tx_enabled_channels     = [0, 1] #channel 1 = [0], channel 2 = [1], channel 1+2 = [0, 1]

sdr.rx_hardwaregain_chan0   = 0 #min = -1, max = 73, N/A unless gain_control_mode_chan0 = "manual"
sdr.rx_hardwaregain_chan1   = 0 #min = -1, max = 73, N/A unless gain_control_mode_chan1 = "manual"

sdr.rx_lo                   = int(96.9e6) #min = 7e7, max = 6e9 [Hz]
sdr.tx_lo                   = int(433e6) #min = 4.6875011e7, max = 6e9 [Hz]

sdr.rx_rf_bandwidth         = int(2e6) #min = 0, max = 2.147483647e9 [Hz]
sdr.tx_rf_bandwidth         = int(2e6) #min = 0, max = 2.147483647e9 [Hz]

sdr.disable_dds() #Disable Direct Digital Synthesizer (DDS)
