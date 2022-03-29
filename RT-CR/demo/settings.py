IP_SDR      = "169.254.92.202" #SDR IP

IQ_ARR_PATH   = "../signalExample.mat" #path to .mat file containing RX IQ data
COMP_ARR_PATH = "../signalExample.mat" #path to .mat file containing signal IQ data for comparison (correlation, cross-spectral density)

IQ_ARR_EN   = False #enable RX IQ from .mat file
COMP_ARR_EN = False #enable signal for comparison from .mat file

PLOT_EN  = True #enable plotting
SENS_EN  = False #enable sensing
TX_EN    = False #enable RF transmission
DEMOD_EN = False #enable demodulation

PLOT_TYPE  = "time" #RX data plot type
SENS_TYPE  = None #sensing algorithm
DEMOD_TYPE = "FM"

SWIN_LEN         = 256 #subwindows length [samples]
SWIN_OVERLAP_LEN = 64 #subwindows overlap [samples]

FFT_LEN  = 4096 #FFT length
COMP_LEN = 1024 #output comparison length

SP_CHUNKS_NUM = 4 #number of chunks shown in spectrogram

TX_SLEEP = 2 #TX ON/OFF switching time [s]

DEMOD_PATH   = "./demod_signal.wav" #path to .wav file containing demodulated signal
DEMOD_FS     = 48000 #sampling rate of demodulated signal [Hz]
DEMOD_FILT_N = 8 #order of low-pass filter used in FM demodulation
