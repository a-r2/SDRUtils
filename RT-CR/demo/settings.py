IP_SDR      = "169.254.92.202" #SDR IP

IQ_ARR_PATH   = "../signalExample.mat" #path to .mat file containing RX IQ data
COMP_ARR_PATH = "../signalExample.mat" #path to .mat file containing signal IQ data for comparison (correlation, cross-spectral density)

IQ_ARR_EN   = False #enable RX IQ from .mat file
COMP_ARR_EN = False #enable signal for comparison from .mat file
SENS_EN     = False #enable sensing
TX_EN       = False #enable RF transmission

PLOT_TYPE = "coherence" #RX data plot type
SENS_TYPE = None #sensing algorithm

SWIN_LEN         = 64 #Subwindows length [samples]
SWIN_OVERLAP_LEN = 16 #Subwindows overlap [samples]

FFT_LEN  = 1024
COMP_LEN = 1024

SP_CHUNKS = 4

TX_SLEEP = 2 #TX ON/OFF switching time [s]
