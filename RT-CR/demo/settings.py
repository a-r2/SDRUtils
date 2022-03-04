IP_SDR      = "169.254.92.202" #SDR IP

IQ_ARR_PATH   = "../signalExample.mat" #path to .mat file containing RX IQ data
CORR_ARR_PATH = "../signalExample.mat" #path to .mat file containing correlation signal IQ data

IQ_ARR_EN   = False #enable RX IQ from .mat file
CORR_ARR_EN = False #enable signal correlation from .mat file
SENS_EN     = False #enable sensing
TX_EN       = False #enable RF transmission

PLOT_TYPE   = 2 #RX data plot type
SENS_TYPE   = None #sensing algorithm

WIN_NUM     = 64 #number of FFT windows
WIN_OVERLAP = 0.5 #FFT windows overlap %

FFT_LEN     = 1024
CORR_LEN    = 1024

SP_CHUNKS   = 4

TX_SLEEP    = 2 #TX ON/OFF switching time [s]
