import scipy.io as sio

from settings import *

""" ADC SCALING """
ADC_SCALE = 2 ** 12 #2 power of ADC bits

""" IQ ARRAY FROM .MAT FILE """
if IQ_ARR_EN:
    IQ_ARR = sio.loadmat(IQ_ARR_PATH) #load .mat
    IQ_ARR = IQ_ARR["a"].flatten()
else:
    IQ_ARR = None

""" CORRELATION ARRAY .MAT FILE """
if CORR_ARR_EN:
    CORR_ARR = sio.loadmat(CORR_ARR_PATH) #load .mat
    CORR_ARR = CORR_ARR["a"].flatten()
else:
    CORR_ARR = None
