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

""" ARRAY .MAT FILE FOR COMPARISON """
if COMP_ARR_EN:
    COMP_ARR = sio.loadmat(COMP_ARR_PATH) #load .mat
    COMP_ARR = COMP_ARR["a"].flatten()
else:
    COMP_ARR = None
