import numpy as np
import scipy.signal as signal
import wave

from settings import *
from constants import *
from utils import *

def demod(sdr, rx2demod_out):
    if DEMOD_TYPE == "FM":
        fm_demod(sdr, rx2demod_out)
    elif DEMOD_TYPE == "AM":
        am_demod(sdr, rx2demod_out)

def am_demod(sdr, rx2demod_out):
    pass

def fm_demod(sdr, rx2demod_out):
    import sounddevice as sd #error when import is out of the function
    df       = 5 #channel bandwidth/max modulating signal frequency
    wav_file = wave.open(DEMOD_PATH, "wb")
    wav_file.setparams((1, 2, DEMOD_FS, 1, "NONE", "not compressed"))
    while True:
        sdr_data  = rx2demod_out.recv()
        IQ_t_1, _, sample_rate, _, _, _,  _, _ = sdr_data
        lpf       = signal.butter(DEMOD_FILT_N, 15e3, btype="low", analog=False, output="sos", fs=int(sample_rate/5))
        IQ_t_1    = signal.decimate(IQ_t_1, 5)
        phi       = np.angle(IQ_t_1)
        phi_delta = np.diff(np.unwrap(phi)/(2*np.pi*df))
        phi_delta = np.append(phi_delta[0], phi_delta)
        phi_delta = signal.sosfilt(lpf, phi_delta)
        phi_delta = signal.decimate(phi_delta, 8)
        s_est     = np.int16((phi_delta/np.max(np.abs(phi_delta))) * 0.5 * ADC_SCALE) #estimated signal
        wav_file.writeframes(s_est)
        sd.play(s_est, DEMOD_FS)
