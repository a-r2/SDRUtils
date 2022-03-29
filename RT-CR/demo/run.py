import multiprocessing as mp
import time

import numpy.matlib as matlib
from scipy import signal

from settings import *
from sdr_init import *
from constants import *
from utils import *

from rt_demod import *
from rt_plot import *
from rt_sens import *
from rt_rx import *
from rt_tx import *

""" EVENTS """
if TX_EN:
    tx_flag = mp.Event()

""" PIPES """
if DEMOD_EN:
    rx2demod_in, rx2demod_out = mp.Pipe()
else:
    rx2demod_in, rx2demod_out = None, None
if PLOT_EN:
    rx2plot_in, rx2plot_out = mp.Pipe()
else:
    rx2plot_in, rx2plot_out = None, None
if SENS_EN:
    rx2sens_in, rx2sens_out = mp.Pipe()
else:
    rx2sens_in, rx2sens_out = None, None
if PLOT_EN & SENS_EN:
    plot2sens_in, plot2sens_out = mp.Pipe()
else:
    plot2sens_in, plot2sens_out = None, None

""" PROCESSES """
if DEMOD_EN:
    demod_proc                = mp.Process(target=demod, args=(sdr, rx2demod_out), daemon=True)
if PLOT_EN:
    plot_proc               = mp.Process(target=plot_data, args=(sdr, rx2plot_out, plot2sens_in), daemon=True)
if SENS_EN:
    sens_proc                   = mp.Process(target=sens_spectrum, args=(rx2sens_out, plot2sens_out), daemon=True)
if TX_EN:
    tx_proc = mp.Process(target=transmit_data, args=(sdr, tx_flag), daemon=True)
rx_proc = mp.Process(target=receive_data, args=(sdr, rx2plot_in, rx2sens_in, rx2demod_in), daemon=True)

""" INITIALIZATION """
if TX_EN:
    print("WARNING!")
    print("Starting RX/TX sequence")
else:
    print("Starting RX only sequence")
time.sleep(2)
if DEMOD_EN:
    demod_proc.start()
if PLOT_EN:
    plot_proc.start()
if SENS_EN:
    sens_proc.start()
if TX_EN:
    tx_proc.start()
rx_proc.start()

""" LOOP """
while True:
    try:
        if TX_EN:
            if tx_flag.is_set():
                tx_flag.clear()
                print("Ending transmission...")
            else: 
                tx_flag.set()
                print("Starting transmission...")
            time.sleep(TX_SLEEP)
    except KeyboardInterrupt:
        if TX_EN:
            tx_proc.close()
        if SENS_EN:
            rx2sens_in.close()
            rx2sens_out.close()
            plot2sens_in.close()
            plot2sens_out.close()
            sens_proc.close()
        if DEMOD_EN:
            rx2demod_in.close()
            rx2demod_out.close()
            demod_proc.close()
        if PLOT_EN:
            rx2plot_in.close()
            rx2plot_out.close()
            plot_proc.close()
        rx_proc.close()
        exit()
