import multiprocessing as mp
import time

import numpy.matlib as matlib
from scipy import signal

from sdr_init import *
from constants import *
from utils import *

from rt_plot import *
from rt_rx import *
from rt_tx import *
from rt_sens import *

""" PROCESSES """
rx2plot_in, rx2plot_out = mp.Pipe()
if TX_EN:
    tx_flag = mp.Event()
    tx_proc = mp.Process(target=transmit_data, args=(sdr, tx_flag), daemon=True)
if SENS_EN:
    rx2sens_in, rx2sens_out     = mp.Pipe()
    plot2sens_in, plot2sens_out = mp.Pipe()
    rx_proc                     = mp.Process(target=receive_data, args=(sdr, rx2plot_in, rx2sens_in), daemon=True)
    plot_proc                   = mp.Process(target=plot_data, args=(sdr, rx2plot_out, plot2sens_in), daemon=True)
    sens_proc                   = mp.Process(target=sens_spectrum, args=(rx2sens_out, plot2sens_out), daemon=True)
else:
    rx_proc   = mp.Process(target=receive_data, args=(sdr, rx2plot_in, None), daemon=True)
    plot_proc = mp.Process(target=plot_data, args=(sdr, rx2plot_out, None), daemon=True)

""" INITIALIZATION """
if TX_EN:
    print("WARNING!")
    print("Starting RX/TX sequence")
else:
    print("Starting RX only sequence")
time.sleep(2)
rx_proc.start()
plot_proc.start()
if TX_EN:
    tx_proc.start()
if SENS_EN:
    sens_proc.start()

""" LOOP """
while True:
    if TX_EN:
        if tx_flag.is_set():
            tx_flag.clear()
            print("Ending transmission...")
        else: 
            tx_flag.set()
            print("Starting transmission...")
        time.sleep(TX_SLEEP)
    else:
        pass
