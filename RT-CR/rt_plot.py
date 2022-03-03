import matplotlib
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import rcParams
import numpy as np
import numpy.matlib as matlib
from scipy import signal
import time

from settings import *
from constants import *
from utils import *

from rt_rx import *

matplotlib.use("qt5agg")
plt.ion()

rcParams["axes.grid"]       = True
rcParams["axes.xmargin"]    = 0
rcParams["axes.ymargin"]    = 0
rcParams["lines.linestyle"] = "-"
rcParams["lines.marker"]    = "."
rcParams["legend.loc"]      = "upper right"
rcParams["legend.fontsize"] = 8

LEGEND_LABELS = np.array(("Channel 1", "Channel 2"))

def plot_data(sdr, rx2plot_out, plot2sens_in):
    """
    Plot data

    Plots received IQ data in real time

    Args:

        sdr: AD9361 object (class)
    """
    # Select plot
    try:
        if PLOT_TYPE == 0 or PLOT_TYPE == "time": #time
            fig, ax = plt.subplots(2, num="SDR RX time-domain IQ")
            fig.suptitle("Time-domain IQ signal")
        elif PLOT_TYPE == 1 or PLOT_TYPE == "constellation": #constellation
            fig, ax = plt.subplots(1, num="SDR RX IQ constellation")
            fig.suptitle("IQ constellation")
            rcParams["lines.linestyle"] = "none"
        elif PLOT_TYPE == 2 or PLOT_TYPE == "power spectrum": #power spectrum
            fig, ax = plt.subplots(1, num="SDR RX IQ power spectrum")
            fig.suptitle("IQ power spectrum")
        elif PLOT_TYPE == 3 or PLOT_TYPE == "spectrogram": #spectrogram
            fig, ax = plt.subplots(1, num="SDR RX IQ spectrogram")
            fig.suptitle("IQ spectrogram")
        elif PLOT_TYPE == 4 or PLOT_TYPE == "histogram": #histogram
            fig, ax = plt.subplots(1, num="SDR RX IQ power spectrum histogram")
            fig.suptitle("IQ power spectrum histogram")
        elif PLOT_TYPE == 5 or PLOT_TYPE == "wavelet": #wavelet
            fig, ax = plt.subplots(1, num="SDR RX IQ wavelet")
            fig.suptitle("IQ wavelet")
        elif PLOT_TYPE == 6 or PLOT_TYPE == "correlation": #correlation
            fig, ax = plt.subplots(2, num="SDR RX correlation IQ")
            fig.suptitle("Correlation IQ signal")
        else:
            raise ValueError("The argument PLOT_TYPE in IQ_plot function shall only take integer values 0, 1 or 2")
        fig_man = plt.get_current_fig_manager()
        fig_man.window.showMaximized()
    except ValueError:
        raise
    # Plot in real time
    if PLOT_TYPE == 0 or PLOT_TYPE == "time": #time
        t_plot(sdr, fig, ax, rx2plot_out)
    elif PLOT_TYPE == 1 or PLOT_TYPE == "constellation": #constellation
        c_plot(sdr, fig, ax, rx2plot_out)
    elif PLOT_TYPE == 2 or PLOT_TYPE == "power spectrum": #power spectrum
        ps_plot(sdr, fig, ax, rx2plot_out, plot2sens_in)
    elif PLOT_TYPE == 3 or PLOT_TYPE == "spectrogram": #spectrogram
        sp_plot(sdr, fig, ax, rx2plot_out)
    elif PLOT_TYPE == 4 or PLOT_TYPE == "histogram": #histogram
        h_plot(sdr, fig, ax, rx2plot_out)
    elif PLOT_TYPE == 5 or PLOT_TYPE == "wavelet": #wavelet
        w_plot(sdr, fig, ax, rx2plot_out)
    elif PLOT_TYPE == 6 or PLOT_TYPE == "correlation": #correlation
        corr_plot(sdr, fig, ax, rx2plot_out, plot2sens_in)

def t_plot(sdr, fig_t, ax_t, rx2plot_out):
    ax_I_t, ax_Q_t = ax_t[0], ax_t[1]
    ax_I_lim       = np.array([[0.,0.],[0.,0.]])
    ax_Q_lim       = np.array([[0.,0.],[0.,0.]])
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            sdr_data = rx2plot_out.recv()
            _, _, sample_rate, samples_num, _, _, _, _ = sdr_data
            IQ_t_1       = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind    += 1
            update_t_fig(fig_t, ax_I_t, ax_Q_t, ax_I_lim, ax_Q_lim, IQ_t_1, None, sample_rate, samples_num, [0], 1)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num,  _, _ = sdr_data
            update_t_fig(fig_t, ax_I_t, ax_Q_t, ax_I_lim, ax_Q_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num)

def update_t_fig(fig_t, ax_I_t, ax_Q_t, ax_I_lim, ax_Q_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num):
    times = np.linspace(0, samples_num / sample_rate, samples_num)
    I_t_1, Q_t_1 = split_IQ(IQ_t_1)
    max_I_t_1    = max(abs(I_t_1))
    max_Q_t_1    = max(abs(Q_t_1))
    if max_I_t_1 > ax_I_lim[1][1]:
        ax_I_lim[1] = [-max_I_t_1, max_I_t_1]
    if max_Q_t_1 > ax_Q_lim[1][1]:
        ax_Q_lim[1] = [-max_Q_t_1, max_Q_t_1]
    if channels_num == 2:
        I_t_2, Q_t_2 = split_IQ(IQ_t_2)
        max_I_t_2    = max(abs(I_t_2))
        max_Q_t_2    = max(abs(Q_t_2))
        if max_I_t_2 > ax_I_lim[1][1]:
            ax_I_lim[1] = [-max_I_t_2, max_I_t_2]
        if max_Q_t_2 > ax_Q_lim[1][1]:
            ax_Q_lim[1] = [-max_Q_t_2, max_Q_t_2]
        I_t_comb = np.array([I_t_1, I_t_2])
        I_t_comb = I_t_comb[channels]
        Q_t_comb = np.array([Q_t_1, Q_t_2])
        Q_t_comb = Q_t_comb[channels]
        ax_I_t.plot(times, I_t_comb[0], times, I_t_comb[1])
        ax_Q_t.plot(times, Q_t_comb[0], times, Q_t_comb[1])
    else:
        ax_I_t.plot(times, I_t_1)
        ax_Q_t.plot(times, Q_t_1)
    legend = LEGEND_LABELS[channels]
    legend = legend.tolist()
    ax_I_t.set_ylim(ax_I_lim[1])
    ax_Q_t.set_ylim(ax_Q_lim[1])
    ax_I_t.set_xlabel("Time [s]")
    ax_I_t.set_ylabel("In-phase [mV]")
    ax_I_t.legend(legend)
    ax_Q_t.set_xlabel("Time [s]")
    ax_Q_t.set_ylabel("Quadrature [mV]")
    ax_Q_t.legend(legend)
    fig_t.canvas.flush_events()
    ax_I_t.clear()
    ax_Q_t.clear()

def c_plot(sdr, fig_c, ax_c, rx2plot_out):
    ax_lim       = np.array([[0.,0.],[0.,0.]])
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            sdr_data = rx2plot_out.recv()
            _, _, _, samples_num, _, _, _, _ = sdr_data
            IQ_t_1       = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind    += 1
            update_c_fig(fig_c, ax_c, ax_lim, IQ_t_1, None, [0], 1)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, _, _, channels, channels_num, _, _ = sdr_data
            update_c_fig(fig_c, ax_c, ax_lim, IQ_t_1, IQ_t_2, channels, channels_num)

def update_c_fig(fig_c, ax_c, ax_lim, IQ_t_1, IQ_t_2, channels, channels_num):
    I_t_1, Q_t_1 = split_IQ(IQ_t_1)
    max_I_1      = max(abs(I_t_1))
    max_Q_1      = max(abs(Q_t_1))
    if max_I_1 > ax_lim[1][1]:
        ax_lim[1] = [-max_I_1, max_I_1]
    if max_Q_1 > ax_lim[1][1]:
        ax_lim[1] = [-max_Q_1, max_Q_1]
    if channels_num == 2:
        I_t_2, Q_t_2 = split_IQ(IQ_t_2)
        max_I_2      = max(abs(I_t_2))
        max_Q_2      = max(abs(Q_t_2))
        if max_I_2 > ax_lim[1][1]:
            ax_lim[1] = [-max_I_2, max_I_2]
        if max_Q_2 > ax_lim[1][1]:
            ax_lim[1] = [-max_Q_2, max_Q_2]
        I_t_comb = np.array([I_t_1, I_t_2])
        I_t_comb = I_t_comb[channels]
        Q_t_comb = np.array([Q_t_1, Q_t_2])
        Q_t_comb = Q_t_comb[channels]
        ax_c.plot(I_t_comb[0], Q_t_comb[0], I_t_comb[1], Q_t_comb[1])
    else:
        ax_c.plot(I_t_1, Q_t_1)
    legend = LEGEND_LABELS[channels]
    legend = legend.tolist()
    ax_c.set_xlim(ax_lim[1])
    ax_c.set_ylim(ax_lim[1])
    ax_c.set_xlabel("In-phase [mV]")
    ax_c.set_ylabel("Quadrature [mV]")
    ax_c.legend(legend)
    fig_c.canvas.flush_events()
    ax_c.clear()

def ps_plot(sdr, fig_ps, ax_ps, rx2plot_out, plot2sens_in):
    ax_lim = np.array([[0.,0.],[0.,0.]])
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            sdr_data = rx2plot_out.recv()
            _, _, sample_rate, samples_num, _, _, _, _ = sdr_data
            IQ_t_1       = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind    += 1
            update_ps_fig(fig_ps, ax_ps, ax_lim, IQ_t_1, None, sample_rate, samples_num, [0], 1, plot2sens_in)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, _, _ = sdr_data
            update_ps_fig(fig_ps, ax_ps, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, plot2sens_in)

def update_ps_fig(fig_ps, ax_ps, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, plot2sens_in):
    freqs, IQ_ps_1  = signal.welch(IQ_t_1, fs=sample_rate, window="hanning", nperseg=None, noverlap=None, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", axis=-1, average="mean")
    IQ_ps_1        = 10 * np.log10(IQ_ps_1 + 1e-16)
    freqs          = np.fft.fftshift(freqs)
    IQ_ps_1        = np.fft.fftshift(IQ_ps_1)
    min_IQ_ps_1    = min(IQ_ps_1)
    max_IQ_ps_1    = max(IQ_ps_1)
    if min_IQ_ps_1 < ax_lim[1][0]:
        ax_lim[1][0] = min_IQ_ps_1
    if max_IQ_ps_1 > ax_lim[1][1]:
        ax_lim[1][1] = max_IQ_ps_1
    if channels_num == 2:
        _, IQ_ps_2  = signal.welch(IQ_t_2, fs=sample_rate, window="hanning", nperseg=None, noverlap=None, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", axis=-1, average="mean")
        IQ_ps_2        = 10 * np.log10(IQ_ps_2 + 1e-16)
        IQ_ps_2        = np.fft.fftshift(IQ_ps_2)
        min_IQ_ps_2    = min(IQ_ps_2)
        max_IQ_ps_2    = max(IQ_ps_2)
        if min_IQ_ps_2 < ax_lim[1][0]:
            ax_lim[1][0] = min_IQ_ps_2
        if max_IQ_ps_2 > ax_lim[1][1]:
            ax_lim[1][1] = max_IQ_ps_2
        IQ_ps_comb = np.array([IQ_ps_1, IQ_ps_2])
        IQ_ps_comb = IQ_ps_comb[channels]
        ax_ps.plot(freqs, IQ_ps_comb[0], freqs, IQ_ps_comb[1])
    else:
        ax_ps.plot(freqs, IQ_ps_1)
    legend = LEGEND_LABELS[channels]
    legend = legend.tolist()
    ax_ps.set_xlabel("Frequency [Hz]")
    ax_ps.set_ylabel("Power [dBW]")
    ax_ps.legend(legend)
    if SENS_EN:
        signal_clas, max_ind, min_ind = plot2sens_in.recv()
        ax_ps.plot(freqs[max_ind], IQ_ps_1[max_ind], "o", freqs[min_ind], IQ_ps_1[min_ind], "o", freqs, signal_clas)
    fig_ps.canvas.flush_events()
    ax_ps.clear()

def sp_plot(sdr, fig_sp, ax_sp, rx2plot_out):
    ax_lim = np.array([[0.,0.],[0.,0.]])
    _, _, _, samples_num, _, _,  _, _ = get_sdr_data(sdr)
    iter_sp  = np.zeros((SP_CHUNKS, FFT_LEN))
    freqs_sp = np.zeros((SP_CHUNKS, FFT_LEN))
    SP_1     = np.zeros((SP_CHUNKS, FFT_LEN))
    SP_2     = np.zeros((SP_CHUNKS, FFT_LEN))
    i        = 0 #iterations
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            i += 1
            sdr_data = rx2plot_out.recv()
            _, _, sample_rate, samples_num, _, _,  _, _ = sdr_data
            IQ_t_1       = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind    += 1
            freqs_sp, SP_1, SP_2 = update_sp_fig(fig_sp, ax_sp, ax_lim, IQ_t_1, None, sample_rate, samples_num, [0], 1, iter_sp, freqs_sp, SP_1, SP_2, i)
    else:
        while True:
            i += 1
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, _, _ = sdr_data
            freqs_sp, SP_1, SP_2 = update_sp_fig(fig_sp, ax_sp, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, iter_sp, freqs_sp, SP_1, SP_2, i)

def update_sp_fig(fig_sp, ax_sp, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, iter_sp, freqs_sp, SP_1, SP_2, i):
    freqs, IQ_sp_1  = signal.welch(IQ_t_1, fs=sample_rate, window="hanning", nperseg=None, noverlap=None, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", axis=-1, average="mean")
    iter_sp          = matlib.repmat(np.arange(0, SP_CHUNKS), FFT_LEN, 1).T
    freqs_c          = np.fft.fftshift(freqs)
    freqs_sp[:-1, :] = freqs_sp[1:]
    freqs_sp[-1, :]  = freqs_c
    IQ_sp_1_c        = np.fft.fftshift(IQ_sp_1)
    IQ_sp_1_c        = 10 * np.log10(IQ_sp_1_c + 1e-16)
    SP_1[:-1, :]     = SP_1[1:]
    SP_1[-1, :]      = IQ_sp_1_c
    if channels_num == 2:
        _, IQ_sp_2  = signal.welch(IQ_t_2, fs=sample_rate, window="hanning", nperseg=None, noverlap=None, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", axis=-1, average="mean")
        IQ_sp_2_c    = np.fft.fftshift(IQ_sp_2)
        IQ_sp_2_c    = 10 * np.log10(IQ_sp_2_c + 1e-16)
        SP_2[:-1, :] = SP_2[1:]
        SP_2[-1, :]  = IQ_sp_2_c
        ax_sp.pcolormesh(iter_sp, freqs_sp, 0.5 * (SP_1 + SP_2), shading="gouraud")
    else:
        ax_sp.pcolormesh(iter_sp, freqs_sp, SP_1, shading="gouraud")
    ax_sp.set_xlabel("Iteration [-]")
    ax_sp.set_ylabel("Frequency [Hz]")
    tick_labels = ["-" + str(i) for i in range(SP_CHUNKS - 1, 0, -1)]
    tick_labels.append("0")
    ax_sp.set_xticks(np.arange(0, SP_CHUNKS), labels=tick_labels)
    fig_sp.canvas.flush_events()
    ax_sp.clear()
    return freqs_sp, SP_1, SP_2

def h_plot(sdr, fig_h, ax_h, rx2plot_out):
    ax_lim = np.array([[0.,0.],[0.,0.]])
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            sdr_data = rx2plot_out.recv()
            _, _, sample_rate, samples_num, _, _, _, _ = sdr_data
            IQ_t_1       = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind    += 1
            update_h_fig(fig_h, ax_h, ax_lim, IQ_t_1, None, sample_rate, samples_num, [0], 1)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, _, _ = sdr_data
            update_h_fig(fig_h, ax_h, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num)

def update_h_fig(fig_h, ax_h, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num):
    N_win = int(FFT_LEN // WIN_NUM)
    freqs, IQ_ps_1  = signal.welch(IQ_t_1, fs=sample_rate, window="hanning", nperseg=None, noverlap=None, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", axis=-1, average="mean")
    IQ_ps_1     = 10 * np.log10(IQ_ps_1 + 1e-16)
    freqs       = np.fft.fftshift(freqs)
    IQ_ps_1     = np.fft.fftshift(IQ_ps_1)
    IQ_ps_win_1 = np.lib.stride_tricks.sliding_window_view(IQ_ps_1, window_shape=N_win)[::int(WIN_OVERLAP * N_win)]
    min_IQ_h_1 = min(IQ_ps_1)
    max_IQ_h_1 = max(IQ_ps_1)
    if min_IQ_h_1 < ax_lim[1][0]:
        ax_lim[1][0] = min_IQ_h_1
    if max_IQ_h_1 > ax_lim[1][1]:
        ax_lim[1][1] = max_IQ_h_1
    if channels_num == 2:
        _, IQ_ps_2  = signal.welch(IQ_t_2, fs=sample_rate, window="hanning", nperseg=None, noverlap=None, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", axis=-1, average="mean")
        IQ_ps_2     = 10 * np.log10(IQ_ps_2 + 1e-16)
        IQ_ps_2     = np.fft.fftshift(IQ_ps_2)
        IQ_ps_win_2 = np.lib.stride_tricks.sliding_window_view(IQ_ps_2, window_shape=N_win)[::int(WIN_OVERLAP * N_win)]
        for windw_ind in range(WIN_NUM):
            ax_h.hist(IQ_ps_win_1[windw_ind] + IQ_ps_win_2[windw_ind], bins="auto", range=None, density=True, weights=None, cumulative=False, bottom=None, histtype="bar", align="mid", orientation="vertical", rwidth=None, log=False, color=None, label=None, stacked=False)
        min_IQ_h_2 = min(IQ_ps_2)
        max_IQ_h_2 = max(IQ_ps_2)
        if min_IQ_h_2 < ax_lim[1][0]:
            ax_lim[1][0] = min_IQ_h_2
        if max_IQ_h_2 > ax_lim[1][1]:
            ax_lim[1][1] = max_IQ_h_2
    else:
        for windw_ind in range(WIN_NUM):
            ax_h.hist(IQ_ps_win_1[windw_ind], bins="auto", range=None, density=True, weights=None, cumulative=False, bottom=None, histtype="bar", align="mid", orientation="vertical", rwidth=None, log=False, color=None, label=None, stacked=False)
    ax_h.set_xlabel("Power [dBW]")
    #ax_h.legend(["Window " + str(i) for i in range(WIN_NUM)])
    fig_h.canvas.flush_events()
    ax_h.clear()

def w_plot(sdr, fig_w, ax_w, rx2plot_out):
    ax_lim = np.array([[0.,0.],[0.,0.]])
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            sdr_data = rx2plot_out.recv()
            _, _, sample_rate, samples_num, _, _, _, _ = sdr_data
            IQ_t_1       = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind    += 1
            update_w_fig(fig_w, ax_w, ax_lim, IQ_t_1, None, sample_rate, samples_num, [0], 1)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, _, _ = sdr_data
            update_w_fig(fig_w, ax_w, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num)

def update_w_fig(fig_w, ax_w, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num):
    wvl_1 = signal.cwt(IQ_t_1, signal.ricker, np.arange(1,64))
    if channels_num == 2:
        wvl_2 = signal.cwt(IQ_t_2, signal.ricker, np.arange(1,64))
        ax_w.pcolormesh(wvl_1 + wvl_2)
    else:
        ax_w.pcolormesh(wvl_1)
    ax_w.set_xlabel("Translation [sample]")
    ax_w.set_ylabel("Scale (1/Frequency) [1/Hz]")
    fig_w.canvas.flush_events()
    ax_w.clear()

def corr_plot(sdr, fig_corr, ax_corr, rx2plot_out, plot2sens_in):
    ax_corr_r, ax_corr_i = ax_corr[0], ax_corr[1]
    ax_lim               = np.array([[0.,0.],[0.,0.]])
    if CORR_ARR_EN:
        sdr_data = rx2plot_out.recv()
        _, _, _, samples_num, _, _, _, _ = sdr_data
        CORR_ARR_LEN = len(CORR_ARR)
        corr_arr = CORR_ARR
        if CORR_ARR_LEN < CORR_LEN:
            corr_arr = np.pad(corr_arr, (0, CORR_LEN - CORR_ARR_LEN))
        elif CORR_ARR_LEN > CORR_LEN:
            corr_arr = corr_arr[:samples_num]
    else:
        corr_arr = None
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            sdr_data = rx2plot_out.recv()
            _, _, _, samples_num, _, _, _, _ = sdr_data
            IQ_t_1       = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind    += 1
            update_corr_fig(fig_corr, ax_corr_r, ax_corr_i, ax_lim, IQ_t_1, None, samples_num, [0], 1, corr_arr, plot2sens_in)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, _, samples_num, channels, channels_num,  _, _ = sdr_data
            update_corr_fig(fig_corr, ax_corr_r, ax_corr_i, ax_lim, IQ_t_1, IQ_t_2, samples_num, channels, channels_num, corr_arr, plot2sens_in)

def update_corr_fig(fig_corr, ax_corr_r, ax_corr_i, ax_lim, IQ_t_1, IQ_t_2, samples_num, channels, channels_num, corr_arr, plot2sens_in):
    IQ_t_1 = signal.detrend(IQ_t_1)
    if CORR_ARR_EN:
        corr_arr_1 = corr_arr
    else:
        corr_arr_1 = IQ_t_1
    if channels_num == 2:
        IQ_t_2 = signal.detrend(IQ_t_2)
        if CORR_ARR_EN:
            corr_arr_2 = corr_arr
        else:
            corr_arr_2 = IQ_t_2
        IQ_t_comb     = np.array([IQ_t_1, IQ_t_2])
        IQ_t_comb     = IQ_t_comb[channels]
        corr_arr_comb = np.array([corr_arr_1, corr_arr_2])
        corr_arr_comb = corr_arr_comb[channels]
        corr_1        = signal.correlate(IQ_t_comb[0], corr_arr_comb[0])
        corr_1        = corr_1[samples_num - 1:samples_num + CORR_LEN]
        corr_2        = signal.correlate(IQ_t_comb[1], corr_arr_comb[1])
        corr_2        = corr_2[samples_num - 1:samples_num + CORR_LEN]
        ax_corr_r.plot(corr_1.real)
        ax_corr_i.plot(corr_1.imag)
        ax_corr_r.plot(corr_2.real)
        ax_corr_i.plot(corr_2.imag)
    else:
        corr_1 = signal.correlate(IQ_t_1, corr_arr_1)
        corr_1 = corr_1[samples_num - 1:samples_num + CORR_LEN]
        corr_2 = None
        ax_corr_r.plot(corr_1.real)
        ax_corr_i.plot(corr_1.imag)
    legend = LEGEND_LABELS[channels]
    legend = legend.tolist()
    ax_corr_r.set_xlabel("Lag [-]")
    ax_corr_r.set_ylabel("Correlation (real) [-]")
    ax_corr_r.legend(legend)
    ax_corr_i.set_xlabel("Lag [-]")
    ax_corr_i.set_ylabel("Correlation (imaginary) [-]")
    ax_corr_i.legend(legend)
    if SENS_EN:
        pass
        #? = plot2sens_in.recv()
        #ax_corr.plot(?)
    fig_corr.canvas.flush_events()
    ax_corr_r.clear()
    ax_corr_i.clear()
