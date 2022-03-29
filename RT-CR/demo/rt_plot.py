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

COLORMAP      = plt.get_cmap("hsv")
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
            fig.suptitle("Time-domain RX IQ signal")
        elif PLOT_TYPE == 1 or PLOT_TYPE == "constellation": #constellation
            fig, ax = plt.subplots(1, num="SDR RX IQ constellation")
            fig.suptitle("RX IQ constellation")
            rcParams["lines.linestyle"] = "none"
        elif PLOT_TYPE == 2 or PLOT_TYPE == "FFT": #FFT
            fig, ax = plt.subplots(2, num="SDR RX IQ FFT")
            fig.suptitle("RX IQ FFT")
        elif PLOT_TYPE == 3 or PLOT_TYPE == "power spectrum": #power spectrum
            fig, ax = plt.subplots(1, num="SDR RX IQ power spectrum")
            fig.suptitle("RX IQ power spectrum")
        elif PLOT_TYPE == 4 or PLOT_TYPE == "spectrogram": #spectrogram
            fig, ax = plt.subplots(1, num="SDR RX IQ spectrogram")
            fig.suptitle("RX IQ spectrogram")
        elif PLOT_TYPE == 5 or PLOT_TYPE == "histogram": #histogram
            fig, ax = plt.subplots(1, num="SDR RX IQ power spectrum histogram")
            fig.suptitle("RX IQ power spectrum histogram")
        elif PLOT_TYPE == 6 or PLOT_TYPE == "wavelet": #wavelet
            fig, ax = plt.subplots(1, num="SDR RX IQ wavelet")
            fig.suptitle("RX IQ wavelet")
        elif PLOT_TYPE == 7 or PLOT_TYPE == "correlation": #correlation
            fig, ax = plt.subplots(2, num="SDR RX IQ correlation")
            fig.suptitle("RX IQ correlation")
        elif PLOT_TYPE == 8 or PLOT_TYPE == "cross-spectral density": #cross-spectral density
            fig, ax = plt.subplots(2, num="SDR RX IQ cross-spectral density")
            fig.suptitle("RX IQ cross-spectral density")
        elif PLOT_TYPE == 9 or PLOT_TYPE == "coherence": #coherence
            fig, ax = plt.subplots(1, num="SDR RX IQ coherence")
            fig.suptitle("RX IQ coherence")
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
    elif PLOT_TYPE == 7 or PLOT_TYPE == "cross-spectral density": #cross-spectral density
        csd_plot(sdr, fig, ax, rx2plot_out, plot2sens_in)
    elif PLOT_TYPE == 8 or PLOT_TYPE == "coherence": #coherence
        coh_plot(sdr, fig, ax, rx2plot_out, plot2sens_in)
    elif PLOT_TYPE == 9 or PLOT_TYPE == "fft": #FFT
        fft_plot(sdr, fig, ax, rx2plot_out, plot2sens_in)

def t_plot(sdr, fig_t, ax_t, rx2plot_out):
    ax_I_t, ax_Q_t     = ax_t[0], ax_t[1]
    ax_I_lim, ax_Q_lim = np.array([[0.,0.], [0.,0.]]), np.array([[0.,0.], [0.,0.]])
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            sdr_data  = rx2plot_out.recv()
            _, _, sample_rate, samples_num, _, _, _, _ = sdr_data
            IQ_t_1    = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind += 1
            ax_I_lim, ax_Q_lim = update_t_axes(fig_t, ax_I_t, ax_Q_t, ax_I_lim, ax_Q_lim, IQ_t_1, None, sample_rate, samples_num, [0], 1)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num,  _, _ = sdr_data
            ax_I_lim, ax_Q_lim = update_t_axes(fig_t, ax_I_t, ax_Q_t, ax_I_lim, ax_Q_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num)

def update_t_axes(fig_t, ax_I_t, ax_Q_t, ax_I_lim, ax_Q_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num):
    times        = np.linspace(0, samples_num / sample_rate, samples_num)
    I_t_1, Q_t_1 = split_IQ(IQ_t_1)
    if channels_num == 2:
        I_t_2, Q_t_2 = split_IQ(IQ_t_2)
        I_t_comb     = np.array([I_t_1, I_t_2])
        I_t_comb     = sort_channels(I_t_comb, channels)
        Q_t_comb     = np.array([Q_t_1, Q_t_2])
        Q_t_comb     = sort_channels(Q_t_comb, channels)
        ax_I_t.plot(times, I_t_comb[0], times, I_t_comb[1])
        ax_I_lim     = scale_axes(ax_I_t, I_t_comb, ax_I_lim, axes="Y", multichan=True, symm=True)
        ax_Q_t.plot(times, Q_t_comb[0], times, Q_t_comb[1])
        ax_Q_lim     = scale_axes(ax_Q_t, Q_t_comb, ax_Q_lim, axes="Y", multichan=True, symm=True)
    else:
        ax_I_t.plot(times, I_t_1)
        ax_I_lim = scale_axes(ax_I_t, I_t_1, ax_I_lim, axes="Y", multichan=False, symm=True)
        ax_Q_t.plot(times, Q_t_1)
        ax_Q_lim = scale_axes(ax_Q_t, Q_t_1, ax_Q_lim, axes="Y", multichan=False, symm=True)
    legend = LEGEND_LABELS[channels]
    legend = legend.tolist()
    ax_I_t.set_xlabel("Time [s]")
    ax_I_t.set_ylabel("In-phase [mV]")
    ax_I_t.legend(legend)
    ax_Q_t.set_xlabel("Time [s]")
    ax_Q_t.set_ylabel("Quadrature [mV]")
    ax_Q_t.legend(legend)
    fig_t.canvas.flush_events()
    ax_I_t.clear()
    ax_Q_t.clear()
    return ax_I_lim, ax_Q_lim

def c_plot(sdr, fig_c, ax_c, rx2plot_out):
    ax_lim = np.array([[0.,0.],[0.,0.]])
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            sdr_data  = rx2plot_out.recv()
            _, _, _, samples_num, _, _, _, _ = sdr_data
            IQ_t_1    = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind += 1
            ax_lim    = update_c_axes(fig_c, ax_c, ax_lim, IQ_t_1, None, [0], 1)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, _, _, channels, channels_num, _, _ = sdr_data
            ax_lim   = update_c_axes(fig_c, ax_c, ax_lim, IQ_t_1, IQ_t_2, channels, channels_num)

def update_c_axes(fig_c, ax_c, ax_lim, IQ_t_1, IQ_t_2, channels, channels_num):
    I_t_1, Q_t_1 = split_IQ(IQ_t_1)
    if channels_num == 2:
        I_t_2, Q_t_2 = split_IQ(IQ_t_2)
        I_t_comb = np.array([I_t_1, I_t_2])
        I_t_comb = sort_channels(I_t_comb, channels)
        Q_t_comb = np.array([Q_t_1, Q_t_2])
        Q_t_comb = sort_channels(Q_t_comb, channels)
        ax_c.plot(I_t_comb[0], Q_t_comb[0], I_t_comb[1], Q_t_comb[1])
        ax_lim = scale_axes(ax_c, np.array([I_t_comb, Q_t_comb]), ax_lim, axes="XY", multichan=True, symm=True)
    else:
        ax_c.plot(I_t_1, Q_t_1)
        ax_lim = scale_axes(ax_c, np.array([I_t_1, Q_t_1]), ax_lim, axes="XY", multichan=False, symm=True)
    legend = LEGEND_LABELS[channels]
    legend = legend.tolist()
    ax_c.set_xlabel("In-phase [mV]")
    ax_c.set_ylabel("Quadrature [mV]")
    ax_c.legend(legend)
    fig_c.canvas.flush_events()
    ax_c.clear()
    return ax_lim

def fft_plot(sdr, fig_fft, ax_fft, rx2plot_out, plot2sens_in):
    ax_fft_r, ax_fft_i = ax_fft[0], ax_fft[1]
    ax_r_lim, ax_i_lim = np.array([[0.,0.], [0.,0.]]), np.array([[0.,0.], [0.,0.]])
    if COMP_ARR_EN:
        sdr_data = rx2plot_out.recv()
        _, _, _, samples_num, _, _, _, _ = sdr_data
        COMP_ARR_LEN = len(COMP_ARR)
        fft_arr = COMP_ARR
        if COMP_ARR_LEN < COMP_LEN:
            fft_arr = np.pad(fft_arr, (0, COMP_LEN - COMP_ARR_LEN))
        elif COMP_ARR_LEN > COMP_LEN:
            fft_arr = fft_arr[:samples_num]
    else:
        fft_arr = None
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            sdr_data = rx2plot_out.recv()
            _, _, sample_rate, samples_num, _, _, _, _ = sdr_data
            IQ_t_1             = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind          += 1
            ax_r_lim, ax_i_lim = update_fft_axes(fig_fft, ax_fft_r, ax_fft_i, ax_r_lim, ax_i_lim, IQ_t_1, None, sample_rate, samples_num, [0], 1, fft_arr, plot2sens_in)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num,  _, _ = sdr_data
            ax_r_lim, ax_i_lim = update_fft_axes(fig_fft, ax_fft_r, ax_fft_i, ax_r_lim, ax_i_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, fft_arr, plot2sens_in)

def update_fft_axes(fig_fft, ax_fft_r, ax_fft_i, ax_r_lim, ax_i_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, fft_arr, plot2sens_in):
    if channels_num == 2:
        if COMP_ARR_EN:
            fft_arr_comb = np.array([fft_arr, fft_arr])
        else:
            fft_arr_comb = np.array([IQ_t_1, IQ_t_2])
        IQ_t_comb        = np.array([IQ_t_1, IQ_t_2])
        fft_arr_comb     = sort_channels(fft_arr_comb, channels)
        IQ_t_comb        = sort_channels(IQ_t_comb, channels)
        fft_1            = np.fft.fft(IQ_t_comb[0], n=FFT_LEN)
        fft_r_1, fft_i_1 = fft_1.real, fft_1.imag
        fft_2            = np.fft.fft(IQ_t_comb[1], n=FFT_LEN)
        fft_r_2, fft_i_2 = fft_2.real, fft_2.imag
        freqs            = np.fft.fftfreq(FFT_LEN, d=1/sample_rate)
        ax_fft_r.plot(freqs, fft_1.real, freqs, fft_2.real)
        ax_r_lim         = scale_axes(ax_fft_r, np.array([fft_r_1, fft_r_2]), ax_r_lim, axes="Y", multichan=True, symm=False)
        ax_fft_i.plot(freqs, fft_1.imag, freqs, fft_2.imag)
        ax_i_lim         = scale_axes(ax_fft_i, np.array([fft_i_1, fft_i_2]), ax_i_lim, axes="Y", multichan=True, symm=False)
    else:
        if COMP_ARR_EN:
            fft_arr = fft_arr
        else:
            fft_arr = IQ_t_1
        fft   = np.fft.fft(IQ_t_1, n=FFT_LEN)
        fft_r = fft.real
        fft_i = fft.imag
        freqs = np.fft.fftfreq(FFT_LEN, d=1/sample_rate)
        ax_fft_r.plot(freqs, fft_r)
        ax_r_lim = scale_axes(ax_fft_r, fft_r, ax_r_lim, axes="Y", multichan=False, symm=False)
        ax_fft_i.plot(freqs, fft_i)
        ax_i_lim = scale_axes(ax_fft_i, fft_i, ax_i_lim, axes="Y", multichan=False, symm=False)
    legend = sort_channels(LEGEND_LABELS, channels)
    legend = legend.tolist()
    ax_fft_r.set_xlabel("Frequency [MHz]")
    ax_fft_r.set_ylabel("Magnitude (real) [-]")
    ax_fft_r.legend(legend)
    ax_fft_i.set_xlabel("Frequency [MHz]")
    ax_fft_i.set_ylabel("Magnitude (imag) [-]")
    ax_fft_i.legend(legend)
    if SENS_EN:
        pass
        #? = plot2sens_in.recv()
        #ax_fft_r.plot(?)
        #ax_fft_i.plot(?)
    fig_fft.canvas.flush_events()
    ax_fft_r.clear()
    ax_fft_i.clear()
    return ax_r_lim, ax_i_lim

def ps_plot(sdr, fig_ps, ax_ps, rx2plot_out, plot2sens_in):
    ax_lim = np.array([[0.,0.],[0.,0.]])
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            sdr_data  = rx2plot_out.recv()
            _, _, sample_rate, samples_num, _, _, _, _ = sdr_data
            IQ_t_1    = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind += 1
            ax_lim = update_ps_axes(fig_ps, ax_ps, ax_lim, IQ_t_1, None, sample_rate, samples_num, [0], 1, plot2sens_in)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, _, _ = sdr_data
            ax_lim = update_ps_axes(fig_ps, ax_ps, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, plot2sens_in)

def update_ps_axes(fig_ps, ax_ps, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, plot2sens_in):
    assert FFT_LEN <= samples_num, "FFT length (" + str(FFT_LEN) + ") must be less than or equal to the buffer size(" + str(samples_num) + ")"
    if channels_num == 2:
        IQ_t_comb      = np.array([IQ_t_1, IQ_t_2])
        IQ_t_comb      = sort_channels(IQ_t_comb, channels)
        freqs, IQ_ps_1 = signal.welch(IQ_t_1, fs=sample_rate, window="hanning", nperseg=SWIN_LEN, noverlap=SWIN_OVERLAP_LEN, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", average="mean")
        _, IQ_ps_2     = signal.welch(IQ_t_2, fs=sample_rate, window="hanning", nperseg=SWIN_LEN, noverlap=SWIN_OVERLAP_LEN, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", average="mean")
        freqs          = np.fft.fftshift(freqs)
        IQ_ps_1        = 10 * np.log10(IQ_ps_1 + 1e-16)
        IQ_ps_2        = 10 * np.log10(IQ_ps_2 + 1e-16)
        IQ_ps_1        = np.fft.fftshift(IQ_ps_1)
        IQ_ps_2        = np.fft.fftshift(IQ_ps_2)
        ax_ps.plot(freqs, IQ_ps_1, freqs, IQ_ps_2)
        ax_lim         = scale_axes(ax_ps, np.array([IQ_ps_1, IQ_ps_2]), ax_lim, axes="Y", multichan=True, symm=False)
    else:
        freqs, IQ_ps = signal.welch(IQ_t_1, fs=sample_rate, window="hanning", nperseg=SWIN_LEN, noverlap=SWIN_OVERLAP_LEN, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", average="mean")
        IQ_ps        = 10 * np.log10(IQ_ps + 1e-16)
        freqs        = np.fft.fftshift(freqs)
        IQ_ps        = np.fft.fftshift(IQ_ps)
        ax_ps.plot(freqs, IQ_ps.real)
        ax_lim       = scale_axes(ax_ps, IQ_ps, ax_lim, axes="Y", multichan=False, symm=False)
    legend = LEGEND_LABELS[channels]
    legend = legend.tolist()
    ax_ps.set_xlabel("Frequency [Hz]")
    ax_ps.set_ylabel("Power [dBW]")
    ax_ps.legend(legend)
    if SENS_EN:
        signal_clas, max_ind, min_ind = plot2sens_in.recv()
        ax_ps.plot(freqs[max_ind], IQ_ps[max_ind], "o", freqs[min_ind], IQ_ps[min_ind], "o", freqs, signal_clas)
    fig_ps.canvas.flush_events()
    ax_ps.clear()
    return ax_lim

def sp_plot(sdr, fig_sp, ax_sp, rx2plot_out):
    ax_lim = np.array([[0.,0.],[0.,0.]])
    _, _, _, samples_num, _, _,  _, _ = get_sdr_data(sdr)
    iter_sp  = np.zeros((SP_CHUNKS_NUM, FFT_LEN))
    freqs_sp = np.zeros((SP_CHUNKS_NUM, FFT_LEN))
    SP_1     = np.zeros((SP_CHUNKS_NUM, FFT_LEN))
    SP_2     = np.zeros((SP_CHUNKS_NUM, FFT_LEN))
    i        = 0 #iterations
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            i += 1
            sdr_data = rx2plot_out.recv()
            _, _, sample_rate, samples_num, _, _,  _, _ = sdr_data
            IQ_t_1       = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind    += 1
            freqs_sp, SP_1, SP_2 = update_sp_axes(fig_sp, ax_sp, ax_lim, IQ_t_1, None, sample_rate, samples_num, [0], 1, iter_sp, freqs_sp, SP_1, SP_2, i)
    else:
        while True:
            i += 1
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, _, _ = sdr_data
            freqs_sp, SP_1, SP_2 = update_sp_axes(fig_sp, ax_sp, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, iter_sp, freqs_sp, SP_1, SP_2, i)

def update_sp_axes(fig_sp, ax_sp, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, iter_sp, freqs_sp, SP_1, SP_2, i):
    freqs, IQ_sp_1  = signal.welch(IQ_t_1, fs=sample_rate, window="hanning", nperseg=SWIN_LEN, noverlap=None, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", average="mean")
    iter_sp          = matlib.repmat(np.arange(0, SP_CHUNKS_NUM), FFT_LEN, 1).T
    freqs_c          = np.fft.fftshift(freqs)
    freqs_sp[:-1, :] = freqs_sp[1:]
    freqs_sp[-1, :]  = freqs_c
    IQ_sp_1_c        = np.fft.fftshift(IQ_sp_1)
    IQ_sp_1_c        = 10 * np.log10(IQ_sp_1_c + 1e-16)
    SP_1[:-1, :]     = SP_1[1:]
    SP_1[-1, :]      = IQ_sp_1_c
    if channels_num == 2:
        _, IQ_sp_2  = signal.welch(IQ_t_2, fs=sample_rate, window="hanning", nperseg=SWIN_LEN, noverlap=None, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", average="mean")
        IQ_sp_2_c    = np.fft.fftshift(IQ_sp_2)
        IQ_sp_2_c    = 10 * np.log10(IQ_sp_2_c + 1e-16)
        SP_2[:-1, :] = SP_2[1:]
        SP_2[-1, :]  = IQ_sp_2_c
        ax_sp.pcolormesh(iter_sp, freqs_sp, 0.5 * (SP_1 + SP_2), shading="gouraud")
    else:
        ax_sp.pcolormesh(iter_sp, freqs_sp, SP_1, shading="gouraud")
    ax_sp.set_xlabel("Iteration [-]")
    ax_sp.set_ylabel("Frequency [Hz]")
    tick_labels = ["-" + str(i) for i in range(SP_CHUNKS_NUM - 1, 0, -1)]
    tick_labels.append("0")
    ax_sp.set_xticks(np.arange(0, SP_CHUNKS_NUM), labels=tick_labels)
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
            update_h_axes(fig_h, ax_h, ax_lim, IQ_t_1, None, sample_rate, samples_num, [0], 1)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, _, _ = sdr_data
            update_h_axes(fig_h, ax_h, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num)

def update_h_axes(fig_h, ax_h, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num):
    freqs, IQ_ps_1 = signal.welch(IQ_t_1, fs=sample_rate, window="hanning", nperseg=SWIN_LEN, noverlap=None, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", average="mean")
    IQ_ps_1        = 10 * np.log10(IQ_ps_1 + 1e-16)
    freqs          = np.fft.fftshift(freqs)
    IQ_ps_1        = np.fft.fftshift(IQ_ps_1)
    IQ_ps_win_1    = subwindows(IQ_ps_1, SWIN_LEN, SWIN_OVERLAP_LEN)
    SWIN_NUM       = len(IQ_ps_win_1)
    ax_h.set_prop_cycle("color", [COLORMAP(1.*i/SWIN_NUM) for i in range(SWIN_NUM)])
    if channels_num == 2:
        _, IQ_ps_2  = signal.welch(IQ_t_2, fs=sample_rate, window="hanning", nperseg=SWIN_LEN, noverlap=None, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", average="mean")
        IQ_ps_2     = 10 * np.log10(IQ_ps_2 + 1e-16)
        IQ_ps_2     = np.fft.fftshift(IQ_ps_2)
        IQ_ps_win_2 = subwindows(IQ_ps_2, SWIN_LEN, SWIN_OVERLAP_LEN)
        for windw_ind in range(SWIN_NUM):
            ax_h.hist(IQ_ps_win_1[windw_ind] + IQ_ps_win_2[windw_ind], bins="auto", range=None, density=True, weights=None, cumulative=False, bottom=None, histtype="bar", align="mid", orientation="vertical", rwidth=None, log=False, color=None, label=None, stacked=False)
    else:
        for windw_ind in range(SWIN_NUM):
            ax_h.hist(IQ_ps_win_1[windw_ind], bins="auto", range=None, density=True, weights=None, cumulative=False, bottom=None, histtype="bar", align="mid", orientation="vertical", rwidth=None, log=False, color=None, label=None, stacked=False)
        ax_lim = scale_axes(ax_c, np.array([I_t_1, Q_t_1]), ax_lim, axes="XY", multichan=False, symm=True)
    ax_h.set_xlabel("Power [dBW]")
    #ax_h.legend(["Window " + str(i) for i in range(SWIN_NUM)])
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
            update_w_axes(fig_w, ax_w, ax_lim, IQ_t_1, None, sample_rate, samples_num, [0], 1)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, _, _ = sdr_data
            update_w_axes(fig_w, ax_w, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num)

def update_w_axes(fig_w, ax_w, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num):
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
    ax_r_lim, ax_i_lim   = np.array([[0.,0.],[0.,0.]]), np.array([[0.,0.],[0.,0.]])
    if COMP_ARR_EN:
        sdr_data = rx2plot_out.recv()
        _, _, _, samples_num, _, _, _, _ = sdr_data
        COMP_ARR_LEN = len(COMP_ARR)
        corr_arr = COMP_ARR
        if COMP_ARR_LEN < COMP_LEN:
            corr_arr = np.pad(corr_arr, (0, COMP_LEN - COMP_ARR_LEN))
        elif COMP_ARR_LEN > COMP_LEN:
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
            ax_r_lim, ax_i_lim = update_corr_axes(fig_corr, ax_corr_r, ax_corr_i, ax_r_lim, ax_i_lim, IQ_t_1, None, samples_num, [0], 1, corr_arr, plot2sens_in)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, _, samples_num, channels, channels_num,  _, _ = sdr_data
            ax_r_lim, ax_i_lim = update_corr_axes(fig_corr, ax_corr_r, ax_corr_i, ax_r_lim, ax_i_lim, IQ_t_1, IQ_t_2, samples_num, channels, channels_num, corr_arr, plot2sens_in)

def update_corr_axes(fig_corr, ax_corr_r, ax_corr_i, ax_r_lim, ax_i_lim, IQ_t_1, IQ_t_2, samples_num, channels, channels_num, corr_arr, plot2sens_in):
    IQ_t_1 = signal.detrend(IQ_t_1)
    if COMP_ARR_EN:
        corr_arr_1 = corr_arr
    else:
        corr_arr_1 = IQ_t_1
    if channels_num == 2:
        IQ_t_2 = signal.detrend(IQ_t_2) #substract mean
        if COMP_ARR_EN:
            corr_arr_2 = corr_arr
        else:
            corr_arr_2 = IQ_t_2
        IQ_t_comb          = np.array([IQ_t_1, IQ_t_2])
        IQ_t_comb          = IQ_t_comb[channels]
        corr_arr_comb      = np.array([corr_arr_1, corr_arr_2])
        corr_arr_comb      = corr_arr_comb[channels]
        corr_1             = signal.correlate(IQ_t_comb[0], corr_arr_comb[0])
        corr_1             = corr_1[samples_num - 1:samples_num + COMP_LEN]
        corr_2             = signal.correlate(IQ_t_comb[1], corr_arr_comb[1])
        corr_2             = corr_2[samples_num - 1:samples_num + COMP_LEN]
        corr_1_r, corr_1_i = corr_1.real, corr_1.imag
        corr_2_r, corr_2_i = corr_2.real, corr_2.imag
        ax_corr_r.plot(corr_1_r)
        ax_corr_r.plot(corr_2_r)
        ax_r_lim           = scale_axes(ax_corr_r, np.array([corr_1_r, corr_2_r]), ax_r_lim, axes="Y", multichan=True, symm=False)
        ax_corr_i.plot(corr_1_i)
        ax_corr_i.plot(corr_2_i)
        ax_i_lim           = scale_axes(ax_corr_i, np.array([corr_1_i, corr_2_i]), ax_i_lim, axes="Y", multichan=True, symm=False)
    else:
        corr_1             = signal.correlate(IQ_t_1, corr_arr_1) #substract mean
        corr_1             = corr_1[samples_num - 1:samples_num + COMP_LEN]
        corr_2             = None
        corr_1_r, corr_1_i = corr_1.real, corr_1.imag
        ax_corr_r.plot(corr_1_r)
        ax_r_lim           = scale_axes(ax_corr_r, corr_1_r, ax_r_lim, axes="Y", multichan=False, symm=False)
        ax_corr_i.plot(corr_1_i)
        ax_i_lim           = scale_axes(ax_corr_i, corr_1_i, ax_i_lim, axes="Y", multichan=False, symm=False)
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
    return ax_r_lim, ax_i_lim

def csd_plot(sdr, fig_csd, ax_csd, rx2plot_out, plot2sens_in):
    ax_csd_r, ax_csd_i = ax_csd[0], ax_csd[1]
    ax_r_lim, ax_i_lim = np.array([[0.,0.],[0.,0.]]), np.array([[0.,0.],[0.,0.]])
    if COMP_ARR_EN:
        sdr_data = rx2plot_out.recv()
        _, _, _, samples_num, _, _, _, _ = sdr_data
        COMP_ARR_LEN = len(COMP_ARR)
        csd_arr = COMP_ARR
        if COMP_ARR_LEN < COMP_LEN:
            csd_arr = np.pad(csd_arr, (0, COMP_LEN - COMP_ARR_LEN))
        elif COMP_ARR_LEN > COMP_LEN:
            csd_arr = csd_arr[:samples_num]
    else:
        csd_arr = None
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            sdr_data = rx2plot_out.recv()
            _, _, sample_rate, samples_num, _, _, _, _ = sdr_data
            IQ_t_1             = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind          += 1
            ax_r_lim, ax_i_lim = update_csd_axes(fig_csd, ax_csd_r, ax_csd_i, ax_r_lim, ax_i_lim, IQ_t_1, None, sample_rate, samples_num, [0], 1, csd_arr, plot2sens_in)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num,  _, _ = sdr_data
            ax_r_lim, ax_i_lim = update_csd_axes(fig_csd, ax_csd_r, ax_csd_i, ax_r_lim, ax_i_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, csd_arr, plot2sens_in)

def update_csd_axes(fig_csd, ax_csd_r, ax_csd_i, ax_r_lim, ax_i_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, csd_arr, plot2sens_in):
    if channels_num == 2:
        if COMP_ARR_EN:
            csd_arr_comb = np.array([csd_arr, csd_arr])
        else:
            csd_arr_comb = np.array([IQ_t_1, IQ_t_2])
        IQ_t_comb    = np.array([IQ_t_1, IQ_t_2])
        csd_arr_comb = sort_channels(csd_arr_comb, channels)
        IQ_t_comb    = sort_channels(IQ_t_comb, channels)
        freqs, Pxy_1 = signal.csd(IQ_t_comb[0], csd_arr_comb[0], fs=sample_rate, window="hanning", nperseg=SWIN_LEN, noverlap=SWIN_OVERLAP_LEN, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", average="mean")
        _, Pxy_2 = signal.csd(IQ_t_comb[1], csd_arr_comb[1], fs=sample_rate, window="hanning", nperseg=SWIN_LEN, noverlap=SWIN_OVERLAP_LEN, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", average="mean")
        freqs            = np.fft.fftshift(freqs)
        Pxy_1            = np.fft.fftshift(Pxy_1)
        Pxy_2            = np.fft.fftshift(Pxy_2)
        Pxy_1_r, Pxy_1_i = Pxy_1.real, Pxy_1.imag
        Pxy_2_r, Pxy_2_i = Pxy_2.real, Pxy_2.imag
        ax_csd_r.plot(freqs, Pxy_1_r, freqs, Pxy_2_r)
        ax_r_lim         = scale_axes(ax_csd_r, np.array([Pxy_1_r, Pxy_2_r]), ax_r_lim, axes="Y", multichan=True, symm=False)
        ax_csd_i.plot(freqs, Pxy_1_i, freqs, Pxy_2_i)
        ax_i_lim         = scale_axes(ax_csd_i, np.array([Pxy_1_i, Pxy_2_i]), ax_i_lim, axes="Y", multichan=True, symm=False)
    else:
        if COMP_ARR_EN:
            csd_arr = csd_arr
        else:
            csd_arr = IQ_t_1
        freqs, Pxy = signal.csd(IQ_t_1, csd_arr, fs=sample_rate, window="hanning", nperseg=SWIN_LEN, noverlap=SWIN_OVERLAP_LEN, nfft=FFT_LEN, detrend="constant", return_onesided=False, scaling="spectrum", average="mean")
        freqs        = np.fft.fftshift(freqs)
        Pxy          = np.fft.fftshift(Pxy)
        Pxy_r, Pxy_i = Pxy.real, Pxy.imag
        ax_csd_r.plot(freqs, Pxy_r)
        ax_r_lim     = scale_axes(ax_csd_r, Pxy_r, ax_r_lim, axes="Y", multichan=False, symm=False)
        ax_csd_i.plot(freqs, Pxy_i)
        ax_i_lim     = scale_axes(ax_csd_i, Pxy_i, ax_i_lim, axes="Y", multichan=False, symm=False)
    legend = sort_channels(LEGEND_LABELS, channels)
    legend = legend.tolist()
    ax_csd_r.set_xlabel("Frequency [MHz]")
    ax_csd_r.set_ylabel("Cross-spectral density (real) [-]")
    ax_csd_r.legend(legend)
    ax_csd_i.set_xlabel("Frequency [MHz]")
    ax_csd_i.set_ylabel("Cross-spectral density (imag) [-]")
    ax_csd_i.legend(legend)
    if SENS_EN:
        pass
        #? = plot2sens_in.recv()
        #ax_csd_r.plot(?)
        #ax_csd_i.plot(?)
    fig_csd.canvas.flush_events()
    ax_csd_r.clear()
    ax_csd_i.clear()
    return ax_r_lim, ax_i_lim

def coh_plot(sdr, fig_coh, ax_coh, rx2plot_out, plot2sens_in):
    ax_lim = np.array([[0.,0.],[0.,0.]])
    if COMP_ARR_EN:
        sdr_data = rx2plot_out.recv()
        _, _, _, samples_num, _, _, _, _ = sdr_data
        COMP_ARR_LEN = len(COMP_ARR)
        coh_arr = COMP_ARR
        if COMP_ARR_LEN < COMP_LEN:
            coh_arr = np.pad(coh_arr, (0, COMP_LEN - COMP_ARR_LEN))
        elif COMP_ARR_LEN > COMP_LEN:
            coh_arr = coh_arr[:samples_num]
    else:
        coh_arr = None
    if not IQ_ARR is None:
        part_ind = 0
        while True:
            sdr_data     = rx2plot_out.recv()
            _, _, sample_rate, samples_num, _, _, _, _ = sdr_data
            IQ_t_1       = array1D_part(IQ_ARR, samples_num, part_ind)
            part_ind    += 1
            ax_lim = update_coh_axes(fig_coh, ax_coh, ax_lim, IQ_t_1, None, sample_rate, samples_num, [0], 1, coh_arr, plot2sens_in)
    else:
        while True:
            sdr_data = rx2plot_out.recv()
            IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num,  _, _ = sdr_data
            ax_lim = update_coh_axes(fig_coh, ax_coh, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, coh_arr, plot2sens_in)

def update_coh_axes(fig_coh, ax_coh, ax_lim, IQ_t_1, IQ_t_2, sample_rate, samples_num, channels, channels_num, coh_arr, plot2sens_in):
    if channels_num == 2:
        if COMP_ARR_EN:
            coh_arr_comb = np.array([coh_arr, coh_arr])
        else:
            coh_arr_comb = np.array([IQ_t_1, IQ_t_2])
        IQ_t_comb    = np.array([IQ_t_1, IQ_t_2])
        coh_arr_comb = sort_channels(coh_arr_comb, channels)
        IQ_t_comb    = sort_channels(IQ_t_comb, channels)
        freqs, Cxy_1 = signal.coherence(IQ_t_comb[0], coh_arr_comb[0], fs=sample_rate, window="hanning", nperseg=SWIN_LEN, noverlap=SWIN_OVERLAP_LEN, nfft=FFT_LEN, detrend="constant")
        _, Cxy_2     = signal.coherence(IQ_t_comb[1], coh_arr_comb[1], fs=sample_rate, window="hanning", nperseg=SWIN_LEN, noverlap=SWIN_OVERLAP_LEN, nfft=FFT_LEN, detrend="constant")
        freqs        = np.fft.fftshift(freqs)
        Cxy_1        = np.fft.fftshift(Cxy_1)
        Cxy_2        = np.fft.fftshift(Cxy_2)
        ax_coh.plot(freqs, Cxy_1, freqs, Cxy_2)
        ax_lim       = scale_axes(ax_coh, np.array([Cxy_1, Cxy_2]), ax_lim, axes="Y", multichan=True, symm=False)
    else:
        if COMP_ARR_EN:
            coh_arr = coh_arr
        else:
            coh_arr = IQ_t_1
        freqs, Cxy = signal.coherence(IQ_t_1, coh_arr, fs=sample_rate, window="hanning", nperseg=SWIN_LEN, noverlap=SWIN_OVERLAP_LEN, nfft=FFT_LEN, detrend="constant")
        freqs      = np.fft.fftshift(freqs)
        Cxy        = np.fft.fftshift(Cxy)
        ax_coh.plot(freqs, Cxy)
        ax_lim     = scale_axes(ax_coh, Cxy, ax_lim, axes="Y", multichan=False, symm=False)
    legend = sort_channels(LEGEND_LABELS, channels)
    legend = legend.tolist()
    ax_coh.set_xlabel("Frequency [MHz]")
    ax_coh.set_ylabel("Coherence (real) [-]")
    ax_coh.legend(legend)
    if SENS_EN:
        pass
        #? = plot2sens_in.recv()
        #ax_coh.plot(?)
    fig_coh.canvas.flush_events()
    ax_coh.clear()
    return ax_lim
