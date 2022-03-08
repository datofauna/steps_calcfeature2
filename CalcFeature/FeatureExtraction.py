# -*- coding: utf-8 -*-
"""
This module contains shared functions for wingbeat analysis.

You can add or change things but remember to administrate in the function
header what you changed.

Created on Fri Jun  9 15:03:04 2017

@author: FP
"""
import numpy as np

import pandas
import scipy.sparse
import scipy.interpolate
import scipy.signal
from sklearn.neighbors import KernelDensity
import warnings

from .Filter import *

# %% Basic functions
def calc_spectrum(time_signal, zero_pad=False):
    """Calculate the spectrum of an event

    Inputs
    ----------
    time_signal : np.array LxD (float)
        time signal of lenght L, with D channels
    zero_pad : Bool/Int
        Zero pad time signal to higher resolution

    Outputs
    ----------
    spectrum : np.array LxD (float)
      Full fourier spectrum, non-shifted
    20170720 made it compatible for higher dimensiosn
    """
    if not zero_pad:
        zp = None
    else:
        zp = zero_pad
    if time_signal.ndim == 1:
        spectrum = np.fft.fft(time_signal, n=zp)
    elif time_signal.ndim > 1:
        spectrum = np.fft.fft(time_signal, axis=1, n=zp)
    return spectrum


# %%
def optical_mass(signal):
    """Calculate the optical mass of an event

    Inputs
    ----------
    dist_signal : np.array M (float)
        distance signal

    Outputs
    ----------
    optical_mass : float
      The total sum of all the signal
    """
    if signal.ndim == 1:
        OM = np.sum(signal)
    elif signal.ndim == 2:
        OM = np.sum(signal, axis=1)
    elif signal.ndim == 3:
        OM = np.sum(signal, axis=(1, 2))
    return OM


# %%
def time_max(time_signal):
    """Calculate the time max of an event

    Inputs
    ----------
    time_signal : np.array M (float)
        time signal

    Outputs
    ----------
    time_max_sig : float
      The maximum signal in the time series
    """
    return time_signal.max()


# %%
def COM(signal):
    """Calculate the center of mass

    Inputs
    ----------
    signal : np.array M (float)
        time or distantce signal

    Outputs
    ----------
    COM : float
        the center of mass of the signal
    """
    # calculate COM in distance
    signal = signal - np.min(signal)
    Z = np.arange(0, len(signal))
    COM = sum(Z*signal)/sum(signal)
    return COM


# %%
def moving_average(signal, n=3):
    """
    Returns the moving average with a period of n

     Inputs
    ----------
    signal : np.array
        Vector that function operates on

    Outputs
    ----------
    mavg: np.array
        Moving average of signal with period n
    """

    ret = np.cumsum(signal, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    rtr_array = np.repeat(ret[0], n - 1)
    mavg = np.append(rtr_array, ret[n - 1:]/n)

    return mavg


# %% Wing beat frequency methods
def autocorr(signal, length="same"):
    """Calculate the autocorrelation of an array, signal,

    Inputs
    ----------
    signal : np.array M (float)
        Data signal to be autocorrelated
    length: {"same","full"}, optional
        Mode for the correlation. Refer to numpy.convolve for describtion of modes.
        Default "same"

    Outputs
    ----------
    auto_correlation: np.array M ("same"), 2*M-1 ("full")
        The auto-correlation of signal
    """

    autocorrelation = np.correlate(signal, signal, mode=length)
    return autocorrelation


# %%
def peaks_index(signal):
    """Calculate the indices that contains a peak in the array signal

    Inputs
    ----------
    signal : np.array M (float)
        Data signal

    Outputs
    ----------
    index: np.array N
        Output the indices of the peaks in a signal
        """

    index = (np.diff(np.sign(np.diff(signal))) < 0).nonzero()[0] + 1
    return index


# %%
def find_fundamental_auto(spectrum, t_start, t_stop, fs, freq_low=25):
    """Calculate the fundamental frequency of a wingbeat signal using
    autocorrelation twice

    Inputs
    ----------
    spectrum : np.array M (float)
        the fourier transform of the timesignal of the event
        (complex values and not shifted)
    t_start : integer
        minimum time pixel of the event
    t_stop : integer
        maximum time pixel of the event
    fs : float
        sampling frequency in Hz
    freq_low : integer, optional
        Frequency threshold below which frequencies are filtered out.
        Default 25 Hz

    Outputs
    ----------
    fund_freq_auto: float
        Fundamental wingbeat frequency of a signal.
        If no frequencies are present it will output NAN

    Changes:
    20160916 - JCP had to add a try except to filter an error,
                needs some attention
    20161207 - JHN changed to input spectrum instead of timesignal

    """
    # stepsize in time [s]
    dt = 1/fs

    # No. of samples in the time signal
    N_fft = spectrum.shape[0]

    # Frequency axis
    freq_ax = fs/(2*N_fft)*np.array(range(-N_fft, N_fft, 2))

    # Shifts spectrum to fit axis
    freq_sig = np.fft.fftshift(spectrum)

    # Filtered spectrum
    freq_filt = (abs(freq_ax) > freq_low)*freq_sig
    freq_peaks = peaks_index(abs(freq_sig))

    # Filtered time signal
    time_filt = np.fft.ifft(np.fft.ifftshift(freq_filt))
    t_sig_filt = np.real(time_filt[t_start:t_stop])
    # signal value below 0 filtered out
    thresh_sig = (t_sig_filt > 0)*t_sig_filt

    # Autocorrelation twice
    autoauto = autocorr(autocorr(thresh_sig))
    auto_thresh = (autoauto > (2*np.median(autoauto) - np.min(autoauto)))*autoauto
    # Peak indices of 2xautocorrelated signal
    auto_peaks = peaks_index(auto_thresh)

    if auto_peaks.shape[0] < 2:
        fund_freq_auto = 0
    else:
        # guess of fundamental frequency
        freq_guess = 1/(np.median(np.diff(auto_peaks))*dt)
        # error catch to quickly work around
        # "attempt to get argmin of an empty sequence"
        try:
            # find frequency peak closest to guess
            peak_ind = np.argmin(abs(freq_ax[freq_peaks] - freq_guess))
            fund_freq_auto = freq_ax[freq_peaks[peak_ind]]

        except ValueError:
            fund_freq_auto = 0

    return fund_freq_auto


# %%
def find_fundamental_cepstrum(spectrum, fs, f_max=1500):
    """ use ceptrum analysis to find the fundamental frequency. The cepstrum
    method is explained at different places in literature or on wiki.

    Inputs
    ----------
    spectrum : np.array L
        the fourier transform of the timesignal of the event
        (complex values and not shifted)
    fs: float
        the sampling frequency of the time signal
    f_max: maximum frequency allowed. The cepstrum will be cut there

    Outputs
    ----------
    fund_freq : float
        The fundamental frequency calculated from idx_d as 1/(idx_d/fs)

    Changes:
    20161207 - JHN changed to input spectrum instead of timesignal
    20161216 - JHN Added method 'new' which find the WBF better.
               csv-files created before this date uses the 'old' method
    20191115 - RARA replaced old/new with f_max having 1500 default
    """
    # calculate the cepstrum
    cepstrum = np.abs(np.fft.ifft(np.log(np.abs(spectrum)**2)))**2
    cutoff = fs // f_max
    
    # find the index of the peak in the cepstrum
    idx = np.argmax(cepstrum[cutoff:len(cepstrum)//2]) + cutoff

    # using the two naigbouring points to refine the accuracy of the peak
    idx_d = idx + np.sum(cepstrum[idx - 1:idx + 2]*np.array([-1, 0, 1]))/np.sum(cepstrum[idx - 1:idx + 2])


    # calculates thhe fundamental frequency from idx_d
    fund_freq = fs/idx_d

    if fund_freq > fs/2:
        fund_freq = 0

    return fund_freq


# %%
def find_fundamental_HPS(spectrum, fs, N=4):
    """ a function to find the fundamental frequency using the harmonic
    product spectrum (HPS). This method takes the product of a fft spectrum
    with its down sampled copies. If harmonics are present this product leads
    to one rather nice peak at the fundamental frequency.

    Inputs
    ----------
    spectrum : np.array
        the fourier transform of the timesignal of the event
        (complex values and not shifted)
    fs : int
        sampling frequency of the signal
    N : int
        the maximum down sampling value


    Outputs
    ----------
    HPS_frequency : float
        the fundamental frequency calculated as:
        freq = index*fs / signal_sample_length

    Changes:
    20161206 - KRY changed the "spectrum" to a input as the function is
                called repeatedly
    20161207 - JHN removed wing_signal input (only used for its length,
                which is the same as spectrum), and changed from absolute
                value to the complex spectrum
    """
    # calculate fft
    # spectrum = np.abs(np.fft.fft(wing_signal))

    real_spec = abs(spectrum)
    real_half = len(real_spec)//2
    max_len = real_half//N
    # create harmonic product spectrum
    HPS = real_spec[:max_len]
    for i in range(N-1):
        down_sample = scipy.signal.decimate(real_spec[:real_half], i + 2,
                                            zero_phase=True)
        HPS = HPS*down_sample[:max_len]
    # find the right peak
    idd1 = scipy.signal.argrelextrema(HPS, np.greater)[0]
    idd2 = scipy.signal.argrelextrema(HPS[idd1], np.greater)[0]
    try:
        index_fund = idd1[idd2[np.argmax(HPS[idd1[idd2]])]]
    except ValueError:
        index_fund = 0
    HPS_frequency = index_fund*fs/len(spectrum)

    if HPS_frequency > fs/2:
            HPS_frequency = 0

    return HPS_frequency


# %%
def find_fundamental_lund(spectrum, fs):
    """ Function to find the fundamental frequency of a wingbeat signal, based
    on the freqeuncy peaks.

    Inputs
    ----------
    spectrum : np.array
        the fourier transform of the timesignal of the event
        (complex values and not shifted)
    fs: float
        the sampling frequency of the time signal

    Outputs
    ----------
    fund_freq : float
        The fundamental frequency

    Changes:
    20161207 - JHN changed to input spectrum instead of timesignal
    """
    # samples in the time signal
    N_fft = spectrum.shape[0]

    # Shifts fft spectrum
    freq_sig = np.fft.fftshift(spectrum)

    # Calculates the power spectrum
    pwr_spec = 2*abs(freq_sig)

    # Generate frequency axis
    freq_ax = fs/(2*N_fft)*np.array(range(-N_fft, N_fft, 2))

    # Threshold for peaks
    threshold = np.median(pwr_spec)
    thresh_spec = (pwr_spec > threshold)*pwr_spec
    freq_peaks = (np.diff(np.sign(np.diff(thresh_spec))) < 0).nonzero()[0] + 1

    if len(freq_peaks > 1):
        # guess of fundamental frequency
        freq_guess = np.median(np.diff(freq_peaks))
        # find frequency peak closest to guess
        peak_ind = np.argmin(abs(freq_ax[freq_peaks]-freq_guess))
        # fundamental frequency
        fund_freq = freq_ax[freq_peaks[peak_ind]]
    else:
        # fundamental frequency if there is only one peak in spectrum
        fund_freq = 0

    return fund_freq, threshold


# %%
def find_fundamental_combined(HPS4, HPS3, HPS2, cepstrum, AC):
    """ Function to find the fundamental frequency as a combination of the methods
    HPS2, HPS3, HPS4, cepstrum and autocorrelation.

    Inputs
    ----------
    HPS4 : np.array L
        Array of frequencies found using the HPS4 method
    HPS3 : np.array L
        Array of frequencies found using the HPS3 method
    HPS2 : np.array L
        Array of frequencies found using the HPS2 method
    cepstrum : np.array L
        Array of frequencies found using the cepstrum method
    AC : np.array L
        Array of frequencies found using the autocorrelation method

    Outputs
    ----------
    fund_freq : float
        The fundamental frequency

    JHN - 20162510: Changed the limit of the filters from 0 to 10.
                    Thus methods that find frequencies ~0 (like 1.6)
                    does not overwrite other methods if they find real
                    frequencies.
    """

    if hasattr(HPS4, "__len__") == 0:
        fund_freq = 0
        isarray = 0
        HPS4 = np.array([HPS4, HPS4])
        HPS3 = np.array([HPS3, HPS3])
        HPS2 = np.array([HPS2, HPS2])
        cepstrum = np.array([cepstrum, cepstrum])
        AC = np.array([AC, AC])
    else:
        fund_freq = np.zeros(len(HPS4))
        isarray = 1

    # Step 1: if HPS4/2*HPS4 and HPS3 are within 10%, frequency is set to HPS4
    filt11 = WBF_compare_filter(HPS4, HPS3)*HPS4
    filt12 = WBF_compare_filter(2*HPS4, HPS3)*HPS4

    filt1 = filt11*(filt11 > 10)
    filt1[filt12 > 10] = filt12[filt12 > 10]

    fund_freq = filt1

    # Step 2: If HPS2 is within 10% of any method,
    # frequency is set to the given method
    filt21 = WBF_compare_filter(HPS2, cepstrum)*HPS2
    filt22 = WBF_compare_filter(HPS2, AC)*AC
    filt23 = WBF_compare_filter(HPS2, HPS3)*HPS3
    filt24 = WBF_compare_filter(HPS2, HPS4)*HPS4

    filt2 = filt21*(filt21 > 10)
    filt2[filt22 > 10] = filt22[filt22 > 10]
    filt2[filt23 > 10] = filt23[filt23 > 10]
    filt2[filt24 > 10] = filt24[filt24 > 10]

    fund_freq[filt2 > 0] = filt2[filt2 > 0]

    # Step 3: If HPS is within 10% of 2*HPS4/HPS3 or
    # 3*HPS4/HPS3, frequency is set to HPS4/HPS3
    filt31 = WBF_compare_filter(AC, 3*HPS3)*HPS3
    filt32 = WBF_compare_filter(AC, 3*HPS4)*HPS4
    filt33 = WBF_compare_filter(AC, 2*HPS3)*HPS3
    filt34 = WBF_compare_filter(AC, 2*HPS4)*HPS4

    filt3 = filt31*(filt31 > 10)
    filt3[filt32 > 10] = filt32[filt32 > 10]
    filt3[filt33 > 10] = filt33[filt33 > 10]
    filt3[filt34 > 10] = filt34[filt34 > 10]

    fund_freq[filt3 > 0] = filt3[filt3 > 0]

    # Step 4: If cepstrum is within 10% of any method,
    # freqeuncy is set to the given method
    filt41 = WBF_compare_filter(cepstrum, AC)*AC
    filt42 = WBF_compare_filter(cepstrum, HPS3)*HPS3
    filt43 = WBF_compare_filter(cepstrum, HPS4)*HPS4

    filt4 = filt41*(filt41 > 10)
    filt4[filt42 > 10] = filt42[filt42 > 10]
    filt4[filt43 > 10] = filt43[filt43 > 10]

    fund_freq[filt4 > 0] = filt4[filt4 > 0]

    # Step 5: If AC is within 10% of HPS4 or
    # HPS3 frequency is set to the given method
    filt51 = WBF_compare_filter(AC, HPS3)*HPS3
    filt52 = WBF_compare_filter(AC, HPS4)*HPS4

    filt5 = filt51*(filt51 > 10)
    filt5[filt52 > 10] = filt52[filt52 > 10]

    fund_freq[filt5 > 0] = filt5[filt5 > 0]

    if isarray == 0:
        fund_freq = fund_freq[0]

    return fund_freq


# %%
def find_fund_peaks(spectrum, fs, HPS4, HPS3, HPS2, cep, AC, peak_thresh=10):
    """ Function to find the fundamental frequency and value, as well as higher
    harmonics up to the first three. This is done by comparing the other
    methods to the peaks of the fourier spectrum.

    Inputs
    ----------
    spectrum : np.array
        the fourier transform of the timesignal of the event
        (complex values and not shifted)
    fs : scalar
        sampling frequency
    HPS4 : np.array L
        Array of frequencies found using the HPS4 method
    HPS3 : np.array L
        Array of frequencies found using the HPS3 method
    HPS2 : np.array L
        Array of frequencies found using the HPS2 method
    cepstrum : np.array L
        Array of frequencies found using the cepstrum method
    AC : np.array L
        Array of frequencies found using the autocorrelation method

    Outputs
    ----------
    fund_freq : float
        The fundamental frequency
    fund_val : float
        Signal strength in counts at fund_freq

    Changes:
    20161207 - JHN changed to input spectrum instead of timesignal
    20161220 - Split function into two. Anything relating to finding harmonics
                is in function 'find_harmonics'
    """
    # No. of samples in the time signal
    N_fft = spectrum.shape[0]
    # Frequency axis
    freq_ax = fs/(2*N_fft)*np.array(range(-N_fft, N_fft, 2))
    # Shifted FFT spectrum
    freq_sig = np.fft.fftshift(abs(spectrum))
    pos_sig = freq_sig[freq_ax >= 0]
    pos_ax = freq_ax[freq_ax >= 0]
    peak_ind = peaks_index(np.append([abs(pos_sig)], [0]))

    peak_val = abs(pos_sig[peak_ind])
    peak_freq = pos_ax[peak_ind]
    peak = {'Apeak_ind': peak_ind, 'Bpeak_val': peak_val,
            'Cpeak_freq': peak_freq}
    peak = pandas.DataFrame(data=peak)
    peak_filt = peak[peak['Cpeak_freq'] > 50]
    peak_desc = peak_filt.sort_values('Bpeak_val', ascending=0)

    if len(peak_desc) > peak_thresh:
        no_of_peaks = peak_thresh
    else:
        no_of_peaks = len(peak_desc)

    mask = np.zeros(no_of_peaks)
    rank = np.zeros(no_of_peaks)
    for i in range(no_of_peaks):
        rank[i] = i+1
        WBFcep = (cep > (peak_desc['Cpeak_freq'].iloc[i] - 10)) and (cep < (peak_desc['Cpeak_freq'].iloc[i] + 10))
        WBFAC = (AC > (peak_desc['Cpeak_freq'].iloc[i] - 5)) and (AC < (peak_desc['Cpeak_freq'].iloc[i] + 5))
        WBFHPS4 = (HPS4 > (peak_desc['Cpeak_freq'].iloc[i] - 5)) and (HPS4 < (peak_desc['Cpeak_freq'].iloc[i] + 5))
        WBFHPS3 = (HPS3 > (peak_desc['Cpeak_freq'].iloc[i] - 5)) and (HPS3 < (peak_desc['Cpeak_freq'].iloc[i] + 5))
        WBFHPS2 = (HPS2 > (peak_desc['Cpeak_freq'].iloc[i] - 5)) and (HPS2 < (peak_desc['Cpeak_freq'].iloc[i] + 5))
        mask[i] = 1*WBFcep + 1*WBFAC + 1*WBFHPS4 + 1*WBFHPS3 + 1*WBFHPS2

    WBF_factor = mask/rank

    if len(peak_desc) == 0:
        fund_freq = 0
        fund_val = 0
    else:
        WBF_factor = mask/rank
        if max(WBF_factor) == 0:
            fund_freq = 0
            fund_val = 0
        else:
            WBF_ind = np.argmax(WBF_factor)
            fund_freq = peak_desc['Cpeak_freq'].iloc[WBF_ind]
            fund_val = peak_desc['Bpeak_val'].iloc[WBF_ind]

    return [fund_freq, fund_val]


# %%
def find_harmonics(spectrum, fs, fund_freq, harm_thresh=0.015):
    """ Function to find harmonics in a spectrum, based on a found
    fundamental frequency. Finds the frequency and value of up to 3 harmonics.

    Inputs
    ----------
    spectrum : np.array
        the fourier transform of the timesignal of the event
        (complex values and not shifted)
    fs : scalar
        sampling freqeuncy
    fund_freq : float
        The estimated fundamental frequency
    harm_thresh : float
        Threshold for when a peak is considered a harmonic.
        Default 0.015 (1.5%)

    Outputs
    ----------
    harm_freq1 : float
        Frequency of the first harmonic (this is the second harmonic, 2*f0)
    harm_val1 : float
        Signal strength in counts at harm_freq1
    harm_freq2 : float
        Frequency of the seconds harmonic
    harm_val2 : float
        Signal strength in counts at harm_freq2
    harm_freq3 : float
        Frequency of the third harmonic
    harm_val3 : float
        Signal strength in counts at harm_freq3

    """
    # No. of samples in the time signal
    N_fft = spectrum.shape[0]

    # Frequency axis
    freq_ax = fs/(2*N_fft)*np.array(range(-N_fft, N_fft, 2))

    # Shifted FFT signal
    freq_sig = np.fft.fftshift(abs(spectrum))
    pos_sig = freq_sig[freq_ax >= 0]
    pos_ax = freq_ax[freq_ax >= 0]
    peak_ind = peaks_index(np.append([abs(pos_sig)], [0]))

    peak_val = abs(pos_sig[peak_ind])
    peak_freq = pos_ax[peak_ind]
    peak = {'Apeak_ind': peak_ind, 'Bpeak_val': peak_val,
            'Cpeak_freq': peak_freq}
    peak = pandas.DataFrame(data=peak)
    peak_filt = peak[peak['Cpeak_freq'] > 50]
    peak_desc = peak_filt.sort_values('Bpeak_val', ascending=0)

    high_peaks = peak_desc[0:4].sort_values('Cpeak_freq', ascending=1)
    peak_matrix = np.zeros(len(high_peaks)) + 100

    harm_freq1 = 0
    harm_freq2 = 0
    harm_freq3 = 0
    harm_val1 = 0
    harm_val2 = 0
    harm_val3 = 0
    harm_sum = 0

    if fund_freq != 0:
        # Remainder relative: abs(round(A/B) - A/B)
        peak_matrix = abs(round(high_peaks['Cpeak_freq']/fund_freq) - high_peaks['Cpeak_freq']/fund_freq)
        harm_matrix = round(high_peaks['Cpeak_freq']/fund_freq)

        harmonic_mask = (peak_matrix < (harm_matrix-1)*harm_thresh) & (harm_matrix > 0)
        harmonic = harmonic_mask*harm_matrix
#        max_factor = max(harm_matrix)
        harm_sum = sum(1*harmonic)

        if harm_sum > 0.5:
            for i in range(len(high_peaks)):
                    if harmonic.iloc[i] == 2:
                        harm_freq1 = high_peaks['Cpeak_freq'].iloc[i]
                        harm_val1 = high_peaks['Bpeak_val'].iloc[i]
                    if harmonic.iloc[i] == 3:
                        harm_freq2 = high_peaks['Cpeak_freq'].iloc[i]
                        harm_val2 = high_peaks['Bpeak_val'].iloc[i]
                    if harmonic.iloc[i] == 4:
                        harm_freq3 = high_peaks['Cpeak_freq'].iloc[i]
                        harm_val3 = high_peaks['Bpeak_val'].iloc[i]

    return harm_freq1, harm_val1, harm_freq2, harm_val2, harm_freq3, harm_val3


# %%
def detect_chg_slope(vector):
    """
    Finds two indeces in a vector when slope changes from - to +.
    Returns the index of the first two occurences. If no slope change is
    detected, it returns 0

     Inputs
    ----------
    vector : np.array
        Vector to find the slope change (typically the FFT signal)

    Outputs
    ----------
    index_slope_pos1 : float
        The index of the first occurence of change of slope from - to +
    index_slope_pos2 : float
        The index of the second occurence of change of slope from - to +
    """
    # Set detection of first time slope changes from - to + to False
    detect_up = False
    # Set detection of first time slope changes from + to - to False
    detect_down = False

    # Number of consecutive points that need to be + or - to detect a
    # slope change
    nPoints = 3
    try:
        # Loop over vector
        for i in range(0, len(vector) - nPoints):
            # If slope has changed from neg to positive once
            if (np.sum(np.diff(vector[i:i + nPoints]) > 0) > 1 and
               detect_up is False):
                detect_up = True
                index_slope_pos1 = i-1

            # If slope has changed to negative
            if (np.sum(np.diff(vector[i:i + nPoints]) < 0) > 1 and
               detect_up is True):

                detect_down = True

            # When slope changes back to positive (break if true)
            if(detect_up and
               detect_down and
               np.sum(np.diff(vector[i: i + nPoints]) > 0) > 1):

                index_slope_pos2 = i + nPoints
                break

            # If reaching the end, then return 0
            if(i == len(vector) - nPoints - 1):
                index_slope_pos1 = 0
                index_slope_pos2 = 0

    # If an error happened, return 0
    except:
        index_slope_pos1 = 0
        index_slope_pos2 = 0

    return index_slope_pos1, index_slope_pos2


# %%
def perc_close_numbers(n1, n2, perc):
    """Determines if two numbers are within a given percentage of each other.
    If true the function returns true, else it returns false

    Inputs
    ----------
    n1 : np.float
        Number 1
    n2 : np.float
        Number 2

    Outputs
    ----------
    result : Boolean
    """

    try:
        if(n1/n2 > (1 - perc) and n1/n2 < (1 + perc)):
            within = True
        else:
            within = False

    except:
        within = False

    return within


# %%
def find_fundamental_sgo(spectrum, fs, f_max = 1500):
    """ Tries to determine the fundamental frequency of a FFT signal using two
    successive linear Savitzky-Golay filters with a long and short period to
    smoothen the signal. Then look at the 1st derivative to find the first
    strong fluctuation and pick that as the fundamental frequency

    Inputs
    ----------
    spectrum : np.array
        the fourier transform of the timesignal of the event
        (complex values and not shifted)
    fs       : float
        the sampling frequency of the time signal


    Outputs
    ----------
    fund_freq : float
        The fundamental frequency
    """
    # Set percentage of number of points in spectrum to use for SGO filter
    s = 1
    # Get the frequencies for spectrum
    fftx = np.fft.fftfreq(len(spectrum), 1/fs)

    try:
        # Get index where freq > f_max
        sind = np.where(fftx > f_max)[0][0]
        # Set number of points used for short window filtering
        filter_short = np.floor((s/100)*sind).astype('int')

        if filter_short < 2:
            filter_short = 2

        if(np.remainder(filter_short, 2) == 0): #(window length must be odd)
            filter_short += 1

        # Set number of points used for long window filtering
        filter_long = np.floor((s*3/100)*sind).astype('int')

        if filter_long <= 2:
            filter_long = 2

        if(np.remainder(filter_long, 2) == 0):
            filter_long += 1

        # Set number of extra points to use after long filtering to avoid
        # spikes in 1st derivative
        sPoints = np.floor((s/2/100)*sind).astype('int')

        if sPoints == 0:
            sPoints = 1

        # Only grab the part of the signal that is below f_max
        yvect = np.abs(np.abs(spectrum[0:sind]))

        # Apply two consecutive linear Savitzky-Golay filters to signal**2
        smooth_yvect1 = scipy.signal.savgol_filter(yvect**2, filter_long, 1)
        smooth_yvect2 = scipy.signal.savgol_filter(smooth_yvect1, filter_short, 1)

        # Get first derivative of 2nd Savitzky-Golay filter
        if len(fftx) < 500:
            der1_sgo = np.diff(smooth_yvect2)
        else:
            der1_sgo = np.diff(smooth_yvect2[filter_long + sPoints:len(smooth_yvect2)])

        # Normalize according to largest change
        der1_sgo = der1_sgo/np.max(der1_sgo)

        if len(fftx) < 500:
            # Try to get index of the first place it crosses 0.4
            ind1 = np.where(der1_sgo > 0.4)[0][0]
            # Try to get index where it crosses -0.4
            ind2 = np.where(der1_sgo < -0.4)[0][np.where(der1_sgo < -0.4)[0] > ind1][0]

        else:
            # Try to get index of the first place it crosses 0.4
            ind1 = np.where(der1_sgo > 0.4)[0][0] + filter_long + sPoints
            # Try to get index where it crosses -0.4
            ind2 = np.where(der1_sgo < -0.4)[0][0] + filter_long + sPoints

        # Get the index of where the max in the spectrum occured between ind1
        # and ind2
        freq_ind = np.argmax(yvect[ind1:ind2]) + ind1

        # Get the frequency corresponding to index freq_ind
        fund_freq = fftx[freq_ind]

#        fig, ax = plt.subplots(3)
#        ax[0].plot(fftx, np.abs(spectrum), label=len(spectrum))
#        ax[1].plot(fftx[0:sind], smooth_yvect1, c='orange')
#        ax[1].plot(fftx[0:sind], smooth_yvect2, c='green')
#
#        ax[2].plot(fftx[:sind-1], der1_sgo, c='red')
##        ax[2].twinx().plot(fftx[0:sind], smooth_yvect2)
#
#        for axis in ax:
#            axis.legend()
#            axis.set_xlim([0, 500])
#            axis.plot([fund_freq, fund_freq], [0, 1], marker='*', color='m')

    # If an error occured, return 0 as guess
    except Exception as e:
        #print('Error in SGO method,', str(e))
        fund_freq = 0

    return fund_freq


# %%
def find_fundamental_invw(spectrum, mavg, fs, p=10, f_max = 1500):
    """
    Tries to find the fundamental frequency of a FFT signal,
    looking for the max amplitude after a crossover from negative to postive
    of the smoothed FFT signal has occured. Then it applies a weight
    function to all points, giving preference to peaks in the lower
    part of the spectrum

    Inputs
    ----------
    spectrum : np.array
        the fourier transform of the timesignal of the event
        (complex values and not shifted)
    fs : float
        the sampling frequency of the time signal in Hz

    Outputs
    ----------
    fund_freq : list
        The fundamental frequency and the cutoff frequency
        [fund_frequency, cutoff_frequency]
    """

    # Get the frequencies for spectrum
    fftx = np.fft.fftfreq(len(spectrum), 1/fs)

    try:

        if max(fftx)>f_max:
            # Get index where freq > f_max
            sind = np.where(fftx > f_max)[0][0]

        else:
            sind = np.where(fftx == max(fftx))[0][0]

        # Only grab the part of the signal that is below f_max
        yvect = np.abs(spectrum[0:sind])

        # Get index where moving average changes from positive to neg
        ind_slope_change = np.where(np.diff(mavg) > 0)[0][0]

        # Construct weighting vector
        w_vect = 1/(np.arange(ind_slope_change, sind)/sind)**1

        # Get index of maximum after slope change
        ind_max = np.argmax(yvect[ind_slope_change:len(yvect)]*w_vect)

        # Add index where slope change occured
        ind = ind_slope_change + ind_max

        # Get the fundamental frequency
        fund_freq = fftx[ind]

        # Also output the index of where the slope change occured
        freq_cutoff = fftx[ind_slope_change]

        # Array returned by function
        return_array = np.array([fund_freq, freq_cutoff])

    # If no frequency can be found, return 0
    except:
        return_array = np.array([0, 0])

    return return_array


# %%
def find_fundamental_mavg(spectrum, fs, p=10, f_max=1500):
    """
    Tries to find the fundamental frequency of a FFT signal, looking for the
    max amplitude after a crossover from negative to postive of the smoothed
    FFT signal has occured

    Inputs
    ----------
    spectrum : np.array
        the fourier transform of the timesignal of the event
        (complex values and not shifted)
    fs       : float
        the sampling frequency of the time signal

    Outputs
    ----------
    fund_freq : float
        The fundamental frequency
    mavg : np.array
        the moving average signal of the frequency spectrum
    """

    # Get the frequencies for spectrum
    fftx = np.fft.fftfreq(len(spectrum), 1/fs)
    try:
        if max(fftx)>f_max:
            # Get index where freq > f_max
            sind = np.where(fftx > f_max)[0][0]

        else:
            sind = np.where(fftx== max(fftx))[0][0]

        # Only grab the part of the signal that is below f_max
        yvect = np.abs(spectrum[0:sind])

        # Get period of mavg signal
        p = np.floor(0.05*len(yvect)).astype('int')

        # Get moving average signal
        mavg_sig = moving_average(yvect, p)

        # Get index where moving average changes from positive to neg
        ind_slope_change = np.where(np.diff(mavg_sig) > 0)[0][0]

        # Get index of maximum after slope change
        ind_max = np.argmax(yvect[ind_slope_change:len(yvect)])

        # Add index where slope change occured
        ind = ind_slope_change + ind_max

        # Get the fundamental frequency
        fund_freq = fftx[ind]
        # Also output the index of where the slope change occured
        freq_cutoff = fftx[ind_slope_change]

    # If no frequency can be found, return 0
    except:
        warnings.warn('Error, no frequency found in FE.find_fundamental_mavg')
        fund_freq = 0
        freq_cutoff = 0
        mavg_sig = 0

    return [fund_freq, freq_cutoff], mavg_sig


# %%
def find_fundamental_mavg_combined(f_mavg, estimates, perc=0.05):
    """
    Tries to find the fundamental frequency of a FFT signal,
    using several methods to determine the best guess.

    Inputs
    ----------
    f_mavg : np.float
        Fundamental frequency guessed by the mavg method

    estimates : list
        List with all the estimates from other WBF methods

    perc : np.float
        Percentage the frequency guesses have to be within eachother for the
        methods to vote for one frequency

    Outputs
    ----------
    fund_freq : float
        The fundamental frequency
    """
    fund_freq = f_mavg
    count_mav = 0
    count_mav2 = 0
    # If one estimate is at half the estimate of MAV, then use MAV/2
    for estimate in estimates:
        if perc_close_numbers(f_mavg/2, estimate, perc) is True:
            fund_freq = f_mavg/2
            count_mav2 += 1

    # Unless 2 estimates agrees with MAV, then use MAV
    for estimate in estimates:
        if perc_close_numbers(f_mavg, estimate, perc) is True:
            count_mav += 1

    if count_mav > 3:
        fund_freq = f_mavg

    # Unless three methods agree with MAV/2
    if count_mav2 >= count_mav:
        fund_freq = f_mavg/2

    return fund_freq


# %% Body wing separation methods
def calc_BW_values(body_sig, wing_sig):

    body_mean = np.mean(body_sig)
    wing_mean = np.mean(wing_sig)
    max_body = np.max(body_sig)
    min_body = np.min(body_sig)
    max_wing = np.max(wing_sig)
    min_wing = np.min(wing_sig)
    body_median = np.median(body_sig)
    wing_median = np.median(wing_sig)
    body_center = np.mean(body_sig[int(len(body_sig)/2)-10:int(len(body_sig)/2)+10])
    wing_center = np.mean(wing_sig[int(len(wing_sig)/2)-10:int(len(wing_sig)/2)+10])

    # Avoid division by zero
    ind = (body_sig > 0) & (wing_sig > 0)
    BW = (body_sig/(body_sig + wing_sig))[ind]

    bw_ratio_mean = np.mean(BW)
    bw_ratio_median = np.median(BW)
    bw_ratio_center = np.mean(BW[int(len(BW)/2)-10:int(len(BW)/2)+10])
    return body_mean, wing_mean, body_median, wing_median, body_center, wing_center, bw_ratio_mean, bw_ratio_median, bw_ratio_center, max_body, min_body, max_wing, min_wing

def body_locmin_method(time_signal, t_start, t_stop, interp_kind='cubic'):
    """ a function to separate body from wing contributions. It works by
    finding the local minima in the time signal (1st order) and then
    finding the local minima again in that series (2nd order). Both result
    in "under lapping" functions of the signal. The data points between the
    local minima are interpolated with a cubic spline.

    The 2nd order result works well for clear oscillatory events with a good
    body contribution.  The 1st order does not work very well for most events.

    When no wing beat is present the method does not give a correct body.

    Inputs
    ----------
    time_signal : np.array length L
        the time signal
    t_start : Int
        start indicies
    t_stop : Int
        end indicies
    interp_kind : string
        argument is passed to scipy.interpolate.interp1d to set for instance
        linear or cubic interpolation

    Outputs
    ----------
    body_1st : np.array  length L
        The extracted body signal based on 1st order local minima
    body_2nd : np.array  length L
        The extracted body signal based on 2nd order local minima

    Changes
    ----------
    20160920 - JCP : changed t_start to t_start-1 in idx to prevent the
    occurance of the same idx value twice
    """
    # underlapping function absed on 1st order local minima
    loc_min_1st = scipy.signal.argrelextrema(time_signal, np.less)[0]
    # prepare values and their indexes for interpolation
    val = np.insert(time_signal[loc_min_1st],
                    [0, 0, len(loc_min_1st),
                     len(loc_min_1st)],
                    [0, 0, 0, 0])

    idx = np.insert(loc_min_1st, [0, 0, len(loc_min_1st), len(loc_min_1st)],
                    [0, t_start - 1, t_stop, len(time_signal) - 1])
    try:
        interp_obj = scipy.interpolate.interp1d(idx, val, kind='cubic')
    except np.linalg.linalg.LinAlgError:
        # with the wrong inputs sometimes the error
        # numpy.linalg.linalg.LinAlgError: SVD did not converge
        print('Caught SVD error in FE.body_locmin_method')
        return 0.0

    # create the body
    body_1st = interp_obj(np.arange(len(time_signal)))
    body_1st[0:t_start] = 0
    body_1st[t_stop:] = 0

    # underlapping function absed on 2nd order local minima
    loc_min_2nd = scipy.signal.argrelextrema(time_signal[loc_min_1st],
                                             np.less)[0]

    # prepare values and their indexes for interpolation
    val = np.insert(time_signal[loc_min_1st[loc_min_2nd]],
                    [0, 0, len(loc_min_2nd), len(loc_min_2nd)],
                    [0, 0, 0, 0])

    idx = np.insert(loc_min_1st[loc_min_2nd], [0, 0, len(loc_min_2nd),
                    len(loc_min_2nd)],
                    [0, t_start, t_stop, len(time_signal) - 1])

    interp_obj = scipy.interpolate.interp1d(idx, val, kind=interp_kind)

    # create the body
    body_2nd = interp_obj(np.arange(len(time_signal)))
    body_2nd[0:t_start] = 0
    body_2nd[t_stop:] = 0

    return body_2nd


# %%
def body_lund_method(time_signal, f0, fs):
    """ a function to separate body from wing contributions. It works by
    using a sliding minimum and sliding maximum filter with width based on the
    period of the signal.

    When no wing beat is present the method set the full signal as body.

    Inputs
    ----------
    time_signal : np.array length L
        the time signal
    f0 : float
        fundamental frequency
    fs : int
        sampling frequency

    Outputs
    ----------
    body : np.array  length L
        The extracted body signal
    """
    # number of pixels in one period of the signal
    period = int(round((fs)/(f0+1e-9)))
    # period needs to be odd to work with order_filter
    if period % 2 == 0:
        period = period + 1
    if period > time_signal.shape[0]:
        body_smooth = time_signal
    else:
        # min sliding filter
        minslide = scipy.signal.order_filter(time_signal, np.ones(period), 0)
        # max sliding filter
        maxslide = scipy.signal.order_filter(time_signal,
                                             np.ones(period), period - 1)
        max_wingbody = np.max(maxslide + minslide)
        maxbody = np.max(minslide)

        body = abs(maxbody*(minslide + maxslide)/max_wingbody)
        # gaussian function for smoothing
        smooth = scipy.signal.gaussian(period, period/(4*np.pi))
        # smoothed body contribution
        body_smooth = np.convolve(body, smooth/sum(smooth), 'same')
    return body_smooth


# %%
def body_slide_method(time_signal, f0, fs, style='min'):
    """Separate body from wing contributions. It works by using a sliding
    minimum or maximum filter whose width is based on the period of the signal.
    There is also an option to isolate the "roof" of the signal, to better
    quantify specularity for example.

    When no wing beat is present the method set the full signal as body.

    Inputs
    ----------
    time_signal : np.array length L
        the time signal
    f0 : float
        fundamental frequency
    fs : int
        sampling frequency
    style : int
        'min' for minimum, 'max' for maximum

    Outputs
    ----------
    body : np.array  length L
        The extracted body signal
    """
    # number of pixels in one period of the signal
    period = int(round((fs)/(f0 + 1e-9)))
    rank = 1
    if style == 'max':
        rank = period-1

    # 1 period needs to be odded to work with order_filter
    if period % 2 == 0:
        period = period + 1
    if period > time_signal.shape[0]:
        body_smooth = time_signal
    else:
        # min sliding filter
        slide = scipy.signal.order_filter(time_signal, np.ones(period), rank)

        body = slide
        # gaussian function for smoothing
        smooth = scipy.signal.gaussian(period, period/(2*np.pi))
        # smoothed body contribution
        body_smooth = np.convolve(body, smooth/sum(smooth), 'same')

    return body_smooth

# %%
def calc_melanine(values, iNIR=0, iSWIR=1):
    """ Calculates the ratio between the NIR and SWIR channel.
    Melanization = NIR/(NIR + SWIR)


    Inputs
    ----------
    values : np.array no_channels, length
        the signal
    NIR : int
        Channel index for NIR channel

    Outputs
    ----------
    mean : float
        The calculated melanization mean
    median: float
        The calculated melanization median
    center: float
        The calculated melanization in the center pixel
    Changes
    ----------
    non yet
    """

    NIR = values[iNIR, :]
    SWIR = values[iSWIR, :]
    length = np.shape(values)[1]

    NIR = np.ndarray.flatten(NIR)
    SWIR = np.ndarray.flatten(SWIR)

    # To avoid division by zero
    ind = (NIR > 0) & (SWIR > 0)
    melanization = NIR[ind]/(NIR[ind] + SWIR[ind])

    np.isnan(melanization)
    mean = np.mean(melanization)
    median = np.median(melanization)
    center = np.mean(melanization[int(length/2)-50:int(length/2)+50])
    return mean, median, center

# %%
def body_loc_mx_erode_method(time_signal, t_start, t_stop):
    """ a function to separate body from wing contributions. It works by
    finding the local maxima in the time signal and removing them iteratively
    untill the total number of maxima is below 1 + (t_stop-t_start)/40. The
    remaining points are interpolated lineraly


    (1st order) and then
    finding the local minima again in that series (2nd order). Both result
    in "under lapping" functions of the signal. The data points between the
    local minima are interpolated with a cubic spline.

    The 2nd order result works well for clear oscillatory events with a good
    body contribution.  The 1st order does not work very well for most events.

    When no wing beat is present the method does not give a correct body.

    Inputs
    ----------
    time_signal : np.array length L
        the time signal
    t_start : int
        start index of event
    t_stop : int
        stop index of event

    Outputs
    ----------
    body : np.array  length L
        The extracted body signal

    Changes
    ----------
    non yet
    """

    new_sig = time_signal

    # find local max and minima in the original signal
    loc_max = scipy.signal.argrelextrema(new_sig, np.greater)[0]
    loc_min = scipy.signal.argrelextrema(new_sig, np.less)[0]

    # interpolating the found extrema to the nearest neighbours
    new_sig[loc_max] = (new_sig[loc_max-1]+new_sig[loc_max+1])/2
    new_sig[loc_min] = (new_sig[loc_min-1]+new_sig[loc_min+1])/2

    # iteratively do the interpolation on the local maxima untill the number
    # of local maxima is below 1 + (t_stop-t_start)/40
    loc_max = scipy.signal.argrelextrema(new_sig, np.greater)[0]

    while len(loc_max) > 1 + (t_stop-t_start)/40:
        new_sig[loc_max] = (new_sig[loc_max-1]+new_sig[loc_max+1])/2
        loc_max = scipy.signal.argrelextrema(new_sig, np.greater)[0]

    # find remaining local extrema
    loc_max = scipy.signal.argrelextrema(new_sig, np.greater)[0]
    loc_min = scipy.signal.argrelextrema(new_sig, np.less)[0]

    # create interpolation object and interpolate data
    idx = np.sort(np.concatenate((loc_max, loc_min,
                                  np.array([0, t_start, t_stop,
                                            len(time_signal) - 1]))))
    val = new_sig[idx]
    interp_obj = scipy.interpolate.interp1d(idx, val)
    body = interp_obj(np.arange(len(time_signal)))
    # put begin and start to zero
    body[0:t_start] = 0
    body[t_stop:] = 0

    return body


# %%
def body_wing_separation(t_start, t_stop, data_matrix, WBF, fs, channels):
    """Separates body and wing signals in multichannel data. Currently uses the
    slidemin method.

    Inputs
    ----------
    t_start : Int
        start indicies in tume vector
    t_stop : Int
        end indicies in tume vector
    data_matrix : np.array L x M
        data matrix, with time on vertical axis
    WBF : Float
        the estimated wing beat frequency in Hz
    fs : Int
        sampling frequency in Hz

    Outputs
    ----------
    spectras : list len 4
        a list of time signals (not spectra!)
        [body_co, body_de, wing_co, wing_de]

    """
    co_sig = data_matrix[t_start:t_stop, channels[0]]
    de_sig = data_matrix[t_start:t_stop, channels[1]]
    signal = np.array([co_sig, de_sig])

    # Calculates the body signals
    body_sig = [body_slide_method(signal[0], WBF, fs, style='min'),
                body_slide_method(signal[1], WBF, fs, style='min')]

    wing_sig = signal - body_sig

    # Renaming to make the code a bit easier to follow
    body_co = body_sig[0] - np.min(body_sig[0])
    body_de = body_sig[1] - np.min(body_sig[1])
    wing_co = wing_sig[0] - np.min(body_sig[0])
    wing_de = wing_sig[1] - np.min(body_sig[1])

    spectras = [body_co, body_de, wing_co, wing_de]
    return spectras


# %% Lab specific functions
def separate_specular(spectras, WBF, fs, plot=False, nr=0, channel=False):
    """Separates the specular part from the co-polarized and de-polarized
    signals. The script is originally based on Alems matlab code
    "Main_program_Mosquito_FP". There is an option to plot the calculated
    spectra which could be useful when investigating single events.

    Inputs
    ----------
    spectras : List of numpy arrays
        list which contains the body and wing signals,
        [co-body, de_body, co_wing, de-wing]
    WBF : Float
        the estimated wing beat frequency in Hz
    fs : Int
        sampling frequency in Hz
    plot : bool
        optional, False or True
    nr : int
        required for plotting, event number
    channel : str
        For plot tilte

    Outputs
    ----------
    spectras : list of numpy arrays
        the output file, [co-body, de_body, co_wing, de-wing,
                          specular-body, specular-wing]

    The function goes trough the following steps:

    Calculates the relative amplitude of the co- and depolarized signals
    Using the calculated amplitude factor, it calculates the pure specular part
    by removing the depolarized light with the same polarization as
    the linearized.

    20161223: KRY created function
    20170102: KRY added documentation and more intuitive variable names
    20170420: KRY changed a lot, moved the bw-separation out of the function
    20170424: KRY, redifined the scale, no goes from 0 to 1
    """

    # Clusters body signals togheter [body_co, body_de]
    body_sig = [spectras[0], spectras[1]]
    # Clusters wing signals togheter [wing_co, wing_de]
    wing_sig = [spectras[2], spectras[3]]

    # Uses the median as a measurment of signal strength,
    # 1E-9 added to prevent division by zero
    scale_body = np.median(body_sig[0]/(body_sig[0] + body_sig[1] + 1E-9))
    scale_wing = np.median(wing_sig[0]/(wing_sig[0] + wing_sig[1] + 1E-9))

    # Specular signal gives only the specular part,
    # which is setting on top of the co-polarized signal.
    # Ask Alem to draw it on the whiteboard... :)
    body_spec = body_sig[0] - scale_body*body_sig[1]
    wing_spec = wing_sig[0] - scale_wing*wing_sig[1]

    # Adds the calculated specular spectra to the output list
    spectras.append(body_spec)
    spectras.append(wing_spec)

    # Plots the spectra
    if plot is True:
        # Recreates the original signal
        signal = [body_sig[0] + wing_sig[0], body_sig[1] + wing_sig[1]]
        body_sig.append(body_spec)
        wing_sig.append(wing_spec)
        signal.append(body_spec + wing_spec)
        time_vector = np.arange(0, len(body_sig[0])*1/fs, 1/fs)*1000
        colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']

        co_power = np.abs(np.fft.fft(wing_sig[0]))**2
        de_power = np.abs(np.fft.fft(wing_sig[1]))**2
        spec_power = np.abs(np.fft.fft(wing_spec))**2

        fig = plt.figure()
        fig.subplots_adjust(hspace=.5)
        titles = ['Co-polarized signal',
                  'De-polarized signal',
                  'Specular signal']

        # Time plots
        for k in range(3):
            ax = plt.subplot(2, 2, k + 1)
            ax.plot(time_vector, signal[k], color=colors[0],
                    label='Original signal')
            ax.plot(time_vector, body_sig[k], color=colors[3 + k],
                    label='Body signal')
            ax.plot(time_vector, wing_sig[k], color=colors[1 + k],
                    label='Wing signal')

            # Plot parameters
            ax.legend()
            ax.set_xlabel('Time (ms)')
            ax.set_title(titles[k])

        # Frequency plots
        ax = plt.subplot(2, 2, 4)
        freq_axis = np.fft.fftfreq(np.shape(signal)[1], 1/fs)
        half_way = np.int(len(freq_axis)/2) # robust half for odd numbers of len
        ax.plot(freq_axis[1:half_way], co_power[1:half_way],
                label='co-polarized wing signal', color=colors[1])
        ax.plot(freq_axis[1:half_way], de_power[1:half_way],
                label='de-polarized wing signal', color=colors[2])
        ax.plot(freq_axis[1:half_way], spec_power[1:half_way],
                label='specular wing signal', color=colors[-1])
        #        ax.set_yscale('log')
        ax.set_xlim([0, 10*WBF])
        ticks = ax.get_xticks()
        plt.locator_params(axis='x', nbins=len(ticks)*2)
        ax.grid()
        ax.legend()
        ax.set_title('Power spectrum')
        if channel is False:
            fig.suptitle('Event ' + str(nr))
        else:
            fig.suptitle('Event ' + str(nr) + ', ' + channel)
        ax.set_xlabel('Hz')

    return spectras

# %%
def body_wing_ratio(spectra, WBF, fs, plot=False):
    """Calculatates the body-wing ratio in all channels. It interpolates the
    wing peak signal using the sliding method and returns the fractions.
    The fraction between body and wing signal is calculated for all points
    where the body signal > than the median of the body signal. This avoids
    very large and weird numbers when the body signal is close to 0.

    There is also an option to plot the returned result for easier
    understanding.

    Inputs
    ----------
    spectras : List of numpy arrays
        list which contains the body and wing signals,
        [body_co, body_de, wing_co, wing_de, body_spec, wing_spec]
    WBF : Float
        the estimated wing beat frequency in Hz
    fs : Int
        sampling frequency in Hz
    plot : bool
        optional, False or True
    nr : int
        required for plotting, event number

    Outputs
    ----------
    spectras : list of numpy arrays
        wing_max signals are numpy arrays, interpolated between the wing beat
        peaks. frac signals are arrays with values between 0 and 1
        [wing_max_co, wing_max_de, wing_max_spec, frac_co, frac_de, frac_spec]
    """

    # Renames channels for readability
    body_co = spectra[0]
    body_de = spectra[1]
    wing_co = spectra[2]
    wing_de = spectra[3]
    body_spec = spectra[4]
    wing_spec = spectra[5]

    # Interpolates the maximum wing signal
    wing_max_co = body_slide_method(body_co + wing_co, WBF, fs, style='max')
    wing_max_de = body_slide_method(body_de + wing_de, WBF, fs, style='max')
    wing_max_spec = body_slide_method(body_spec + wing_spec, WBF, fs, style='max')

    # Calculates the fraction between body and wing signal
    frac_co = (wing_max_co + body_co/wing_max_co)[body_co > np.median(body_co)]
    frac_de = (wing_max_de + body_de/wing_max_de)[body_de > np.median(body_de)]
    frac_spec = (wing_max_spec + body_spec/wing_max_spec)[body_spec > np.median(body_spec)]

    if plot is True:
        plt.figure()
        xvec = np.arange(0, len(body_co))
        # Upper plot
        ax = plt.subplot2grid((2, 1), (0, 0))
        # Co parts
        ax.plot(wing_max_co, color='b', ls='--', label='Wing co')
        ax.plot(body_co, color='b', label='body co')
        ax.plot([0, len(body_co)], [np.median(body_co), np.median(body_co)])
        # De parts
        ax.plot(wing_max_de, color='r', ls='--', label='Wing de')
        ax.plot(body_de, color='r', label='body de')
        ax.plot([0, len(body_de)], [np.median(body_de), np.median(body_de)])
        # Specular parts
        ax.plot(wing_max_spec, color='g', ls='--', label='wing spec')
        ax.plot(body_spec, color='g', label='body spec')
        ax.plot([0, len(body_co)], [np.median(body_spec),
                                    np.median(body_spec)])
        # Plot parameters
        ax.set_xlim([0, len(body_co)])
        ax.legend()

        # Lower plot
        ax2 = plt.subplot2grid((2, 1), (1, 0))
        # Co parts
        ax2.plot(xvec[body_co > np.median(body_co)], frac_co,
                 color='b', marker='*', label='Copolarized')
        ax2.plot([0, len(body_co)], [np.median(frac_co), np.median(frac_co)],
                 color='b', ls='--')
        # De parts
        ax2.plot(xvec[body_de > np.median(body_de)], frac_de,
                 color='r', marker='*', label='Depolarized')
        ax2.plot([0, len(body_co)], [np.median(frac_de), np.median(frac_de)],
                 color='r', ls='--')
        # specular parts
        ax2.plot(xvec[body_spec > np.median(body_spec)], frac_spec,
                 color='g', marker='*', label='Specular')
        ax2.plot([0, len(body_co)], [np.median(frac_spec),
                                     np.median(frac_spec)], color='g', ls='--')
        # Plot parameters
        ax2.set_xlim([0, len(body_co)])
        ax2.legend()

    return [wing_max_co, wing_max_de, wing_max_spec,
            frac_co, frac_de, frac_spec]


# %%
def specularity(spectra, wing_max_spectra, fs, plot=False):
    """Calculatates the specularity of the body and wing signals. Specularity
    is defined as the fraction the co-polarized or specular signal composes
    of the total signal and is a value between 0 and 1.

    There is also an option to plot the returned result for easier
    understanding.

    Inputs
    ----------
    spectra : List of numpy arrays
        list which contains the body and wing signals,
        [body_co, body_de, wing_co, wing_de, body_spec, wing_spec]
    wing_max_spectra : List of numpy arrays
        Interpolated wing beat signals, [wing_max_co,
                                         wing_max_de,
                                         wing_max_spec]
    fs : Int
        sampling frequency in Hz
    plot : bool
        optional, False or True

    Outputs
    ----------
    spectras : list of numpy arrays
        the output array, frac_co, frac_de and frac_spec is added to the
        [frac_body_code, frac_body_specde, frac_wing_code, frac_wing_specde]
    """

    # Renaming for readability
    body_co = spectra[0]
    body_de = spectra[1]
    body_spec = spectra[4]

    wing_co = wing_max_spectra[0]
    wing_de = wing_max_spectra[1]
    wing_spec = wing_max_spectra[2]

    # Defines a region where body signal is larger than its medium
    region = np.array(body_de > np.median(body_de))
    if sum(region) < 5:
        region = region + 1

    # calculates fractions
    frac_body_code = (body_co/(body_co + body_de))[region]
    frac_body_specde = (body_spec/(body_spec + body_de))[region]
    frac_wing_code = (wing_co/(wing_co + wing_de))[region]
    frac_wing_specde = (wing_spec/(wing_spec + wing_de))[region]

    if plot is True:
        plt.figure()
        xvec = np.arange(0, len(body_co))
        # Upper plot
        ax = plt.subplot2grid((2, 1), (0, 0))
        ax.plot([0, len(body_co)], [np.median(body_de), np.median(body_de)])
        # Body parts
        ax.plot(body_co, color='b', ls='--', label='body_co')
        ax.plot(body_de, color='r', ls='--', label='body_de')
        ax.plot(body_spec, color='g', ls='--', label='body_spec')

        # Wing parts
        ax.plot(wing_co, color='b', label='wing_co')
        ax.plot(wing_de, color='r', label='wing_de')
        ax.plot(wing_spec, color='g', label='wing_spec')

        # Plot parameters
        ax.set_xlim([0, len(body_co)])
        ax.legend()

        # Lower plot
        ax2 = plt.subplot2grid((2, 1), (1, 0))
        # Body parts
        ax2.plot(xvec[region], frac_body_code, color='b', marker='d',
                 ls='--', label='body_code')
        ax2.plot([0, len(body_co)], [np.median(frac_body_code),
                                     np.median(frac_body_code)],
                 color='b', ls='--')
        ax2.plot(xvec[region], frac_body_specde, color='g', marker='*',
                 ls='--', label='body_specde')
        ax2.plot([0, len(body_co)], [np.median(frac_body_specde),
                                     np.median(frac_body_specde)],
                 color='g', ls='--')

        # wing parts
        ax2.plot(xvec[region], frac_wing_code, color='r', marker='d',
                 label='wing_code')
        ax2.plot([0, len(body_co)], [np.median(frac_wing_code),
                                     np.median(frac_wing_code)],
                 color='r', ls='--')
        ax2.plot(xvec[region], frac_wing_specde, color='y', marker='*',
                 label='wing_specde')
        ax2.plot([0, len(body_co)], [np.median(frac_wing_specde),
                                     np.median(frac_wing_specde)],
                 color='y', ls='--')

        # Plot parameters
        ax2.set_xlim([0, len(body_co)])
        ax2.legend()

    return [frac_body_code, frac_body_specde, frac_wing_code, frac_wing_specde]


# %% LIDAR specific functions
def event_slope(event_img, range_v, time_v, method='old', plot=False):
    """ determines the "slope" of the event see below for mor explanation

    Inputs
    ----------
    event_img : np.array LxH
        the 2D event data
    range_v : np.array length L (or was it H?)
        the range indices axis
    time_v : np.array length H (or was it L?)
        the time indices axis

    Outputs
    ----------
    slope : np.float
        the calculated slope, gives 0 for short events for instance if t_start
        and t_stop differ only by 1 or 2
    y_axis_crossing : np.float
        the y_axis crossing of the fitted curve

    Events both span a range in time
    and in range. Thus an insect flying e.g. left-right gives a mirrored event
    in range compared to an insect flying right-left. This gives an apperent
    slope in the 2D depiction of the event. This function gives the slope of
    this event. If the slope is 0 the event extends in the range but is
    occuring only in one time frame. If the slope is high the event is limited
    in range put takes long.

    The fitting is done by taking all data and using the positive values
    as weights in a polynomial fit. This is reasonably robust but not perfect.
    Gives NaN for short events for instance if t_start and t_stop differ only
    by 1 or 2

    Gives regularly RankWarnings for events that are very short.
    2016 09 22 - JCP :added y_axis_crossing output

    """
    event_img_flat = event_img.flatten()

    if method == 'old':
        # R (range) and T (time) coordinates
        R, T = np.meshgrid(range_v, time_v)
        # fit a linear curve to the picure, using weigths. The weights
        # are squared and the negative weights are put to 0
        offset = 1e-9
        # this small offset can eb used to avoid poorly conditioned cases with
        # many zeros to give an error in that case put it to 1e-9
        w = offset + event_img_flat**2*(event_img_flat > 0)

        try:
            # Old method
            p = np.polyfit(R.flatten(), T.flatten(), 1, w=w)
            slope = p[0]
            y_axis_crossing = p[1]

        except TypeError:
            print('Type Error')
            slope = 0
            y_axis_crossing = 0

        if plot is True:
            # Old method
            fig = plt.figure()
            ax0 = plt.subplot2grid((1, 1), (0, 0))
            ax0.imshow(event_img, interpolation='none')
            pol = np.poly1d(p)
            ax0.plot(range_v - range_v[0], pol(range_v) - time_v[0])

    if method == 'new':
        # New method
        event_img_t = event_img.transpose()
        max_pix = np.sort(event_img_flat)[::-1]
        no_of_pixels = np.round(event_img_t.size*0.015).astype(int)
        if no_of_pixels < 5:
            no_pixels = 5
        thresh = max_pix[no_of_pixels]
        y, x = np.where(event_img_t > thresh)
        A = np.vstack([x, np.ones(len(y))]).T
        slope, y_axis_crossing = np.linalg.lstsq(A, y)[0]

        if plot is True:
            # new method
            fig = plt.figure()
            ax1 = plt.subplot2grid((1, 1), (0, 0))
            xv = time_v - time_v[0]
            ax1 = plt.subplot2grid((1, 1), (0, 0))
            ax1.imshow(event_img_t, interpolation='none')
            ax1.plot(x, y, 'o', label='Original data', markersize=5)
            ax1.plot(xv, slope*xv + y_axis_crossing, 'r', label='Fitted line')
            ax1.legend()

    return slope, y_axis_crossing


# %%
def generate_general_features(channel_nr):
    """ function that combines all of the above feature axtraction methods into
    one function.

    Inputs
    ----------
    channel_nr : list of channels?
        not sure but it seems that for many of the methods we sould select one
        or more channels

    Outputs
    ----------
    feature_names : list of strings
        list of feature names as strings
    feature_values : list of feature values
        list with the values of each feature
    """
    feature_names = []
    feature_values = []

    feature_names = feature_names + ['time']
    feature_values = feature_values + [datetime]

    feature_names = feature_names + ['t_start']
    feature_values = feature_values + [t_start]

    feature_names = feature_names + ['opt_mass']
    feature_values = feature_values + [opt_mass]

    feature_names = feature_names + ['time_max']
    feature_values = feature_values + [time_max]

    feature_names = feature_names + ['COM_time']
    feature_values = feature_values + [COM_time]

    feature_names = feature_names + ['saturation']
    feature_values = feature_values + [saturation]


# %%
def generate_WBF_features(spectrum, fs, t_start, t_stop):
    """ function that combines all of the WBF feature axtraction methods into
    one function.

    Inputs
    ----------
    spectrum : np.array 1D
        ??? must be fixed length?
    fs : float
        sample rate Hz
    tstart : int

    ?? can we avoid t_start.. we have to, what is the spectrum exactly, zeropadded or not?
        sample rate Hz

    Outputs
    ----------
    feature_names : list of strings
        list of feature names as strings
    feature_values : list of feature values
        list with the values of each feature
    """
    feature_names = []
    feature_values = []

    WBF_cepstrum = find_fundamental_cepstrum(spectrum,fs)
    feature_names = feature_names + ['WBF_cepstrum']
    feature_values = feature_values + [WBF_cepstrum]

    WBF_HPS4 = find_fundamental_HPS(spectrum, fs)
    feature_names = feature_names + ['WBF_HPS4']
    feature_values = feature_values + [WBF_HPS4]

    WBF_HPS3 = find_fundamental_HPS(spectrum, fs, N=3)
    feature_names = feature_names + ['WBF_HPS3']
    feature_values = feature_values + [WBF_HPS3]

    WBF_HPS2 = find_fundamental_HPS(spectrum, fs, N=2)
    feature_names = feature_names + ['WBF_HPS2']
    feature_values = feature_values + [WBF_HPS2]

    WBF_autocor = find_fundamental_auto(spectrum, t_start, t_stop, fs)
    feature_names = feature_names + ['WBF_autocor']
    feature_values = feature_values + [WBF_autocor]

    WBF_combined = find_fundamental_combined(WBF_HPS4, WBF_HPS3, WBF_HPS2,
                                             WBF_cepstrum, WBF_autocor)
    feature_names = feature_names + ['WBF_combined']
    feature_values = feature_values + [WBF_combined]

    WBF_peak = find_fund_peaks(spectrum,fs,WBF_HPS4,WBF_HPS3,WBF_HPS2,
                               WBF_cepstrum, WBF_autocor)
    feature_names = feature_names + ['WBF_peak']
    feature_values = feature_values + [WBF_peak]

    harmonics = find_harmonics(spectrum, fs, WBF_peak[0])
    feature_names = feature_names + ['harmonics']
    feature_values = feature_values + [harmonics]

    # maybe here change something, calc mavg separately and get output of functions more in line with the others
    WBF_mavg, mavg_sig = find_fundamental_mavg(spectrum, fs)
    feature_names = feature_names + ['WBF_mavg']
    feature_values = feature_values + [WBF_mavg[0]]

    WBF_inv = find_fundamental_invw(spectrum, mavg_sig,fs)
    feature_names = feature_names + ['WBF_inv']
    feature_values = feature_values + [WBF_inv]

    WBF_sgo = find_fundamental_sgo(spectrum,fs)
    feature_names = feature_names + ['WBF_sgo']
    feature_values = feature_values + [WBF_sgo]

    WBF_mavg_comb = find_fundamental_mavg_combined(WBF_mavg[0],
                                                   [WBF_peak[0], WBF_HPS2,
                                                    WBF_HPS3,WBF_HPS4,
                                                    WBF_autocor, WBF_inv[0]],
                                                          perc=0.05)
    feature_names = feature_names + ['WBF_mavg_comb']
    feature_values = feature_values + [WBF_mavg_comb]

    return feature_names, feature_values


# %%
def generate_body_features(channel_nr):
    """ function that combines all of the above feature axtraction methods into
    one function.

    Inputs
    ----------
    channel_nr : list of channels?
        not sure but it seems that for many of the methods we sould select one
        or more channels

    Outputs
    ----------
    feature_names : list of strings
        list of feature names as strings
    feature_values : list of feature values
        list with the values of each feature
    """


    feature_names = feature_names + ['WBF_mavg_comb']
    feature_values = feature_values + [WBF_mavg_comb]


#def generate_LAB_features(channel_nr):
#    """ function that combines all of the above feature axtraction methods into
#    one function. NO THIS MAKES NO SENSE THIS FUNCTION WOULD NEED TO MANY INPUTS
#
#    Inputs
#    ----------
#    channel_nr : list of channels?
#        not sure but it seems that for many of the methods we sould select one
#        or more channels
#
#    Outputs
#    ----------
#    feature_names : list of strings
#        list of feature names as strings
#    feature_values : list of feature values
#        list with the values of each feature
#    """
#    feature_names = []
#    feature_values = []
#
#
#def generate_LIDAR_features(channel_nr):
#    """ function that combines all of the above feature axtraction methods into
#    one function.
#
#    Inputs
#    ----------
#    channel_nr : list of channels?
#        not sure but it seems that for many of the methods we sould select one
#        or more channels
#
#    Outputs
#    ----------
#    feature_names : list of strings
#        list of feature names as strings
#    feature_values : list of feature values
#        list with the values of each feature
#    """
#    feature_names = []
#    feature_values = []
#
#    feature_names = feature_names + ['r_start']
#    feature_values = feature_values + [r_start]
#
#    feature_names = feature_names + ['slot_defect_count']
#    feature_values = feature_values + [slot_defect_count]
#
#    feature_names = feature_names + ['slope']
#    feature_values = feature_values + [slope]
#
#    feature_names = feature_names + ['COM_range']
#    feature_values = feature_values + [COM_range]


##    header = np.array(['filename', 'dirname', 'time', 'total_event_nr', 'event_nr',
##                   'seconds', 't_start', 't_stop', 'opt_mass', 'time_max',
##                   'COM_time', 'saturation', 'WBF_cepstrum', 'WBF_HPS4',            # WBF estimations
##                   'WBF_HPS3', 'WBF_HPS2', 'WBF_autocor', 'WBF_combined',
##                   'WBF_peak_f0', 'WBF_peak_f0_amp', 'WBF_peak_2f0',
##                   'WBF_peak_2f0_amp', 'WBF_peak_3f0', 'WBF_peak_3f0_amp',
##                   'WBF_peak_4f0', 'WBF_peak_4f0_amp', 'WBF_inv', 'WBF_mavg',
##                   'WBF_mavg_comb', 'WBF_sgo', 'body_mx_NIR_co',                    # NIR parameters
##                   'body_mx_NIR_de', 'body_sm_NIR_co', 'body_sm_NIR_de',
##                   'wing_mx_NIR_co', 'wing_mx_NIR_de', 'wing_sm_NIR_co',
##                   'wing_sm_NIR_de', 'bw_ratio_NIR_co', 'bw_ratio_NIR_de',
##                   'bw_ratio_NIR_spec', 'body_spec_NIR_code',
##                   'body_spec_NIR_specde', 'wing_spec_NIR_code',
##                   'wing_spec_NIR_specde', 'body_mx_SWIR_co',
##                   'body_mx_SWIR_de', 'body_sm_SWIR_co', 'body_sm_SWIR_de',         # SWIR parameters
##                   'wing_mx_SWIR_co', 'wing_mx_SWIR_de', 'wing_sm_SWIR_co',
##                   'wing_sm_SWIR_de', 'bw_ratio_SWIR_co', 'bw_ratio_SWIR_de',
##                   'bw_ratio_SWIR_spec', 'body_spec_SWIR_code',
##                   'body_spec_SWIR_specde', 'wing_spec_SWIR_code',
##                   'wing_spec_SWIR_specde', 'melanization_body',                    # Melanization
##                   'melanization_wing'])
##
##
##
##    ['filename', 'dirname', 'time', 'total_event_nr', 'event_nr',
##                   'seconds', 't_start', 't_stop', 'd_start', 'd_stop',
##                   'term_pix', 'term_sig', 'opt_mass', 'time_max', 'COM_range',
##                   'COM_time', 'slope', 'saturation', 'slot_defect_count',
##                   'WBF_cepstrum', 'WBF_HPS4', 'WBF_HPS3', 'WBF_HPS2',
##                   'WBF_autocor', 'WBF_combined', 'WBF_peak_f0',
##                   'WBF_peak_f0_amp', 'WBF_peak_2f0', 'WBF_peak_2f0_amp',
##                   'WBF_peak_3f0', 'WBF_peak_3f0_amp', 'WBF_peak_4f0',
##                   'WBF_peak_4f0_amp', 'body_mx_LC', 'wing_mx_LC',
##                   'body_sum_LC', 'wing_sum_LC', 'body_mx_LL', 'wing_mx_LL',
##                   'body_sum_LL', 'wing_sum_LL', 'body_mx_LU', 'wing_mx_LU',
##                   'body_sum_LU', 'wing_sum_LU', 'body_mx_SL', 'wing_mx_SL',
##                   'body_sum_SL', 'wing_sum_SL'])
##                       for nr in range(nr_of_events):
##        total_event_nr = total_event_nr + 1  # increment the event number
##        print(nr + 1, ' off ', nr_of_events)
##
##        # check for saturation in event
##        saturation = np.max(signal) > 65536
##
##        # get some other features
##        opt_mass = FE.optical_mass(time_sig[nr, :])
##        time_max = FE.time_max(time_sig[nr, :])
##        COM_time = FE.COM(time_sig[nr, :])
##
##        # get fundamental frequency from complete signal
##        WBF_cepstrum = FE.find_fundamental_cepstrum(spectrum[nr, :], fs)
##        WBF_HPS4 = FE.find_fundamental_HPS(spectrum[nr, :], fs)
##        WBF_HPS3 = FE.find_fundamental_HPS(spectrum[nr, :], fs, N=3)
##        WBF_HPS2 = FE.find_fundamental_HPS(spectrum[nr, :], fs, N=2)
##        WBF_autocor = FE.find_fundamental_auto(spectrum[nr, :], t_start[nr],
##                                               t_stop[nr], fs)
##
##        WBF_combined = FE.find_fundamental_combined(WBF_HPS4,
##                                                    WBF_HPS3,
##                                                    WBF_HPS2,
##                                                    WBF_cepstrum,
##                                                    WBF_autocor)
##
##        WBF_peak = FE.find_fund_peaks(spectrum[nr, :], fs, WBF_HPS4, WBF_HPS3,
##                                      WBF_HPS2, WBF_cepstrum, WBF_autocor)
##
##        harmonics = FE.find_harmonics(spectrum[nr, :], fs, WBF_peak[0])
##
##        WBF_mavg, mavg_sig = FE.find_fundamental_mavg(spectrum[nr, :], fs)
##
##        WBF_inv = FE.find_fundamental_invw(spectrum[nr, :], mavg_sig, fs)
##
##        WBF_sgo = FE.find_fundamental_sgo(spectrum[nr, :], fs)
##
##        WBF_mavg_comb = FE.find_fundamental_mavg_combined(WBF_mavg[0],
##                                                          [WBF_peak[0],
##                                                           WBF_HPS2,
##                                                           WBF_HPS3,
##                                                           WBF_HPS4,
##                                                           WBF_autocor,
##                                                           WBF_sgo,
##                                                           WBF_inv[0]],
##                                                          0.05)
##
##        # Calculate & wing and body separation
##        # co = copolarized, de = depolarized
##        #               0       1           2       3
##        # spectra = [body_co, body_de, wing_co, wing_de]
##        spectra_NIR_temp = FE.body_wing_separation(t_start[nr], t_stop[nr],
##                                              data_matrix, WBF_mavg_comb,
##                                              fs, [0, 1])
##
##        spectra_SWIR_temp = FE.body_wing_separation(t_start[nr], t_stop[nr],
##                                               data_matrix, WBF_mavg_comb,
##                                               fs, [2, 3])
##
##        # Separate specular channel and add to spectra list
##        #               0       1           2       3       4           5
##        # spectra = [body_co, body_de, wing_co, wing_de, body_spec, wing_spec]
##        spectra_NIR = FE.separate_specular(spectra_NIR_temp, WBF_mavg_comb,
##                                           fs, plot=False, nr=total_event_nr,
##                                           channel='NIR')
##
##        spectra_SWIR = FE.separate_specular(spectra_SWIR_temp, WBF_mavg_comb,
##                                            fs, plot=False, nr=total_event_nr,
##                                            channel='SWIR')
##
##        # Calculates body parameters, (Maximum, sum & ratio of signals)
##        body_mx_NIR_co = np.max(spectra_NIR[0])
##        body_mx_NIR_de = np.max(spectra_NIR[1])
##        body_mx_SWIR_co = np.max(spectra_SWIR[0])
##        body_mx_SWIR_de = np.max(spectra_SWIR[1])
##
##        body_sm_NIR_co = np.sum(spectra_NIR[0])
##        body_sm_NIR_de = np.sum(spectra_NIR[1])
##        body_sm_SWIR_co = np.sum(spectra_SWIR[0])
##        body_sm_SWIR_de = np.sum(spectra_SWIR[1])
##
##        # Calculates wing parameters, (Maximum and sum of signal)
##        wing_mx_NIR_co = np.max(spectra_NIR[2])
##        wing_mx_NIR_de = np.max(spectra_NIR[3])
##        wing_mx_SWIR_co = np.max(spectra_SWIR[2])
##        wing_mx_SWIR_de = np.max(spectra_SWIR[3])
##
##        wing_sm_NIR_co = np.sum(spectra_NIR[2])
##        wing_sm_NIR_de = np.sum(spectra_NIR[3])
##        wing_sm_SWIR_co = np.sum(spectra_SWIR[2])
##        wing_sm_SWIR_de = np.sum(spectra_SWIR[3])
##
##        # Body/wing ratio
##        #                   0           1           2
##        # bw_ratio = [wing_max_co, wing_max_de, wing_max_spec
##        #               3           4       5
##        #             frac_co, frac_de, frac_spec]
##        bw_ration_NIR = FE.body_wing_ratio(spectra_NIR, WBF_mavg_comb,
##                                           fs, plot=False)
##
##        bw_ration_SWIR = FE.body_wing_ratio(spectra_SWIR, WBF_mavg_comb,
##                                            fs, plot=False)
##
##        # Stores wing max spectra for further use when calculating specularity
##        wing_max_spectras_NIR = [bw_ration_NIR[0],
##                                 bw_ration_NIR[1],
##                                 bw_ration_NIR[2]]
##
##        wing_max_spectras_SWIR = [bw_ration_SWIR[0],
##                                  bw_ration_SWIR[1],
##                                  bw_ration_SWIR[2]]
##
##        # Reduces to one value by taking the median
##        bw_ratio_NIR_co = np.median(bw_ration_NIR[3])
##        bw_ratio_NIR_de = np.median(bw_ration_NIR[4])
##        bw_ratio_NIR_spec = np.median(bw_ration_NIR[5])
##
##        bw_ratio_SWIR_co = np.median(bw_ration_SWIR[3])
##        bw_ratio_SWIR_de = np.median(bw_ration_SWIR[4])
##        bw_ratio_SWIR_spec = np.median(bw_ration_SWIR[5])
##
##        # Calculate specularity
##        spec_frac_NIR = FE.specularity(spectra_NIR, wing_max_spectras_NIR,
##                                       fs, plot=False)
##
##        spec_frac_SWIR = FE.specularity(spectra_SWIR, wing_max_spectras_SWIR,
##                                        fs, plot=False)
##
##        # Reduces to one value by taking the median
##        body_spec_NIR_code = np.median(spec_frac_NIR[0])
##        body_spec_NIR_specde = np.median(spec_frac_NIR[1])
##        wing_spec_NIR_code = np.median(spec_frac_NIR[2])
##        wing_spec_NIR_specde = np.median(spec_frac_NIR[3])
##
##        body_spec_SWIR_code = np.median(spec_frac_SWIR[0])
##        body_spec_SWIR_specde = np.median(spec_frac_SWIR[1])
##        wing_spec_SWIR_code = np.median(spec_frac_SWIR[2])
##        wing_spec_SWIR_specde = np.median(spec_frac_SWIR[3])
##
##        # max([body_co, body_de, wing_co, wing_de, specular])
##        max_NIR = np.max(spectra_NIR, 1)
##        max_SWIR = np.max(spectra_SWIR, 1)
##
##        # maximum of co- & depolarized signal in SWIR /
##        # maximum of co- & depolarized signal in SWIR and NIR
##        melanization_body = (max_SWIR[0] + max_SWIR[1])/(max_NIR[0] +
##                                                         max_NIR[1] +
##                                                         max_SWIR[0] +
##                                                         max_SWIR[1])
##
##        melanization_wing = (max_SWIR[2] + max_SWIR[3])/(max_NIR[2] +
##                                                         max_NIR[3] +
##                                                         max_SWIR[2] +
##                                                         max_SWIR[3])
##
##        # Stores all data in an array
##        event_array = np.array([files[i],               # Metadata
##                                dirs[i],
##                                time_datetime,
##                                total_event_nr,
##                                nr, seconds,
##                                t_start[nr],
##                                t_stop[nr],
##                                opt_mass,
##                                time_max,
##                                COM_time,
##                                saturation,
##                                WBF_cepstrum,           # WBF estimations
##                                WBF_HPS4,
##                                WBF_HPS3,
##                                WBF_HPS2,
##                                WBF_autocor,
##                                WBF_combined,
##                                WBF_peak[0],
##                                WBF_peak[1],
##                                harmonics[0],
##                                harmonics[1],
##                                harmonics[2],
##                                harmonics[3],
##                                harmonics[4],
##                                harmonics[5],
##                                WBF_inv[0],
##                                WBF_mavg[0],
##                                WBF_mavg_comb,
##                                WBF_sgo,
##                                body_mx_NIR_co,         # NIR parameters
##                                body_mx_NIR_de,
##                                body_sm_NIR_co,
##                                body_sm_NIR_de,
##                                wing_mx_NIR_co,
##                                wing_mx_NIR_de,
##                                wing_sm_NIR_co,
##                                wing_sm_NIR_de,
##                                bw_ratio_NIR_co,
##                                bw_ratio_NIR_de,
##                                bw_ratio_NIR_spec,
##                                body_spec_NIR_code,
##                                body_spec_NIR_specde,
##                                wing_spec_NIR_code,
##                                wing_spec_NIR_specde,
##                                body_mx_SWIR_co,        # SWIR parameters
##                                body_mx_SWIR_de,
##                                body_sm_SWIR_co,
##                                body_sm_SWIR_de,
##                                wing_mx_SWIR_co,
##                                wing_mx_SWIR_de,
##                                wing_sm_SWIR_co,
##                                wing_sm_SWIR_de,
##                                bw_ratio_SWIR_co,
##                                bw_ratio_SWIR_de,
##                                bw_ratio_SWIR_spec,
##                                body_spec_SWIR_code,
##                                body_spec_SWIR_specde,
##                                wing_spec_SWIR_code,
##                                wing_spec_SWIR_specde,
##                                melanization_body,      # Melanization
##                                melanization_wing])
##
##
#%% NEW FEATURES -------------------
    # helper functions
def savgol(spectrum, fs=20000, f_max=1500):
    """Applies two consecutive Savitzky-Golay filters (one short, one longer)
    on the signal spectrum. This is a helper function designed to be used in 
    feature extraction methods."""
    s = 1
    fftx = np.fft.fftfreq(len(spectrum), 1/fs)
    # Get index where freq > f_max
    sind = np.where(fftx > f_max)[0][0]
    # Set number of points used for short window filtering
    filter_short = np.floor((s/100)*sind).astype('int')

    if filter_short < 2:
        filter_short = 2

    if(np.remainder(filter_short, 2) == 0): #(window length must be odd)
        filter_short += 1

    # Set number of points used for long window filtering
    filter_long = np.floor((s*3/100)*sind).astype('int')

    if filter_long <= 2:
        filter_long = 2

    if(np.remainder(filter_long, 2) == 0):
        filter_long += 1

    # Set number of extra points to use after long filtering to avoid
    # spikes in 1st derivative
    sPoints = np.floor((s/2/100)*sind).astype('int')

    if sPoints == 0:
        sPoints = 1

    # Only grab the part of the signal that is below f_max
    yvect = np.abs(np.abs(spectrum[0:sind]))

    # Apply two consecutive linear Savitzky-Golay filters to signal**2
    smooth_yvect1 = scipy.signal.savgol_filter(yvect**2, filter_long, 1)
    smooth_yvect2 = scipy.signal.savgol_filter(smooth_yvect1, filter_short, 1)
    
    return smooth_yvect2, fftx


def get_peaks(signal, **kwargs):
    """Returns peak positions and prominences in a signal or spectrum.
    Standard peak-finding method.
    Parameters:
        signal: the signal to find peaks
        kwargs: keyword arguments passed to set peak prominence, distance etc.""" 
        
    peaks = scipy.signal.find_peaks(signal, **kwargs)[0]
    proms = scipy.signal.peak_prominences(signal, peaks)[0]
    return peaks, proms

def interbeat(peak, fftx):
    """Gets the exact frequency at the peak point from the frequency list.
    Parameters:
        peak: Peak position, int
        fftx: Frequencies, numpy array"""
    
    fl = int(np.floor(peak))
    unit = fftx[int(np.ceil(peak))] - fftx[fl]
    dec = peak - fl
    return fftx[fl] + dec * unit

# KDE algorithms
def kde_peaks(diffs, bandwidth=1, drawPlots=False, method="max"):
    """Returns the average expected distance between peaks using Kernel Density
    Estimation. Uses a list of peak differences. Parameters:
        diffs: The numpy.array of peak position differences.
        bandwidth: KDE-parameter, has to be set manually. Default=1
        drawPlots: Set to True if you want displays of the (max) method.
        method: {'max', 'cluster'}: The 'max' method simply returns the maximum
            of the density plot, while the 'cluster' method looks for the 
            largest cluster between local minima.
            """

    difpts = diffs.reshape(-1, 1)

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(difpts)
    s = np.linspace(min(difpts),max(difpts), num=150)#.reshape(-1,1)
    e = kde.score_samples(s.reshape(-1,1))
    ix = np.argmax(e)
    if drawPlots:
        plt.figure()
        plt.plot(s, e)
        plt.scatter(s[ix], e[ix], c="r")
        plt.annotate(str(s[ix][0]), (s[ix][0], e[ix]))
        plt.title(f"sklearn-KDE, {method}, bandwidth={bandwidth}")
        
    if method == "max":
        return s[ix][0]
    
    elif method == "cluster":
        mi = scipy.signal.argrelextrema(e, np.less)[0]
        mi = np.insert(mi,0,0)
        mi = np.append(mi,len(s)-1)
        
        largest_cluster = 0
        for i in range(len(mi)-1):
            cluster = difpts[(difpts >= s[mi[i]]) & (difpts <= s[mi[i+1]])]
            cluster_length = len(cluster)
            meanValue = cluster.mean()
            clusterWeight = meanValue*cluster_length
            if clusterWeight > largest_cluster:
                largest_mi = i
                largest_cluster = clusterWeight
    
        if largest_cluster == 0:
            return 0
    
        mostPoints = difpts[(difpts >= s[mi[largest_mi]]) & (difpts <= s[mi[largest_mi+1]])]
        meanDist = mostPoints.mean()
    
        return meanDist
    
    else:
        raise ValueError("Method must be 'max' or 'cluster'.")

# KDE-algorithm for raw signal WB
def rawkde_numbeat_fundamental(data, fs=20000, f_max=1000, prominence=None, **kdekw):
    """Estimation of the number of wingbeats and fundamental frequency based on
    the event signal. Scipy find_peaks and Kernel Density Estimation are used
    for the fundamental frequency.
    Parameters:
        data: np.array, 1xN, data signal
        fs: int, sampling frequency, default 20000
        f_max: int, maximum allowed frequency, sets the distance of the peaks,
            default 1000
        prominence: Peak prominences, default is 3 times the standard deviation
            of the signal
        kdekw: KDE algorithm keyword arguments"""
    if not prominence:
        prominence = 3 * data.std()
    try:
        peaks, _ = get_peaks(data, prominence=prominence, distance=fs//f_max)
        if len(peaks) == 0:
            return 0, 0
        diffs = np.insert(np.diff(peaks), 0, peaks[0])
        kde = kde_peaks(diffs, **kdekw)
        return len(peaks), fs / kde
    except Exception as e:
        print(e)

        return 0, 0

# SGO-KDE algorithm for FFT
def find_fundamental_sgokde(spectrum, fs=20000, inter=True, **kdekw):
    """Fundamental frequency estimation using the SGO filter, then selecting 
    the peak using Kernel Density Estimation. For details see the sgo and the 
    kde_peaks functions.
    Parameters:
        spectrum: np.array, the spectrum whose fundamental we want
        fs: int, sampling frequency
        inter: boolean, if True, it attemps to give a more precise frequency 
            by refining the resolution
    Returns:
        fund, peaks, proms: the fundamental frequency, the peaks and their 
            prominences in the Fourier domain"""
    try:    
        filt, fftx = savgol(spectrum, fs=fs)
        cutoff = 50
        peaks, proms = get_peaks(filt, prominence=3*filt[cutoff:].std())
        diffs = np.insert(np.diff(peaks), 0, peaks[0])
        kde = kde_peaks(diffs, **kdekw)
        if not inter:
            return fftx[kde], peaks, proms
        else:
            return interbeat(kde, fftx), peaks, proms

    except Exception as e:
        print(e)
        return 0, [0], 0
    
#def sgokde_extras(peaks, proms, num=2):
#    """Extracts the extra features from the peaks and prominences of the SGO
#    KDE algorithm (it's just a reorder and count etc.)
#    Has to be developed"""
#    num_harmonics = len(peaks)
#    sorter = np.argsort(proms)[::-1]
#    return num_harmonics

#
def waviness(arr):
    return sum(np.square((pandas.Series(arr).rolling(5, center=True).mean() - arr).dropna())) / np.std(arr)
    
def checkshift(array, maxlag=25, min_corr=.6, min_ratio=1.01, 
               max_wavy=.03, min_slope=1.2,
               plots=False):
    """Checks whether there is a significant cross-correlation between two 
    channels in the same segment, that has a lag other than zero."""
    
    long_mid = array.shape[1]
    m = maxlag - 1
    xo = array[0,:] - array[0,:].mean()
    yo = array[1,:] - array[1,:].mean()
    corr = np.correlate(xo, yo, 'full')
    factor = np.correlate(np.ones(long_mid), np.ones(long_mid), 'full')
    std_factor = np.std(xo) * np.std(yo)
    cfun = (corr / factor / std_factor)[long_mid - maxlag : long_mid + maxlag]
    
    # Check for shape
    diffirst = (cfun[m] - cfun[0]) / np.std(cfun)
    difflast = (cfun[m] - cfun[-1]) / np.std(cfun)
    if plots:
        plt.plot(cfun)
    
    if diffirst < min_slope or difflast < min_slope or waviness(cfun) > max_wavy:
        return None
    if np.argmax(cfun) != m: # Higher value somewhere other than center
        if max(cfun) > min_corr and max(cfun)/cfun[m] > min_ratio:
            # Further conditions
            lag = np.argmax(cfun) - m
            return lag