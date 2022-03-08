# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:37:51 2021

@author: JosefineNielsen
"""
import numpy as np
import scipy.signal

def find_fundamental_sgo_combined(time_sig, fs = 20831, f_max = 1500, f_min = 25, conf = 5):
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
    try:
        #%%
        # Set percentage of number of points in spectrum to use for SGO filter
        s = 1
        # Get the frequencies for spectrum
        spectrum = np.fft.rfft(time_sig, len(time_sig))
        spectrum = np.abs(spectrum)
        fftx = np.fft.fftfreq(len(time_sig), 1/fs)
        # Define empty vector for frequency estimations
        freq_ind = np.zeros(5)
        freq_guess = np.zeros(5)
        low_freq_ind = np.zeros(2)
        low_freq_guess = np.zeros(2)
        
        # Get index where freq > f_max
        sind = np.where(fftx > f_max)[0][0]
        # Get index where freq > f_low
        lind = np.where(fftx > f_min)[0][0]
        
        # Set number of points used for short window filtering
        filter_short1 = np.floor((s/100)*sind).astype('int')
        filter_short2 = np.floor(lind/3).astype('int')

        if filter_short1 < 2:
            filter_short1 = 2
        if filter_short2 < 2:
            filter_short2 = 2

        if(np.remainder(filter_short1, 2) == 0): #(window length must be odd)
            filter_short1 += 1
        if(np.remainder(filter_short2, 2) == 0): #(window length must be odd)
            filter_short2 += 1

        # Set number of points used for long window filtering
        filter_long1 = np.floor((s*3/100)*sind).astype('int')
        filter_long2 = np.floor(lind).astype('int')

        if filter_long1 <= 2:
            filter_long1 = 2
        if filter_long2 <= 2:
            filter_long2 = 2

        if(np.remainder(filter_long1, 2) == 0):
            filter_long1 += 1
        if(np.remainder(filter_long2, 2) == 0):
            filter_long2 += 1

        # Set number of extra points to use after long filtering to avoid
        # spikes in 1st derivative
        sPoints = np.floor((s/2/100)*sind).astype('int')

        if sPoints == 0:
            sPoints = 1

        # Only grab the part of the signal that is below f_max
        yvect = np.abs(np.abs(spectrum[0:sind]))

        # Apply two consecutive linear Savitzky-Golay filters to signal**2
        smooth_yvect11 = scipy.signal.savgol_filter(yvect**2, filter_long1, 1)
        smooth_yvect12 = scipy.signal.savgol_filter(smooth_yvect11, filter_short1, 1)
        smooth_yvect21 = scipy.signal.savgol_filter(yvect**2, filter_long2, 1)
        smooth_yvect22 = scipy.signal.savgol_filter(smooth_yvect21, filter_short2, 1)

        # Get first derivative of 2nd Savitzky-Golay filter
        # if len(fftx) < 500:
        slope1 = np.diff(smooth_yvect12)
        slope2 = np.diff(smooth_yvect22)
        # else:
        #     slope1 = np.diff(smooth_yvect12[(filter_long1 + sPoints):len(smooth_yvect12)])
        #     slope2 = np.diff(smooth_yvect22[(filter_long2 + sPoints):len(smooth_yvect22)])

        # Normalize according to largest change
        slope1 = slope1/np.max(slope1)
        slope2 = slope2/np.max(slope2)
        
        # Calculate slope statistics
        slope1_med = np.median(abs(slope1))
        slope1_iqr = np.percentile(abs(slope1), 75) - np.percentile(abs(slope1), 25)
        slope2_med = np.median(abs(slope2))
        slope2_iqr = np.percentile(abs(slope2), 75) - np.percentile(abs(slope2), 25)

        # Set slope thresholds for defining spectral peaks
        thresh1 = slope1_med + 1.5 * slope1_iqr
        thresh2 = slope1_med + 2 * slope1_iqr
        thresh3 = slope2_med + 2.5 * slope2_iqr
        thresh4 = slope2_med + 3.5 * slope2_iqr
        thresh5 = slope2_med + 4.5 * slope2_iqr

        # if len(fftx) < 500:
            # Try to get index of the first place it crosses the threshold, then 
            # try to get index where it crosses the negative threshold, the first 
            # time after ind1
        ind11 = np.where(slope1 > thresh1)[0][0]
        ind12 = np.where(slope1 < -thresh1)[0][np.where(slope1 < -thresh1)[0] > ind11][0] + 1
        ind21 = np.where(slope1 > thresh2)[0][0]
        ind22 = np.where(slope1 < -thresh2)[0][np.where(slope1 < -thresh2)[0] > ind11][0] + 1
        ind31 = np.where(slope2 > thresh3)[0][0]
        ind32 = np.where(slope2 < -thresh3)[0][np.where(slope2 < -thresh3)[0] > ind11][0] + 1
        ind41 = np.where(slope2 > thresh4)[0][0]
        ind42 = np.where(slope2 < -thresh4)[0][np.where(slope2 < -thresh4)[0] > ind11][0] + 1
        ind51 = np.where(slope2 > thresh5)[0][0]
        ind52 = np.where(slope2 < -thresh5)[0][np.where(slope2 < -thresh5)[0] > ind11][0] + 1

        # Get the index of where the max in the spectrum occured between ind1
        # and ind2
        freq_ind[0] = np.argmax(yvect[ind11:ind12]) + ind11
        freq_ind[1] = np.argmax(yvect[ind21:ind22]) + ind21
        freq_ind[2] = np.argmax(yvect[ind31:ind32]) + ind31
        freq_ind[3] = np.argmax(yvect[ind41:ind42]) + ind41
        freq_ind[4] = np.argmax(yvect[ind51:ind52]) + ind51
        

        # Get the frequency corresponding to index freq_ind
        freq_guess[0] = fftx[int(freq_ind[0])]
        freq_guess[1] = fftx[int(freq_ind[1])]
        freq_guess[2] = fftx[int(freq_ind[2])]
        freq_guess[3] = fftx[int(freq_ind[3])]
        freq_guess[4] = fftx[int(freq_ind[4])]
        
        # Define fundamental frequency as the median of the calculated values
        nz_freq_ind = np.argwhere(freq_guess > 0)
        nz_freq_ind = nz_freq_ind[:,0]
        fund_freq_ind = int(np.round(np.median(freq_ind[nz_freq_ind])))
        fund_freq = np.median(freq_guess[nz_freq_ind])
        #%%
        low_thresh1 = slope1_med + 1 * slope1_iqr
        low_thresh2 = slope2_med + 1 * slope2_iqr
        
        low_ind11 = np.where(slope1 > low_thresh1)[0][0]
        low_ind12 = np.where(slope1 < -low_thresh1)[0][np.where(slope1 < -low_thresh1)[0] > low_ind11][0] + 1
        low_ind21 = np.where(slope2 > low_thresh2)[0][0]
        low_ind22 = np.where(slope2 < -low_thresh2)[0][np.where(slope2 < -low_thresh2)[0] > low_ind21][0] + 1
        
        low_freq_ind[0] = np.argmax(yvect[low_ind11:low_ind12]) + low_ind11
        low_freq_ind[1] = np.argmax(yvect[low_ind21:low_ind22]) + low_ind21
        
        low_freq_guess[0] = fftx[int(low_freq_ind[0])]
        low_freq_guess[1] = fftx[int(low_freq_ind[1])]
        
        power_thresh = np.median(smooth_yvect22) + 2 * (np.percentile(smooth_yvect22,75) - np.percentile(smooth_yvect22,25))
        
        low_half_bound = fund_freq / 2 * (1 - conf/100)
        high_half_bound = fund_freq / 2 * (1 + conf/100)
        
        half_picks = np.argwhere((low_freq_guess > low_half_bound) & (low_freq_guess < high_half_bound))
        half_picks = half_picks[:,0]
        half_picks = (smooth_yvect22[int(np.round(fund_freq_ind / 2))] > power_thresh) + len(half_picks)
        
        if half_picks >= 2:
            fund_freq = fund_freq / 2
            fund_freq_ind = int(np.round(fund_freq_ind / 2))
        #%%
    # If an error occured, return 0 as guess
    except Exception as e:
        #print('Error in SGO method,', str(e))
        fund_freq = 0

    return fund_freq

