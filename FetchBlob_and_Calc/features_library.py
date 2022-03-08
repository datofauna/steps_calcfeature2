import numpy as np
import pandas as pd
from .FeatureExtraction import body_slide_method, calc_spectrum, calc_BW_values, find_fundamental_sgo, find_fundamental_mavg, find_fundamental_invw, \
    find_fundamental_cepstrum, find_fundamental_sgokde, rawkde_numbeat_fundamental, calc_melanine
from .wbf import find_fundamental_sgo_combined
import scipy.signal as ss
import scipy.ndimage as sn

def calc_melanine(ts_array):
    mel_mean, mel_median, mel_center = calc_melanine(ts_array, iNIR=0, iSWIR=1)
    return mel_mean, mel_median, mel_center

def stat_moment(ts, n = 1, mu = 0, sigma = 1):  
    
    if np.min(ts) < 0:
        zero_ind = np.argwhere(ts < 0).flatten()
        ts[zero_ind] = 0
    
    weights = ts / np.sum(ts)
    ind = ((np.arange(0, len(ts)) - mu) / sigma) ** n
    moment = np.sum(np.multiply(ind, weights))
    
    return moment

def body_wing_separation(ts, wbf, fs = 20831): # New 
    
    try:
        W = int(np.round(fs / wbf))
        if W%2 == 0:
            W+=1
        
        smooth = ss.gaussian(2 * W + 1, W / (2 * np.sqrt(2 * np.log(2))))
        smooth = smooth / np.sum(smooth)
        
        sliding_minimum = sn.grey_erosion(ts, W)
        sliding_minimum = ss.convolve(sliding_minimum,smooth,'same')
        
        sliding_median = ss.medfilt(ts, W)
        sliding_median = ss.convolve(sliding_median,smooth,'same')
        
        sliding_maximum = sn.grey_dilation(ts, W)
        sliding_maximum = ss.convolve(sliding_maximum,smooth,'same')
        
        body_vector = sliding_minimum
        diffuse_wing_vector = 2 * (sliding_median - body_vector)
        specular_wing_vector = sliding_maximum - diffuse_wing_vector - body_vector
        
        mu = stat_moment(body_vector)
        sigma = np.sqrt(stat_moment(body_vector, n=2, mu=mu))
        
        start_ind = int(np.ceil(mu-sigma))
        stop_ind = int(np.ceil(mu+sigma))
        
        body_magnitude = np.median(body_vector[start_ind:stop_ind])
        diffuse_wing_magnitude = np.median(diffuse_wing_vector[start_ind:stop_ind])
        specular_wing_magnitude = np.median(specular_wing_vector[start_ind:stop_ind])
        
    except Exception as e:        
        body_magnitude = 0
        diffuse_wing_magnitude = 0
        specular_wing_magnitude = 0
    
    return body_magnitude, diffuse_wing_magnitude, specular_wing_magnitude

def average_WBF_func(ts, fs): #  average_WBF is bound to WBF_SGO_combined, being this the best option
    WBF_SGO_combined =  WBF_SGO_combined_func(ts, fs)
    return [WBF_SGO_combined, 100][WBF_SGO_combined==0]

def split_body_wing_sig(ts, fs):  #OLD
    WBF = average_WBF_func(ts, fs)
    body_sig = body_slide_method(ts, WBF, fs, style='min')
    wing_sig = body_slide_method(ts, WBF, fs, style='max')
    wing_sig = wing_sig - body_sig
    return body_sig, wing_sig
def BW_values_func(ts, fs):
    body_sig, wing_sig = split_body_wing_sig(ts, fs)
    BW_values = calc_BW_values(body_sig, wing_sig)
    return BW_values

def body_wing_sig_pair(ts_array):
    body_sigs = []
    wing_sigs = []
    for ts in ts_array:
        body_sig, wing_sig = split_body_wing_sig(ts, fs = 20831)
        body_sigs.append(body_sig)
        wing_sigs.append(wing_sig)
    body_sigs = np.array(body_sigs)
    wing_sigs = np.array(wing_sigs)

    return body_sigs, wing_sigs

def waviness(arr):
    return sum(np.square((pd.Series(arr).rolling(5, center=True).mean() - arr).dropna())) / np.std(arr)
    
def checkshift(array, maxlag=25, min_corr=.6, min_ratio=1.01, 
               max_wavy=.03, min_slope=1.2):
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
    
    if diffirst < min_slope or difflast < min_slope or waviness(cfun) > max_wavy:
        return None
    if np.argmax(cfun) != m: # Higher value somewhere other than center
        if max(cfun) > min_corr and max(cfun)/cfun[m] > min_ratio:
            # Further conditions
            lag = np.argmax(cfun) - m
            return lag



# Ordinal 0 functions
def length_func(ts, fs):
    return len(ts)/fs

# Ordinal 1 functions
def mean_func(ts, fs):
    return np.mean(ts)

def max_func(ts, fs):
    return np.max(ts) 

def min_func(ts, fs):
    return np.min(ts) 

def median_func(ts, fs):
    return np.median(ts) 

def WBF_mavg_func(ts, fs):
    FT = calc_spectrum(np.array(ts))
    [WBF_mavg, WBF_mavg_cutoff], mavg_sig = find_fundamental_mavg(FT, fs)
    return WBF_mavg

def WBF_mavg_cutoff_func(ts, fs):
    FT = calc_spectrum(np.array(ts))
    [WBF_mavg, WBF_mavg_cutoff], mavg_sig = find_fundamental_mavg(FT, fs)
    return WBF_mavg_cutoff

def WBF_sgo_func(ts, fs):
    FT = calc_spectrum(np.array(ts))
    WBF_sgo = find_fundamental_sgo(FT, fs)
    return WBF_sgo

def WBF_SGO_combined_func(ts, fs):
    WBF_SGO_combined = find_fundamental_sgo_combined(ts, fs)
    return WBF_SGO_combined

def WBF_inw_func(ts, fs):
    FT = calc_spectrum(np.array(ts))
    [WBF_mavg, WBF_mavg_cutoff], mavg_sig = find_fundamental_mavg(FT, fs)
    WBF_inw, WBF_inw_cutoff = find_fundamental_invw(FT, mavg_sig, fs)
    return WBF_inw

def WBF_inw_cutoff_func(ts, fs):
    FT = calc_spectrum(np.array(ts))
    [WBF_mavg, WBF_mavg_cutoff], mavg_sig = find_fundamental_mavg(FT, fs)
    WBF_inw, WBF_inw_cutoff = find_fundamental_invw(FT, mavg_sig, fs)
    return WBF_inw_cutoff

def WBF_sgokde_func(ts, fs):
    FT = calc_spectrum(np.array(ts))
    WBF_sgokde, peaks, _ = find_fundamental_sgokde(FT, fs)
    return WBF_sgokde

def WBF_cepstrum_func(ts, fs):
    FT = calc_spectrum(np.array(ts))
    WBF_cepstrum = find_fundamental_cepstrum(FT, fs)
    return WBF_cepstrum

def WBF_HPS_func(ts, fs):
    FT = calc_spectrum(np.array(ts))
    WBF_HPS = find_fundamental_HPS(FT, fs)
    return WBF_HPS

def WBF_rawkde_func(ts, fs):
    num_wingbeats, WBF_rawkde = rawkde_numbeat_fundamental(ts, method="cluster")
    return WBF_rawkde

def num_wingbeats_func(ts, fs):
    num_wingbeats, WBF_rawkde = rawkde_numbeat_fundamental(ts, method="cluster")
    return num_wingbeats

def num_harmonics_func(ts, fs):
    FT = calc_spectrum(np.array(ts))
    WBF_sgokde, peaks, _ = find_fundamental_sgokde(FT, fs)
    num_harmonics = [len(peaks), 0][type(peaks) is int]
    return num_harmonics

def body_magnitude_func(ts, fs):
    WBF_SGO_combined = find_fundamental_sgo_combined(ts, fs)
    body_magnitude, diffuse_wing_magnitude, specular_wing_magnitude = body_wing_separation(ts, WBF_SGO_combined)
    return body_magnitude

def diffuse_wing_magnitude_func(ts, fs):
    WBF_SGO_combined = find_fundamental_sgo_combined(ts, fs)
    body_magnitude, diffuse_wing_magnitude, specular_wing_magnitude = body_wing_separation(ts, WBF_SGO_combined)
    return diffuse_wing_magnitude

def specular_wing_magnitude_func(ts, fs):
    WBF_SGO_combined = find_fundamental_sgo_combined(ts, fs)
    body_magnitude, diffuse_wing_magnitude, specular_wing_magnitude = body_wing_separation(ts, WBF_SGO_combined)
    return specular_wing_magnitude

def BW_ratio_func(ts, fs):
    WBF_SGO_combined = find_fundamental_sgo_combined(ts, fs)
    body_magnitude, diffuse_wing_magnitude, specular_wing_magnitude = body_wing_separation(ts, WBF_SGO_combined)
    
    if (body_magnitude > 10) and (diffuse_wing_magnitude > 0):
        BW_ratio = body_magnitude / (body_magnitude + diffuse_wing_magnitude)
    else:
        BW_ratio = 0
    return BW_ratio

def SW_ratio_func(ts, fs):
    WBF_SGO_combined = find_fundamental_sgo_combined(ts, fs)
    body_magnitude, diffuse_wing_magnitude, specular_wing_magnitude = body_wing_separation(ts, WBF_SGO_combined)
    
    if (specular_wing_magnitude > 10) and (diffuse_wing_magnitude > 0):
        SW_ratio = specular_wing_magnitude / (specular_wing_magnitude + diffuse_wing_magnitude)
    else:
        SW_ratio = 0
    return SW_ratio

def BS_ratio_func(ts, fs):
    WBF_SGO_combined = find_fundamental_sgo_combined(ts, fs)
    body_magnitude, diffuse_wing_magnitude, specular_wing_magnitude = body_wing_separation(ts, WBF_SGO_combined)
    
    if (body_magnitude > 10) and (specular_wing_magnitude > 0):
        BS_ratio = body_magnitude / (body_magnitude + specular_wing_magnitude)
    else:
        BS_ratio = 0
    return BS_ratio

# Ordinal 2 functions

def body_mean_func(ts, fs):
    body_mean = BW_values_func(ts, fs)[0]
    return body_mean

def wing_mean_func(ts, fs):
    wing_mean = BW_values_func(ts, fs)[1]
    return wing_mean

def body_median_func(ts, fs):
    body_median = BW_values_func(ts, fs)[2]
    return body_median

def wing_median_func(ts, fs):
    wing_median = BW_values_func(ts, fs)[3]
    return wing_median

def body_center_func(ts, fs):
    body_center = BW_values_func(ts, fs)[4]
    return body_center

def wing_center_func(ts, fs):
    wing_center = BW_values_func(ts, fs)[5]
    return wing_center

def bw_ratio_mean_func(ts, fs):
    bw_ratio_mean = BW_values_func(ts, fs)[6]
    return bw_ratio_mean

def bw_ratio_median_func(ts, fs):
    bw_ratio_median = BW_values_func(ts, fs)[7]
    return bw_ratio_median

def bw_ratio_center_func(ts, fs):
    bw_ratio_center = BW_values_func(ts, fs)[8]
    return bw_ratio_center

def max_body_func(ts, fs):
    max_body = BW_values_func(ts, fs)[9]
    return max_body

def min_body_func(ts, fs):
    min_body = BW_values_func(ts, fs)[10]
    return min_body

def max_wing_func(ts, fs):
    max_wing = BW_values_func(ts, fs)[11]
    return max_wing

def min_wing_func(ts, fs):
    min_wing = BW_values_func(ts, fs)[12]
    return min_wing

# Ordinal 3 functions
def body_melanisation_func(ts_array, fs, iNIR = 0, iSWIR = 1):
    
    tsNIR = ts_array[iNIR, :].flatten()
    tsSWIR = ts_array[iSWIR, :].flatten()
        
    wbfNIR = find_fundamental_sgo_combined(tsNIR, fs)
    wbfSWIR = find_fundamental_sgo_combined(tsSWIR, fs)
        
    bNIR, dwNIR, swNIR = body_wing_separation(tsNIR, wbfNIR, fs)
    bSWIR, dwSWIR, swSWIR = body_wing_separation(tsSWIR, wbfSWIR, fs)
        
    if (bNIR > 0) & (bSWIR > 0):
        body_melanisation = bSWIR / (bSWIR+ bNIR)
    else:
        body_melanisation = 0

    return body_melanisation

def wing_melanisation_func(ts_array, fs, iNIR = 0, iSWIR = 1):
    
    tsNIR = ts_array[iNIR, :].flatten()
    tsSWIR = ts_array[iSWIR, :].flatten()
        
    wbfNIR = find_fundamental_sgo_combined(tsNIR, fs)
    wbfSWIR = find_fundamental_sgo_combined(tsSWIR, fs)
        
    bNIR, dwNIR, swNIR = body_wing_separation(tsNIR, wbfNIR, fs)
    bSWIR, dwSWIR, swSWIR = body_wing_separation(tsSWIR, wbfSWIR, fs)
        
    if (dwNIR > 0) & (dwSWIR > 0):
        wing_melanisation = dwSWIR / (dwSWIR + dwNIR)
    else:
        wing_melanisation = 0
    
    return wing_melanisation

def specular_ratio_func(ts_array, fs, iNIR = 0, iSWIR = 1):
    
    tsNIR = ts_array[iNIR, :].flatten()
    tsSWIR = ts_array[iSWIR, :].flatten()
        
    wbfNIR = find_fundamental_sgo_combined(tsNIR, fs)
    wbfSWIR = find_fundamental_sgo_combined(tsSWIR, fs)
        
    bNIR, dwNIR, swNIR = body_wing_separation(tsNIR, wbfNIR, fs)
    bSWIR, dwSWIR, swSWIR = body_wing_separation(tsSWIR, wbfSWIR, fs)
        
    if (swNIR > 0) & (swSWIR > 0):
        specular_ratio = swSWIR / (swSWIR + swNIR)
    else:
        specular_ratio = 0
    
    return specular_ratio

def mel_mean_body_func(ts_array, fs):
    if ts_array.shape[0] != 2:
        return None
    body_sigs, wing_sigs = body_wing_sig_pair(ts_array)
    mel_mean_body, mel_median_body, mel_center_body = calc_melanine(body_sigs)
    return mel_mean_body

def mel_median_body_func(ts_array, fs):
    if ts_array.shape[0] != 2:
        return None
    body_sigs, wing_sigs = body_wing_sig_pair(ts_array)
    mel_mean_body, mel_median_body, mel_center_body = calc_melanine(body_sigs)
    return mel_median_body

def mel_center_body_func(ts_array, fs):
    if ts_array.shape[0] != 2:
        return None
    body_sigs, wing_sigs = body_wing_sig_pair(ts_array)
    mel_mean_body, mel_median_body, mel_center_body = calc_melanine(body_sigs)
    return mel_center_body

def mel_mean_wing_func(ts_array, fs):
    if ts_array.shape[0] != 2:
        return None
    body_sigs, wing_sigs = body_wing_sig_pair(ts_array)
    mel_mean_wing, mel_median_wing, mel_center_wing = calc_melanine(wing_sigs)
    return mel_mean_wing


def mel_median_wing_func(ts_array, fs):
    if ts_array.shape[0] != 2:
        return None
    body_sigs, wing_sigs = body_wing_sig_pair(ts_array)
    mel_mean_wing, mel_median_wing, mel_center_wing = calc_melanine(wing_sigs)
    return mel_median_wing

def mel_center_wing_func(ts_array, fs):
    if ts_array.shape[0] != 2:
        return None
    body_sigs, wing_sigs = body_wing_sig_pair(ts_array)
    mel_mean_wing, mel_median_wing, mel_center_wing = calc_melanine(wing_sigs)
    return mel_center_wing

def mel_mean_tot_func(ts_array, fs):
    if ts_array.shape[0] != 2:
        return None
    mel_mean_tot, mel_median_tot, mel_center_tot = calc_melanine(ts_array)
    return mel_mean_tot

def mel_median_tot_func(ts_array, fs):
    if ts_array.shape[0] != 2:
        return None
    mel_mean_tot, mel_median_tot, mel_center_tot = calc_melanine(ts_array)
    return mel_median_tot

def mel_center_tot_func(ts_array, fs):
    if ts_array.shape[0] != 2:
        return None
    mel_mean_tot, mel_median_tot, mel_center_tot = calc_melanine(ts_array)
    return mel_center_tot


def phase_shift_error_func(ts_array, fs):
    phase_shift_error = checkshift(ts_array)
    return phase_shift_error



feature_dict = {0: {'length': length_func},
                1: {'mean': mean_func,
                    'max': max_func,
                    'min': min_func,
                    'median': median_func,
                    'WBF_SGO_combined': WBF_SGO_combined_func,
                    'body_magnitude': body_magnitude_func,
                    'diffuse_wing_magnitude': diffuse_wing_magnitude_func,
                    'specular_wing_magnitude': specular_wing_magnitude_func,
                    'BW_ratio': BW_ratio_func,
                    'SW_ratio': SW_ratio_func,
                    'BS_ratio': BS_ratio_func,
                    'WBF_sgo': WBF_sgo_func,
                    'WBF_sgokde': WBF_sgokde_func,
                    'WBF_mavg': WBF_mavg_func,
                    'WBF_cepstrum': WBF_cepstrum_func,
                    'WBF_rawkde': WBF_rawkde_func,
                    'WBF_mavg_cutoff': WBF_mavg_cutoff_func,
                    'WBF_HPS': WBF_HPS_func,
                    'WBF_inw': WBF_inw_func,
                    'WBF_inw_cutoff': WBF_inw_cutoff_func,
                    'num_harmonics': num_harmonics_func,
                    'num_wingbeats': num_wingbeats_func},
                2: {'body_mean': body_mean_func,
                    'wing_mean': wing_mean_func,
                    'body_median': body_median_func,
                    'wing_median': wing_median_func,
                    'body_center': body_center_func,
                    'wing_center': wing_center_func,
                    'bw_ratio_mean': bw_ratio_mean_func,
                    'bw_ratio_median': bw_ratio_median_func,
                    'bw_ratio_center': bw_ratio_center_func,
                    'max_body': max_body_func,
                    'min_body': min_body_func,
                    'max_wing': max_wing_func,
                    'min_wing': min_wing_func},
                3: {'body_melanisation': body_melanisation_func,
                    'wing_melanisation': wing_melanisation_func,
                    'specular_ratio': specular_ratio_func,
                    'mel_mean_tot': mel_mean_tot_func,
                    'mel_median_tot': mel_median_tot_func,
                    'mel_center_tot': mel_center_tot_func,
                    'mel_mean_body': mel_mean_body_func,
                    'mel_median_body': mel_median_body_func,
                    'mel_center_body': mel_center_body_func,
                    'mel_mean_wing': mel_mean_wing_func,
                    'mel_median_wing': mel_median_wing_func,
                    'mel_center_wing': mel_center_wing_func,
                    'phase_shift_error': phase_shift_error_func}}

