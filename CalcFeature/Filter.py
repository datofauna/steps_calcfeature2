# -*- coding: utf-8 -*-
"""
This module contains shared functions for filtering events from csv data files
created based on FaunaPhotonics LIDAR data

You can add or change things but remember to administrate in the function header
what you changed.

"""

import numpy as np

# %%
def slope_filter(slope):
    """Function that generates a mask that is 0 for events with slope value 'nan',
    and 1 for all other events

    Inputs
    ----------
    slope : np.array N
        Array of slope values for N events

    Outputs
    ----------
    mask_slope : np.array N
        Array with 0 for events that has slope 'nan', and 1 for all other events
    """

    mask = np.isnan(slope) == 0
    mask_slope = 1*mask

    return mask_slope

def duration_filter(duration, min_dur):
    """Function that generates a mask that is 0 for events with duration less than
    min_dur pixels, and 1 for all other events

    Inputs
    ----------
    duration : np.array N
        Array of duration values for N events
    min_dur : integer
        Minimum duration in pixels for an event to be considered a 'real' event

    Outputs
    ----------
    mask_duration : np.array N
        Array with 0 for events that has durations less than min_dur, and 1 for
        all other events
    """

    mask = duration>min_dur
    mask_duration = 1*mask

    return mask_duration

def mass_filter(mass, min_mass=0):
    """Function that generates a mask that is 0 for events with mass less than
    min_mass, and 1 for all other events

    Inputs
    ----------
    mass : np.array N
        Array of mass values for N events
    min_mass : integer
        Optional. Minimum mass for an event to be considered a 'real' event.
        Standard set to 0

    Outputs
    ----------
    mask_mass : np.array N
        Array with 0 for events that has mass less than min_mass, and 1 for
        all other events
    """

    mask = mass>min_mass
    mask_mass = 1*mask

    return mask_mass

def COM_filter(COM, min_COM=0, max_COM=2048):
    """Function that generates a mask that is 0 for events where COM is out of bounds
    (lower than min_COM or higher than max_COM) or nan, and 1 for all other events

    Inputs
    ----------
    COM : np.array N
        Array of center of mass values for N events
    min_COM : integer
        Optional. Minimum COM for an event to be considered a 'real' event.
        Standard set to 0
    max_COM : integer
        Optional. Max COM for an event to be considered a 'real' event.
        Standard set to 2048

    Outputs
    ----------
    mask_COM : np.array N
        Array with 0 for events that has COM lower than min_COM, higher than max_COM,
        or nan, and 1 for all other events
    """

    mask_nan = np.isnan(COM) == 0
    mask_low = COM>min_COM
    mask_high = COM<max_COM
    mask_COM = (1*mask_nan)*(1*mask_low)*(1*mask_high)

    return mask_COM

def WBF_filter(WBF, min_WBF=50, max_WBF=750):
    """Function that generates a mask that is 0 for events outsize the region
    min_WBF to max_WBF (WBF is wing beat frequency)

    Inputs
    ----------
    WBF : np.array N
        Array of WBF values for N events
    min_WBF : float
        Optional. Minimum WBF for an event to be considered a 'real' event.
        Standard set to 50
    max_WBF : float
        Optional. Minimum WBF for an event to be considered a 'real' event.
        Standard set to 750

    Outputs
    ----------
    mask_mass : np.array N
        Array with 0 for events that has WBF less than min_WBF or more than
        max_WBF, and 1 for all other events
    """

    mask = (WBF > min_WBF) & (WBF < max_WBF)

    return mask*1

def WBF_compare_filter(WBF1,WBF2,percentage = 10):
    """Function that generates a mask that is 0 for events where the two
    WBF measures defer more than a setr percentage (WBF is wing beat frequency)

    Inputs
    ----------
    WBF1 : np.array N
        Array of WBF values for N events
    WBF1 : np.array N
        Array of WBF values for N events
    percentage : float
        Optional. Percentage of WBF difference allowed to pass filter


    Outputs
    ----------
    mask_mass : np.array N
        Array with 1 for events that has WBFs that are within the given
        percentage regime the same, and 0 for all other events
    """
    p_plus = 1.0 + percentage/100
    p_minus = 1.0 - percentage/100

    mask = (WBF2 < WBF1*p_plus) & (WBF2 > WBF1*p_minus)

    return mask*1

def standard_filter(dt, slope='on', duration='on', mass='on',COM='on'):
    """Function creates a unified filter consisting of all the individual filters.

    Inputs
    ----------
    dt : np.array MxN
        Array of M values for N events
    slope : 'on' or 'off'
        Optional. Determines if slope_filter should be used. Standard set to 'on'
    duration : 'on' or 'off'
        Optional. Determines if duration_filter should be used. Standard set to 'on'
    mass : 'on' or 'off'
        Optional. Determines if mass_filter should be used. Standard set to 'on'
    COM : 'on' or 'off'
        Optional. Determines if COM_filter should be used. Standard set to 'on'

    Outputs
    ----------
    mask_full : np.array N
        Array with 0 for 'false' events determined by the accepted filters, 1 for all others

    20161003 - JHN: Added the wingbeat frequency filter, working on the combined frequency found
                with function find_fundamental_combined
    20161110 - JHN: Removed the WBF filter again and added it seperately in a new function
                standard_filter_new, so standard_filter can still be run with older csv-files
    """
    length = dt.shape[0]

    if slope == 'on':
        mask_slope = slope_filter(dt['slope'])
    else:
        mask_slope = np.ones(length)
        print('slope not filtered')

    if duration == 'on':
        duration = dt['t_stop'] - dt['t_start']
        min_dur = 40
        mask_duration = duration_filter(duration,min_dur)
    else:
        mask_duration = np.ones(length)
        print('duration not filtered')

    if mass == 'on':
        mask_mass = mass_filter(dt['opt_mass'])
    else:
        mask_mass = np.ones(length)
        print('mass not filtered')

    if COM == 'on':
        mask_COM = COM_filter(dt['COM'])
    else:
        mask_COM = np.ones(length)
        print('COM not filtered')

    mask_full = (1*mask_slope)*(1*mask_duration)*(1*mask_mass)*(1*mask_COM)

    return mask_full

def standard_filter_new(dt, slope='on', duration='on', mass='on',COM='on', WBF='on', fs=1500):
    """Function creates a unified filter consisting of all the individual filters.

    Inputs
    ----------
    dt : np.array MxN
        Array of M values for N events
    slope : 'on' or 'off'
        Optional. Determines if slope_filter should be used. Standard set to 'on'
    duration : 'on' or 'off'
        Optional. Determines if duration_filter should be used. Standard set to 'on'
    mass : 'on' or 'off'
        Optional. Determines if mass_filter should be used. Standard set to 'on'
    COM : 'on' or 'off'
        Optional. Determines if COM_filter should be used. Standard set to 'on'
    WBF: 'on' or 'off'
        Optional. Determines if WBF_filter should be used. Standard set to 'on'
    fs: integer
        Optional. Sampling frequency, only used if WBF='on'. Standard set to 1500 Hz

    Outputs
    ----------
    mask_full : np.array N
        Array with 0 for 'false' events determined by the accepted filters, 1 for all others

    20162510 - JHN: Added the WBF filter. Also added fs, to make sure the the
                    WBF filter uses the right limits independent of the sampling frequency
    """
    length = dt.shape[0]

    if slope == 'on':
        mask_slope = slope_filter(dt['slope'])
    else:
        mask_slope = np.ones(length)
        print('slope not filtered')

    if duration == 'on':
        duration = dt['t_stop'] - dt['t_start']
        min_dur = 40
        mask_duration = duration_filter(duration,min_dur)
    else:
        mask_duration = np.ones(length)
        print('duration not filtered')

    if mass == 'on':
        mask_mass = mass_filter(dt['opt_mass'])
    else:
        mask_mass = np.ones(length)
        print('mass not filtered')

    if COM == 'on':
        mask_COM = COM_filter(dt['COM_range'])
    else:
        mask_COM = np.ones(length)
        print('COM not filtered')

    if WBF == 'on':
        mask_WBF = WBF_filter(dt['WBF_combined'],max_WBF=fs/2)
    else:
        mask_WBF = np.ones(length)
        print('COM not filtered')

    mask_full = (1*mask_slope)*(1*mask_duration)*(1*mask_mass)*(1*mask_COM)*(1*mask_WBF)

    return mask_full


def use_mask(dt,mask):
    """Filters out events from a csv data set dt using mask which is an array
    that has 0 for events to be filtered away, and 1 for events that should be kept

    Inputs
    ----------
    dt : np.array MxN
        Array of M values for N events
    mask : np.array N
        Mask containing 0 for events to be filtered out and 1 for all other events

    Outputs
    ----------
    dt_filtered : np.array MxL
        Array of M values for L events, with all 'false' events filtered out
    """
    dt_filtered = dt[mask>0.5]

    return dt_filtered

def time_before_filter(time,end_date):
    """Function that generates a mask that is 0 for events that are after an
    given end date

    Inputs
    ----------
    time : np.array N
        Array or series of datetime objects for N events
    end_date : datetime object
        single time event for instance cerated as pandas.datetime(2016,06,22)

    Outputs
    ----------
    mask : np.array N
        Array with 1 for events that are before end_date
    """
    mask = time < end_date
    return mask*1

def time_after_filter(time,start_date):
    """Function that generates a mask that is 0 for events that are after an
    given end date

    Inputs
    ----------
    time : np.array N
        Array or series of datetime objects for N events
    end_date : datetime object
        single time event for instance cerated as pandas.datetime(2016,06,22)

    Outputs
    ----------
    mask : np.array N
        Array with 1 for events that are before end_date

    Usage example
    ----------
    day = pandas.datetime(2016,11,11)
    time_after(time,day)
    """
    mask = time > start_date
    return mask*1

def time_between_filter(time,start_date,end_date):
    """Function that generates a mask that is 0 for events that are outside a
    given date range

    Inputs
    ----------
    time : np.array N
        Array or series of datetime objects for N events
    end_date : datetime object
        single time event for instance cerated as pandas.datetime(2016,06,22)

    Outputs
    ----------
    mask : np.array N
        Array with 1 for events that are before end_date

    Usage example
    ----------
    day = pandas.datetime(2016,11,11)
    time_after(time,day)
    """
    mask = (time > start_date) & (time < end_date)
    return mask*1