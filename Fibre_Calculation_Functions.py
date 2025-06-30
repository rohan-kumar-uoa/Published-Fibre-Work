### Stationary Enlight imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import os
import pandas as pd
import glob
import time
from multimethod import multimethod
from IPython.display import HTML
from typing import Union
from tqdm import tqdm

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

prop_cycle = plt.rcParams['axes.prop_cycle']
color_sequence = prop_cycle.by_key()['color']

master_wvl_range = [1545,1560]

### Time Dependent Enlight Imports
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, iirfilter, sosfiltfilt
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.stats import linregress

import plotly.subplots as sp
import plotly.graph_objects as go
import plotly.express as px

from datetime import datetime
from nptdms import TdmsFile

plt.rcParams.update({'font.size': 12})  # Apply a default font size

### TicToc
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def get_colourlist(n):
    return color_sequence[:n]

### File Management functions
def rm_ext(fname):
    return os.path.splitext(fname)[0]

def get_ext(fname):
    return os.path.splitext(fname)[1]

def get_rootdir():
    return os.path.split(os.getcwd())[0]

def get_common_fileext(dir, ext = "", filt = None):
    flist = glob.glob(os.path.join(dir,'*' + ext))
    
    if filt:
        flist = [fname for fname in flist if (filt in fname)]
    return flist 

def get_user_dir():
    return os.path.split(os.getcwd())[0]

### db-mw conversion functions
def db_to_mw(array_db):
    """
    Converts an array of values from decibels (dB) to milliwatts (mW).
    
    Parameters:
        array_db (numpy.ndarray): Input array with values in dB.
        
    Returns:
        numpy.ndarray: Converted array with values in mW.
    """
    return 10 ** (array_db / 10)

def mw_to_db(array_mw):
    """
    Converts an array of values from milliwatts (mW) to decibels (dB).
    
    Parameters:
        array_mw (numpy.ndarray): Input array with values in mW.
        
    Returns:
        numpy.ndarray: Converted array with values in dBm.
    """
    return 10*np.log10(array_mw)

def normalize(arr):
    return arr/np.max(arr)

### Read stationary spectra from Enlight
def sortbyval(d):
    """
    Sorts dictionary by its values, useful when calling .items(), .keys() and .values()
    
    Parameters:
        d (dictionary): dictionary with values numbers
        
    Returns:
        dictionary: Sorted dictionary by values
    """
    return dict(sorted(d.items(), key=lambda item: item[1]))

def read_spectrum(fname, specname = False):
    """
    Reads a stationary spectrum from the ENLIGHT software as a dataframe.
    
    Parameters:
        fname (string): filename to read
        specname (bool): name of spectrum. Set to the filename-extension as default.
        
    Returns:
        dict : {spectrum name : dataframe with wvl and corresponding power in db/mW}
    """
    if not specname:
        specname = os.path.split(fname)[-1]
        
    df = pd.read_csv(fname, sep = '\t', names = ['wvl', 'db'], header = 52, dtype = float, usecols = [0,1])
    df['power'] = 10**(0.1*df['db'])
    return {specname:df}

def read_spectrum_dir(fdir):
    """
    Reads a folder of stationary spectra from the Enlight software
    
    Parameters:
        fdir (string): folder name to read
        
    Returns:
        dict : dict with spectra names as keys, and spectrum dataframe as values.
    """
    fold_name = os.path.split(fdir)[1]

    specdic = {}
    for root,dirs,files in os.walk(fdir):
        files = files
        for fname in files:
            fname_full = os.path.join(root,fname)

            spdic = read_spectrum(fname_full, specname=fname[:-4])       
            
            for k,df in spdic.items():
                specdic[k] = df
    
    return specdic

def plot_spectrum(specdic, plot_mode = "db", normal = False, xlims = [1547,1553]):
    """
    Plot spectra stored in a dictionary. Can plot power in dB or mW.
    
    Parameters:
        specdic (dict): dictionary with spectra df as values.
        plot_mode (str): either db or mW for y-axis scale
        normal (bool): If True, all y-values are normalized so the maximum is 1.
        xlims (array): xlimits for the plots
        
        
    Returns:
        None
    """
    scale = .5
    plot_rat = np.array([25,9])
    plt.figure(figsize=scale*plot_rat)

    ylabs = {"db":"Power (dBm)", "power": "Power (W)"}
    for specname, df in specdic.items():
        data = df[plot_mode]
        if normal:
            data = normalize(data)
        plt.plot(df['wvl'], data, label = specname, linewidth = 2)
    
    plt.xlim(xlims)
    plt.xlabel('Wavelength (nm)')
    if normal:
        plt.ylabel("Normalized")
    else:
        plt.ylabel(ylabs[plot_mode])
    plt.ticklabel_format(useOffset = False)
    plt.legend()
    return

### Gaussian Functions
def gaussian(x, a, x0, fwhm):
    """
    Returns an array storing a gaussian signal evaluated at an array of wavelengths
    
    Parameters:
        x (numpy.ndarray): array of wavelengths 
        a (float): Amplitude of gaussian
        x0 (float): Mean (centre) of gaussian
        fwhm (float): Full width at half maximum [same units as x]
        
    Returns:
        (numpy.ndarray) gaussian signal
    """
    return a*np.exp(-4*np.log(2)*((x-x0)/fwhm)**2) 

def unit_gaussian(x,x0, fwhm):
    """Gaussian signal with unit amplitude. See gaussian() for parameters.
    """
    return gaussian(x,1,x0,fwhm)

def sum_ngauss(x,kwargs):
    kwargs = np.array(kwargs)
    if kwargs.ndim == 1:
        return gaussian(x,*kwargs)
    else:
        contrib = np.array([gaussian(x, *p) for p in kwargs])
        return np.sum(contrib, axis = 0)
    
### Superposition of gaussians
def G_1(x, F, R1, x0, fwhm):
    return F * R1* unit_gaussian(x, x0, fwhm)

def G_i(x, F, R:np.ndarray, x0:np.ndarray, fwhm:np.ndarray):
    if not (len(R) == len(x0) and len(x0) == len(fwhm)):
        raise ValueError("R,x0,fwhm must have same length")
    n = len(R)
    if n == 1:
        return F * R[0]* unit_gaussian(x, x0[0], fwhm[0])
    else:
        Ri = R[-1]
        arr = np.array([1-R[i]*unit_gaussian(x,x0[i], fwhm[i]) for i in range(n-1)])    
        arr = np.prod(np.vstack(arr), axis = 0)**2
        return Ri*unit_gaussian(x, x0[-1], fwhm[-1])*F * arr

def R_ORS(x, F, R, x0, fwhm):
    if not(isinstance(F,(int, float)) or (len(F) == len(x))):
        raise ValueError("F must either be a float, or have the same length as x")
    
    if not (len(R) == len(x0) and len(x0) == len(fwhm)):
        raise ValueError("R,x0,fwhm must have same length")
    n = len(R)

    factor = np.ones(len(x))
    combined = np.zeros(len(x))
    # for i in range(1,n+1):
    #         combined += G_i(x,F, R[:i], x0[:i], fwhm[:i])

    for i in range(n):
        currgauss = unit_gaussian(x,x0[i], fwhm[i])

        combined += F*R[i]*currgauss*factor

        factor *= (1-R[i]*currgauss)**2
    
    return combined

### Superposition of spectra
def realG_1(x, F, R1, S1):
    return F * R1* S1

def realG_i(x, F, R:np.ndarray, S:np.ndarray):
    if not (len(R) == len(S)):
        raise ValueError("R and S (spectra) must have same length")
    n = len(R)
    
    if n == 1:
        return F * R[0]* S[0]
    else:
        Ri = R[-1]
        Si = S[-1]
        arr = np.array([1-R[i]*S[i] for i in range(n-1)])    
        arr = np.vstack(arr)
        arr = np.prod(arr, axis = 0)**2
        return Ri*Si*F * arr

def combine_spectra(x, F, R, S, losses = [], shifts =[]):
    """
    Calculates the approximate combined spectrum from an array composed of individual spectra.
    Mainly used to simulate a FBG array using the individual FBG reflection spectra.
    Main formula from:
    [1] W. Liu, W. Zhou, Y. Wang, W. Zhang, and G. Yan, “Wavelength demodulation method for FBG overlapping spectrum utilizing bidirectional long short-term memory neural network,” Measurement, vol. 242, p. 115918, Jan. 2025, doi: 10.1016/j.measurement.2024.115918.
    
    Parameters:
        x (numpy.ndarray): Array of wavelengths for combined signal to be evaluated
        F (float OR numpy.ndarray): Output power of interrogator, assuming it is uniform over all wavelengths
                                    OR an array storing the output power as a function of wavelength
        R (list[float]): array storing amplitude of the individual spectra.
        S (list[numpy.ndarray]): list storing spectral responses to combine
        losses (list): list of coupling (or other such) losses between each spectra [dB]
        shifts (list): wavelength shift to apply to each spectra, same units as x. Positive shift values correspond to positive spectral shifts
        
    Returns:
        combined : combined spectrum, factoring in all losses and shifts
    """
    if not(isinstance(F,(int, float)) or (len(F) == len(x))):
        raise ValueError("F must either be a float, or have the same length as x")
    if not (len(R) == len(S)):
        raise ValueError("R and S (spectra) must have same length")
    n = len(R)  
    
    if isinstance(F,np.ndarray):
        print(R)
        ind = [F[r == x] for r in R]
        print(ind)
        Ref = R / F
    else:
        Ref = R / F

    if not ((len(shifts) == 0) or (len(shifts) == (n))):
        raise ValueError("Shifts must either be empty, or have the same length as S")
    if len(shifts) == 0:
        shifts = np.zeros(n, dtype = int)

    if not ((len(losses) == 0) or (len(losses) == (n))):
        raise ValueError("losses must either be empty, or have the same length as S")
    # Set coupling losses to zero if none were 
    # loss[i] is the loss from FBG_{i} -> FBG_{i+1}
    if len(losses) == 0:
        losses = np.zeros(n)

    # print(f"{R=}")

    # Normalize the spectral signals
    Snorm = np.array([s/np.max(s) for s in S])

    loss_rat = 10**(-.1*np.array(losses, dtype= float))

    factor = np.ones(len(x))
    combined = np.zeros(len(x))
    delta_wvl = np.diff(x)[0]
    # for i in range(1,n+1):
    #     combined += realG_i(x,F, R[:i], S[:i])
    # tf, tax = plt.subplots()
    for i in range(n):
        Stoadd = shift_signal(Snorm[i], shifts[i]/delta_wvl)
        # tax.plot(x,Stoadd, label = i)

        combined += F*Ref[i]*Stoadd*factor*(loss_rat[i]**2)

        factor*= ((1-Ref[i]*Stoadd)*(loss_rat[i]))**2
            # tax.plot(x, factor, label = i)

    # tax.legend()
    return combined

def analytic_triplesum(x,F,R,S, losses = [0,0,0], shifts =[0.,0.,0.]):
    """
    Calculates a more accurate combined spectrum than combine_spectra()
    Contribution from 2nd spectrum is the exact contribution. Contribution from 3rd spectrum is the same approximation as combine_spectra()
    Main formula from:
    [1] W. Liu, W. Zhou, Y. Wang, W. Zhang, and G. Yan, “Wavelength demodulation method for FBG overlapping spectrum utilizing bidirectional long short-term memory neural network,” Measurement, vol. 242, p. 115918, Jan. 2025, doi: 10.1016/j.measurement.2024.115918.
    
    Parameters:
        x (numpy.ndarray): Array of wavelengths for combined signal to be evaluated
        F (float OR numpy.ndarray): Output power of interrogator, assuming it is uniform over all wavelengths
                                    OR an array storing the output power as a function of wavelength
        R (list[float]): array storing amplitude of the individual spectra.
        S (list[numpy.ndarray]): list storing spectral responses to combine
        losses (list): list of coupling (or other such) losses between each spectra [dB]
        shifts (list): wavelength shift to apply to each spectra, same units as x. Positive shift values correspond to positive spectral shifts
        
    Returns:
        combined : combined spectrum, factoring in all losses and shifts
    """
    delta_wvl = np.diff(x)[0]
    Ref = R/F
    R1,R2,R3 =  Ref
    # print(R1,R2,R3)
    Snorm = np.array([s/np.max(s) for s in S])
    S_shifted = np.array([shift_signal(Snorm[i], shifts[i]/delta_wvl) for i in range(3)])
    S1,S2,S3 = S_shifted

    l1,l2,l3 = db_to_mw(-1*np.array(losses, dtype = float))
    # print(l1,l2,l3)

    term1 = R1*S1 * (l1**2)

    term2 = ((1-R1*S1)**2) * (R2*S2) * (1/(1-R1*R2*S1*S2*(l2**2))) * (l1**2)
    # term2 = ((1-R1*S1)**2) * (R2*S2)

    term3 = ((1-R1*S1)**2) * ((1-R2*S2)**2) * (R3*S3) * (l1*l2*l3)**2
    # term3 = 0
    return F*(term1 + term2 + term3)


### Helper functions for spectrum class
def get_peak(wvl, power):
    """Returns properties of the highest peak in a spectrum

    Parameters:
        wvl (numpy.ndarray): wavelength array
        power (numpy.ndarray): spectral response
        
    Returns:
        a (float): amplitude of highest peak
        x0 (float): x-location of corresponding peak
        fwhm (float): full width at half maximum of peak
    """
    delta_wvl = np.diff(wvl)[0]
    # peak_ind, peak_dict = find_peaks(power, height = 0.01, rel_height=0.5, width = 0)
    # a = power[peak_ind][0]
    # x0 = wvl[peak_ind][0]
    # fwhm = delta_wvl*peak_dict['widths'][0]
    
    peak_ind = [np.argmax(power)]
    widths, width_heights, left_ips, right_ips = peak_widths(power, peak_ind, rel_height = 0.5)

    a = power[peak_ind]
    x0 = wvl[peak_ind]
    fwhm = delta_wvl*(right_ips-left_ips)

    return a, x0, fwhm

def fit_gauss(p0,wvl, power):
    """
    Fits a gaussian to a spectrum given an initial guess of amplitude, location and FWHM. Useful to use in conjuction with get_peak()
    
    Parameters:
        p0 (list): initial guess for amplitude, location and FWHM of gaussian
        wvl (numpy.ndarray): wavelength array
        power (numpy.ndarray): spectral response
        
    Returns:
        a (float): gaussian amplitude
        x0 (float): gaussian mean
        fwhm (float): FWHM of gaussian
    """
    popt, pcov = curve_fit(gaussian, wvl, power, p0 = p0)

    a,x0,fwhm = popt
    return a,x0,fwhm

### Load output spectrum of si255
def si255_F(plotbool = False, wvl_range = None):
    """Output spectrum of si255 interrogator"""
    df = pd.read_csv('../rr.csv').convert_dtypes()

    wvl = np.arange(1460.,1620,.008)
    smoothdb = gaussian_filter1d(df.db, 1)
    inn = np.interp(wvl, df['wvl'], smoothdb)

    spl = UnivariateSpline(df.wvl, df.db, k = 2)
    spl.set_smoothing_factor(8)
    spl_out = spl(wvl)
    if plotbool:
        plt.plot(df.wvl, db_to_mw(df.db),label = "Data", linewidth = 5)
        
        # plt.plot(wvl,inn, label = "Gaussian smoothing")
        plt.plot(wvl, db_to_mw(spl_out), 'r', label = 'Spline')
        plt.legend()
        # plt.plot(wvl, pow)

    final_db = spl_out

    if wvl_range:
        new_sig, new_wvl = crop_data_by_wavelength(final_db,wvl,wvl_range = wvl_range)
        return new_wvl, db_to_mw(new_sig)
    
    return wvl, db_to_mw(final_db)

def si155_F(plotbool = False, wvl_range = [1500,1600]):
    """Output spectrum of si155 interrogator"""
    # df = pd.read_csv('../rr.csv').convert_dtypes()

    wvl = np.arange(1520.35, 1579.99,.01)

    d, new_wvl = crop_data_by_wavelength(wvl,wvl, wvl_range)
    F = .2
    return d, F

### Time Dependent Functions

def get_agg_file_list(all = False):
    r_dir = r"Fibre Work"
    fdir = r"Strain Testing Tidied Data"
    data_dir = os.path.join(get_rootdir(),r_dir,fdir)
    tensile_files = get_common_fileext(data_dir, ext = ".xls")
    responses_files = get_common_fileext(data_dir, ext = ".csv")
    peaks_files = get_common_fileext(data_dir, ext = ".txt", filt = "Peaks")
    agg_file_list = np.array(list(zip(tensile_files, responses_files, peaks_files)))

    if all:
        return agg_file_list
    else:
        kept_files = np.array([1,4,5,8,11])-1
        agg_file_list = agg_file_list[kept_files]
        return agg_file_list

### Timestamp Management

def elongate_times(times, start, end):
    """Stretch out an array of relative times between a specified start and end time"""
    deltaT = .143

    time_corr = deltaT
    for i in range(len(times)):
        if (times[i] > start) & (times[i] < end):
            times[i]+= time_corr
        if times[i] > start:
            time_corr += deltaT
    return times

def AbsTime_to_RelTime(abstimearr):
    """Convert an array of absolute timestamps into relative times
    
    Parameters:
        abstimearr (array[pd.Timestamp]): Array of absolute timestamps stored in pd.Timestamp format

    Returns:
        reltimearr (array[float]): Array of relative times as floats.    
        
    """
    if isinstance(abstimearr, np.ndarray):
        abstimearr = pd.Series(abstimearr)

    return np.array((abstimearr-abstimearr.iloc[0]).dt.total_seconds())

def RelTime_to_AbsTime(first_time, times):
    """Converts an array of relative times and a start time into absolute timestamps
    
    Parameters:
        first_time (pd.Timestamp): Absolute timestamp of first element
        times (array[float]): array of relative times

    Returns:
        tarr (numpy.ndarray[pd.Timestamp]): Array of absolute times  
        
    """
    tarr = pd.to_datetime(first_time.tz_localize(None).timestamp() + times,unit = 's')
    tarr = [t.tz_localize('Pacific/Auckland') for t in tarr]
    return np.array(tarr)

### Default loading functions for time-dependent spectra (from Bart)

def load_spec(lines):
    """Parses a 'Responses' type output file from si255 and si155 hyperion interrogators
    
    Parameters:
        lines (list): Lines of Responses file

    Returns:
        timestamps (numpy.ndarray): Relative timestamps of spectral response
        data_ch1 (numpy.ndarray): 2D array with rows being spectral responses of Channel 1 at different times 
        data_ch2 (numpy.ndarray): 2D array with rows being spectral responses of Channel 2 at different times  
        
    """
    timestamps = []
    data_ch1 = []
    data_ch2 = []

    for i in range(len(lines)):
        line = lines[i]
        
        # Match the timestamp
        match = re.match(r"(\d{1,2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}\.\d{5})", line)
        if match:
            # Parse the timestamp
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, "%d/%m/%Y %H:%M:%S.%f")
            timestamps.append(timestamp)
            
            # Check that there are at least two lines of data after the timestamp
            if i + 2 < len(lines):
                # Parse the first line of data
                values_str1 = lines[i + 1].strip()
                values1 = np.array([float(val) for val in values_str1.split('\t')])
                data_ch1.append(values1)
                

                ### COMMENTED OUT TO IGNORE PARSING CHANNEL 2
                # Parse the second line of data
                # values_str2 = lines[i + 2].strip()
                # values2 = np.array([float(val) for val in values_str2.split('\t')])
                # data_ch2.append(values2)

    # Convert lists to numpy arrays
    return np.array(timestamps), np.array(data_ch1), np.array(data_ch2)

def load_csv_to_dataframe(file_path):
    """
    Opens a file selection dialog, reads the chosen CSV file, and stores it in a variable with the given name.
    
    Parameters:
        df_name (str): The name of the DataFrame variable to assign the data to.
        
    Returns:
        DataFrame: The loaded pandas DataFrame.
    """
    # Check if a file was selected
    if file_path:
        try:
            # Load the file into a DataFrame
            df = pd.read_csv(file_path, delimiter=',', encoding='utf-8')
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df.fillna(0, inplace=True)  # Replace any NaNs with 0
            print(f"\nDataFrame loaded successfully from file: {os.path.basename(file_path)}")
            # print(df.head())  # Display the first few rows
            return df  # Return the loaded DataFrame
        except FileNotFoundError:
            print("\nError: File not found. Please check the file path and try again.")
        except pd.errors.EmptyDataError:
            print("\nError: The file is empty or has no valid data.")
        except Exception as e:
            print(f"\nError loading file: {e}")
    else:
        print("No file selected. Exiting.")
        return None

def convert_to_elapsed_time(timestamps):
    """Convert an array of Unix timestamps to a first time and relative timestamps
    
    Parameters:
        timestamps (numpy.ndarray): Array of Unix timestamps

    Returns:
       first_timestamp (pd.Timestamp): First timestamp
       elapsed_time (numpy.ndarray): Relative timestamps
        
    """
    # Convert the first timestamp to Unix time and store it
    first_timestamp = timestamps[0]
    first_timestamp_unix = int(first_timestamp.timestamp())
    
    # Calculate elapsed time in seconds for each timestamp
    elapsed_time = np.array([(ts - first_timestamp).total_seconds() for ts in timestamps])
    
    return pd.to_datetime(first_timestamp).tz_localize('Pacific/Auckland'), elapsed_time

def crop_data_by_columns(data, wl, start_col, end_col):
    """
    Crop the dataset array and wavelength array based on specified column numbers.

    Parameters:
    - data (np.array): Array containing the loaded data (Ch1 or Ch2)
    - wl (np.array): Array containing the second line of data for each timestamp.
    - start_col (int): Starting column index to keep (inclusive).
    - end_col (int): Ending column index to keep (exclusive).

    Returns:
    - tuple: Cropped data and wavelength arrays.
    """
    # Ensure the specified columns are within bounds of the arrays
    if start_col < 0 or end_col > data.shape[1]:
        raise ValueError("Column indices are out of bounds for the data arrays.")
    if start_col >= end_col:
        raise ValueError("start_col must be less than end_col.")
    
    # Crop the datasets
    cropped_data = data[:, start_col:end_col]
    cropped_wl = wl[start_col:end_col]
    
    return cropped_data, cropped_wl

def crop_data_by_wavelength(data, wvl, wvl_range):
    """
    Crop the dataset array and wavelength array based on specified wavelength numbers.

    Parameters:
    - data (np.array): Array containing the loaded data (Ch1 or Ch2)
    - wl (np.array): Array containing the second line of data for each timestamp.
    - wvl_range (numpy.ndarray): [start_wavelength (inclusive), end_wavelength (exclusive)]

    Returns:
    - tuple: Cropped data and wavelength arrays.
    """
    # Ensure the specified columns are within bounds of the arrays
    start_wvl, end_wvl = wvl_range
    if start_wvl >= end_wvl:
        raise ValueError("start_col must be less than end_col.")
    
    # Crop the datasets
    wvl_to_keep = (start_wvl<=wvl) & (wvl<end_wvl)
    cropped_wvl = wvl[wvl_to_keep]

    if data.ndim == 1:
        cropped_data = data[wvl_to_keep]
    elif data.ndim == 2:
        cropped_data = data[:,wvl_to_keep]
    else: 
        raise ValueError("data must be either 1D or 2D")
    
    
    return cropped_data, cropped_wvl

### Additional loading functions for .csv or .tdms spectra
def load_csvtdms_spectrum(fname, wvl_range = [1545,1560]):
    """Loads a Responses type file from Enlight saved as a .tdms file from Labview, or .csv file from signal_to_csv()
    
    Parameters:
        fname (str): filename
        fdir (str): file directory name
        root_dir (stir): absolute path of file directory)
        start_col, end_col: column range to keep.

    Returns:
        reltimearr (array[float]): Array of relative times as floats.    
        
    """

    ext = get_ext(fname)
    if ext == ".tdms":
        data = TdmsFile.pdread(fname)
        data_ch1 = data['Channel 1'].as_dataframe()
        data_ch1 = data_ch1.drop_duplicates()
        AbsTime = pd.to_datetime(data_ch1['Time (ns)']).dt.tz_localize('Pacific/Auckland')
    elif ext == '.csv':
        data_ch1 = pd.read_csv(fname)
        AbsTime = pd.to_datetime(data_ch1['Time (ns)']).dt.tz_convert("Pacific/Auckland")
    first_timestamp = AbsTime[0]
    elapsed_time = np.array((AbsTime-AbsTime[0]).dt.total_seconds())
    wvl = np.array(data_ch1.columns[1:], dtype = 'float')
    signal = np.array(data_ch1.loc[:, data_ch1.columns != 'Time (ns)'], dtype = 'float')
    signal, wvl = crop_data_by_wavelength(signal, wvl, wvl_range)

    return signal, wvl,(first_timestamp, elapsed_time)

def load_spectrum_timedep(fname, wvl_range = [1545,1560]):
    """NOTE: ONLY READS FIRST CHANNEL
    """

    if os.path.exists(rm_ext(fname)+".csv"):
        print("Loaded from .csv")
        return load_csvtdms_spectrum(rm_ext(fname)+'.csv', wvl_range = [1545,1560])
    else:
        if fname:
            with open(fname, 'r') as f:
                lines = f.readlines()
            timestamps, data_ch1, data_ch2 = load_spec(lines)
        
        first_timestamp, elapsed_time = convert_to_elapsed_time(timestamps)

        ### Add functionality to automatically import wavelength information
        for i in range(len(lines)):
            if "Wavelength Start (nm):" in lines[i]:
                start = float(lines[i].split()[-1])
                step = float(lines[i+1].split()[-1])
                numpoints = float(lines[i+2].split()[-1])
                break

        wavelength = np.arange(start, start + step*numpoints, step)

        ###S255
        # # Define the wavelength arrays
        # start = 1460
        # step = 0.008
        # end = 1460 + (step * 20000)  # Adjust end as needed to define length
        # wavelength = np.arange(start, end, step)

        ###SI55
        # wavelength = np.arange(1520, 1520+6000*0.01, 0.01)
        
        # data_ch1, wl_ch1 = crop_data_by_columns(data_ch1, wavelength, 10600, 12600)
        if wvl_range == None:
            wvl_range = [1545,1560]
        data_ch1, wl_ch1 = crop_data_by_wavelength(data_ch1, wavelength, wvl_range)

    ### Automatic Filtering (disable if not wanted)
    return data_ch1, wl_ch1, (first_timestamp,elapsed_time)
 
def signal_to_csv(signal, wvl, first_time,times, fname:str):

    df = pd.DataFrame(signal, columns = wvl)
    df['Time (ns)'] = RelTime_to_AbsTime(first_time, times)
    df = df[np.concat([df.columns[-1:],df.columns[:-1]])]
    save_filename = os.path.join(rm_ext(fname) + '.csv')
    df.to_csv(save_filename, index = False)
    return

### Convert signal to df, to be passed into animate_signal
def signal_to_df(signal, wvl, times):
    # df = pd.DataFrame()
    # ncols = signal.shape[1]
    # for time,sig in zip(times,signal):
    #     fs = 125
    #     sos = scipy.signal.iirfilter(4, Wn=4, fs=fs, btype="low", ftype="butter", output = 'sos')
    #     sig_filtered = scipy.signal.sosfiltfilt(sos, sig)


    #     # newrow = pd.DataFrame({'time': np.repeat([time], ncols), 'wvl': wvl, 'db':sig, 'power':db_to_mw(sig), 'db_filt':sig_filtered})
    #     newrow = pd.DataFrame({'time': np.repeat([time], ncols), 'wvl': wvl, 'db':sig, 'power':db_to_mw(sig),})

    #     df = pd.concat([df,newrow])

    testdf = pd.DataFrame(signal, columns = wvl)
    testdf['time'] = times
    testdf = testdf.melt(id_vars=['time'], value_vars=testdf.columns[0:len(testdf.columns)-1],var_name='wvl',value_name='db').sort_values(['time', 'wvl']).reset_index(drop=True)
    testdf['power'] = db_to_mw(testdf.db)

    return testdf

### Default plotting functions for time-dependent spectra
def plot_data_colormap(data,time,wl, plot_name = "ColorMap"):
    """Plots a time-dependent spectral response as a colormap
    
    Parameters: 
        data (numpy.ndarray): 2D array storing spectral response
        time (numpy.ndarray): 1D array storing relative timestamps. NOTE: colormap assumes timestamps are equally spaced
        wl (numpy.ndarray): wavelength array
        plot_name: (str, optional): Optional name of colourmap
    """
    plt.figure(figsize=(10, 4))
    
    # Display data as a colormap
    plt.imshow(data,extent=[wl[0], wl[-1], time[-1], time[0]], aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Data Value')  # Add a color bar for reference
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Time [s]')
    plt.title(plot_name)
    plt.tight_layout()
    plt.show()
    
def plot_spectrum_attimes(data,time,wl):
    plt.figure(figsize=(10, 6))
    # Display data as a colormap
    for i, y in enumerate(time):
        plt.plot(wl,data[i,:])
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity [dBm]')
#     plt.title('Data Line 1 as Colormap')
    
    plt.show()

def plot_interactive_colormap(data, time, wl, plot_name = "ColorMap"):
    """
    Plots an interactive colormap using Plotly.

    Parameters:
    - data (np.ndarray): 2D array of data values.
    - time (np.ndarray): 1D array of time values.
    - wl (np.ndarray): 1D array of wavelength values.
    """
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=wl,
        y=time,
        colorscale='Viridis',
        colorbar=dict(title='Data Value')
    ))

    fig.update_layout(
        xaxis_title='Wavelength [nm]',
        yaxis_title='Time [s]',
        title=plot_name,
        template='plotly_white',
        autosize = True
    )

    fig.update_layout(
    font=dict(
        size=18,  # Set the font size here
    ))
    fig.update_layout( autosize=False, width=800, height=300,margin={'t':50,'l':0,'b':0,'r':0} ) 
    fig.show()

def spectrum_surface_timedep(signal, wvl, times):
    fig = go.Figure(data=[go.Surface(z=signal, y=times, x=wvl)])
    fig.update_layout(title=dict(text='Spectrum Surface'), autosize=True,
                    width=800, height=700,
                    margin=dict(l=65, r=50, b=65, t=90))
    fig.update_layout(
        scene=dict(
            xaxis_title_text='Wavelength(nm)',
            yaxis_title_text='Time (s)',
            zaxis_title_text='Data'
        )
    )
    fig.update_scenes(aspectratio = dict( x = 1.5, y = 1, z = .5))
    fig.show()
    return 

# animate_signal is deprecated for now due to all the errors
# def animate_signal(signal_df, peak_df, mode = "power", title = "Animated Signal"):
    # if not (mode == "power" or mode == "db"):
    #     print("mode must be either power or db")

    fig1 = px.line(signal_df, x="wvl", y=mode, animation_frame="time",hover_name="wvl")

    # fig2 = px.line(signal_df, x="wvl", y="db_filt", animation_frame="time", color_discrete_sequence=['red'])

    fig2 = px.scatter(peak_df, x="wvl", y="power", animation_frame="time", color_discrete_sequence=['red'])

    fig3 = px.scatter(peak_df, x="left_wvl", y="width_height", animation_frame="time", color_discrete_sequence=['green'], animation_group = 'peak_id', hover_name="peak_id")

    fig4 = px.scatter(peak_df, x="right_wvl", y="width_height", animation_frame="time", color_discrete_sequence=['green'], animation_group = 'peak_id', hover_name="peak_id")


    # build frames to be animated from two source figures.
    frames = [go.Frame(data=f.data + fig2.frames[i].data + fig3.frames[i].data + fig4.frames[i].data, name=f.name) for i, f in enumerate(fig1.frames)]
    # frames = combine_frames([fig1, fig2])

    combined_fig = go.Figure(data=frames[0].data, frames=frames, layout=fig1.layout)
    combined_fig.update_layout(
        title=title,
        template='plotly_white',
        autosize = True
    )
    combined_fig.show()
    return

def combine_frames(figarr):

    n = len(figarr[0].frames)
    for i in range(len(figarr)):
        if len(figarr[i].frames) != n:
            print("Plots don't have the same amount of frames")
    
    frames = [go.Frame(data = sum([f.frames[i].data for f in figarr]), name = figarr[0].frames.name) for i in range(n)]
    return frames

### Time dependent signal processing
def fourier_signal(wvl, signal:np.ndarray, index = 0):
    N = len(wvl)
    T = 0.008
    sig = signal[index,:]
    sigf = fft(sig)
    wvlf = fftfreq(N, T)[:N//2]
    plt.plot(wvlf, 2.0/N * np.abs(sigf[0:N//2]))
    plt.grid()
    plt.yscale(value = 'log')
    plt.show()

def smooth_signal(data_ch1):
    fs = 125
    sos = iirfilter(4, Wn=4, fs=fs, btype="low", ftype="butter", output = 'sos')
    for i in range(len(data_ch1)):
        data_ch1[i,:] = sosfiltfilt(sos, data_ch1[i,:])
    return data_ch1

### Managing tensile data and Enlight PEAKS files
def load_tensile_data(fname):
    df = pd.read_csv(fname, names = ['AbsTime', 'RelTime', 'position', 'force', 'stress'], header = 1, sep = '\t')
    df.AbsTime = pd.to_datetime(df['AbsTime']).dt.tz_localize('UTC').dt.tz_convert('Pacific/Auckland')
    df['RelTime'] = AbsTime_to_RelTime(df['AbsTime'])
    forces = np.array(df['force'])
    first_time = df['AbsTime'][0]
    times = np.array(df['RelTime'])
    return forces, first_time, times

def load_enlight_peaks(fname, fdir = "", root_dir = ""):
    if not root_dir:
        root_dir = get_rootdir()
    spec_to_load = os.path.join(root_dir,fdir, fname)

    num_peaks = pd.read_csv(spec_to_load, skiprows=65, sep = '\t', names = range(2), usecols = [0,1])[1]
    max_num_peaks = np.max(num_peaks)

    peak_col_ind = list(range(9, max_num_peaks+9))

    df = pd.read_csv(spec_to_load, skiprows=65, sep = '\t', names = range(9 + max_num_peaks))
    df = df[[0] + peak_col_ind]

    df[0] = pd.to_datetime(df[0], dayfirst = True).dt.tz_localize('Pacific/Auckland')
    df[1] = AbsTime_to_RelTime(df[0])
    melted_df = df.melt(id_vars=[0, 1], value_vars=peak_col_ind,var_name='peak',value_name='wvl')
    melted_df.columns = ['AbsTime', 'RelTime', 'peak', 'wvl']

    melted_df = melted_df.dropna()
    return melted_df

def convert_responsesfolder_to_csv(fdir: str, wvl_range = [1540,1565], override_responses = True):
    print("Converting .txt files in ", fdir)
    for root,dirs,files in os.walk(fdir):
        for fname in files:
            if override_responses:
                check = (get_ext(fname) == '.txt')
            else: 
                check = ('Responses' in fname) and (get_ext(fname) == '.txt')
            if check:
                abs_fname = os.path.join(root, fname)
                signal, wvl, (first_time, times) = load_spectrum_timedep(abs_fname, wvl_range=wvl_range)
                print(fname, ": Loaded")
                signal_to_csv(signal, wvl, first_time, times, fname = abs_fname) 
                print(fname, ": Converted")
    print(fdir, " files converted to .csv")
    return

### Helper functions for peak tracking in data_culster.calc_peaks() method. 
### Using scipy.signal.find_peaks() and scipy.optimize.curve_fit()
def shifted_peaks_bg(signal, wvl, times, first_time):
    peak_df = pd.DataFrame()
    wvl_interper = interp1d(np.arange(len(wvl)),wvl, kind = 'linear')
    
    for (time,spectrum) in zip(times,signal):
        peak_ind, peak_dict = find_peaks(spectrum, height = 120e-6,)
        peak_results = np.array(peak_widths(spectrum, peak_ind, rel_height = .5))
        peak_results[2:4,:] = wvl_interper(peak_results[2:4,:])
        npeaks = len(peak_ind)
        
        if npeaks>0 :
            kept_peaks = np.argmax(peak_dict['peak_heights'])
            peak_ind = [peak_ind[kept_peaks]]
            peak_results = peak_results[:,kept_peaks]

            # print(peak_results)
            widths, width_heights, left_wvl, right_wvl = peak_results
            peak_id_arr = ['main']*len(peak_ind)
            newrow = pd.DataFrame({'time':[time]*len(peak_ind), 'bwid': widths*0.008, 'wvl': wvl[peak_ind], 'db':mw_to_db(spectrum[peak_ind]), 'power': spectrum[peak_ind], 'peak_id': peak_id_arr, 'left_wvl':left_wvl,'right_wvl':right_wvl, 'width_height':width_heights})

            peak_df = pd.concat([peak_df,newrow])
        else:
            nan_ent =float('NaN')
            newrow = pd.DataFrame({'time':[time], 'bwid': nan_ent, 'wvl': nan_ent, 'db':nan_ent, 'power': nan_ent, 'peak_id': nan_ent, 'left_wvl':nan_ent,'right_wvl':nan_ent, 'width_height':nan_ent})
            peak_df = pd.concat([peak_df,newrow])

    peak_df['AbsTime'] = np.array(RelTime_to_AbsTime(first_time, peak_df['time']))
    peak_df['RelTime'] = peak_df['time']
    return peak_df

def sum_1gauss(x,a1,x1,sigma1):
    return gaussian(x,a1,x1,sigma1)
def sum_2gauss(x,a1,x1,sigma1,a2,x2,sigma2):
    return gaussian(x,a1,x1,sigma1) +gaussian(x,a2,x2,sigma2)
def sum_3gauss(x,a1,x1,sigma1,a2,x2,sigma2,a3,x3,sigma3):
    return gaussian(x,a1,x1,sigma1) +gaussian(x,a2,x2,sigma2) + gaussian(x,a3,x3,sigma3)

def calculate_gaussian(signal, wvl,id):
    main_popt = []
    secondary_popt = []
    power = db_to_mw(signal)
    # popt0 = [0.0124, 1550.17, 0.465, 0.008, 1555,0.2]
    popt0, pcov = curve_fit(gaussian, wvl, power[0],bounds=([0, 1549,0,], [1, 1551., 1000]))
    popt = popt0.copy()

    func_to_fit = lambda x,a,x0,fwhm: gaussian(x,*popt0)+gaussian(x,a,x0,fwhm)
    for i in tqdm(range(signal.shape[0]), desc = "Fitting Gaussian Peaks"):
        popt, pcov = curve_fit(func_to_fit, wvl, power[i], p0 = popt,bounds=([0, popt0[1],0],[1, 1565., 10]))

        main_popt.append(popt0)
        secondary_popt.append(popt)

    return np.array(main_popt), np.array(secondary_popt)

### Time dependent spectra classes
class exp_datafile():
    def __init__(self, fname, fdir,root_dir, filetype) -> None:
        self.fname = fname
        self.fdir =fdir
        self.root_dir = root_dir
        self.abspath = os.path.join(root_dir, fdir,fname)
        self.filetype = filetype
        if filetype == "Tensile":
            self.frame = load_tensile_data(self.fname, fdir= self.fdir, root_dir = self.root_dir)
        elif filetype == "Enlight Peaks":
            self.frame = load_enlight_peaks(self.fname, fdir= self.fdir, root_dir = self.root_dir)
        else:
            raise ValueError("Filetype not supported")
    
    def __repr__(self) -> str:
        return f"{self.filetype} Data located at {self.abspath}"

class data_cluster():
    def __init__(self,fdir, root_dir) -> None:
        self.fdir = fdir
        self.root_dir = root_dir
        self.abspath = os.path.join(root_dir, fdir)

    def __repr__(self) -> str:
        return f"Data Cluster of files in {self.abspath}"

    def load_tensile(self, tensile_fname):
        self.tensile_fname = tensile_fname
        self.fibreid = rm_ext(os.path.split(tensile_fname)[1])[4:]
        self.ten = exp_datafile(tensile_fname, self.fdir, self.root_dir, "Tensile")

    def load_enlight(self, enlightpeaks_fname):
        self.enlightpeaks_fname = enlightpeaks_fname
        self.enpks = exp_datafile(enlightpeaks_fname, self.fdir, self.root_dir, "Enlight Peaks")
    
    def load_responses(self, resp_fname, wvl_range = None):
        self.resp_fname = resp_fname
        self.id = rm_ext(self.resp_fname)
        signal, wvl, (first_time, times) = load_spectrum_timedep(resp_fname, fdir = self.fdir, root_dir=self.root_dir, wvl_range = wvl_range)
        self.signal, self.wvl, (self.first_time, self.times) = (signal, wvl, (first_time, times))
        self.signal_to_plot()
        self.set_signal_df()

    def power(self):
        return db_to_mw(self.signal)
    
    def plot_colourmap(self):
        plot_data_colormap(self.signal, self.times, self.wvl, plot_name = self.id)

    def load_cluster(self, fname_group):
        tensile_fname, resp_fname, enlightpeaks_fname = fname_group
        self.load_tensile(tensile_fname)
        self.load_responses(resp_fname)
        self.load_enlight(enlightpeaks_fname)
    
    def calc_peaks(self):
        # self.calcpks = shifted_peaks_testing(self.signal, self.wvl, self.times, hthresh = hthresh, prominence = prominence, first_time = self.first_time)
        self.pkframe = shifted_peaks_bg(self.signalplot, self.wvl, self.times, self.first_time)

    def signal_to_plot(self):
        self.signalplot = db_to_mw(self.signal) - db_to_mw(self.signal[0,:])
        # self.signalplot = db_to_mw(self.signal[0,:]) - db_to_mw(self.signal)

    def set_signal_df(self):
        testdf = pd.DataFrame(self.signalplot, columns = self.wvl)
        testdf['time'] = self.times
        testdf = testdf.melt(id_vars=['time'], value_vars=testdf.columns[0:len(testdf.columns)-1],var_name='wvl',value_name='data value').sort_values(['time', 'wvl']).reset_index(drop=True)
        self.signal_df = testdf

    def animate(self):
        self.set_signal_df()
        animate_signal(self.signal_df, self.pkframe, mode = "data value", title = self.tensile_fname)
    
    def gaussian_peaks(self):
        main_popt, secondary_popt = calculate_gaussian(self.signal, self.wvl, self.fibreid)
        self.popt1 = main_popt
        self.popt2 =  secondary_popt

### Plotting force-time, peakwvl-peakpower, force-peakwvl
def modify_pkframe(pkframe):
    copypkframe = pkframe.copy()
    ind_to_rep = copypkframe['RelTime']>=78
    changestarttime = copypkframe['AbsTime'].iloc[np.argwhere(ind_to_rep)[0]]
    keptabstime = copypkframe['AbsTime'][np.logical_not(ind_to_rep)]
    newabstime = np.repeat([changestarttime], sum(ind_to_rep)) + pd.to_timedelta(np.linspace(0,76,sum(ind_to_rep)), unit = 's')
    copypkframe['AbsTime'] = np.concat([keptabstime, newabstime])
    return copypkframe

def plot_peakvsforce(tensile_df, peaks_df, peak_filt, plot_name, smoothness = 50):
    if "005_FBG3.xls" in plot_name:
        peaks_df = modify_pkframe(peaks_df)


    colseq = get_colourlist(2)
    min_start_time = np.min([tensile_df['AbsTime'].iloc[0], peaks_df['AbsTime'].iloc[0]])
    tensile_df['NewRelTime'] = tensile_df.AbsTime - min_start_time
    tensile_df['NewRelTime'] = tensile_df['NewRelTime'].dt.total_seconds()
    peaks_df['NewRelTime'] = peaks_df.AbsTime - min_start_time
    peaks_df['NewRelTime'] = peaks_df['NewRelTime'].dt.total_seconds()
    
    if peak_filt and ('peak_id' in peaks_df.columns):
        peaks_df = peaks_df[peaks_df['peak_id'] == peak_filt]

    smoother_force = gaussian_filter1d(tensile_df.force, smoothness)

    # fill_value = tuple(tensile_df.iloc[[0,-1]].force)
    force_interper = interp1d(tensile_df['NewRelTime'], smoother_force, bounds_error=False, fill_value = 'extrapolate', kind = 'linear')
    force_int = force_interper(peaks_df['NewRelTime'])

    fig, axs = plt.subplots(1,3, figsize = 0.5*np.array([30,10]))

    # plt.scatter(peaks_df['NewRelTime'], forces, s=3)
    axs[0].plot(tensile_df.NewRelTime, tensile_df.force, color = colseq[1], label = 'Force')
    axs[0].plot(tensile_df.NewRelTime, smoother_force, 'r-', label = 'Smooth Force')
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Force [N]", color = colseq[1])
    axs[0].legend(fontsize = 15)
    axs[0].tick_params(axis = 'y', colors = colseq[1]) 
    ax0_2 = axs[0].twinx()
    ax0_2.set_ylabel('Wavelength [nm]', color = colseq[0])
    ax0_2.tick_params(axis = 'y', colors = colseq[0]) 
    ax0_2.scatter(peaks_df.NewRelTime, peaks_df.wvl, color = colseq[0], label = 'Wavelength', s=7)
    ax0_2.ticklabel_format(style = 'plain', axis='y', useOffset = False)
    ### PLOT WAVELENGTH AGAINST FORCE
    # axs[1].scatter(force_int, peaks_df.wvl, color = 'r', s=3)
    # axs[1].set(xlabel = "Force [N]", ylabel = "Wavelength [nm]")

    axs[1].scatter(peaks_df.wvl, peaks_df.power, color = colseq[1], s=5)
    # axs[1].scatter(force_int, peaks_df.power, s= 3)
    axs[1].set_xlabel("Shifted Peak Wavelength [nm]")
    axs[1].set_ylabel("Peak Power[nm]", color = colseq[1])
    axs[1].tick_params(axis = 'y', colors = colseq[1]) 
    axs[1].ticklabel_format(style = 'plain', axis='y', useOffset = False)

    ax1_2 = axs[1].twinx()
    ax1_2.set_ylabel('Peak Bandwidth [nm]', color = colseq[0])
    ax1_2.tick_params(axis = 'y', colors = colseq[0]) 
    ax1_2.scatter(peaks_df.wvl, peaks_df.bwid, color = colseq[0], label = 'Bandwidth', s=5)
    ax1_2.ticklabel_format(style = 'plain', axis='y', useOffset = False)

    axs[2].scatter(force_int, peaks_df.wvl, color = 'r', label = 'force vs wvl', s= 4)
    axs[2].set_xlabel("Force [N]")
    axs[2].set_ylabel("Wavelength [nm]")
    axs[2].ticklabel_format(style = 'plain', axis='y', useOffset = False)

    if plot_name:
        plt.suptitle(plot_name)


    fitting_ind = peaks_df.wvl >= 1550.5
    fitting_ind = force_int >= 0.5
    try:
        result = linregress(force_int[fitting_ind], peaks_df['wvl'][fitting_ind])
        intercept_witherr = f'{result.intercept:.3f} +/- {result.intercept_stderr:.3f}'
        print(intercept_witherr)
        axs[2].plot(force_int[fitting_ind], np.polyval([result.slope, result.intercept], force_int[fitting_ind]), label = intercept_witherr)
        axs[2].legend(fontsize = 15)
    except ValueError:
        print(f"Error in f{plot_name}")

    # neg_wvl = np.mean(peaks_df['wvl'][force_int >= 0.5])
    # print(f"mean wvl is {neg_wvl}")
    # ax0_2.hlines([neg_wvl], xmin = 0, xmax = tensile_df.NewRelTime.iloc[-1], color = 'tab:purple', linewidth = 5)
    plt.tight_layout()
    return

### Shifting a spectrum
@multimethod
def shift_signal(signal: Union[list, np.ndarray],shift: Union[int, np.int64, np.int32]):
    """Shifts a signal left or right by an integer number of entries. Empty values in shifted signal are populated by smallest value in original signal.
    
    Parameters:
        signal (list, np.ndarray): signal array.
        shift (int): integer number of indices to shift signal

    Returns:
        rolled (numpy.ndarray): Shifted signal
        
    """
    # print('intshift')
    if shift == 0:
        return signal

    rolled = np.roll(signal, shift)
    if shift>0:
        rolled[:shift] = np.min(signal)
    elif shift<0:
        rolled[shift:] = np.min(signal)
    return rolled

@multimethod
def shift_signal(signal: Union[list, np.ndarray],shift: float):
    """Shifts a signal left or right by an floating point number of indices. Empty values in shifted signal are populated by smallest value in original signal.
    
    Parameters:
        signal (list, np.ndarray): signal array.
        shift (int): integer number of indices to shift signal

    Returns:
        rolled (numpy.ndarray): Shifted signal
        
    """
    # print('floatshift')
    if shift == 0:
        return signal
    n = len(signal)

    filler = np.min(signal)

    xval = np.arange(n)
    
    rolled = np.interp(xval - shift, xval, signal, left = filler, right = filler)

    return rolled

class fibredata():
    def __init__(self, folderpath, wvl_range = None) -> None:
        self.abspath = folderpath
        root, foldername = os.path.split(folderpath)
        self.foldername = foldername

        self.signals = []
        self.first_times = []
        self.timess = []
        self.ids = []

        self.force_first_times = []
        self.force_timess = []
        self.forces = []

        for root, dirs, files in os.walk(folderpath):
            for i,fname in enumerate(files):
                if get_ext(fname) == '.tdms':
                    fname_full = os.path.join(root, fname)
                    spectral_data, force_data = read_force_spectrum(fname_full, wvl_range)
                    (signal, wvl, first_time, times) = spectral_data
                    self.first_times.append(first_time)
                    self.timess.append(times)
                    self.signals.append(mw_to_db(signal)) 
                    self.ids.append(rm_ext(fname))
                    if i == 0:
                        self.wvl = wvl
                    elif (wvl == self.wvl).all():
                        pass
                    else:
                        print("Current wvl:", self.wvl)
                        print("New wvl:", wvl)
                        raise ValueError("Inconsistent Wavelength Arrays.")
                    
                    (force_first_time, force_times, forces) = force_data
                    self.force_first_times.append(force_first_time)
                    self.force_timess.append(force_times)
                    self.forces.append(forces)
                    print(fname, " Loaded")
        return
    
    def numdatasets(self):
        return len(self.first_times)

    def signal(self):
        return np.concatenate(self.signals)

    def times(self):
        master_time = np.array([])
        for t0, arrtimes in zip(self.first_times, self.timess):
            master_time = np.append(master_time, RelTime_to_AbsTime(t0, arrtimes))
                                    
        master_time = np.concatenate([self.first_times[:1], master_time])

        return AbsTime_to_RelTime(master_time)[1:]

    def force(self):
        return np.concatenate(self.forces)
    
    def force_first_time(self):
        return self.force_first_times[0]
    
    def force_times(self):
        master_time = np.array([])
        for t0, arrtimes in zip(self.force_first_times, self.force_timess):
            master_time = np.append(master_time, RelTime_to_AbsTime(t0, arrtimes))
                                    
        master_time = np.concatenate([self.force_first_times[:1], master_time])
        return AbsTime_to_RelTime(master_time)[1:]
    
    def first_time(self):
        return self.first_times[0]

    def plot_colourmap(self, index = None):
        if index != None:
            plot_data_colormap(self.signals[index], self.timess[index], self.wvl, plot_name = self.ids[index])
        else:
            plot_data_colormap(self.signal(), self.times(), self.wvl, plot_name = f"Combined from {self.abspath}")

    def power(self):
        return db_to_mw(self.signal)
    
    def plot_force(self, i = None):
        if i != None:
            plt.plot(self.force_timess[i], self.forces[i], label = self.ids[i])
        else:
            plt.plot(self.force_times(), self.force(), label = "all force")

        plt.xlabel("Time (s)")
        plt.ylabel("Force (N)")
        plt.legend()
        return
    
    def save_signal_npy(self):
        np.save(f'{self.foldername} - signal_data', self.signal())

def read_force_spectrum(fname, wvl_range = None):
    """Obtains data from a 'TandFBG_Tensile_edit.vi' type Labview file. Returns force and channel 1 spectral data.
    
    Parameters:
        fname (str): Absolute location of .tdms datafile
        wvl_range (numpy.ndarray, optional): If provided, crops signal into [lower_wvl, upper_wvl] range.

    Returns:
        tuple1:
            signal (numpy.ndarray): 2D array storing time-dependent signal for Channel 1
            wvl (numpy.ndarray): wavelength array
            first_time (pd.Timestamp): first-timestamp of signal
            times (numpy.ndarray): array of relative timestamps

        tuple2:
            force_first_time (pd.Timestamp): first-timestamp of force data
            force_times (numpy.ndarray): array of relative timestamps
            forces (numpy.ndarray): array of forces [N]
    """
    # with TdmsFile.open(fname) as tdms_file:
    #     for chunk in tdms_file.data_chunks():
    #         pass

    data = TdmsFile.read(fname)

    # Extract experiment start time
    first_wvlchannel = data['Spectra 1'].channels()[1].name
    first_time = data['Spectra 1'][first_wvlchannel].properties["wf_start_time"]
    first_time = pd.to_datetime(first_time).tz_localize("Pacific/Auckland")

    # Extract spectral data
    data_ch1 = data['Spectra 1'].as_dataframe()
    wvl = np.array(data_ch1.columns[1:], dtype = 'float')
    signal = np.array(data_ch1.loc[:, data_ch1.columns != 'Timestamps'], dtype = 'float')
    times = np.array(data_ch1['Timestamps'].astype('float'))

    if wvl_range:
        signal, wvl = crop_data_by_wavelength(signal, wvl, wvl_range)

    # Get Force Data
    # get force start time
    force_first_time = data['Tensile']['Force'].properties["wf_start_time"]
    force_first_time = pd.to_datetime(force_first_time).tz_localize("Pacific/Auckland")

    forcedata = data['Tensile'].as_dataframe()
    force_times = np.array(forcedata['Timestamps'], dtype = 'float')
    forces = np.array(forcedata['Force'], dtype = 'float')
    positions = np.array(forcedata['Position'], dtype = 'float')


    return (signal, wvl, first_time, times), (force_first_time, force_times, forces)

@multimethod
def recalibrate_times(first_timeA:pd.Timestamp, timesA:np.ndarray,
                       first_timeB:pd.Timestamp, timesB:np.ndarray):
    """Given two relative time arrays with two corresponding start times, the start times are matched with the relative timestamps updated, i.e. the returned two absolute timestamps are both equal to min(first_timeA, first_timeB)
    
    Parameters:
        first_timeA (pd.Timestamp): Start time A
        first_timeB (pd.Timestamp): Start time B
        timesA (numpy.ndarray): array of times beginning at first_timeA
        timesB (numpy.ndarray): array of times beginning at first_timeB

    Returns:
        first_timeA (pd.Timestamp): New start time A
        newtimesA (numpy.ndarray): updated relative times array
        first_timeB (pd.Timestamp): New start time B
        newtimesB (numpy.ndarray): updated relative times array
        
    """
    newtimesA = timesA.copy()
    newtimesB = timesB.copy()
    endA = RelTime_to_AbsTime(first_timeA, timesA)[-1]
    endB = RelTime_to_AbsTime(first_timeB, timesB)[-1]

    minstart = min(first_timeA, first_timeB)
    if first_timeA == first_timeB:
        return first_timeA, newtimesA, first_timeB, newtimesB
    
    if minstart == first_timeA:
        # tdelta = first_timeB-endA
        tdelta = first_timeB-first_timeA

        newtimesB += tdelta.total_seconds()
        first_timeB = first_timeA
    elif minstart == first_timeB:
        # tdelta = first_timeA-endB
        tdelta = first_timeA-first_timeB

        newtimesA += tdelta.total_seconds()
        first_timeA = first_timeB

    return first_timeA, newtimesA, first_timeB, newtimesB

class data():
    """A class to store data recorded at different times.

    Attributes:
        vals (numpy.ndarray): array storing data values
        times (numpy.ndarray): array storing relative timestamps
        first_time (pd.Timestamp): absolute timestamp reference
    """
    def __init__(self, vals = np.array([]),
                  times = np.array([]),
                    first_time = pd.Timestamp.now()) -> None:
        """Initialises the data class

        Attributes:
            vals (numpy.ndarray): array storing data values
            times (numpy.ndarray): array storing relative timestamps
            first_time (pd.Timestamp): absolute timestamp reference
        """
        self.vals = vals
        self.times = times
        self.first_time = first_time
        pass

    def __repr__(self):
        return "Data Cluster"

class spectral():
    """A class to store an optical spectrum for a single timestamp.

    Attributes:
        sig (data): structure to store signal information
        fname (str): filename of spectral datafile
        abspath (str): absolute path of spectral datafile
        id (str): Name of spectrum, defaults to fname without the extension
        wvl (numpy.ndarray): array of wavelengths for spectrum
        p0 (list): stores the output of the analyse method. list with estimates of the peak amplitude, location and FWHM
        self.a (float): amplitude of fitted gaussian
        self.x0 (float): loction of fitted gaussian
        self.fwhm (float): FWHM of fitted gaussian
    """

    def __init__(self):
        """Standard __init__ method of class
        """
        self.fname = ""
        self.abspath = ""
        self.id = ""
        self.wvl = np.array([])
        self.sig = data()
        return None

    @classmethod
    def from_arrays(cls, signal, wvl):
        """Creates a spectral class from an existing signal. Useful to store a spectrum at a single time snapshot from a time dependent spectral sample

        Parameters:
            signal (numpy.ndarray): 1D array storing the spectrum
            wvl (numpy.ndarray): array of wavelengths

        Returns:
            spec (spectral): spectral class object
        """
        spec = cls()
        spec.sig.vals = signal
        spec.wvl = wvl
        return spec

    def __repr__(self) -> str:
        return f"Spectrum ID: \"{self.id}\" \nFileLocation: \"{self.abspath}\""

    @classmethod
    def load_from_enlight(cls,filename, wvl_range = None):
        """Creates a spectral class from an ENLIGHT file (file obtained by clicking the save spectrum button). Can read both si155 and si255 interrogators spectrum.

        Parameters:
            filename (str): Absolute file path of the ENLIGHT file.
            wvl_range (list, optional): If a list of [lower_wvl, upper_wvl] is provided, the wavelength and spectral data is cropped into this range

        Returns:
            spec (spectral): spectral class object
        """
        spec = cls()
        spec.fname = os.path.split(filename)[1]
        spec.abspath = filename
        spec.id = rm_ext(spec.fname)
        df = pd.read_csv(spec.abspath, sep = '\t', names = ['wvl', 'db'], header = 52, dtype = float, usecols = [0,1])

        spec.wvl = np.array(df.wvl)
        spec.sig.vals = np.array(df.db)
        if wvl_range:
            # crop_data_by_wavelength only accepts 2D arrays, so need to make
            # self.db 2D first
            cropped_data, cropped_wvl = crop_data_by_wavelength(spec.sig.vals,spec.wvl, wvl_range)

            spec.wvl = cropped_wvl
            spec.sig.vals = cropped_data

        return spec

    def power(self):
        """Returns the spectrum in units of mW"""
        return db_to_mw(self.sig.vals)
    
    def analyse(self):
        """Naive peak finding method. Finds the tallest (usually main) peak and extracts location and FWHM using scipy.signal.peak_widths()"""
        a, x0, fwhm = get_peak(self.wvl, self.power())
        self.p0 = [a,x0,fwhm]
        return self

    def fit_gaussian(self):
        """Fits a gaussian to the spectrum, using the naive peak information as initial guesses."""
        self.analyse()
        a, x0, fwhm = fit_gauss(self.p0,self.wvl, self.power())
        self.a = a
        self.x0 = x0
        self.fwhm = fwhm
        return self

    def setup_figure(self, init_fig = False):
        """Sets up the figure for the plot_db() and plot_mw() methods. If the figure already exists, retrieves the figure.

        Parameters:
            init_fig (bool, Optional): if True, a new figure is created. Otherwise, the existing figure is retrieved.

        Returns:
            fig: (plt.Figure): figure object
            axs: (plt.Axes): axes object
        """
        if init_fig:
            scale = .3
            plot_rat = np.array([25,9])
            fig, axs = plt.subplots(figsize=scale*plot_rat)

        fig, axs = plt.gcf(), plt.gca()
        return fig, axs

    def plot_db(self, init_fig = False):
        """Plots the stored spectrum in units of dBm

        Parameters:
            init_fig (bool, optional): if True, initialises the figure. Otherwise, plots on an existing figure.
        """
        fig,axs = self.setup_figure(init_fig)
        axs.plot(self.wvl, self.sig.vals, label = self.id)        
        axs.set_xlim([1549,1551])
        axs.set_xlabel('Wavelength (nm)')
        axs.set_ylabel('Power (dbm)')
        axs.ticklabel_format(useOffset = False)
        axs.legend()
        return

    def plot_mw(self, init_fig = False):
        """Plots the stored spectrum in units of mW

        Parameters:
            init_fig (bool, optional): if True, initialises the figure. Otherwise, plots on an existing figure.
        """
        fig, axs = self.setup_figure(init_fig)
        axs.plot(self.wvl, self.power(), label = self.id)        
        axs.set_xlim(*[1549,1551])
        axs.set_xlabel('Wavelength (nm)')
        axs.set_ylabel('Power (mW)')
        axs.ticklabel_format(useOffset = False)
        axs.legend()
        return

    def plot_powers(self, init_fig = False):
        """Plots the stored spectrum in both dBm and mW on side-by-side subplots

        Parameters:
            init_fig (bool, optional): if True, initialises the figure. Otherwise, plots on an existing figure.
        """
        if init_fig:
            scale = .3
            plot_rat = np.array([45,12])
            fig, axs = plt.subplots(1,2,figsize=scale*plot_rat)
        else:
            fig,axs = plt.gcf(), plt.gcf().get_axes()

        axs[0].plot(self.wvl, self.sig.vals, label = self.id)  
        axs[1].plot(self.wvl, self.power(), label = self.id)
        axs[0].set(xlabel = 'Wavelength (nm)', ylabel = 'Power (dBm)')
        axs[1].set(xlabel = 'Wavelength (nm)', ylabel = 'Power (mW)')

        for ax in axs:
            ax.set_xlim([1549,1551])
            ax.ticklabel_format(useOffset = False)
            ax.legend()
        # fig.tight_layout()
        return
    
    def si255_db(self):
        """Returns the spectrum in units of dBm evaluated at wavelength intervals of 8pm, the default of the si255(). Wavelength range is [1545,1560].
        """ 
        new_wvl = np.arange(1545,1560, 0.008)
        signal = np.interp(new_wvl, self.wvl, self.sig.vals)
        return signal
    
    def si255_mw(self):
        """Returns the spectrum in units of mW evaluated at wavelength intervals of 8pm, the default of the si255(). Wavelength range is [1545,1560].
        """ 
        return db_to_mw(self.si255_db())
    
    def si255_wvl(self):
        """Returns the wavelength array of the si255() interrogator in 8pm increments. Wavelength range is [1545,1560]
        """ 
        return np.arange(1545,1560, 0.008)

from abc import ABC
class abs_spectral_array(ABC, np.ndarray):
    def __new__(cls, run_array, array_name = ""):
        # run_array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(run_array).view(cls)
        # add the new attribute to the created instance
        obj.array_name = array_name
        # Finally, we must return the newly created object:
        return obj
    
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.array_name = getattr(obj, 'array_name', "No Name")

@multimethod
def recalibrate_times(data1: data, data2: data):
    """ Recalibrate two data classes to have the same start times

    Parameters:
        data1 (data): first data class
        data2 (data): second data class

    Returns
        data1 (data): updated with same first_time as data2
        data2 (data): updated with same first_time as data1
    """
    ftA, tA, ftB, tB = recalibrate_times(data1.first_time, data1.times, data2.first_time, data2.times)
    data1.first_time = ftA
    data1.times = tA
    data2.first_time = ftB
    data2.times = tB
    return data1, data2

class spectral_summation(np.ndarray):
    """A class to store a multiple stationary spectra. Mainly used for the __call__ method, which returns a function that simulates the spectral response of a FBG array with constituents having the spectral responses stored.

    NOTE: implementation is overcomplicated as the class is a subclass of numpy.ndarray. It is far simpler to just subclass the spectral() class and overwrite all the methods for a list of stationary spectra.

    Attributes:
        sig (data): structure to store signal information
        fname (str): filename of spectral datafile
        abspath (str): absolute path of spectral datafile
        id (str): Name of spectrum, defaults to fname without the extension
        wvl (numpy.ndarray): array of wavelengths for spectrum
        p0 (list): stores the output of the analyse method. list with estimates of the peak amplitude, location and FWHM
        self.a (float): amplitude of fitted gaussian
        self.x0 (float): loction of fitted gaussian
        self.fwhm (float): FWHM of fitted gaussian
    """
    def __new__(cls, run_array, array_name = ""):
        # run_array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(run_array).view(cls)
        # add the new attribute to the created instance
        obj.array_name = array_name
        # Finally, we must return the newly created object:
        return obj
    
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.array_name = getattr(obj, 'array_name', "No Name")

    @classmethod
    def load_from_files(cls, rootdir, fnames, wvl_range = None):
        """Reads specified .txt files from ENLIGHT stationary spectrum snapshots in a directory. 

        Parameters:
            rootdir (str): directory path
            fnames (numpy.ndarray): array of filenames to read
            wvl_range (numpy.ndarray, optional): if provided, the wavelength range of the spectra is cropped.  

        Returns:
            arr (spectral_summation): spectral summation class object
        """
        arr = []
        for fname in fnames:
            abs_path = os.path.join(rootdir, fname)
            spec = spectral.load_from_enlight(abs_path, wvl_range = wvl_range)
            arr.append(spec)
        
        return cls(arr)
    
    @classmethod
    def load_from_enlight_folder(cls,foldername, wvl_range = None):
        """Reads ALL .txt files from ENLIGHT stationary spectrum snapshots in a directory. 

        Parameters:
            foldername (str): directory path
            wvl_range (numpy.ndarray, optional): if provided, the wavelength range of the spectra is cropped.  

        Returns:
            arr (spectral_summation): spectral summation class object
        """
        arr = []

        for root, dirs, files in os.walk(foldername):
            for file in files:
                abs_path = os.path.join(root,file)
                spec = spectral.load_from_enlight(abs_path, wvl_range = wvl_range)
                arr.append(spec)

        return cls(arr,array_name = os.path.split(foldername)[1])

    def signal(self):
        """Returns a 2D array with rows being each of the stored stationary spectra.
        """
        sigs = np.array([x.sig.vals for x in self])
        return np.concatenate(sigs)
    
    def power(self):
        """Returns 2D array of spectra stored in units of mW.
        """
        return db_to_mw(self.signal())
    
    def plot_powers(self, init_fig = True):
        """Plots the stored spectrum in units of dBm and mW side by side as subplots

        Parameters:
            init_fig (bool, optional): if True, initialises the figure. Otherwise, plots on an existing figure.
        """
        self[0].plot_powers(init_fig)

        for i in range(1,len(self)):
            self[i].plot_powers()

        return
    
    def plot_mw(self, init_fig = True):
        """Plots the stored spectrum in units of mW.

        Parameters:
            init_fig (bool, optional): if True, initialises the figure. Otherwise, plots on an existing figure.
        """
        self[0].plot_mw(init_fig)
        for i in range(1,len(self)):
            self[i].plot_mw()

        return
    
    def wvl(self):
        """Returns the wavelength array of the stored spectra
        NOTE: Assumes all wavelengths of the spectra are equal
        """
        return self[0].wvl
    
    def R(self):
        """Returns an array storing the amplitudes of the stored spectra. Used as a surrogate for the  reflectivity of each FBG. 
        """
        return np.array([run.a for run in self])
    
    def S(self):
        """Returns the wavelength array of the stored spectra
        NOTE: Assumes all wavelengths of the spectra are equal
        """
        return np.array([run.power() for run in self])
    
    def __call__(self, func = combine_spectra):
        """Returns a function that simulates a FBG array made out of the stored spectral responses. Currently supports summations of two or three spectra. 

        Parameters:
            func (function, optional): default combination function is combine_spectra(). Another option is analytic_triplesum. This is the function used to calculate the spectral summation

        Returns:
            local_func (function): combination function. Takes as inputs an array of wavelengths, the output power of the interrogator (in float or array format), and the losses/shifts of each FBG in the array.
        
        """
        
        if len(self) == 2:
            local_func = lambda x, F, l1,l2, s1,s2: func(x,F,self.R(), self.S(), losses = [l1,l2], shifts = [s1,s2])
        elif len(self) == 3:
            local_func = lambda x, F, l1,l2,l3, s1,s2,s3: func(x,F,self.R(), self.S(), losses = [l1,l2,l3], shifts = [s1,s2,s3])
        elif len(self) == 4:
            local_func = lambda x, F, l1,l2,l3,l4, s1,s2,s3,s4: func(x,F,self.R(), self.S(), losses = [l1,l2,l3,l4], shifts = [s1,s2,s3,s4])
        elif len(self) == 5:
            local_func = lambda x, F, l1,l2,l3,l4,l5, s1,s2,s3,s4,s5: func(x,F,self.R(), self.S(), losses = [l1,l2,l3,l4,l5], shifts = [s1,s2,s3,s4,s5])
        else:
            raise ValueError("Not Supported")
        return local_func
    
    def fit_gaussian(self):
        """Fits a gaussian to each of the stored spectra. Uses the spectral.fit_gaussian() method.
        """
        for run in self:
            run.fit_gaussian()
        return self

class spectral_td():
    """A class to store a single time dependent spectrum. 

    Attributes:
        fname (str): filename of spectral datafile
        abspath (str): absolute path of spectral datafile
        id (str): Name of spectrum, defaults to fname without the extension
        wvl (numpy.ndarray): array of wavelengths for spectrum

        sig (data): structure to store signal information
        force (data): structure to store force information (if needed)
        force_bool (bool): True if force data has been added.

        source_power (numpy.ndarray) : 1D array of fitted source powers
        losses (numpy.ndarray) : 2D array of fitted coupling losses
        bragg_wvls (numpy.ndarray) : 2D array of fitted Bragg wavelength values

    """
    def __init__(self):
        """Standard __init__ method
        """
        self.fname = ""
        self.abspath = ""
        self.id = ""

        currtime = pd.Timestamp.now("Pacific/Auckland")

        self.wvl = np.array([])
        self.sig = data()

        self.force_bool = False
        self.force = data()
        return

    def __repr__(self):
        return f"Time-Dep Spec \"{self.id}\" at {self.abspath}"

    @classmethod
    def load_from_responses(cls, filename, wvl_range=None):
        """Load a time dependent spectrum from a Responses.txt file, obtained using the save time-dependent spectrum feature on ENLIGHT.

        Parameters:
            filename (str): absolute path of responses.txt file
            wvl_range (numpy.ndarray,optional): optional range to crop the wavelength and spectral arrays.

        Returns:
            spec (spectral_td): time dependent spectral_td object 
        
        """
        spec = cls()
        leftover, spec.fname = os.path.split(filename)
        spec.abspath = filename
        spec.id = rm_ext(spec.fname)

        signal, wvl, (first_time, times) = load_spectrum_timedep(spec.fname, wvl_range = wvl_range)

        spec.wvl = wvl
        spec.sig.vals = signal
        spec.sig.first_time = first_time
        spec.sig.times = times
        spec.drop_duplicate_signal()
        return spec
    
    @classmethod
    def load_from_array(cls, signal, wvl, times, first_time = pd.Timestamp.now("Pacific/Auckland"), force_first_time = pd.Timestamp.now("Pacific/Auckland"), forces = np.array([]), force_times = np.array([]), id = "None"):
        """Directly initialise spectral_td object from previously stored arrays. Can be used to concatenate multiple time dependent spectra together.

        Parameters:
            signal (numpy.ndarray): 2D array with each row being an optical spectrum at different times
            wvl (numpy.ndarray): 1D array of corresponding wavelengths
            times (numpy.ndarray): 1D array relative timestamps for each row of signal
            first_time (pd.Timestamp, optional): absolute timestamp reference for the signal  times array
            force_first_time (pd.Timestamp, optional): absolute timestamp reference for force times array
            forces (numpy.ndarray, optional): 1D array of forces (recorded by Tinius Olsen tensile tester)
            force_times (numpy.ndarray, optional): 1D array of relative timestamps for forces
            id (str, optional): name for new spectral_td object

        Returns:
            spec (spectral_td): time dependent spectral_td object 
        
        """
        spec = cls()
        spec.sig.vals = signal
        spec.wvl = wvl
        spec.sig.times = times
        spec.sig.first_time = first_time
        spec.id = id
        spec.drop_duplicate_signal()
        if len(forces>0):
            spec.force.vals = forces
            spec.force.first_time = force_first_time
            spec.force.times = force_times
            spec.drop_duplicate_signal()
            spec.force_bool = True
            spec.match_firsttimes()
        return spec

    @classmethod
    def load_from_tensilelabview(cls,filename, wvl_range = None):
        """Initialise spectral_td object using an output file from a 'TandFBG_Tensile _____.vi'. Only supports reading of a single channel of spectral data (Channel 1).

        Parameters:
            filename (str): absolute path of datafile
            wvl_range (numpy.ndarray, optional): optional wavelength and signal cropping

        Returns:
            spec (spectral_td): time dependent spectral_td object 
        """
        spec = spectral_td()
        spec.fname = os.path.split(filename)[1]
        spec.abspath = filename
        spec.id = rm_ext(spec.fname)
        
        (signal, wvl, first_time, times), (force_first_time, force_times, forces) = read_force_spectrum(filename, wvl_range = wvl_range)

        spec.sig.vals = mw_to_db(signal)
        spec.wvl = wvl
        spec.sig.first_time = first_time
        spec.sig.times = times

        spec.force.first_time = force_first_time
        spec.force.times = force_times
        spec.force.vals = forces
        spec.force_bool = True

        # spec.match_firsttimes()
        spec.drop_duplicate_signal()
        spec.match_firsttimes()
        print(spec.fname, " Loaded")

        return spec

    def numpoints(self):
        """Returns the number of timestamps of data stored"""
        return len(self.sig.times)
    
    def __call__(self, i):
        """Select a single spectrul from the full time dependent spectrum and return as a spectral class object

        Parameters:
            i (int): index of spectrum to retrieve. A value of i extracts the i-th row of the signal 2D array with the corresponding i-th timestamp. Negative indices are supported.

        Returns:
            spec (spectral): spectral class object
        """
        if abs(i) >= self.numpoints():
            return IndexError(f"Index {i} is out of range of spectrum {self.id} with {self.numpoints()} timestamps")
        ti = self.sig.times[i]
        spec = spectral.from_arrays(self.sig.vals[i], self.wvl)

        spec.fname = spec.fname
        spec.abspath = self.abspath
        spec.id = self.id + f"_{i}_{ti}s"
        spec.sig.first_time = ti
        return spec

    def plot_colourmap(self):
        """Plots colourmap using stored spectral data"""
        plot_data_colormap(self.sig.vals, self.sig.times, self.wvl, plot_name = self.id)

    def smooth_force(self,smoothness):
        """Applies a 1D gaussian filter to the stored force data. Useful to use since Tinius Olsen tensile tester has too coarse of a force resolution for data analysis needs
        
        Parameters:
            smoothness (float): standard deviation for gaussian kernal, i.e. degree of smoothness
            
        Returns:
            smoothed spectrum
            
        """
        return gaussian_filter1d(self.force.vals, smoothness)
    
    def power(self):
        """Converts a stored signal in dBm to mW"""
        return db_to_mw(self.sig.vals)
    
    def gaussian_peaks(self):
        main_peak, secondary_peak = calculate_gaussian(self.sig.vals, self.wvl,id = "")

        self.main_peak = data(vals = main_peak, times = self.sig.times, first_time = self.sig.first_time)

        self.sec_peak = data(vals = secondary_peak, times = self.sig.times, first_time = self.sig.first_time)

        df = pd.DataFrame({'db': mw_to_db(secondary_peak[:,0]), 'wvl': secondary_peak[:,1], 'fwhm':secondary_peak[:,2]})
        self.peak_df = df
        return 

    def shifted_peaks_bg(self):
        sig_minus_bg = db_to_mw(self.sig.vals) - db_to_mw(self.sig.vals[0,:])
        df = shifted_peaks_bg(sig_minus_bg, self.wvl, self.sig.times, self.sig.first_time)
        newdf = df[['bwid', 'wvl', 'db']]
        newdf.columns = ['fwhm', 'wvl', 'db']
        self.peak_df = newdf
        return df
    
    def old_peak_v_force(self, smoothness = 1):
        fig, axs = plt.subplots(1,3)
        sec_axs = [ax.twinx() for ax in axs]

        smoother_force = self.smooth_force(smoothness)
        axs[0].plot(self.force.times, self.force.vals, label = "Raw Force")
        axs[0].plot(self.force.times, smoother_force, label = "Smooth")
        sec_axs[0].scatter(self.sig.times, self.peak_df.wvl, label = "Wavelength", s = 5, color = 'tab:green')

        axs[0].set(xlabel = "Time (s)", ylabel = "Force (N)")
        sec_axs[0].set(ylabel = "Shifted Peak Wavelength [nm]")

        axs[1].scatter(self.peak_df.wvl, self.peak_df.fwhm, s = 5, label = 'FWHM')
        sec_axs[1].scatter(self.peak_df.wvl, db_to_mw(self.peak_df.fwhm), color = 'tab:orange', s = 5, label = "Power")
    
        axs[1].set(xlabel = "Shifted Peak Wavelength [nm]", ylabel = "FWHM")
        sec_axs[1].set(ylabel = "Shifted Peak Power [mW]")

        interp_sm_force = np.interp(self.sig.times, self.force.times, smoother_force)
        axs[2].scatter(interp_sm_force, self.peak_df.wvl, s = 5, label = "Force vs Wvl", color = 'tab:red')
        axs[2].set(xlabel = 'Force [N]', ylabel = 'Shifted Peak Wavelength [nm]')

        [ax.legend(loc = 'upper left') for ax in axs]
        [ax.legend(loc = 'lower right') for ax in sec_axs]

        fig.set_size_inches(12, 4)
        fig.tight_layout()
        return

    def plot_force(self, smoothness = 1):
        """Plot the stored force data as a function of time. Overlays a smoothed version of the force with a desired smoothness

        Parameters:
            smoothness (float, optional): standard deviation of gaussian filter kernal.
        """
        fig, ax1 = plt.subplots()
        ax1.plot(self.force.times, self.force.vals, label = "Raw Force")

        smoother_force = self.smooth_force(smoothness)
        
        plt.plot(self.force.times, smoother_force, label = "Smooth")

        ax2 = ax1.twinx()

        ax1.set(xlabel = "Time (s)", ylabel = "Force (N)")
        ax2.set(ylabel = "Wavelength [nm]")
        ax1.legend()
        # ax2.legend()
        return 
        
    def match_firsttimes(self):
        """Alters relative timestamps of both signal and force data to ensure they are both with reference to the same starting time. Changes are made to the data class object with the later first_time attribute.
        """
        # minstart = min(self.first_time, self.force_first_time)

        # if minstart == self.first_time:
        #     tdelta = self.force_first_time-self.first_time

        #     self.force_times += tdelta.total_seconds()
        #     self.force_first_time = self.first_time

        # elif minstart == self.force_first_time:
        #     tdelta = self.first_time-self.force_first_time

        #     self.times += tdelta.total_seconds()
        #     self.first_time = self.force_first_time
        self.sig, self.force = recalibrate_times(self.sig, self.force)
        return

    def drop_duplicate_signal(self):
        """Deletes any duplicate spectra in the saved data. If the acquisition time between spectral samples on Labview is set to below ~0.15s, duplicate signal information is recorded which should be deleted.
        """
        df = pd.DataFrame(self.sig.vals, columns = self.wvl)
        df['times'] = self.sig.times
        df.drop_duplicates(subset = df.columns[:-1], inplace = True)
        self.sig.times = df['times'].to_numpy(dtype = float)
        self.sig.vals = df[self.wvl].to_numpy(dtype=float)
        return self

    def signal_animate(self, indA = 0, indB = -1, spacing = 1, wvlA = 1549.5, wvlB = 1553):
        """Animates a time dependent signal. Animated figure is displayed under jupyter code cells.
        
        Parameters:
            indA (int, optional): starting row index for animation
            indB (int, optional): ending row index for animation
            spacing (int, optional): spacing between plotted timestamps, i.e. spacing = n plots every n-th row.
            wvlA (float, optional): lower wavelength bound to animate
            wvlB (float, optional): upper wavelength ound to animate

        Returns:
            ani (plt.animation.FuncAnimation): animation
        """
        plt.rcParams['animation.embed_limit'] = 2**128
    
        if indB <0:
            indB = len(self.sig.times)
        plt.rcParams["animation.html"] = "jshtml"
        scale = .3
        plot_rat = np.array([25,9])
        fig,ax = plt.subplots(figsize = scale*plot_rat)
        plot_signal, plot_wvl = crop_data_by_wavelength(self.sig.vals, self.wvl, wvl_range = [wvlA, wvlB])
    
        line = ax.plot(plot_wvl, plot_signal[indA], label = f"t = {self.sig.times[indA]} s")[0]
        ax.set(xlabel='Wavelength [nm]', ylabel='Intensity [dBm]')
        ax.legend()
    
        frames = np.arange(indA, indB, spacing)
        # frames = self.numpoints()
        # print(frames)
        def update(frame):
            # for each frame, update the data
            line.set_ydata(plot_signal[frame])
            line.set(label = f"t = {self.sig.times[frame]} s")
            ax.legend()
            # print(frame)
            return line

        ani = animation.FuncAnimation(fig = fig, func = update, frames = frames, interval = 30)
        plt.close()
        # HTML(ani.to_jshtml())
        return ani

    def plot_forcevswvl(self, smoothness = 1):
        """Plots smoothed force against calculated Bragg wavelength of signal.

        Parameters:
            smoothness (float, optional): standard deviation of 1D gaussian filter kernel

        Returns:
            fig (plt.Figure): figure object
            ax (plt.Axes): axes object
        """
        fig,ax = plt.subplots()
        interpedforces = np.interp(self.sec_peak.times, self.force.times, self.smooth_force(smoothness))
        ax.scatter(interpedforces, self.sec_peak.vals[:,1], label = "wvl")
        ax.set(xlabel = "Force (N)", ylabel = "wavelength")
        ax.legend()
        return fig, ax
    
    def fitting_combined_spectra(self, run_array, fitfunc = combine_spectra):
        """Using a spectral_summation class, fits a summation of individual spectra to each timestamp of a time-dependent signal. This method refits output and losses at each timestamp.

        Parameters:
            run_array (spectral_summation): object storing the stationary spectra to sum up
            fitfunc (function, optional): fitting function type. Either combine_spectra or analytic_triplesum.

        Returns:
            source_power (numpy.ndarray): array of fitted source powers for each timestamp
            losses (numpy.ndarray): 2D array storing the coupling losses for each timestamp
            bragg_wvls (numpy.ndarray): 1D array storing the fitted bragg wavelength values for each timestamp

        """
        wvl = run_array.wvl()
        x0s = np.array([run.x0 for run in run_array])
        nspecs = len(run_array)
        self.constituent_ids = [run.id for run in run_array]
        self.num_constituents = nspecs
        
        func_to_fit = run_array(func = fitfunc)

        popt = [.2] + nspecs*[0] + nspecs*[0.]
        lower_bound = [0] + nspecs*[0] + nspecs*[-5]
        upper_bound = [20] + nspecs*[2] + nspecs*[10]
        # bounds = ([0,0,0,0,-5,-5,-5], [20,10,10,10,10,10,10])
        bounds = (lower_bound, upper_bound)
        master_popt = []

        for i in tqdm(range(self.numpoints())):
            popt, pcov = curve_fit(func_to_fit, wvl,db_to_mw(self.sig.vals[i]), p0 = popt, bounds = bounds)
            master_popt.append(popt)

        master_popt = np.array(master_popt)
        self.source_power = master_popt[:,0]
        self.losses = master_popt[:,1:(1+nspecs)]
        self.bragg_wvls = master_popt[:,(1+nspecs):] + x0s
        return self.source_power, self.losses, self.bragg_wvls
    
    def fitting_combined_spectra_SHIFTSONLY(self, run_array, fitfunc = combine_spectra):
        """Using a spectral_summation class, fits a summation of individual spectra to each timestamp of a time-dependent signal. This method obtains the output power and losses from the first_timestamp and fixes these for all the remaining curve fits.

        Parameters:
            run_array (spectral_summation): object storing the stationary spectra to sum up
            fitfunc (function, optional): fitting function type. Either combine_spectra or analytic_triplesum.

        Returns:
            source_power (numpy.ndarray): array of fitted source powers for each timestamp
            losses (numpy.ndarray): 2D array storing the coupling losses for each timestamp
            bragg_wvls (numpy.ndarray): 1D array storing the fitted bragg wavelength values for each timestamp

        """
        wvl = run_array.wvl()
        x0s = np.array([run.x0 for run in run_array])
        nspecs = len(run_array)
        self.constituent_ids = [run.id for run in run_array]
        self.num_constituents = nspecs
        
        func_to_fit = run_array(func = fitfunc)

        popt = [.2] + nspecs*[0] + nspecs*[0.]
        lower_bound = [0] + nspecs*[0] + nspecs*[-5]
        upper_bound = [20] + nspecs*[10] + nspecs*[10]
        # bounds = ([0,0,0,0,-5,-5,-5], [20,10,10,10,10,10,10])
        bounds = (lower_bound, upper_bound)
        master_popt = []


        popt, pcov = curve_fit(func_to_fit, wvl,db_to_mw(self.sig.vals[0]), p0 = popt, bounds = bounds)
        master_popt.append(popt)

        bounds_new = (bounds[0][(1+nspecs):], bounds[1][(1+nspecs):])
        p0_new = popt[(1+nspecs):]

        saved_popt = popt.copy()
        stored_losses = popt[1:(1+nspecs)].copy()

        if nspecs == 3:
            updated_func = lambda x, s1,s2,s3: func_to_fit(x, popt[0], *stored_losses, *[s1, s2, s3])
        elif nspecs == 5:
            updated_func = lambda x, s1,s2,s3,s4,s5: func_to_fit(x, popt[0], *stored_losses, *[s1, s2, s3, s4, s5])
        popt = p0_new

        for i in tqdm(range(1,self.numpoints())):
            popt, pcov = curve_fit(updated_func, wvl,db_to_mw(self.sig.vals[i]), p0 = popt, bounds = bounds_new)
            master_popt.append(np.concatenate([saved_popt[:(1+nspecs)],popt]))


        master_popt = np.array(master_popt)
        self.source_power = master_popt[:,0]
        self.losses = master_popt[:,1:(1+nspecs)]
        self.bragg_wvls = master_popt[:,(1+nspecs):] + x0s
        return self.source_power, self.losses, self.bragg_wvls

    def plot_fitting_values(self):
        """Plot the fitted output power, losses, and Bragg wavelengths as a function of time.
        """
        fig,axs = plt.subplots(1,2)
        fig.set_size_inches(10, 5)


        axs[0].set(xlabel = "Time (s)", ylabel = "Coupling Loss (db)")
        ax_power = axs[0].twinx()
        ax_power.set(ylabel = "Output Power (mW)")
        ax_power.plot(self.sig.times, self.source_power, 'r',label = 'output power')
        ax_power.legend()
        ax_power.tick_params(axis='y', colors='red')
        axs[1].set(xlabel = "Time (s)", ylabel = "Bragg Wvl (db)")

        axs[0].plot(self.sig.times, self.losses, label = self.constituent_ids)
        axs[1].plot(self.sig.times, self.bragg_wvls, label = self.constituent_ids)

        axs[0].legend()
        axs[1].legend()
        fig.tight_layout()
        axs[1].ticklabel_format(useOffset = False)
        return

class spectral_array(np.ndarray):
    """A class to store multiple time-dependent spectra.

    NOTE: implementation is overcomplicated as the class is a subclass of numpy.ndarray. It is far simpler to just subclass the spectral_td() class and overwrite all the methods for a list of time dependent.

    Attributes:
        array_name (str, optional): nickname for stored spectral data

        sig (data): structure to store signal information
        force (data): structure to store force information (if needed)
        force_bool (bool): True if force data has been added.

        source_power (numpy.ndarray) : 1D array of fitted source powers
        losses (numpy.ndarray) : 2D array of fitted coupling losses
        bragg_wvls (numpy.ndarray) : 2D array of fitted Bragg wavelength values

    """
    def __new__(cls, run_array, array_name = ""):
        # run_array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(run_array).view(cls)
        # add the new attribute to the created instance
        obj.array_name = array_name
        obj.force_bool = False
        # Finally, we must return the newly created object:
        return obj
    
    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.array_name = getattr(obj, 'array_name', "No Name")
        self.force_bool = getattr(obj, "force_bool", False)

    @classmethod
    def load_from_folder(cls, foldername, wvl_range = None):
        """Used to load a whole folder filled with output files from a 'TandFBG_Tensile _____.vi' type file. In most cases, this will be from a single run, with each file corresponding to a different tensile cycle.

        Parameters:
            foldername (str): absolute path of directory containing the tensile/spectral data .tdms files.
            wvl_range (numpy.ndarray, optional): wavelength cropping range

        Returns:
            newarr (spectral_array): array-like object with entries being spectral_td instances.
        """
        speclist = []

        tdms_filelist = get_common_fileext(foldername, ext = '.tdms')

        for absfilename in tdms_filelist:
            nextspec = spectral_td.load_from_tensilelabview(absfilename, wvl_range=wvl_range)
            speclist.append(nextspec)
        
        newarr = cls(speclist, array_name = os.path.split(foldername)[1])
        newarr.force_bool = True
        return newarr
    
    @classmethod
    def load_from_folder_txt(cls, foldername, wvl_range = None):
        """Used to load a whole folder filled with Responses.txt type files from ENLIGHT. These are obtained using the save time-dependent spectrum feature on ENLIGHT.

        Parameters:
            foldername (str): absolute path of directory containing the spectral data .txt files.
            wvl_range (numpy.ndarray, optional): wavelength cropping range

        Returns:
            newarr (spectral_array): array-like object with entries being spectral_td instances.
        """
        speclist = []

        txt_filelist = get_common_fileext(foldername, ext = '.txt')

        for absfilename in txt_filelist:
            nextspec = spectral_td.load_from_responses(absfilename, wvl_range=wvl_range)
            speclist.append(nextspec)
        
        newarr = cls(speclist, array_name = os.path.split(foldername)[1])
        return newarr

    def dimension_check(self):
        """Check whether the wavelength arrays of each spectral_td element are the same"""
        holder = [x.wvl for x in self]
        if not (holder == holder[0]).all():
            raise ValueError("Inconsistent Dimensions of spectra")

    def signal(self):
        """Concatenates the 2D signal arrays from each spectral_td element into a new 2D array"""
        self.dimension_check()
        holder = [x.sig.vals for x in self]
        return np.concatenate(holder)

    def forces(self):
        """Concatenates the 1D force arrays from each spectral_td element"""
        if self.force_bool:
            return np.concatenate([x.force.vals for x in self])
        else:
            raise ValueError("No Force Data Found")

    def times(self):
        """Using the first_time and relative timestamp arrays of each element, returns a single array of relative times, each with reference to the first_time of the first spectral_td item.
        """
        holder = []
        mspec = self[0]
        master_first_time = mspec.sig.first_time
        
        timebuffer = 0
        
        for spec in self:
            ftA, tA, ftB, tB = recalibrate_times(mspec.sig.first_time, mspec.sig.times, spec.sig.first_time, spec.sig.times)

            holder.append(tB+timebuffer)
            timebuffer += spec.sig.times[-1]

        return np.concatenate(holder)

    def force_times(self):
        """Using the force first_time and relative force timestamp arrays of each element, returns a single array of relative force times, each with reference to the force first_time of the first spectral_td item.
        """
        if self.force_bool:
            holder = []
            mspec = self[0]
            timebuffer = 0
            
            for spec in self:
                ftA, tA, ftB, tB = recalibrate_times(mspec.force.first_time, mspec.force.times, spec.force.first_time, spec.force.times)

                holder.append(tB+timebuffer)
                timebuffer += spec.sig.times[-1]

            return np.concatenate(holder)
        else:
            raise ValueError("No Force Data Found")

    def first_time(self):
        """Returns the signal first_time of the first spectral_td element"""
        return self[0].sig.first_time

    def force_first_time(self):
        """Returns the force first_time of the first spectral_td element"""
        if self.force_bool:
            return self[0].force.first_time
        else:
            raise ValueError("No Force Data Found")

    def wvl(self):
        """If all elements have the same wavelength arrays, return this wavelength array."""
        self.dimension_check()
        return self[0].wvl
        
    def aggregate(self, combined_id = "combined"):
        """Combines/concatenates all spectral_td objects into a single spectral_td object. The spectral information is concatenated, with the first_time and times arrays being updated. If force data is stored, the returned spectral_td object has this information also added. 

        Parameters: 
            combined_id (str, optional): name of new aggregated/concatenated signal 

        Returns:
            spec (spectral_td): single spectral_td instance storing all the concatenated data
        """
        if self.force_bool:
            spec = spectral_td.load_from_array(self.signal(), self.wvl(), self.times(), self.first_time(), force_first_time= self.force_first_time(), forces = self.forces(), force_times = self.force_times(),id = combined_id)
        else:
            spec = spectral_td.load_from_array(self.signal(), self.wvl(), self.times(), self.first_time(), id = combined_id)
        print("aggregated")
        return spec

    def plot_colourmap(self):
        """Aggregates the stored runs and plots them on a colourmap
        """
        self.aggregate().plot_colourmap()

    def plot_force(self, smoothness = 1):
        """Aggregates the stored runs and plot the resulting force data as a function of time. A smoothed force is overlayed.

        Parameters:
            smoothness (float, optional): standard deviation of 1D gaussian filter kernel. 
        """
        self.aggregate().plot_force(smoothness = smoothness)