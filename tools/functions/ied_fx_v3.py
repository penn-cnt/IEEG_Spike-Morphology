#establishing environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy import signal as sig
import mat73
from scipy.io import loadmat
import pickle as pkl
import matplotlib.patches as mpatches
from os import path

import random

def load_pt(ptname, data_directory):
    """
    loads the pickle object of a single patient
    input: ptname -> name of patient (ex. "HUP101"), data_directory -> list where data_directory[0] contains the folder of results and data_directory[1] contains RID data
    output: spike object containing values, chlabels, fs, soz information, and select (meta data of each spike)
    """
    filename = data_directory[0] + '/pickle_spike/{}_obj.pkl'.format(ptname)
    with open(filename, 'rb') as f:
        spike = pkl.load(f)    
    return spike

# line length
def LL(x):
    return np.sum(np.absolute(np.ediff1d(x)))

# energy
def E(x):
    return np.sum(x ** 2)

#RMS
def RMS(x):
    return np.sqrt(np.mean(x**2))

# area
def A(x):
    return np.sum(np.absolute(x))

# spectral amp
def spectral_amplitude(x):
    x_fft = np.fft.fft(x)
    return np.mean(x_fft)

def mean_amplitude_freq(X, fs, lF, uF):
    time_step = 1/fs
    ps = np.abs(np.fft.fft(X)) ** 2
    freqs = np.fft.fftfreq(X.size, time_step)
    mask = np.logical_and(freqs >= lF, freqs <= uF )
    avgValue = ps[mask].mean()
    return avgValue

# number of crossings (zero) - not in
def ZX(x, demean_flag = False):
    if demean_flag:
        x_demean = x - np.mean(x)
    else:
        x_demean = x
    num_crossings = 0
    for i in range(1, len(x)):
        fromAbove = False
        fromBelow = False
        if x_demean[i - 1] > 0 and x_demean[i] < 0:
            fromAbove = True
        if x_demean[i - 1] < 0 and x_demean[i] > 0:
            fromBelow = True

        if fromAbove or fromBelow:
            num_crossings += 1
    return num_crossings

def MEAN(x):
    return np.mean(x)

# def bandpower(x, fs, fmin, fmax):
#     f, Pxx = sig.periodogram(x, fs=fs)
#     ind_min = np.argmax(f > fmin) - 1
#     ind_max = np.argmax(f > fmax) - 1
#     return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

import numpy as np
from scipy.signal import welch
from scipy.integrate import simps

def bandpower_matrix(data, sf, band, window_sec=None):
    """Compute the relative bandpower of the signal x in a specific frequency band for multiple channels.

    Parameters
    ----------
    data : 2d-array
        Input signal in the time-domain. Shape should be (n_samples, n_channels).
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2

    Return
    ------
    rel_bp : 1d-array
        Relative band power for each channel.
    """
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = int(window_sec * sf)
    else:
        nperseg = int((2 / low) * sf)

    # Get the number of channels
    n_samples, n_channels = data.shape

    # Initialize array to store relative bandpower for each channel
    rel_bp = np.zeros(n_channels)

    for ch in range(n_channels):
        # Compute the modified periodogram (Welch)
        freqs, psd = welch(data[:, ch], sf, nperseg=nperseg)

        # Frequency resolution
        freq_res = freqs[1] - freqs[0]

        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using Simpson's rule
        bp = simps(psd[idx_band], dx=freq_res)
        total_power = simps(psd, dx=freq_res)

        # Calculate relative bandpower
        rel_bp[ch] = bp / total_power

    return rel_bp

def create_feat_list(values):
    feats = []
    for val in values:
        feat_per_spike = []
        val_t = val.transpose()
        for list in val_t:
            list = [0 if math.isnan(x) else x for x in list]
            maxes = np.max(np.absolute(list[750:1251])) #calculate max values around peak
            linelen = LL(list)
            area = A(list)
            feat_per_spike.append([maxes,linelen, area])
        feats.append(feat_per_spike)
    
    return feats
    
def hifreq_ch_spike(select_spikes):
    """ 
    function to find the frequency of spiking for a unique channel
    input: 1000 random spike file (randi)
    output: 2x1 list containing the unique channels[0] and the frequency in which they are spiking [1]
    """

    spiking_ch = [] #create list of spiking channels from spike.select (1000 random spikes)
    for spike in select_spikes:
        spiking_ch.append(spike[1])

    uniq_chs = np.unique(spiking_ch) #generate unique channels

    counts=[] #find the frequency
    for ch in uniq_chs:
        x = spiking_ch.count(ch)
        counts.append(x)

    total = 0 #sanity check - the frequency of ch's should add up to 1000
    for ele in range(0, len(counts)):
        total = total + counts[ele]
    if total != 1000:
        print('not working correct')

    spiking_chs = [int(x) for x in spiking_ch]

    return [uniq_chs, counts], spiking_chs

def find_spike_ch(select_spikes,values):
    #this function will find the values for the spiking channel.
    #should return a 1000 x #ofsamples matrix

    select_spikes_ch = []
    for spikes in select_spikes:
        ch = spikes[1]
        ch = int(ch)-1
        select_spikes_ch.append(ch)
    
    spike_value_all = []
    for i in range(np.size(values,0)):
        val = values[i]
        val_t = val.transpose()
        spike_value = val_t[select_spikes_ch[i]]
        spike_value_all.append(spike_value)

    return select_spikes_ch, spike_value_all #list of spiking channels, list of values for each spiking channel.

def average_waveform_hifreq(spike_select, spike_values):
    """
    This function serves to find channel which created the most spikes and plot their average waveform
    input: 1000 random spikes file and the corresponding values
    output: figure with the average waveform, and a list with the average waveform of the top 5 highest frequent channels (if theres ties, it adds more channels)
    """

    counts, chs = hifreq_ch_spike(spike_select)
    sorted_counts = np.sort(counts[1])
    x = sorted_counts[-1]
    y = sorted_counts[-5:-1]
    high_freq_count = np.append(y,x) #finds the highest spiking counts

    loc_high_counts = []
    for i in range(len(high_freq_count)):
        loc_high_counts.append((np.where(counts[1] == high_freq_count[i])[0]))
    loc_high_counts = (np.concatenate(loc_high_counts)) #find where the highest spiking counts are located.

    high_chs = counts[0][loc_high_counts] #find the channel number of the highest spiking count.

    #index of the channel with highest freq, from the 1000 spikes
    idx_of_all_chs = []
    for ch in high_chs:
        idx_of_all_chs.append(np.where(chs == ch)[0])
    #idx_of_all_chs = np.concatenate(idx_of_all_chs) 

    #find the average waveform for each spike.chlabels
    waveforms = []
    for i in range(len(high_chs)):
        spike_at_ch = []
        for spike_x in idx_of_all_chs[i]:
            spike_xs = (spike_values[spike_x])
            spike_fix = spike_xs[:,int(high_chs[i])-1]
            spike_at_ch.append(spike_fix)
        waveforms.append(spike_at_ch)

    #calculate the average
    avg_waveforms = []
    for waves in waveforms:
        avg_waveforms.append(np.mean(waves,axis=0))

    #plot average waveform
    fig, axs = plt.subplots(len(avg_waveforms), 1, figsize=(7,15))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.74)
    time = np.linspace(0,4,2001)
    for i in range(len(waveforms)):
        axs[i].plot(time, avg_waveforms[i], 'k') #plot nerve data - unfiltered
        axs[i].set_ylabel("Voltage (millivolts)")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_title("Average Waveform for Channel {}, Freq = {}/1000".format(int(high_chs[i]), len(waveforms[i])))

    return fig, avg_waveforms

def load_rid(ptname, data_directory):
    ptids = pd.read_csv(data_directory[0] + '/pt_data/all_ptids.csv')
    rid = ptids['r_id'].loc[ptids['hup_id'] == ptname].astype('string')
    rid = np.array(rid)
    if np.size(rid) == 0:
        print('not on table???')
        rid = [0,0]
        brain_df = 0
    elif path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])) == True:
        dkt_directory = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])
        brain_df = pd.read_csv(dkt_directory)
        brain_df['name'] = brain_df['name'].astype(str) + '-CAR'
    elif path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-implant01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])) == True:
        dkt_directory = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-implant01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])
        brain_df = pd.read_csv(dkt_directory)
        brain_df['name'] = brain_df['name'].astype(str) + '-CAR'
    elif path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-clinical01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])) == True:
        dkt_directory = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-clinical01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])
        brain_df = pd.read_csv(dkt_directory)
        brain_df['name'] = brain_df['name'].astype(str) + '-CAR'
    else: 
        print('no image')
        rid = [0,0]
        brain_df = 0

    return rid[0], brain_df

def unnesting(df, explode, axis):
    '''
    code that expands lists in a column in a dataframe.
    '''
    if axis==1:
        idx = df.index.repeat(df[explode[0]].str.len())
        df1 = pd.concat([
            pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
        df1.index = idx

        return df1.join(df.drop(explode, 1), how='right')
    else :
        df1 = pd.concat([
                         pd.DataFrame(df[x].tolist(), index=df.index).add_prefix(x) for x in explode], axis=1)
        return df1.join(df.drop(explode, 1), how='right')

def load_rid_forjson(ptname, data_directory):
    ptids = pd.read_csv(data_directory[0] + '/pt_data/all_ptids.csv')
    rid = ptids['r_id'].loc[ptids['hup_id'] == ptname].astype('string')
    rid = np.array(rid)

    if path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])) == True:
        dkt_directory = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])
    elif path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-implant01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])) == True:
        dkt_directory = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-implant01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])
    elif path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-clinical01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])) == True:
        dkt_directory = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-clinical01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])

    brain_df = pd.read_csv(dkt_directory)
    brain_df['name'] = brain_df['name'].astype(str) + '-CAR'
    return rid[0], brain_df

'''     
#outdated code (changed to a newer version that looks for gray matter)   
def label_fix(pt, data_directory, threshold = 0.20):
    
    #label_fix reassigns labels overlapping brain regions to "empty labels" in our DKTantspynet output from IEEG_recon
    #input:  pt - name of patient. example: 'HUP100' 
    #        data_directory - directory containing CNT_iEEG_BIGS folder. (must end in '/')
    #        threshold - arbitrary threshold that r=2mm surround of electrode must overlap with a brain region. default: threshold = 25%, Brain region has a 25% or more overlap.
    #output: relabeled_df - a dataframe that contains 2 extra columns showing the second most overlapping brain region and the percent of overlap. 
    

    rid, brain_df = load_rid_forjson(pt, data_directory)
    if path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)) == True:
        json_labels = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)
    elif path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-implant01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)) == True:
        json_labels = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-implant01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)
    elif path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-clinical01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)) == True:
        json_labels = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-clinical01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)
        
    workinglabels = pd.read_json(json_labels, lines=True)
    empty = (workinglabels[workinglabels['label'] == 'EmptyLabel'])
    empty = unnesting(empty, ['labels_sorted', 'percent_assigned'], axis=0)
    empty = empty[np.isnan(empty['percent_assigned1']) == False]
    changed = empty[empty['percent_assigned1'] >= threshold]
    
    brain_df['name'] = brain_df['name'].str.replace('-CAR','')
    relabeled_df = brain_df.merge(changed[['labels_sorted1', 'percent_assigned1']], left_on=brain_df['name'], right_on=changed['name'], how = 'left', indicator=True)   
    relabeled_df['final_label'] = relabeled_df['labels_sorted1'].fillna(relabeled_df['label'])
    relabeled_df['name'] = relabeled_df['name'].astype(str) + '-CAR' #added for this version for our analysis

    return relabeled_df
'''

def gray_labels(row):
    '''
    function looks for gray matter, if it finds it then it will look to return the label in which it is not Empty or None
    '''
    #check if a dataframe contains a column by the name of 'hi'

    if 'AT_labels1' in row:
        if (row['AT_labels0'] == 'gray matter') | (row['AT_labels1'] == 'gray matter'): #change this to 3 == stATEMENTS    MY GROUPING IS INCORRECT

            if (row['labels_sorted1'] != "EmptyLabel") | (row['labels_sorted1'] != None):
                return row['labels_sorted1']
            
            elif (row['labels_sorted2'] != "EmptyLabel") | (row['labels_sorted2'] != None):
                return row['labels_sorted2']
            
            elif (row['labels_sorted3'] != "EmptyLabel") | (row['labels_sorted3'] != None):
                return row['labels_sorted3']
            
            elif (row['labels_sorted4'] != "EmptyLabel") | (row['labels_sorted1'] != None):
                return row['labels_sorted4']

    if 'AT_labels2' in row:
        if (row['AT_labels0'] == 'gray matter') | (row['AT_labels1'] == 'gray matter') | (row['AT_labels2'] == 'gray matter'): #change this to 3 == stATEMENTS    MY GROUPING IS INCORRECT

            if (row['labels_sorted1'] != "EmptyLabel") | (row['labels_sorted1'] != None):
                return row['labels_sorted1']
            
            elif (row['labels_sorted2'] != "EmptyLabel") | (row['labels_sorted2'] != None):
                return row['labels_sorted2']
            
            elif (row['labels_sorted3'] != "EmptyLabel") | (row['labels_sorted3'] != None):
                return row['labels_sorted3']
            
            elif (row['labels_sorted4'] != "EmptyLabel") | (row['labels_sorted1'] != None):
                return row['labels_sorted4']
    

def label_fix(pt, data_directory, threshold = 0.20):
    '''
    label_fix reassigns labels overlapping brain regions to "empty labels" in our DKTantspynet output from IEEG_recon
    input:  pt - name of patient. example: 'HUP100' 
            data_directory - directory containing CNT_iEEG_BIGS folder. (must end in '/')
            threshold - arbitrary threshold that r=2mm surround of electrode must overlap with a brain region. default: threshold = 25%, Brain region has a 25% or more overlap.
    output: relabeled_df - a dataframe that contains 2 extra columns showing the second most overlapping brain region and the percent of overlap. 
    '''

    rid, brain_df = load_rid_forjson(pt, data_directory)
    if path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)) == True:
        json_labels = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)
    if path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-implant01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)) == True:
        json_labels = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-implant01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)
    if path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-clinical01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)) == True:
        json_labels = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-clinical01_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)
    
    #try pd.read_json with lines = True, if that doesn't work then do it without it.
    try: 
        workinglabels = pd.read_json(json_labels, lines=True)
    except:
        print('trying again without lines=True')
        workinglabels = pd.read_json(json_labels)

    empty = (workinglabels[workinglabels['label'] == 'EmptyLabel'])
    empty = unnesting(empty, ['labels_sorted', 'percent_assigned'], axis=0)
    empty = empty[np.isnan(empty['percent_assigned1']) == False]
    changed = empty[empty['percent_assigned1'] >= threshold]
    
    brain_df['name'] = brain_df['name'].str.replace('-CAR','')
    relabeled_df = brain_df.merge(changed[['labels_sorted1', 'percent_assigned1']], left_on=brain_df['name'], right_on=changed['name'], how = 'left', indicator=True)   
    relabeled_df['final_label'] = relabeled_df['labels_sorted1'].fillna(relabeled_df['label'])
    
    relabeled_df = relabeled_df[['name', 'x', 'y', 'z', 'label', 'labels_sorted1', 'percent_assigned1', 'final_label']]
    relabeled_df = relabeled_df.rename(columns={'final_label': 'threshold_label'})

    #load atropos
    if path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-atropos_radius-2_desc-vox_coordinates.json'.format(rid,rid)) == True:
        at_json_labels = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-atropos_radius-2_desc-vox_coordinates.json'.format(rid,rid)
    elif path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-implant01_space-T00mri_atlas-atropos_radius-2_desc-vox_coordinates.json'.format(rid,rid)) == True:
        at_json_labels = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-implant01_space-T00mri_atlas-atropos_radius-2_desc-vox_coordinates.json'.format(rid,rid)
    elif path.exists(data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-clinical01_space-T00mri_atlas-atropos_radius-2_desc-vox_coordinates.json'.format(rid,rid)) == True:
        at_json_labels = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-clinical01_space-T00mri_atlas-atropos_radius-2_desc-vox_coordinates.json'.format(rid,rid)

    #try pd.read_json with lines = True, if that doesn't work then do it without it.
    try: 
        atropos_labels = pd.read_json(at_json_labels, lines=True)
    except:
        print('trying again without lines=True')
        atropos_labels = pd.read_json(at_json_labels)

    atropos_labels = atropos_labels.rename(columns={'labels_sorted': 'AT_labels', 'percent_assigned':'AT_perc'})
    #MERGE ATROPOS AND DKT
    merge_keys = workinglabels.merge(atropos_labels[['AT_labels', 'AT_perc']], left_on = workinglabels['name'], right_on= atropos_labels['name'], how='left')
    
    empty_mergekeys = (merge_keys[merge_keys['label'] == 'EmptyLabel'])
    empty_mergekeys = unnesting(empty_mergekeys, ['labels_sorted', 'percent_assigned','AT_labels','AT_perc'], axis=0)
    empty_mergekeys['at_dkt_labels'] = empty_mergekeys.apply(gray_labels, axis=1)
    relabeled_df_2 = relabeled_df.merge(empty_mergekeys['at_dkt_labels'], left_on=relabeled_df['name'], right_on=empty_mergekeys['name'], how = 'left', indicator='merge_indicator')
    relabeled_df_2['final_label'] = relabeled_df_2['at_dkt_labels'].fillna(relabeled_df_2['threshold_label'])

    relabeled_df_2['name'] = relabeled_df_2['name'].astype(str) + '-CAR' #added for this version for our analysis

    return relabeled_df_2

def replace_letters(string, replace_dict):
    for original, replacement in replace_dict.items():
        string = string.replace(original, replacement)
    return string


def changelabel(ptname, brain_df, data_directory):
    directions = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/label_change/{}.csv'.format(ptname))
    replace_dict = dict(zip(directions['name'], directions['change_to']))
    brain_df['name'] = brain_df['name'].apply(replace_letters, replace_dict = replace_dict)
    #do the same for key_0
    brain_df['key_0'] = brain_df['key_0'].apply(replace_letters, replace_dict = replace_dict)
    return brain_df
    
def load_ptall(ptname, data_directory):
    """ load_ptall combines all the functions together, loading both the RID and the IEEG data just using the Patient NAME
        Will create a dataframe, and a spike object containing: values, fs, chlabels, 
    """

    spike = load_pt(ptname,data_directory)
    rid, brain_df = load_rid(ptname, data_directory)
    if isinstance(brain_df, pd.DataFrame) == False:
        spike = spike
        relabeled_df = 0
        ptname = ptname
        rid = rid
    else:
        relabeled_df = label_fix(ptname, data_directory, threshold = 0.20)
        if ptname == 'HUP106':
            relabeled_df = changelabel(ptname, relabeled_df, data_directory)

    soz_region = pd.read_csv("/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/soz_locations.csv")
    soz_region_pt = soz_region[soz_region['name'] == ptname]['region'].to_list()
    soz_lateralization_pt = soz_region[soz_region['name'] == ptname]['lateralization'].to_list()
    onsetzone = [soz_region_pt, soz_lateralization_pt]
    onsetzone = [x for x in onsetzone for x in x]
    if (pd.isna(onsetzone[0]) == True) & (pd.isna(onsetzone[1]) == False):
        onsetzone = ["no soz", onsetzone[1]]
    elif (pd.isna(onsetzone[0]) == True) & (pd.isna(onsetzone[0]) == True): 
        onsetzone = ["no soz", "no soz"]
    else:
        onsetsone = [onsetzone[0], "no soz"]

    return spike, relabeled_df, onsetzone, [ptname,rid]

def load_cleaned_braindf(ptname, data_directory):
    rid, brain_df = load_rid(ptname, data_directory)
    if isinstance(brain_df, pd.DataFrame) == False:
        relabeled_df = 0
        ptname = 0
        rid = 0
    else:
        relabeled_df = label_fix(ptname, data_directory, threshold = 0.25)
    return [ptname, rid], relabeled_df

def downsample_to_2001(vals):
    """
    Use only on the per sample level (samples x channels), in our stuff it works on values[0]
    """
    if len(vals) > 4000:
        downsamp_vals = vals[::2]
        biglen = len(downsamp_vals)
        diff = int((biglen - 2001)/2)
        new_sample = downsamp_vals[diff:-diff]
    elif (len(vals) > 2001) & (len(vals) < 3000):
        biglen = len(vals)
        diff = int((biglen-2001)/2)
        new_sample = vals[diff:-diff]
    else:
        print('fs<500. Recalculate')
        new_sample = "ERROR"

    return new_sample
    
def value_basis(spike, brain_df, roi):
    """
    Function that takes in all values, the DKT atlas dataframe, and a region of interest (ROI)
    returns a tailored, truncated list of the all the values given a specific ROI
    input: spike object, brain dataframe, roi list
    output: correlated values, channel number (matlab), indices of channels
    
    OLD CODE WAS LOGICALLY INCORRECT
    roi_ch = pd.DataFrame()
    for x in roi:
        roi_ch = roi_ch.append(brain_df[(brain_df['final_label'] == x)])
        #roi_ch = roi_ch.concat([roi_ch, brain_df[(brain_df['label'] == x )]])

    #roi_ch = brain_df.loc[brain_df['label']== roi] #creates truncated dataframe of ROI labels
    roi_chlist = np.array(roi_ch['name']) #converts DF to array

    #finds the index of where to find the channel
    idx_roich = []
    for ch in roi_chlist:
        x = np.where(spike.chlabels[0][-1] == ch)[0]
        idx_roich.append(x)

    idx_roich = [x[0] for x in idx_roich if np.size(x)!=0]
    chnum = [x+1 for x in idx_roich if np.size(x)!=0]

    counts,chs = hifreq_ch_spike(spike.select)

    select_oi = []
    for chroi in chnum:a
        idx = np.where(chs == chroi)[0]
        select_oi.append(idx)

    values_oi = []
    if np.size(select_oi) == 0:
        values_oi = 0
        print("NO MATCHES")
    else:
        for soi in select_oi:
            holder = []
            for x in soi:
                x  = int(x)
                y = spike.values[x]
                if len(x) > 2001: 
                    spike_down = downsample_to_2001(y)
                else:
                    spike_down = y
                holder.append(spike_down)
            
            values_oi.append(holder)

    based_values = values_oi
    """
    roi_ch = pd.DataFrame()
    for x in roi:
        roi_ch = roi_ch.append(brain_df[(brain_df['final_label'] == x)])

    roi_chlist = np.array(roi_ch['name'])

    idx_roich = []
    for i in range(len(spike.chlabels[0])):
        idx_holder = []
        for ch in roi_chlist:
            x = np.where(spike.chlabels[0][i] == ch)[0]
            idx_holder.append(x)
        idx_holder = [x for x in idx_holder for x in x]
        idx_roich.append(idx_holder)

    counts,chs = hifreq_ch_spike(spike.select)

    select_oi = []

    for i, list_of_idx in enumerate(idx_roich):
        if chs[i]-1 in list_of_idx:
            select_oi.append(i)

    values_oi = []
    if np.size(select_oi) == 0:
        values_oi = 0
        print("NO MATCHES {}".format(roi[0]))
    else:
        for soi in select_oi:
            y = spike.values[soi]
            if len(y) > 2001:
                spike_down = downsample_to_2001(y)
            else:
                spike_down = y
            values_oi.append(spike_down)

    based_values = values_oi

    return based_values, idx_roich, chs, select_oi

def value_basis_multiroi(spike, brain_df, region_of_interests):
    """
    Function that takes in all values, the DKT atlas dataframe, and a region of interest (ROI)
    returns a tailored, truncated list of the all the values given a specific ROI
    input: spike object, brain dataframe, roi list
    output: correlated values, channel number (matlab), indices of channels
    """
    all_vals = []
    all_idx_roich = []
    all_chs = []
    all_select_oi = []

    for roi in region_of_interests:
        based_values, idx_roich, chs, select_oi = value_basis(spike, brain_df, roi)
        
        all_vals.append(based_values)
        all_chs.append(chs)
        all_idx_roich.append(idx_roich)
        all_select_oi.append(select_oi)

    return all_vals, all_idx_roich, all_chs, all_select_oi

def avgroi_wave(idx_roich, based_vals):
    #per channel
    avg_waveforms2 = []
    for i in range(len(idx_roich)):
        ch = []
        for spikes2 in based_vals:
            spike_t = np.transpose(spikes2)
            ch.append((spike_t[idx_roich[i]]))
        avg_waveforms2.append(np.nanmean(ch, axis=0))
    return avg_waveforms2

def plot_avgroiwave(avg_waveform, roi,chnum,brain_df):
    roi_ch = pd.DataFrame()
    for x in roi:
        roi_ch = roi_ch.append(brain_df[(brain_df['label'] == x)])

    roi_labels = np.array(roi_ch['label']) #converts DF to array
    fig, axs = plt.subplots(len(avg_waveform), 1, figsize=(7,15))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)
    time = np.linspace(0,4,2001)
    for i in range(len(avg_waveform)):
        axs[i].plot(time, avg_waveform[i], 'k') #plot nerve data - unfiltered
        axs[i].set_ylabel("Voltage (millivolts)")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_title("Average Waveform for Channel {} in {}".format(int(chnum[i]),roi_labels[i]))
    return fig

def totalavg_roiwave(idxch, vals):
    #total per ROI
    all_waveforms = []
    for i in range(len(idxch)):
        waveform = []
        for spikes2 in vals[i]:
            if len(spikes2) > 2001:
                spike_down = downsample_to_2001(spikes2)
            else:
                spike_down = spikes2
            spike_t = np.transpose(spike_down)
            waveform.append(spike_t[idxch[i]])
        all_waveforms.append(waveform)
        
    stacked = [x for x in all_waveforms if np.size(x) != 0]
    all_chs_stacked = np.concatenate(stacked)
    avg_waveform =  np.nanmean(all_chs_stacked,axis=0)
    abs_avg_waveform = np.nanmean(np.abs(all_chs_stacked-avg_waveform), axis=0) #changed this added the subtraction of avg_waveform

    flip_wave = []
    for ch in all_chs_stacked:
        if (ch[1000] < 0):
            flip_wave.append(np.multiply(ch, -1))
        else: 
            flip_wave.append(ch)

    flip_avgwaveform = np.nanmean(all_chs_stacked, axis=0)

    avg_per_ch = []
    abs_avg_per_ch = []
    for waves in all_waveforms:
        avg_per_ch.append(np.nanmean(waves, axis=0))
        abs_avg_per_ch.append(np.nanmean(np.abs(waves-np.nanmean(waves, axis=0)), axis=0)) #added the subtraction of the mean

    return avg_waveform, abs_avg_waveform, flip_avgwaveform, all_chs_stacked, flip_wave, avg_per_ch, abs_avg_per_ch

def plot_avgroiwave(avg_waveform, title_label):
    time = np.linspace(0,4,2001)
    fig = plt.figure(figsize = (6,6))
    plt.plot(time,avg_waveform,'k')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (millivolts)')
    plt.title('Average Waveform for {}'.format(title_label))
    plt.tight_layout()
    plt.show()
    return fig


def plot_avgROIwave_multipt(ptnames, data_directory, roi, title, ymin, ymax):
    all_indiv_vals = []
    all_flip_vals = []
    avgwaves = []
    abs_avgwaves = []
    flip_avgwaves =[]
    count = 0
    for pt in ptnames:
        print(pt)
        spike, brain_df, _, ids = load_ptall(pt, data_directory)
        if isinstance(brain_df, pd.DataFrame) == False: #checks if RID exists
            count += 1
            continue
        if spike.fs[0][-1] < 500:
            print("low fs - skip")
            continue
        vals, chnum, idxch = value_basis(spike, brain_df, roi)    
        if vals == 0: #checks if there is no overlap
            count += 1
            continue
        avg_waveform, abs_avg_waveform, flip_avgwaveform, indiv_vals, flipwave, _, _ = totalavg_roiwave(idxch, vals)
        avgwaves.append(avg_waveform)
        all_indiv_vals.append(indiv_vals)
        all_flip_vals.append(flipwave) 
        abs_avgwaves.append(abs_avg_waveform)
        flip_avgwaves.append(flip_avgwaveform)

    all_chs = np.concatenate(all_indiv_vals)
    flip_allchs = np.concatenate(all_flip_vals)
    total_avg = np.nanmean(all_chs, axis=0)
    abs_total_avg = np.nanmean(np.abs(all_chs-total_avg), axis=0)
    flip_total_avg = np.nanmean(flip_allchs, axis =0)

    single = mpatches.Patch(color='grey', label='Patient Average') #imported matplotlib.patches -> manually creates legend since auto-detection of the legend via plt.legend() didn't work
    cohort = mpatches.Patch(color='r', label='Cohort Average')
    samps = len(ptnames) - count
    if samps == 0:
        print('no matches!')
    time = np.linspace(0,4,len(total_avg))
    fig = plt.figure(figsize=(10,10))
    for vals in avgwaves:
        plt.plot(time, vals, color='grey')
    plt.plot(time, total_avg,'r')
    plt.title('{} Average Spike in {} Patients'.format(title, samps))
    plt.ylim([ymin,ymax])
    plt.xlabel('time (s)')
    plt.ylabel('Voltage (mV)')
    plt.legend(handles=[single,cohort])   
    plt.show()
    
    #abs = True
    fig2 = plt.figure(figsize=(10,10))
    for vals in abs_avgwaves:
        plt.plot(time, vals, color='grey')
    plt.plot(time, abs_total_avg,'r')
    plt.title('[ABS] {} Average Spike in {} Patients'.format(title, samps))
    plt.ylim([ymin,ymax])
    plt.xlabel('time (s)')
    plt.ylabel('Voltage (mV)')
    plt.legend(handles=[single,cohort]) 
    plt.show()

    #FLIP (multiply by -1)
    fig3 = plt.figure(figsize=(10,10))
    for vals in flip_avgwaves:
        plt.plot(time, vals, color='grey')
    plt.plot(time, flip_total_avg,'r')
    plt.title('[FLIP] {} Average Spike in {} Patients'.format(title, samps))
    plt.ylim([ymin,ymax])
    plt.xlabel('time (s)')
    plt.ylabel('Voltage (mV)')
    plt.legend(handles=[single,cohort]) 
    plt.show()

    return fig, fig2, fig3, total_avg, abs_total_avg, flip_total_avg, all_chs, flip_allchs

def find_soz_region(SOZ, brain_df):
    """ returns a list with all the regions of the seizure onset zone """
    brain_df['SOZ'] = brain_df['name'].apply(lambda x: 1 if x in SOZ else 0)
    region = brain_df['final_label'][brain_df['SOZ'] == 1].to_list()
    return region

def biglist_roi(ptnames, roilist, data_directory):
    """
    generates a big list of all relative values based on region of interest.
    takes in multiple regions, thanks to value_basis_multiroi > the results are indexed in order of the roi list we use as the import. *track this*
    """
    roiLlat_values = []
    roiLlat_idxch = []
    infer_spike_soz = []
    clinic_soz = []
    count = 0
    for pt in ptnames:
        print(pt)
        spike, brain_df, soz_region, _  = load_ptall(pt, data_directory)
        if isinstance(brain_df, pd.DataFrame) == False: #checks if RID exists
            count += 1
            continue
        if spike.fs[0][-1] < 500:
            count += 1
            print("low fs - skip")
            continue
        vals, _, idxch = value_basis_multiroi(spike, brain_df, roilist)
        #here instead of storing the val, you want to grab the feature of interest. 
        roiLlat_values.append(vals)
        roiLlat_idxch.append(idxch)    
        region = find_soz_region(spike.soz, brain_df)
        infer_spike_soz.append([spike.soz, region]) #should get skipped if there isn't any 
        clinic_soz.append(soz_region)
        if vals == 0: #checks if there is no overlap
            count += 1
            continue
    return roiLlat_values, roiLlat_idxch, infer_spike_soz, clinic_soz

#THIS DOCUMENT CONTAINS THE MOST UP TO DATE VERSION OF EACH CODE:
#THE OLD VERSION OF THIS IS LOCATED IN ARCHIVE FOLDER : REGION DISTRIBUTION
#THIS CODE WAS MADE IN THE REGION_DISTRIBUTION_WORK IN THE MAIN FOLDER.

def spike_count_perregion(select_oi): #could be generalized to multiple features easily. 

    total_count_perpt = []
    count_roi = []
    count_perpt = []
    for pt in select_oi:
        count_roi = []
        for roi in pt:
            count_roi.append(np.size(roi))
        count_perpt.append(count_roi)

    total_count_perpt = np.sum(count_perpt, axis=1)
        
    return count_perpt, total_count_perpt

def spike_amplitude_perregion(values, chs, select_oi):

    perpt = []
    perpt_mean = []
    for i, pt in enumerate(values):
        #print('new pt {}'.format(i))
        perroi_mean = []
        perroi = []

        for j, roi in enumerate(pt):
            #print('roi {}'.format(j))
            spikefeat = []

            if not roi:
                perroi.append(np.nan)
                perroi_mean.append(np.nan)
                continue

            for l, xs in enumerate(roi):

                val_want = np.transpose(xs)
                val_want = val_want[chs[i][j][select_oi[i][j][l]]-1]
                feat = np.max(np.absolute(val_want[750:1251]))
                spikefeat.append(feat)

            perroi.append(spikefeat)
            perroi_mean.append(np.nanmean(spikefeat))
        perpt.append(perroi)
        perpt_mean.append(perroi_mean)  


    return perpt, perpt_mean

def spike_LL_perregion(values, chs, select_oi): #could be generalized to multiple features easily. 
    """
    returns a list of the features of interest of each pt, per channel in ROI
    """
    perpt = []
    perpt_mean = []
    for i, pt in enumerate(values):
        perroi_mean = []
        perroi = []

        for j, roi in enumerate(pt):
            spikefeat = []

            if not roi:
                perroi.append(np.nan)
                perroi_mean.append(np.nan)
                continue
            for l, xs in enumerate(roi):
                    val_want = np.transpose(xs)
                    val_want = val_want[chs[i][j][select_oi[i][j][l]]-1]
                    feat = LL(val_want[750:1500]) #added a constraint to hopefully capture the spike
                    spikefeat.append(feat)
            perroi.append(spikefeat)
            perroi_mean.append(np.nanmean(spikefeat))
        perpt.append(perroi)
        perpt_mean.append(perroi_mean)

    return perpt, perpt_mean

def divide_chunks(l, n):
    # looping till length l
    biglist = []
    for i in range(0, len(l), n):
        list = l[i:i + n]
        biglist.append(list)
    return biglist

def feat_extract(lists_ptnames, roilist, data_directory):
    clinic_soz_all = []
    Aperpt_mean_all = []
    totalcount_perpt_all = []
    LLperpt_mean_all = []

    for list in lists_ptnames:
        #clear at the start to reduce memory load
        values = []
        idxch = []
        infer_spike_soz = []
        print('cleared + new pt list')

        #values
        values, idxch, infer_spike_soz, clinic_soz = biglist_roi(list, roilist, data_directory)
        clinic_soz_all.append(clinic_soz)

        #features
        Aperpt, Aperpt_mean = spike_amplitude_perregion(values, idxch)
        spike_count_perpt, totalcount_perpt, fullcount_perroi = spike_count_perregion(values)
        LLperpt, LLperpt_mean = spike_LL_perregion(values, idxch)

        Aperpt_mean_all.append(Aperpt_mean)
        totalcount_perpt_all.append(totalcount_perpt)
        LLperpt_mean_all.append(LLperpt_mean)
    

    Aperpt_mean = [x for x in Aperpt_mean_all for x in x]
    LLperpt_mean = [x for x in LLperpt_mean_all for x in x]
    totalcount_perpt = [x for x in totalcount_perpt_all for x in x]
    clinic_soz = [x for x in clinic_soz_all for x in x]

    return Aperpt_mean, LLperpt_mean, totalcount_perpt, clinic_soz

from numpy import var, mean, sqrt
from pandas import Series

def cohend(d1: Series, d2: Series) -> float:
    """
    takes 2 series from pandas and returns the cohens d
    """
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)

    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)

    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)

    # return the effect size
    return (u1 - u2) / s