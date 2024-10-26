#establishing environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy import signal as sig
import mat73
from scipy.io import loadmat

import random

def prep_clean_soz(SOZ_ch_labels):
    """
    function will clean the SOZ labels for each patient.
    input: SOZ_ch_labels from load pt data function. 
    output: list of cleaned names for each patient --> essentially adds '-CAR' to the end of each label.
    """
    clean_soz_labels = []
    for i in range(len(SOZ_ch_labels)):
        labels = SOZ_ch_labels.iloc[i]
        labels_to_list = [x for x in labels]
        if type(labels_to_list[0]) == str:
            labels_squish = labels_to_list[0].replace(" ","")
            labels_split = labels_squish.split(',')
            clean_labels_per_pt = [x+'-CAR' for x in labels_split]
            clean_soz_labels.append(clean_labels_per_pt)
        else: 
            clean_soz_labels.append('SOZ not defined')

    return clean_soz_labels

def load_pt(ptname, data_directory):
    """
    input: ptname, a string containing the name of the patient you want to load. example: 'HUP100'
    output: object: spike ---- contains: List of 1000 random spikes: spike.select   and their subsequent:
    values: spike.values, chlabels: spike.chlabels, fs: spike.fs, soz channels: spike.soz
    """        

    val = mat73.loadmat(data_directory[0] + '/values/values_{}.mat'.format(ptname))
    val2 = val['values_all']
    select_spikes = loadmat(data_directory[0] + '/randi_lists/randi_{}.mat'.format(ptname))
    select_spikes = select_spikes['select_spikes']
    ch_labels = loadmat(data_directory[0] + '/chlabels/chlabels_{}.mat'.format(ptname))
    ch_labels = ch_labels['ch_labels_all']
    fs_all = loadmat(data_directory[0] + '/fs/fs_{}.mat'.format(ptname))
    fs_all = fs_all['fs_all']
    SOZ_chlabels = pd.read_csv(data_directory[0] + '/pt_data/SOZ_channels.csv')
    pt_all = pd.read_csv(data_directory[0] + '/pt_data/ptname_all.csv') #'/Users/carlosaguila/PycharmProjects/CNT_Interictal_Spikes/Patient/pt_database/pt_data/ptname_all.csv')
    pt_name = ("'{}'".format(ptname))
    whichpt = pt_all.index[pt_all['ptname'] == pt_name].tolist()
    clean_SOZ_chlabels = prep_clean_soz(SOZ_chlabels)
    
    class spike:
        values = val2
        select=select_spikes
        chlabels = ch_labels
        fs = fs_all
        soz = clean_SOZ_chlabels[whichpt[0]]

    return spike


#function that takes random value matrix and plots
def plot_rand_spikes(values):
    plot_idx = range(5)
    fig, axs = plt.subplots(len(plot_idx), 2, figsize=(10,10))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.74)

    time = np.linspace(0,4,2001)
    for i in plot_idx:
        
        choice1 = random.choice(range(len(values))) #picks from what run_time to pull from
        choice2 = random.choice(np.transpose(values[choice1])) #picks what spike to plot in run_time
        
        choice3 = random.choice(range(len(values))) #picks from what run_time to pull from
        choice4 = random.choice(np.transpose(values[choice3])) #picks what spike to plot in run_time
        
        axs[i,0].plot(time, choice2, 'k') #plot nerve data - unfiltered
        axs[i,0].set_ylabel("Voltage (millivolts)")
        axs[i,0].set_xlabel("Time (s)")
        axs[i,0].set_title("Random Spike")
        
        axs[i,1].plot(time, choice4,'k') #plot new axs. 
        axs[i,1].set_ylabel("Voltage (millivolts)")
        axs[i,1].set_xlabel("Time (s)")
        axs[i,1].set_title("Random Spike")

    return

#function that takes random matrix, and plots down the channels
def plot_rand_eeg(values):
    set = random.choice(range(len(values)))
    values_for_all = (values[set])
    time = np.linspace(0,4,np.size(values_for_all,0))
    offset = 0
    plt.figure(figsize = (15,15))

    for i in range(np.size(values_for_all,1)):
        values_per_ch = values_for_all.transpose()
        plot_values = values_per_ch[i]
        val = plot_values.flatten()
        val = [0 if math.isnan(x) else x for x in val]
        offset = offset + np.max(val)
        plt.plot(time, val - offset,'k')
    
    return set


#function that takes random matrix, and plots down the channels
def plot_eeg(values, set):
    values_for_all = (values[set])
    time = np.linspace(0,4,np.size(values_for_all,0))
    offset = 0
    plt.figure(figsize = (15,15))

    for i in range(np.size(values_for_all,1)):
        values_per_ch = values_for_all.transpose()
        plot_values = values_per_ch[i]
        val = plot_values.flatten()
        if math.isnan(np.mean(val)) == True:
            continue
        #val = [0 if math.isnan(x) else x for x in val]
        offset = offset + np.max(val)
        plt.plot(time, val - offset,'k')
        plt.title('EEG for random spike {}'.format(set))
        plt.xlabel('Time (seconds)')
        plt.vlines(2,0,-offset-100,colors='r',linestyles='dotted')
    
    return set

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
def ZX(x):
    x_demean = x - np.mean(x)
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

def bandpower(x, fs, fmin, fmax):
    f, Pxx = sig.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

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

#function that takes random value matrix and plots
def plot_rand_onlyspikes(values,select_spikes):
    select_spikes_ch, spike_value_all = find_spike_ch(select_spikes,values)

    plot_idx = range(5)
    #fig, axs = plt.subplots(len(plot_idx), 2, figsize=(10,10))
    #fig.tight_layout()
    #fig.subplots_adjust(hspace=0.74)

    time = np.linspace(0,4,2001)
    columns = 5
    rows = 10
    fig, ax_array = plt.subplots(rows,columns, squeeze=True,figsize=(40,40),dpi=80)
    fig.subplots_adjust(hspace=0.74)
    for i, ax_row in enumerate(ax_array):
        for j, axes in enumerate(ax_row):
            choice=random.choice(spike_value_all)
            axes.plot(time, choice, 'k')
            axes.set_title("Random Spike")
            axes.set_ylabel("Voltage (millivolts)")
            axes.set_xlabel('Time (s)')

    for i in plot_idx:
        continue
        
        choice1 = (random.choice(spike_value_all)) #picks from what run_time to pull from
        
        choice3 = (random.choice(spike_value_all)) #picks from what run_time to pull from
        
        axs[i,0].plot(time, (choice1), 'k') #plot nerve data - unfiltered
        axs[i,0].set_ylabel("Voltage (millivolts)")
        axs[i,0].set_xlabel("Time (s)")
        axs[i,0].set_title("Random Spike")
        
        axs[i,1].plot(time, (choice3),'k') #plot new axs. 
        axs[i,1].set_ylabel("Voltage (millivolts)")
        axs[i,1].set_xlabel("Time (s)")
        axs[i,1].set_title("Random Spike")

    return fig

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
    dkt_directory = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])
    brain_df = pd.read_csv(dkt_directory)
    brain_df['name'] = brain_df['name'].astype(str) + '-CAR'
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
    dkt_directory = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])
    brain_df = pd.read_csv(dkt_directory)
    brain_df['name'] = brain_df['name'].astype(str) + '-CAR'
    return rid[0], brain_df
        
def label_fix(pt, data_directory, threshold = 0.25):
    '''
    label_fix reassigns labels overlapping brain regions to "empty labels" in our DKTantspynet output from IEEG_recon
    input:  pt - name of patient. example: 'HUP100' 
            data_directory - directory containing CNT_iEEG_BIGS folder. (must end in '/')
            threshold - arbitrary threshold that r=2mm surround of electrode must overlap with a brain region. default: threshold = 25%, Brain region has a 25% or more overlap.
    output: relabeled_df - a dataframe that contains 2 extra columns showing the second most overlapping brain region and the percent of overlap. 
    '''

    rid, brain_df = load_rid_forjson(pt, data_directory)
    json_labels = data_directory[1] + '/CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)
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

    
def load_ptall(ptname, data_directory):
    """ load_ptall combines all the functions together, loading both the RID and the IEEG data just using the Patient NAME
        Will create a dataframe, and a spike object containing: values, fs, chlabels, 
    """
    spike = load_pt(ptname,data_directory)
    rid, brain_df = load_rid(ptname, data_directory)
    relabeled_df = label_fix(ptname, data_directory, threshold = 0.25)

    return spike, relabeled_df, [ptname,rid]

def value_basis(spike, brain_df, roi):
    """
    Function that takes in all values, the DKT atlas dataframe, and a region of interest (ROI)
    returns a tailored, truncated list of the all the values given a specific ROI
    input: spike object, brain dataframe, roi list
    output: correlated values, channel number (matlab), indices of channels
    """
    roi_ch = pd.DataFrame()
    for x in roi:
        roi_ch = roi_ch.append(brain_df[(brain_df['final_label'] == x)])
        #roi_ch = roi_ch.concat([roi_ch, brain_df[(brain_df['label'] == x )]])

    #roi_ch = brain_df.loc[brain_df['label']== roi] #creates truncated dataframe of ROI labels
    roi_chlist = np.array(roi_ch['name']) #converts DF to array

    #finds the index of where to find the channel
    idx_roich = []
    for ch in roi_chlist:
        x = np.where(spike.chlabels[0][0] == ch)[0]
        idx_roich.append(x)

    idx_roich = [x[0] for x in idx_roich]
    chnum = [x+1 for x in idx_roich]

    counts,chs = hifreq_ch_spike(spike.select)

    select_oi = []
    for chroi in idx_roich:
        idx = np.where(chs == chroi)[0]
        select_oi.append(idx)

    select_oi = np.concatenate(select_oi)
    select_oi = [int(x) for x in select_oi]
    values_oi = []
    for soi in select_oi:
        values_oi.append(spike.values[soi])

    based_values = values_oi

    return based_values, chnum, idx_roich

