#%% required packages
import pandas as pd
import numpy as np
from ieeg.auth import Session
from resampy import resample

# Import custom functions
import sys, os
code_v2_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/spike_detector/')
sys.path.append(code_v2_path)
from get_iEEG_data import *
from spike_detector import *
from iEEG_helper_functions import *
from spike_morphology_v2 import *

code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

data_directory = ['/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2', '/mnt/leif/littlab/data/Human_Data']

#load the list of patients to exclude
blacklist = ['HUP101' ,'HUP112','HUP115','HUP119','HUP124','HUP144','HUP147','HUP149','HUP155','HUP176','HUP193','HUP194','HUP195','HUP198','HUP208','HUP212','HUP216','HUP217','HUP064','HUP071','HUP072','HUP073','HUP085','HUP094']

#load all the filenames (long form IEEG filenames)
# will_stim_pts = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/spike_detector/will_stim_pts.csv')

# remove the patients in the blacklist from filenames_w_ids
# filenames_w_ids = will_stim_pts[~will_stim_pts['hup_id'].isin(blacklist)]


MUSC_pts = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/MUSC_Emory_LEN_SOZ_type.xlsx')
MUSC_pts_cleaned = MUSC_pts[MUSC_pts['Site_1MUSC_2Emory'] == 1]
# MUSC_pts_cleaned2 = MUSC_pts_cleaned[((MUSC_pts_cleaned['MTL'] == 1) & (MUSC_pts_cleaned['Neo'] == 0))| ((MUSC_pts_cleaned['MTL'] == 0) & (MUSC_pts_cleaned['Neo'] == 1))]

filenames_w_ids = MUSC_pts_cleaned
pt_files_split = np.array_split(filenames_w_ids, 2)
type = 'MUSC' #stim_pts

sz_times = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/MUSC_seizure_times.xlsx')

#%% load the session
#use Carlos's Session
password_bin_filepath = "/mnt/leif/littlab/users/aguilac/tools/agu_ieeglogin.bin"
with open(password_bin_filepath, "r") as f:
    session = Session("aguilac", f.read())

#%% loop through each patient
#pick a split dataframe from the 7 from pt_files_split to process       #CHANGE IN THE FUTURE FOR DIFFERENT BATCHES
pt_files = pt_files_split[0]

pt_files = pt_files.iloc[29:]

#loop through each patient
for index, row in pt_files.iterrows():
    hup_id = row['ParticipantID'] #['hup_id']
    dataset_name = row['filename'] #['filename']

    sz_time_file = sz_times[sz_times['File'] == dataset_name]


    print("\n")
    print(f"------Processing HUP {hup_id} with dataset {dataset_name}------")

    ########################################
    # Get the data from IEEG
    ########################################

    dataset = session.open_dataset(dataset_name)

    all_channel_labels = np.array(dataset.get_channel_labels())
    channel_labels_to_download = all_channel_labels[
        electrode_selection(all_channel_labels)
    ]
    if channel_labels_to_download.size > 0:
        duration_usec = dataset.get_time_series_details(
            channel_labels_to_download[0]
        ).duration
        duration_secs = duration_usec / 1e6
    else:
        continue

    #create a range spanning from 0 to duration_secs in 600 second intervals   
    intervals = np.arange(0, duration_secs, 600)
    #create a list of tuples where each tuple is a start and stop time for each interval
    intervals = list(zip(intervals[:-1], intervals[1:]))
    # for each tuple range, pick a random 60 second interval
    random_intervals = [np.random.randint(i[0], i[1] - 60) for i in intervals]
    #create a list of tuples where each tuple is a start and stop time +- 30 seconds from the random interval
    random_intervals = [(i - 30, i + 30) for i in random_intervals]
    #make sure the first interval is >0
    random_intervals[0] = (150, 210)
    correct_i = 0

    # Iterate over each interval and check for overlap with events
    for i, interval in enumerate(random_intervals):
        for _, row in sz_time_file.iterrows():
            if (row['Onset time']-(60*30) <= interval[0]) & (interval[-1] <= row['Offset time']+(60*30)): #give about a 30 minute padding
                # Interval overlaps with an event, mark for removal
                random_intervals[i] = None
                break

    random_intervals = [x for x in random_intervals if x is not None]

    #check to see if save file exists:
    if os.path.exists(f'{data_directory[0]}/spike_leaders/{type}/{dataset_name}_spike_output.csv'):
        print(f"------{dataset_name}_spike_output.csv already exists------")

        #load the file
        spike_output_DF = pd.read_csv(f'{data_directory[0]}/spike_leaders/{type}/{dataset_name}_spike_output.csv', header=None)
        #set the column names
        spike_output_DF.columns = ['peak_index', 'channel_index', 'channel_label', 'spike_sequence', 'peak', 'left_point', 'right_point','slow_end','slow_max','rise_amp','decay_amp','slow_width','slow_amp','rise_slope','decay_slope','average_amp','linelen', 'interval number', 'peak_index_samples', 'peak_time_usec']
        #get the number of intervals already processed
        num_intervals = spike_output_DF['interval number'].max()

        #remove the intervals already processed from random_intervals
        random_intervals = random_intervals[num_intervals+1:]

        #check to see if there are any intervals left to process
        if len(random_intervals) < 2:
            print(f"------No intervals left to process for {hup_id}_{dataset_name}------")
            continue

        #correct the index for the intervals if it exists.
        correct_i = num_intervals + 1
    
    # Loop through each minute interval
    for i, interval in enumerate(random_intervals):
        
        i = i + correct_i

        print(
            f"Getting iEEG data for interval {i} out of {len(random_intervals)} for HUP {hup_id}"
        )
        duration_usec = 6e7  # 1 minute
        start_time_sec, stop_time_sec = interval[0], interval[1]
        start_time_usec = int(start_time_sec * 1e6)
        stop_time_usec = int(stop_time_sec * 1e6)

        try:
            ieeg_data, fs = get_iEEG_data(
                "aguilac",
                password_bin_filepath,
                dataset_name,
                start_time_usec,
                stop_time_usec,
                channel_labels_to_download,
            )
            fs = int(fs)
        except:
            continue

        # check is FS is too low
        if fs <= 499:
            print(f"Sampling rate is {fs}")
            print("Sampling rate is too low, skip...")
            continue
        
        # Check if ieeg_data dataframe is all NaNs
        if ieeg_data.isnull().values.all():
            print(f"Empty dataframe after download, skip...")
            continue

        good_channels_res = detect_bad_channels_optimized(ieeg_data.to_numpy(), fs)
        good_channel_indicies = good_channels_res[0]
        good_channel_labels = channel_labels_to_download[good_channel_indicies]
        ieeg_data = ieeg_data[good_channel_labels].to_numpy()

        # Check if ieeg_data is empty after dropping bad channels
        if ieeg_data.size == 0:
            print(f"Empty dataframe after artifact rejection, skip...")
            continue

        ieeg_data = common_average_montage(ieeg_data)

        # Apply the filters directly on the DataFrame
        ieeg_data = notch_filter(ieeg_data, 60, fs) 
        ieeg_data = new_bandpass_filt(ieeg_data, 1, 70, fs, order = 4) 

        ##############################
        # Detect spikes
        ##############################

        spike_output = spike_detector(
            data=ieeg_data,
            fs=fs,
            electrode_labels=good_channel_labels,
        )
        spike_output = spike_output.astype(int)
        actual_number_of_spikes = len(spike_output)

        if len(spike_output) == 0:
            print("No spikes detected, skip saving...")
            continue
        else:
            print(f"Detected {len(spike_output)} spikes")

        ##############################
        # Extract spike morphologies
        ##############################

        #                                                                                           FIX HERE. TRY TO FIGURE OUT THAT THIS WILL WORK.
        # Preallocate the result array
        spike_output_to_save = np.empty((spike_output.shape[0], 17), dtype=object)
        spike_output_to_save[:, :] = np.NaN  # Fill with NaNs

        sequence_to_skip = None
        
        for z, spike in enumerate(spike_output):
            peak_index = int(spike[0])
            channel_index = int(spike[1])
            spike_sequence = spike[2]

            if spike_sequence == sequence_to_skip:
                continue

            # Fill the first two columns with peak_index and channel_index
            spike_output_to_save[z, 0] = peak_index
            spike_output_to_save[z, 1] = channel_index
            spike_output_to_save[z, 2] = good_channel_labels[channel_index]
            spike_output_to_save[z, 3] = spike_sequence

            # Extract the spike signal
            spike_signal = ieeg_data[
                peak_index - (2*fs) : peak_index + (2*fs), channel_index
            ]

            #check for edge case and remove all spikes in that same sequence
            if len(spike_signal) != 4*fs:
                print("Edge case, skip...")
                sequence_to_skip = spike_sequence
                continue

            sequence_to_skip = None

            if fs >= 500:
                spike_signal = resample(spike_signal, fs, 500)
            elif fs < 499:
                print(f"Sampling rate is too low ({fs}), skip...")
                continue
            else: 
                print(f"Sampling rate is {fs}")
                print("Sampling rate is too low, skip...""")
                continue                

            try:
                (
                    basic_features,
                    advanced_features,
                    is_valid,
                    bad_reason,
                ) = extract_spike_morphology(spike_signal)

                if is_valid:
                    # Fill the rest of the columns with computed features
                    spike_output_to_save[z, 4:9] = basic_features
                    spike_output_to_save[z, 9:17] = advanced_features
            except Exception as e:
                print(f"Error extracting spike features: {e}")
                continue
        
        # convert to dataframe
        spike_output_DF = pd.DataFrame(spike_output_to_save, columns = ['peak_index', 'channel_index', 'channel_label', 'spike_sequence', 'peak', 'left_point', 'right_point','slow_end','slow_max','rise_amp','decay_amp','slow_width','slow_amp','rise_slope','decay_slope','average_amp','linelen'])
        spike_output_DF['interval number'] = i
        start_time_samples = (start_time_usec/(1e6)) * fs
        spike_output_DF['peak_index_samples'] = spike_output_DF['peak_index'] + start_time_samples
        spike_output_DF['peak_time_usec'] = spike_output_DF['peak_index_samples'] * (1e6/fs)

        if i == 0: 
            #save spike_output_DF as a new csv file
            spike_output_DF.to_csv(f'{data_directory[0]}/spike_leaders/{type}/{dataset_name}_spike_output.csv', index = False)
        else: 
            #save spike_output_DF, append to existing csv file
            spike_output_DF.to_csv(f'{data_directory[0]}/spike_leaders/{type}/{dataset_name}_spike_output.csv', index = False, header = False, mode = 'a')
