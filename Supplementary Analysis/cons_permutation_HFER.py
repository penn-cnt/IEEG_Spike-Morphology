#%%
#%%
import pandas as pd
import numpy as np
# from ieeg.auth import Session
from resampy import resample
import re
import scipy.stats as stats
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
from concurrent.futures import ProcessPoolExecutor
import ast

warnings.filterwarnings('ignore')
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

hup_ei = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/6-seizure_analysis/data/self_run/hfer_HUP_11s.csv')
musc_ei = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/6-seizure_analysis/data/self_run/hfer_MUSC_11s.csv')

musc_ei['EI'] = musc_ei['EI'].apply(lambda x: list(map(float, x.strip('[]').split())))
# Function to convert string representation of list to actual list
def convert_to_list(string_list):
    return ast.literal_eval(string_list)
# Apply the function to the 'name' column
musc_ei['name'] = musc_ei['name'].apply(convert_to_list)

# Remove rows where 'EI' is a list of all zeros
df_filtered = musc_ei[musc_ei['EI'].apply(lambda x: not np.all(np.array(x) == 0.0))]

##############
#HERE WE WILL ADD THE NORMALIZATION ACROSS EACH SEIZURE (PER ROW)
def z_score(values):
    mean = sum(values) / len(values)
    std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
    z_scores = [(x - mean) / std_dev for x in values]
    return z_scores

df_filtered['EI'] = df_filtered['EI'].apply(z_score)
##############

# Explode the EI and name columns to unnest the lists
df_exploded = df_filtered.explode(['EI', 'name'])

# Reset index if needed
df_exploded = df_exploded.reset_index(drop=True)

musc_ei = df_exploded #reassign back to this variable for the rest of the analysis.

musc_ei['EI'] = pd.to_numeric(musc_ei['EI'], errors='coerce')
musc_ei = musc_ei.dropna(subset=['EI'])

# grouped = musc_ei.groupby(['MUSC_ID','name'])['EI'].quantile(0.75).reset_index()
grouped = musc_ei
grouped['name'] = grouped['name'].str.replace("'", "", regex=False)

grouped = grouped.rename(columns={
    'MUSC_ID': 'pt_id',
    'name': 'channel_label',
    'EI': 'EI'  # EI remains the same, but included for clarity
})

grouped['channel_label'] = grouped.apply(
    lambda row: decompose_labels(row['channel_label'], row['pt_id']),
    axis=1
)

musc_files = pd.read_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/6-seizure_analysis/data/MUSC_files.csv', index_col = 0)
musc_files = musc_files[['filename','channel_label']]
musc_files = musc_files.rename(columns = {'filename':'pt_id'})
all_spikes = grouped.merge(musc_files, on = 'pt_id', how = 'inner')

# Function to convert string representation of list to actual list if needed
def convert_to_list(string_list):
    if isinstance(string_list, str):
        return ast.literal_eval(string_list)
    return string_list

# Convert string representations of lists to actual lists if needed
all_spikes['channel_label_y'] = all_spikes['channel_label_y'].apply(convert_to_list)

def is_channel_in_list(row):
    return row['channel_label_x'] in row['channel_label_y']

# Filter rows where channel_label_x is in channel_label_y
all_spikes = all_spikes[all_spikes.apply(is_channel_in_list, axis=1)].reset_index(drop=True)

chs_tokeep = ['RA','LA','LPH','RPH','LAH','RAH']

#if channel_label contains any of the strings in chs_tokeep, keep it
all_spikes = all_spikes[all_spikes['channel_label_x'].str.contains('|'.join(chs_tokeep))].reset_index(drop=True)

all_spikes = all_spikes[~all_spikes['channel_label_x'].str.contains('I|LAP|T|S|C')].reset_index(drop=True)

#strip the letters from the channel_label column and keep only the numerical portion
# all_spikes['channel_label_x'] = all_spikes['channel_label_x'].str.replace('L|R|A|H|P', '', regex = True)

musc_sozs = pd.read_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/6-seizure_analysis/data/MUSC_sozs.csv', index_col = 0)
#add musc_files back so that you can pair up the filenames with the Pt_id
musc_files = pd.read_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/6-seizure_analysis/data/MUSC_files.csv', index_col = 0)
musc_data = musc_files.merge(musc_sozs, on='pt_id', how = 'inner')
musc_files = musc_data[['filename','region']]
musc_files = musc_files.rename(columns = {'filename':'pt_id'})

uniques_sozs = musc_files[['pt_id','region']].drop_duplicates()

# merge them
all_spikes = all_spikes.merge(uniques_sozs, on = 'pt_id', how='inner')
all_spikes = all_spikes.drop(columns = 'channel_label_y')

musc_all_spikes = all_spikes

def SOZ_assigner1(row):
    if row['region'] == 1:
        return 'mesial temporal'
    elif row['region'] == 2:
        return 'temporal neocortical'

musc_all_spikes['region'] = musc_all_spikes.apply(SOZ_assigner1, axis = 1)
musc_all_spikes = musc_all_spikes.rename(columns = {'region':'SOZ'})

###################################
# GET HUP 
###################################

# remove any row that has all 0's for EI# Assuming 'df' is your DataFrame
hup_ei['EI'] = hup_ei['EI'].apply(lambda x: list(map(float, x.strip('[]').split())))

# Function to convert string representation of list to actual list
def convert_to_list(string_list):
    return ast.literal_eval(string_list)
# Apply the function to the 'name' column
hup_ei['name'] = hup_ei['name'].apply(convert_to_list)

# Remove rows where 'EI' is a list of all zeros
df_filtered = hup_ei[hup_ei['EI'].apply(lambda x: not np.all(np.array(x) == 0.0))]

##############
#HERE WE WILL ADD THE NORMALIZATION ACROSS EACH SEIZURE (PER ROW)
def z_score(values):
    mean = sum(values) / len(values)
    std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
    z_scores = [(x - mean) / std_dev for x in values]
    return z_scores

df_filtered['EI'] = df_filtered['EI'].apply(z_score)
##############

# Explode the EI and name columns to unnest the lists
df_exploded = df_filtered.explode(['EI', 'name'])

# Reset index if needed
df_exploded = df_exploded.reset_index(drop=True)

##%%
hup_ei = df_exploded #reassign back to this variable for the rest of the analysis.

hup_ei['EI'] = pd.to_numeric(hup_ei['EI'], errors='coerce')
hup_ei = hup_ei.dropna(subset=['EI'])


# grouped = hup_ei.groupby(['hupID','name'])['EI'].quantile(0.75).reset_index()
grouped = hup_ei

grouped['name'] = grouped['name'].str.replace("'", "", regex=False)

grouped = grouped.rename(columns={
    'hupID': 'pt_id',
    'name': 'channel_label',
    'EI': 'EI'  # EI remains the same, but included for clarity
})

grouped['channel_label'] = grouped.apply(
    lambda row: decompose_labels(row['channel_label'], row['pt_id']),
    axis=1
)

hup_files = pd.read_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/6-seizure_analysis/data/HUP_files.csv', index_col = 0)
hup_files = hup_files[['pt_id','channel_label']]
# hup_files = hup_files.rename(columns = {'filename':'pt_id'})
all_spikes = grouped.merge(hup_files, on = 'pt_id', how = 'inner')

# Function to convert string representation of list to actual list if needed
def convert_to_list(string_list):
    if isinstance(string_list, str):
        return ast.literal_eval(string_list)
    return string_list

# Convert string representations of lists to actual lists if needed
all_spikes['channel_label_y'] = all_spikes['channel_label_y'].apply(convert_to_list)

def is_channel_in_list(row):
    return row['channel_label_x'] in row['channel_label_y']

# Filter rows where channel_label_x is in channel_label_y
all_spikes = all_spikes[all_spikes.apply(is_channel_in_list, axis=1)].reset_index(drop=True)

chs_tokeep = ['RA','LA','RDA','LDA','LH','RH','LDH','RDH','DA','DH','DHA','LB','LDB','LC','LDC','RB','RDB','RC','RDC']

#if channel_label contains any of the strings in chs_tokeep, keep it
all_spikes = all_spikes[all_spikes['channel_label_x'].str.contains('|'.join(chs_tokeep))].reset_index(drop=True)

all_spikes = all_spikes[~all_spikes['channel_label_x'].str.contains('T|F|P|RCC|RCA|RAD|LAD|LHD|RHD|LDAH|RDAH|RCB|Z')].reset_index(drop=True)

#strip the letters from the channel_label column and keep only the numerical portion
# all_spikes['channel_label_x'] = all_spikes['channel_label_x'].str.replace('L|R|A|H|B|C|D', '', regex = True)

hup_sozs = pd.read_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/6-seizure_analysis/data/SOZ_hup.csv', index_col = 0)
hup_files = pd.read_csv('/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/6-seizure_analysis/data/HUP_files.csv', index_col = 0)
hup_data = hup_files.merge(hup_sozs, on='pt_id',how='inner')

hup_files = hup_data[['pt_id','SOZ']]
# hup_files = hup_files.rename(columns = {'filename':'pt_id'})
# hup_data = hup_files
uniques_sozs = hup_files[['pt_id','SOZ']].drop_duplicates()



# musc_files = musc_data[['filename','region']]
# musc_files = musc_files.rename(columns = {'filename':'pt_id'})

# uniques_sozs = musc_files[['pt_id','region']].drop_duplicates()


# merge them
all_spikes = all_spikes.merge(uniques_sozs, on = 'pt_id', how='inner')
all_spikes = all_spikes.drop(columns = 'channel_label_y')

hup_all_spikes = all_spikes

#%% 
#combine them
all_spikes = pd.concat([hup_all_spikes, musc_all_spikes])
all_spikes = all_spikes[['pt_id','ictal_start_time','EI','channel_label_x','SOZ']]
all_spikes = all_spikes.rename(columns = {'channel_label_x':'channel_label'})

# %%
#lets calculate Overall DP for each patient in each cohort

def process_patients(pt_id):
    print(pt_id)
    yo = all_spikes[all_spikes['pt_id'] == pt_id]
    patient_df = yo[['channel_label', 'EI']].groupby('channel_label').agg(['mean', 'count']).reset_index()
    patient_df.columns = ['channel_label', 'EI', 'count']
    patient_df['pt_id'] = pt_id

    # Calculate cumulative sum and percentage
    patient_df = patient_df.sort_values(by='count', ascending=False)
    patient_df['cumulative_count'] = patient_df['count'].cumsum()
    total_count = patient_df['count'].sum()
    patient_df['cumulative_percentage'] = patient_df['cumulative_count'] / total_count

    # Filter to keep channel_labels that make up 90% of the spikes
    # patient_df = patient_df[patient_df['cumulative_percentage'] <= 0.90]

    return patient_df

# # Parallel processing for HUP data
# with ProcessPoolExecutor() as executor:
#     overall_HFER = pd.concat(executor.map(process_patients, all_spikes['pt_id'].unique()), ignore_index=True)

# overall_HFER.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/overall_hfer.csv')

overall_HFER = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/overall_hfer_11s.csv', index_col = 0)
#%%
#Get time bins - every 30 minutes.

# Function to calculate average DP and count per channel_label in 30-minute chunks
def calculate_averages_and_counts(df):
    # Convert 'peak_time_min' to float if it is not already
    # df['peak_time_min'] = df['peak_time_min'].astype(float)
    
    # Create a new column for 30-minute chunks
    # df['time_chunk'] = (df['peak_time_min'] // 30).astype(int)
    
    # Group by 'filename', 'time_chunk', and 'channel_label'
    grouped = df.groupby(['pt_id', 'ictal_start_time', 'channel_label'])
    
    # Calculate average DP and count for each group
    result = grouped.agg(avg_DP=('EI', 'mean'), count=('channel_label', 'size')).reset_index()
    
    return result

all_segment_HFER = calculate_averages_and_counts(all_spikes)

# HUP_segment_DP = HUP_segment_DP[HUP_segment_DP['count'] > 10]
# MUSC_segment_DP = MUSC_segment_DP[MUSC_segment_DP['count']>10]
#%%

# make a list of all channel_labels per time chunk and for each filename.
all_elecs = (overall_HFER[['pt_id', 'channel_label']]
                 .groupby('pt_id')['channel_label']
                 .apply(lambda x: list(x.unique()))
                 .reset_index())

segment_elecs = (all_segment_HFER[['pt_id', 'channel_label','ictal_start_time']]
                 .groupby(['pt_id','ictal_start_time'])['channel_label']
                 .apply(lambda x: list(x.unique()))
                 .reset_index())


# %%
#Find the time chunks that will be analyzed

all_elecs['num_elecs'] = all_elecs['channel_label'].apply(lambda x: len(x))
segment_elecs['num_elecs'] = segment_elecs['channel_label'].apply(lambda x: len(x))

# Function to filter segments based on the 70% threshold
def filter_segments(df_all, df_seg, threshold=0.7):
    results = []
    
    for index, row in df_all.iterrows():
        filename = row['pt_id']
        num_elecs = row['num_elecs']
        threshold_num = threshold * num_elecs
        
        # Filter the segment dataframe for the current filename
        segments = df_seg[df_seg['pt_id'] == filename]
        
        time_chunks = []
        for seg_index, seg_row in segments.iterrows():
            if len(seg_row['channel_label']) >= threshold_num:
                time_chunks.append(seg_row['ictal_start_time'])
        
        results.append({'pt_id': filename, 'ictal_start_time': time_chunks})
    
    return pd.DataFrame(results)

# Apply the function
filtered_segments = filter_segments(all_elecs, segment_elecs, threshold = 0)

# Function to filter the segments
def filter_time_chunks(df_seg, df_filt):
    results = []
    
    for index, row in df_filt.iterrows():
        filename = row['pt_id']
        time_chunks = set(row['ictal_start_time'])
        
        # Filter the segment dataframe for the current filename and time_chunks
        segments = df_seg[(df_seg['pt_id'] == filename) & (df_seg['ictal_start_time'].isin(time_chunks))]
        
        results.append(segments)
    
    return pd.concat(results, ignore_index=True)

# Apply the function
filtered_segment_elecs= filter_time_chunks(segment_elecs, filtered_segments)

#%%
#Merge to only keep the time_chunks of interest (>25% electrode overlap)
segment_HFER = all_segment_HFER.merge(filtered_segment_elecs, on=['pt_id','ictal_start_time'], how = 'inner').drop(columns = ['channel_label_y','num_elecs']).rename(columns={'channel_label_x':'channel_label'})

# %%

all_corrs = pd.DataFrame()
for filename in segment_HFER['pt_id'].unique():
    yo = segment_HFER[segment_HFER['pt_id'] == filename]
    pearson = yo.groupby(['ictal_start_time','channel_label']).mean('avg_DP').reset_index()
    yo_ovr = overall_HFER[overall_HFER['pt_id'] == filename]
    pearson = pearson.merge(yo_ovr[['channel_label','EI']], on = 'channel_label', how='inner') 
    correlations = pearson.groupby('ictal_start_time').apply(lambda x: x['avg_DP'].corr(x['EI'], method = 'spearman')).reset_index()
    # Rename the columns for clarity
    correlations.columns = ['ictal_start_time', 'spearman_correlation']
    correlations['pt_id'] = filename
    all_corrs = pd.concat([all_corrs, correlations])


# %%
group_correlation = []
for filename in all_corrs['pt_id'].unique():
    yo = all_corrs[all_corrs['pt_id'] == filename]
    group_correlation.append(yo['spearman_correlation'].median())

print('mean corr:', np.mean((group_correlation)))
print('std corr:', np.std((group_correlation)))

print('median corr:', np.median((group_correlation)))
print('IQR:', np.percentile(group_correlation, 75) - np.percentile(group_correlation, 25))

og_corrs = group_correlation

#%%
#template for the permutation test
n_permutations = 1000
permuted_stats = []

for _ in range(n_permutations):
    all_spikes['EI'] = np.random.permutation(all_spikes['EI'])

    with ProcessPoolExecutor() as executor:
        overall_HFER = pd.concat(executor.map(process_patients, all_spikes['pt_id'].unique()), ignore_index=True)   

    all_segment_HFER = calculate_averages_and_counts(all_spikes)

    all_elecs = (overall_HFER[['pt_id', 'channel_label']]
                 .groupby('pt_id')['channel_label']
                 .apply(lambda x: list(x.unique()))
                 .reset_index())

    segment_elecs = (all_segment_HFER[['pt_id', 'channel_label','ictal_start_time']]
                    .groupby(['pt_id','ictal_start_time'])['channel_label']
                    .apply(lambda x: list(x.unique()))
                    .reset_index())

    all_elecs['num_elecs'] = all_elecs['channel_label'].apply(lambda x: len(x))
    segment_elecs['num_elecs'] = segment_elecs['channel_label'].apply(lambda x: len(x))

    filtered_segments = filter_segments(all_elecs, segment_elecs, threshold = 0)
    filtered_segment_elecs= filter_time_chunks(segment_elecs, filtered_segments)

    segment_HFER = all_segment_HFER.merge(filtered_segment_elecs, on=['pt_id','ictal_start_time'], how = 'inner').drop(columns = ['channel_label_y','num_elecs']).rename(columns={'channel_label_x':'channel_label'})

    all_corrs = pd.DataFrame()
    for filename in segment_HFER['pt_id'].unique():
        yo = segment_HFER[segment_HFER['pt_id'] == filename]
        pearson = yo.groupby(['ictal_start_time','channel_label']).mean('avg_DP').reset_index()
        yo_ovr = overall_HFER[overall_HFER['pt_id'] == filename]
        pearson = pearson.merge(yo_ovr[['channel_label','EI']], on = 'channel_label', how='inner') 
        correlations = pearson.groupby('ictal_start_time').apply(lambda x: x['avg_DP'].corr(x['EI'], method = 'spearman')).reset_index()
        # Rename the columns for clarity
        correlations.columns = ['ictal_start_time', 'spearman_correlation']
        correlations['pt_id'] = filename
        all_corrs = pd.concat([all_corrs, correlations])

    group_correlation = []
    for filename in all_corrs['pt_id'].unique():
        yo = all_corrs[all_corrs['pt_id'] == filename]
        group_correlation.append(yo['spearman_correlation'].median())

    permuted_stats.append(group_correlation)

#%%
# Flatten the permuted correlations array
np.save('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/permutation_test/1000_permuted_hfer_11s.npy',permuted_stats)
