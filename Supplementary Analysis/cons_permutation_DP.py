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

warnings.filterwarnings('ignore')


MUSC_cons = pd.read_csv('../dataset/consistency/MUSC_clean.csv', index_col=0)
HUP_cons = pd.read_csv('../dataset/consistency/HUP_clean.csv', index_col =0)

# %%
#lets calculate Overall DP for each patient in each cohort

def process_patient_HUP(pt_id):
    print(pt_id)
    yo = HUP_cons[HUP_cons['filename'] == pt_id]
    patient_df = yo[['channel_label', 'DP']].groupby('channel_label').agg(['mean', 'count']).reset_index()
    patient_df.columns = ['channel_label', 'DP', 'count']
    patient_df['filename'] = pt_id

    # Calculate cumulative sum and percentage
    patient_df = patient_df.sort_values(by='count', ascending=False)
    patient_df['cumulative_count'] = patient_df['count'].cumsum()
    total_count = patient_df['count'].sum()
    patient_df['cumulative_percentage'] = patient_df['cumulative_count'] / total_count

    # Filter to keep channel_labels that make up 90% of the spikes
    patient_df = patient_df[patient_df['cumulative_percentage'] <= 0.90]

    return patient_df

# # Parallel processing for HUP data
# with ProcessPoolExecutor() as executor:
#     HUP_Overall_DP = pd.concat(executor.map(process_patient_HUP, HUP_cons['filename'].unique()), ignore_index=True)

# HUP_Overall_DP.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/hup_overall_dp.csv')

def process_patient_MUSC(pt_id):
    print(pt_id)
    yo = MUSC_cons[MUSC_cons['filename'] == pt_id]
    patient_df = yo[['channel_label', 'DP']].groupby('channel_label').agg(['mean', 'count']).reset_index()
    patient_df.columns = ['channel_label', 'DP', 'count']
    patient_df['filename'] = pt_id

    # Calculate cumulative sum and percentage
    patient_df = patient_df.sort_values(by='count', ascending=False)
    patient_df['cumulative_count'] = patient_df['count'].cumsum()
    total_count = patient_df['count'].sum()
    patient_df['cumulative_percentage'] = patient_df['cumulative_count'] / total_count

    # Filter to keep channel_labels that make up 90% of the spikes
    patient_df = patient_df[patient_df['cumulative_percentage'] <= 0.90]

    return patient_df

# # Parallel processing for HUP data
# with ProcessPoolExecutor() as executor:
#     MUSC_Overall_DP = pd.concat(executor.map(process_patient_MUSC, MUSC_cons['filename'].unique()), ignore_index=True)

# MUSC_Overall_DP.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/musc_overall_dp.csv')

HUP_Overall_DP = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/hup_overall_dp.csv', index_col=0)
MUSC_Overall_DP = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/musc_overall_dp.csv',index_col=0)

#%%
#Get time bins - every 30 minutes.

# Function to calculate average DP and count per channel_label in 30-minute chunks
def calculate_averages_and_counts(df):
    # Convert 'peak_time_min' to float if it is not already
    df['peak_time_min'] = df['peak_time_min'].astype(float)
    
    # Create a new column for 30-minute chunks
    df['time_chunk'] = (df['peak_time_min'] // 30).astype(int)
    
    # Group by 'filename', 'time_chunk', and 'channel_label'
    grouped = df.groupby(['filename', 'time_chunk', 'channel_label'])
    
    # Calculate average DP and count for each group
    result = grouped.agg(avg_DP=('DP', 'mean'), count=('channel_label', 'size')).reset_index()
    
    return result

HUP_segment_DP = calculate_averages_and_counts(HUP_cons)
MUSC_segment_DP = calculate_averages_and_counts(MUSC_cons)

# HUP_segment_DP = HUP_segment_DP[HUP_segment_DP['count'] > 10]
# MUSC_segment_DP = MUSC_segment_DP[MUSC_segment_DP['count']>10]
#%%

# make a list of all channel_labels per time chunk and for each filename.
all_elecs_hup = (HUP_Overall_DP[['filename', 'channel_label']]
                 .groupby('filename')['channel_label']
                 .apply(lambda x: list(x.unique()))
                 .reset_index())

segment_elecs_hup = (HUP_segment_DP[['filename', 'channel_label','time_chunk']]
                 .groupby(['filename','time_chunk'])['channel_label']
                 .apply(lambda x: list(x.unique()))
                 .reset_index())

all_elecs_musc = (MUSC_Overall_DP[['filename', 'channel_label']]
                 .groupby('filename')['channel_label']
                 .apply(lambda x: list(x.unique()))
                 .reset_index())

segment_elecs_musc = (MUSC_segment_DP[['filename', 'channel_label','time_chunk']]
                 .groupby(['filename','time_chunk'])['channel_label']
                 .apply(lambda x: list(x.unique()))
                 .reset_index())


# %%
#Find the time chunks that will be analyzed

all_elecs_hup['num_elecs'] = all_elecs_hup['channel_label'].apply(lambda x: len(x))
segment_elecs_hup['num_elecs'] = segment_elecs_hup['channel_label'].apply(lambda x: len(x))

all_elecs_musc['num_elecs'] = all_elecs_musc['channel_label'].apply(lambda x: len(x))
segment_elecs_musc['num_elecs'] = segment_elecs_musc['channel_label'].apply(lambda x: len(x))

# Function to filter segments based on the 70% threshold
def filter_segments(df_all, df_seg, threshold=0.7):
    results = []
    
    for index, row in df_all.iterrows():
        filename = row['filename']
        num_elecs = row['num_elecs']
        threshold_num = threshold * num_elecs
        
        # Filter the segment dataframe for the current filename
        segments = df_seg[df_seg['filename'] == filename]
        
        time_chunks = []
        for seg_index, seg_row in segments.iterrows():
            if len(seg_row['channel_label']) >= threshold_num:
                time_chunks.append(seg_row['time_chunk'])
        
        results.append({'filename': filename, 'time_chunks': time_chunks})
    
    return pd.DataFrame(results)

# Apply the function
filtered_segments_hup = filter_segments(all_elecs_hup, segment_elecs_hup, threshold = 0.25)
filtered_segments_musc = filter_segments(all_elecs_musc, segment_elecs_musc, threshold = 0.25)

# Function to filter the segments
def filter_time_chunks(df_seg, df_filt):
    results = []
    
    for index, row in df_filt.iterrows():
        filename = row['filename']
        time_chunks = set(row['time_chunks'])
        
        # Filter the segment dataframe for the current filename and time_chunks
        segments = df_seg[(df_seg['filename'] == filename) & (df_seg['time_chunk'].isin(time_chunks))]
        
        results.append(segments)
    
    return pd.concat(results, ignore_index=True)

# Apply the function
filtered_segment_elecs_hup = filter_time_chunks(segment_elecs_hup, filtered_segments_hup)
filtered_segment_elecs_musc = filter_time_chunks(segment_elecs_musc, filtered_segments_musc)

#%%
#Merge to only keep the time_chunks of interest (>25% electrode overlap)
hup_segment_dp = HUP_segment_DP.merge(filtered_segment_elecs_hup, on=['filename','time_chunk'], how = 'inner').drop(columns = ['channel_label_y','num_elecs']).rename(columns={'channel_label_x':'channel_label'})
musc_segment_dp = MUSC_segment_DP.merge(filtered_segment_elecs_musc, on=['filename','time_chunk'], how = 'inner').drop(columns = ['channel_label_y','num_elecs']).rename(columns={'channel_label_x':'channel_label'})

# %%

hup_corrs = pd.DataFrame()
for filename in hup_segment_dp['filename'].unique():
    yo = hup_segment_dp[hup_segment_dp['filename'] == filename]
    pearson = yo.groupby(['time_chunk','channel_label']).mean('avg_DP').reset_index()
    yo_ovr = HUP_Overall_DP[HUP_Overall_DP['filename'] == filename]
    pearson = pearson.merge(yo_ovr[['channel_label','DP']], on = 'channel_label', how='inner') 
    correlations = pearson.groupby('time_chunk').apply(lambda x: x['avg_DP'].corr(x['DP'], method = 'spearman')).reset_index()
    # Rename the columns for clarity
    correlations.columns = ['time_chunk', 'spearman_correlation']
    correlations['filename'] = filename
    hup_corrs = pd.concat([hup_corrs, correlations])


musc_corrs = pd.DataFrame()
for filename in musc_segment_dp['filename'].unique():
    yo = musc_segment_dp[musc_segment_dp['filename'] == filename]
    pearson = yo.groupby(['time_chunk','channel_label']).mean('avg_DP').reset_index()
    yo_ovr = MUSC_Overall_DP[MUSC_Overall_DP['filename'] == filename]
    pearson = pearson.merge(yo_ovr[['channel_label','DP']], on = 'channel_label', how='inner') 
    correlations = pearson.groupby('time_chunk').apply(lambda x: x['avg_DP'].corr(x['DP'], method = 'spearman')).reset_index()
    # Rename the columns for clarity
    correlations.columns = ['time_chunk', 'spearman_correlation']
    correlations['filename'] = filename
    musc_corrs = pd.concat([musc_corrs, correlations])

all_corrs = pd.concat([musc_corrs, hup_corrs])

# %%

filename = all_corrs['filename'].unique()[0]
correlations = all_corrs[all_corrs['filename'] == filename]
# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(correlations['time_chunk'], correlations['spearman_correlation'], color = 'r', label = 'Corr of Segment to Overall')
plt.axhline(y=correlations['spearman_correlation'].median(), linestyle='-', color='k', label='Median = DP-Stability')
plt.title(f'DP Stability Calculation for {filename}')
plt.xlabel('Segment (30 - min)')
plt.ylabel('Corr Coef')
plt.ylim([0,1])
plt.legend()
plt.grid(True)


group_correlation = []
for filename in all_corrs['filename'].unique():
    yo = all_corrs[all_corrs['filename'] == filename]
    group_correlation.append(yo['spearman_correlation'].median())

print('mean corr:', np.mean((group_correlation)))
print('std corr:', np.std((group_correlation)))


print('median corr:', np.median((group_correlation)))
print('IQR:', np.percentile(group_correlation, 75) - np.percentile(group_correlation, 25))


og_corrs = group_correlation

#%%

#We want to shuffle the labels

MUSC_cons = pd.read_csv('../dataset/consistency/MUSC_clean.csv', index_col=0)
HUP_cons = pd.read_csv('../dataset/consistency/HUP_clean.csv', index_col =0)

n_permutations = 1000
permuted_stats = []

for _ in range(n_permutations):
    # Shuffle the 'DP' column
    MUSC_cons['DP'] = np.random.permutation(MUSC_cons['DP'])
    HUP_cons['DP'] = np.random.permutation(HUP_cons['DP'])

    # Parallel processing for MUSC data
    with ProcessPoolExecutor() as executor:
        MUSC_Overall_DP = pd.concat(executor.map(process_patient_MUSC, MUSC_cons['filename'].unique()), ignore_index=True)  

    # Parallel processing for HUP data
    with ProcessPoolExecutor() as executor:
        HUP_Overall_DP = pd.concat(executor.map(process_patient_HUP, HUP_cons['filename'].unique()), ignore_index=True)

    HUP_segment_DP = calculate_averages_and_counts(HUP_cons)
    MUSC_segment_DP = calculate_averages_and_counts(MUSC_cons)
    

    # make a list of all channel_labels per time chunk and for each filename.
    all_elecs_hup = (HUP_Overall_DP[['filename', 'channel_label']]
                    .groupby('filename')['channel_label']
                    .apply(lambda x: list(x.unique()))
                    .reset_index())

    segment_elecs_hup = (HUP_segment_DP[['filename', 'channel_label','time_chunk']]
                    .groupby(['filename','time_chunk'])['channel_label']
                    .apply(lambda x: list(x.unique()))
                    .reset_index())

    all_elecs_musc = (MUSC_Overall_DP[['filename', 'channel_label']]
                    .groupby('filename')['channel_label']
                    .apply(lambda x: list(x.unique()))
                    .reset_index())

    segment_elecs_musc = (MUSC_segment_DP[['filename', 'channel_label','time_chunk']]
                    .groupby(['filename','time_chunk'])['channel_label']
                    .apply(lambda x: list(x.unique()))
                    .reset_index())

    all_elecs_hup['num_elecs'] = all_elecs_hup['channel_label'].apply(lambda x: len(x))
    segment_elecs_hup['num_elecs'] = segment_elecs_hup['channel_label'].apply(lambda x: len(x))

    all_elecs_musc['num_elecs'] = all_elecs_musc['channel_label'].apply(lambda x: len(x))
    segment_elecs_musc['num_elecs'] = segment_elecs_musc['channel_label'].apply(lambda x: len(x))

    # Apply the function
    filtered_segments_hup = filter_segments(all_elecs_hup, segment_elecs_hup, threshold = 0.25)
    filtered_segments_musc = filter_segments(all_elecs_musc, segment_elecs_musc, threshold = 0.25)

    # Apply the function
    filtered_segment_elecs_hup = filter_time_chunks(segment_elecs_hup, filtered_segments_hup)
    filtered_segment_elecs_musc = filter_time_chunks(segment_elecs_musc, filtered_segments_musc)

    #merge across
    hup_segment_dp = HUP_segment_DP.merge(filtered_segment_elecs_hup, on=['filename','time_chunk'], how = 'inner').drop(columns = ['channel_label_y','num_elecs']).rename(columns={'channel_label_x':'channel_label'})
    musc_segment_dp = MUSC_segment_DP.merge(filtered_segment_elecs_musc, on=['filename','time_chunk'], how = 'inner').drop(columns = ['channel_label_y','num_elecs']).rename(columns={'channel_label_x':'channel_label'})

    #
    hup_corrs = pd.DataFrame()
    for filename in hup_segment_dp['filename'].unique():
        yo = hup_segment_dp[hup_segment_dp['filename'] == filename]
        pearson = yo.groupby(['time_chunk','channel_label']).mean('avg_DP').reset_index()
        yo_ovr = HUP_Overall_DP[HUP_Overall_DP['filename'] == filename]
        pearson = pearson.merge(yo_ovr[['channel_label','DP']], on = 'channel_label', how='inner') 
        correlations = pearson.groupby('time_chunk').apply(lambda x: x['avg_DP'].corr(x['DP'], method = 'spearman')).reset_index()
        # Rename the columns for clarity
        correlations.columns = ['time_chunk', 'spearman_correlation']
        correlations['filename'] = filename
        hup_corrs = pd.concat([hup_corrs, correlations])


    musc_corrs = pd.DataFrame()
    for filename in musc_segment_dp['filename'].unique():
        yo = musc_segment_dp[musc_segment_dp['filename'] == filename]
        pearson = yo.groupby(['time_chunk','channel_label']).mean('avg_DP').reset_index()
        yo_ovr = MUSC_Overall_DP[MUSC_Overall_DP['filename'] == filename]
        pearson = pearson.merge(yo_ovr[['channel_label','DP']], on = 'channel_label', how='inner') 
        correlations = pearson.groupby('time_chunk').apply(lambda x: x['avg_DP'].corr(x['DP'], method = 'spearman')).reset_index()
        # Rename the columns for clarity
        correlations.columns = ['time_chunk', 'spearman_correlation']
        correlations['filename'] = filename
        musc_corrs = pd.concat([musc_corrs, correlations])

    all_corrs = pd.concat([musc_corrs, hup_corrs])

    group_correlation = []
    for filename in all_corrs['filename'].unique():
        yo = all_corrs[all_corrs['filename'] == filename]
        group_correlation.append(yo['spearman_correlation'].median())

    permuted_stats.append(group_correlation)

np.save('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/permutation_test/1000_permuted_DP.npy',permuted_stats)
