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

# Parallel processing for HUP data
with ProcessPoolExecutor() as executor:
    overall_HFER = pd.concat(executor.map(process_patients, all_spikes['pt_id'].unique()), ignore_index=True)

overall_HFER.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/overall_hfer_11s.csv')

# overall_HFER = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/overall_hfer_11s.csv', index_col = 0)
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

file = all_corrs['pt_id'].unique()[17]

# for file in filenames:
correlations = all_corrs[all_corrs['pt_id'] == file]
# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(correlations['ictal_start_time'], correlations['spearman_correlation'], color = 'r', label = 'Corr of Segment to Overall')
plt.axhline(y=correlations['spearman_correlation'].median(), linestyle='-', color='k', label='Median = HFER-Stability')
plt.title(f'HFER Stability Calculation for {file}')
plt.xlabel('Seizure time (sec)')
plt.ylabel('Corr Coef')
plt.ylim([0,1])
plt.legend()
plt.grid(True)

group_correlation = []
for filename in all_corrs['pt_id'].unique():
    yo = all_corrs[all_corrs['pt_id'] == filename]
    group_correlation.append(yo['spearman_correlation'].median())

print('mean corr:', np.mean((group_correlation)))
print('std corr:', np.std((group_correlation)))

print('median corr:', np.median((group_correlation)))
print('75% - ', np.percentile(group_correlation, 75))
print('25% - ', np.percentile(group_correlation, 25))

#%%

#add a pt_id, and then average those with multiple pt_id's
cor_coefs_w_id = pd.DataFrame(data = {'corr':group_correlation, 'filenames':all_corrs['pt_id'].unique()})

drop_pts = ['HUP093','HUP108','HUP113','HUP114','HUP116','HUP123','HUP087','HUP099','HUP111','HUP121','HUP105','HUP106','HUP107','HUP159'] #These are the patients with less than 8 contacts.
spikes_thresh = pd.read_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/complete_dfs/hup_thresholded.csv', index_col= 0)
HUP_thresh = spikes_thresh[~spikes_thresh['pt_id'].isin(drop_pts)]


#load SOZ corrections
MUSC_sozs = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/MUSC-soz-corrections.xlsx')
MUSC_sozs = MUSC_sozs[MUSC_sozs['Site_1MUSC_2Emory'] == 1]
MUSC_sozs = MUSC_sozs.drop(columns=['Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14'])

#find the patients that should be null, and remove them for the full dataset
nonnan_mask = MUSC_sozs.dropna()
pts_to_remove = nonnan_mask[nonnan_mask['Correction Notes'].str.contains('null')]['ParticipantID'].array

## load the spike data
MUSC_spikes = pd.read_csv('../dataset/complete_dfs/MUSC_thresholded.csv', index_col=0)

#load SOZ corrections
MUSC_sozs = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/MUSC-soz-corrections.xlsx')
MUSC_sozs = MUSC_sozs[MUSC_sozs['Site_1MUSC_2Emory'] == 1]
MUSC_sozs = MUSC_sozs.drop(columns=['Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13','Unnamed: 14'])

#fix SOZ and laterality
MUSC_spikes = MUSC_spikes.merge(MUSC_sozs, left_on = 'pt_id', right_on = 'ParticipantID', how = 'inner')
MUSC_spikes = MUSC_spikes.drop(columns=['ParticipantID','Site_1MUSC_2Emory','IfNeocortical_Location','Correction Notes','lateralization_left','lateralization_right','region'])

#remove the patients that should be NULL for the thresholded dataset
MUSC_spikes = MUSC_spikes[~MUSC_spikes['pt_id'].isin(pts_to_remove)]
MUSC_thresh = MUSC_spikes


MUSC_names = MUSC_thresh[['filename']].drop_duplicates()
MUSC_names = MUSC_names.rename(columns = {'filename':'pt_id'})
HUP_names = HUP_thresh[['pt_id']].drop_duplicates()
all_ids = pd.concat([MUSC_names, HUP_names]).reset_index(drop=True)

all_ids = all_ids.rename(columns = {'pt_id':'filenames'})
cor_coefs_w_id = cor_coefs_w_id.merge(all_ids,on='filenames', how='inner')
cor_coefs_w_id['filenames'] = cor_coefs_w_id['filenames'].str.replace('3T_MP0', '').str.replace('HUP', '').str.replace('_D01', '').str.replace('_D02', '').str.replace('_D03', '').str.replace('MP0', '')
cor_coefs_w_id['filenames'] = cor_coefs_w_id['filenames'].astype(int)

# %%

#use the pearson_df dataframe to merge SOZ's

pearson_df = pd.read_csv('../dataset/ML_data/MUSC/pooled_pearson_all_norm.csv', index_col = 0) #THIS GIVES AUC = 0.8 

cor_coefs_w_id = cor_coefs_w_id.rename(columns = {'filenames':'pt_id'})
soz_w_corrs = cor_coefs_w_id.merge(pearson_df[['pt_id','SOZ']], on='pt_id', how='inner')

soz_w_corrs['mean_corr'] = soz_w_corrs.groupby('pt_id')['corr'].transform('mean')

soz_w_corrs = soz_w_corrs.drop_duplicates(subset='pt_id')

# %%

plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'
test = 'Mann-Whitney'
plt.axhline(y=np.mean((group_correlation)), color='k', linestyle='--', label = 'Avg. Cohort DP Stability')

my_palette = {1:'#E64B35FF', 3:'#7E6148FF', 2:'#3C5488FF'}
pairs=[(1, 2), (2,3), (1,3)]
order = [1,2,3]
ax = sns.boxplot(x='SOZ', y='mean_corr', data=soz_w_corrs, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="SOZ", y="mean_corr", data=soz_w_corrs, color="black", alpha=0.5)
annotator = Annotator(ax, pairs, data=soz_w_corrs, x="SOZ", y="mean_corr", order=order)
annotator.configure(test=test, text_format='star', loc='inside', verbose = True, comparisons_correction='Benjamini-Hochberg')
annotator.apply_and_annotate()

plt.xlabel('SOZ Type', fontsize=12)
plt.ylabel('Average Corr Coef', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(3), ['Mesial Temporal', 'Neo', 'Other'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title('DP Stability per SOZ', fontsize=16)
sns.despine()
# plt.legend(loc = 'upper right')
plt.show()

from scipy.stats import kruskal, mannwhitneyu
print(kruskal(soz_w_corrs[soz_w_corrs['SOZ'] == 1]['mean_corr'], soz_w_corrs[soz_w_corrs['SOZ'] == 2]['mean_corr'],soz_w_corrs[soz_w_corrs['SOZ'] == 3]['mean_corr']))

#%%

# ADD OUTCOME DATA

#NEW outcomes - look through them
pec_outcomes = pd.read_csv('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/PEC_outcomes.csv')
pec_outcomes = pec_outcomes.dropna(subset='HUP Number')
pec_outcomes = pec_outcomes.drop(columns = 'Follow Up #1 Status (9-15 months from surgery):')
pec_outcomes.columns = ['rid', 'hup_id', 'procedure','resection_laterality','resection_target','ablation_target',
                        'ablation_specific_target','months_f1','ilae_f1','engel_f1',
                        'months_f2','ilae_f2','engel_f2']

pec_outcomes = pec_outcomes[pec_outcomes['procedure'] == 'resection or laser']
pec_outcomes = pec_outcomes.dropna(subset=['ablation_target'])
other_tokeep = ['amygdala and hippocampus','laser ablation of left temporal lobe, hippocampus, and amygdala','hippocampus',
                'Right laser thermal ablation of the hippocampus and amygdala','left hippocampal ablation','amygdala',
                'left Planum Polare and Amygdala (partial)', 'hippocampal', 'left hippocampal ablation and amygdala cyst biopsy',
                'Right Hippocampus', 'left amygdala-hippocampal ablation', 'left parahippocampal focal cortical dysplasia']

pec_outcomes = pec_outcomes[(pec_outcomes['ablation_specific_target'].isin(other_tokeep)) | (pec_outcomes['ablation_target'] == 'Mesial Temporal')]

#load in the outcome data
redcap = pd.read_excel('/mnt/leif/littlab/users/aguilac/Projects/FC_toolbox/results/mat_output_v2/pt_data/Erin_Carlos_RedCAP_data.xlsx')

#create our 2 search queries. We want to look down LOCATION and SURGERY NOTES to get Mesial Temporal targets
outcomes = redcap[~redcap['Location?'].isna()]
outcomes_2 = redcap[~redcap['Surgery NOTES'].isna()]

#grab only mesial temporal structure targetted interventions
outcomes = outcomes[outcomes['Location?'].str.contains('Mesial|mesial|Hippo|hippo|amygd|Amygd')]
outcomes = outcomes[~outcomes['Procedure?'].str.contains('Resection|resection')]
outcomes = outcomes[outcomes['Location?'].str.contains('Mesial Temporal')]

#now merge them to see what we have
# mesial_pts = pd.concat([outcomes, outcomes_2])
mesial_pts = outcomes
mesial_pts = mesial_pts.drop_duplicates().reset_index(drop = True)
mesial_pts = mesial_pts[~mesial_pts['Outcomes?'].isna()]
mesial_pts = mesial_pts[~mesial_pts['Outcomes?'].str.contains('NONE|OTHER|None')]

#seperate between good and bad
split_outcomes = mesial_pts['Outcomes?'].str.split()
ilae_indices = split_outcomes.apply(lambda x: x[-1] for x in split_outcomes)
mesial_pts['ilae'] = ilae_indices.iloc[:, 1]

def map_ilae_to_go(ilae_value, which):
    if ilae_value in which:
        return 1
    else:
        return 0
    
which = ['1','1a']
mesial_pts['G/O v1'] = mesial_pts['ilae'].apply(lambda x: map_ilae_to_go(x, which))
which = ['1','2','1a']
mesial_pts['G/O v2'] = mesial_pts['ilae'].apply(lambda x: map_ilae_to_go(x, which))

pts_oi = mesial_pts[['HUP_id','G/O v1','G/O v2']]
pts_oi['HUP_id'] = pts_oi['HUP_id'].str.replace('3T_MP0', '').str.replace('HUP', '')
pts_oi = pts_oi.rename(columns = {'HUP_id':'hup_id'})

pts_oi['hup_id'] = pts_oi['hup_id'].astype(int)
pec_outcomes['hup_id'] = pec_outcomes['hup_id'].astype(int)

pec_outcomes = pec_outcomes.merge(pts_oi, on= 'hup_id', how = 'left')
pec_outcomes = pec_outcomes.rename(columns = {'hup_id':'pt_id'})
soz_w_corrs = soz_w_corrs.merge(pec_outcomes, on='pt_id', how = 'inner')

ILAE_conversion = {'Rare seizures (1-3 seizure days per year)': 2,
                   'Seizure free since surgery, no auras':1,
                   'Seizure reduction >50% (but >3 seizure days/year)':3,
                   'Auras only, no other seizures':1,
                   'No change (between 50% seizure reduction and 100% seizure increase': 4
}


Engel_good = ['IA','IB']
soz_w_corrs['ilae_f1'] = soz_w_corrs['ilae_f1'].map(ILAE_conversion)
soz_w_corrs['engel_f1'] = soz_w_corrs['engel_f1'].str.split(':').str[0]
soz_w_corrs['ilae_f2'] = soz_w_corrs['ilae_f2'].map(ILAE_conversion)
soz_w_corrs['engel_f2'] = soz_w_corrs['engel_f2'].str.split(':').str[0]

soz_w_corrs = soz_w_corrs.drop(columns = ['resection_laterality','resection_target','rid','procedure'])

#outcomes that basically say that are based off ILAE1 being good and everything else BAD
soz_w_corrs['outcome1_f1'] = soz_w_corrs.apply(lambda row: 1 if row['ilae_f1'] == 1 else (row['G/O v1'] if pd.isna(row['ilae_f1']) else 0), axis=1)

#outcomes that are based off ILAE 2 being good and everything else BAD
soz_w_corrs['outcome2_f1'] = soz_w_corrs.apply(lambda row: 1 if (row['ilae_f1'] == 1) | (row['ilae_f1'] == 2) else (row['G/O v2'] if pd.isna(row['ilae_f1']) else 0), axis=1)

#outcomes that are based on anything with an engel classification of A to be GOOD, everything else under is BAD.
# Define the function to create outcome3
def calculate_outcome(row):
    if pd.isna(row['engel_f1']):
        return row['G/O v1']
    elif str(row['engel_f1']) in Engel_good:
        return 1
    else:
        return 0

# Apply the function to each row to create the outcome3 column
soz_w_corrs['outcome3_f1'] = soz_w_corrs.apply(calculate_outcome, axis=1)

def calculate_outcome(row):
    if pd.isna(row['engel_f2']):
        return row['G/O v1']
    elif str(row['engel_f2']) in Engel_good:
        return 1
    else:
        return 0

# Apply the function to each row to create the outcome3 column
soz_w_corrs['outcome3_f2'] = soz_w_corrs.apply(calculate_outcome, axis=1)

#PLOT
x = 'outcome3_f2'

plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'
test = 'Mann-Whitney'
plt.axhline(y=np.mean((group_correlation)), color='k', linestyle='--', label = 'Avg. Cohort DP Stability')

my_palette = {1:'#E64B35FF', 0:'#3C5488FF'}
pairs=[(1,0)]
order = [1,0]
ax = sns.boxplot(x=x, y='mean_corr', data=soz_w_corrs, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x=x, y="mean_corr", data=soz_w_corrs, color="black", alpha=0.5, order = order)
annotator = Annotator(ax, pairs, data=soz_w_corrs, x=x, y="mean_corr", order=order)
annotator.configure(test=test, text_format='simple', loc='inside')
annotator.apply_and_annotate()

plt.xlabel('Outcome', fontsize=12)
plt.ylabel('Average Corr Coef', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(2), ['Good', 'Bad'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title('DP Stability per Outcome', fontsize=16)
sns.despine()
plt.show()


# %%
