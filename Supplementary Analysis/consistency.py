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

#%%
#we may want to load in the HUP_files and MUSC_files

# %%
#lets calculate Overall DP for each patient in each cohort

# def process_patient_HUP(pt_id):
#     print(pt_id)
#     yo = HUP_cons[HUP_cons['filename'] == pt_id]
#     patient_df = yo[['channel_label', 'DP']].groupby('channel_label').agg(['mean', 'count']).reset_index()
#     patient_df.columns = ['channel_label', 'DP', 'count']
#     patient_df['filename'] = pt_id

#     # Calculate cumulative sum and percentage
#     patient_df = patient_df.sort_values(by='count', ascending=False)
#     patient_df['cumulative_count'] = patient_df['count'].cumsum()
#     total_count = patient_df['count'].sum()
#     patient_df['cumulative_percentage'] = patient_df['cumulative_count'] / total_count

#     # Filter to keep channel_labels that make up 90% of the spikes
#     patient_df = patient_df[patient_df['cumulative_percentage'] <= 0.90]

#     return patient_df

# # Parallel processing for HUP data
# with ProcessPoolExecutor() as executor:
#     HUP_Overall_DP = pd.concat(executor.map(process_patient_HUP, HUP_cons['filename'].unique()), ignore_index=True)

# HUP_Overall_DP.to_csv('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/hup_overall_dp.csv')

# def process_patient_MUSC(pt_id):
#     print(pt_id)
#     yo = MUSC_cons[MUSC_cons['filename'] == pt_id]
#     patient_df = yo[['channel_label', 'DP']].groupby('channel_label').agg(['mean', 'count']).reset_index()
#     patient_df.columns = ['channel_label', 'DP', 'count']
#     patient_df['filename'] = pt_id

#     # Calculate cumulative sum and percentage
#     patient_df = patient_df.sort_values(by='count', ascending=False)
#     patient_df['cumulative_count'] = patient_df['count'].cumsum()
#     total_count = patient_df['count'].sum()
#     patient_df['cumulative_percentage'] = patient_df['cumulative_count'] / total_count

#     # Filter to keep channel_labels that make up 90% of the spikes
#     patient_df = patient_df[patient_df['cumulative_percentage'] <= 0.90]

#     return patient_df

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
print('75% - ', np.percentile(group_correlation, 75))
print('25% - ', np.percentile(group_correlation, 25))


#%%
#add a pt_id, and then average those with multiple pt_id's
cor_coefs_w_id = pd.DataFrame(data = {'corr':group_correlation, 'filenames':all_corrs['filename'].unique()})

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


MUSC_names = MUSC_thresh[['pt_id','filename']].drop_duplicates()
HUP_names = HUP_thresh[['pt_id','filename']].drop_duplicates()
all_ids = pd.concat([MUSC_names, HUP_names]).reset_index(drop=True).rename(columns = {'filename':'filenames'})

cor_coefs_w_id = cor_coefs_w_id.merge(all_ids,on='filenames', how='inner')
cor_coefs_w_id['pt_id'] = cor_coefs_w_id['pt_id'].str.replace('3T_MP0', '').str.replace('HUP', '')
cor_coefs_w_id['pt_id'] = cor_coefs_w_id['pt_id'].astype(int)

# %%

#use the pearson_df dataframe to merge SOZ's

pearson_df = pd.read_csv('../dataset/ML_data/MUSC/pooled_pearson_all_norm.csv', index_col = 0) #THIS GIVES AUC = 0.8 

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
