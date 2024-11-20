#%%
import pandas as pd
import numpy as np
import scipy.stats as stats
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')
import ast

# Import custom functions
import sys, os
code_v2_path = os.path.dirname('../../tools/Spike-Detector/')
sys.path.append(code_v2_path)
from get_iEEG_data import *
from spike_detector import *
from iEEG_helper_functions import *
from spike_morphology_v2 import *

# code_path = os.path.dirname('../../tools/functions/')
# sys.path.append(code_path)
# from ied_fx_v3 import *

#HFER per time
type = '11s-hfer-'
hup_ei = pd.read_csv('../../Dataset/hfer_HUP_11s.csv')
musc_ei = pd.read_csv('../../Dataset/hfer_MUSC_11s.csv')

#%% 

###################################
# GET MUSC 
###################################

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

grouped = musc_ei.groupby(['MUSC_ID','name'])['EI'].median().reset_index()
# grouped = musc_ei.groupby(['MUSC_ID','name'])['EI'].quantile(0.8).reset_index()

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

musc_files = pd.read_csv('../../Dataset/MUSC_files.csv', index_col = 0)

all_spikes = grouped.merge(musc_files[['pt_id','channel_label']], on = 'pt_id', how = 'inner')

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
all_spikes['channel_label_x'] = all_spikes['channel_label_x'].str.replace('L|R|A|H|P', '', regex = True)

musc_sozs = pd.read_csv('../../Dataset/MUSC_sozs.csv', index_col = 0)
uniques_sozs = musc_sozs[['pt_id','region']].drop_duplicates()
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


#%%
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


# grouped = hup_ei.groupby(['hupID','name'])['EI'].quantile(0.8).reset_index()
grouped = hup_ei.groupby(['hupID','name'])['EI'].median().reset_index()

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

hup_files = pd.read_csv('../../Dataset/HUP_files.csv', index_col = 0)

all_spikes = grouped.merge(hup_files[['pt_id','channel_label']], on = 'pt_id', how = 'inner')

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
all_spikes['channel_label_x'] = all_spikes['channel_label_x'].str.replace('L|R|A|H|B|C|D', '', regex = True)

hup_sozs = pd.read_csv('../../Dataset/SOZ_hup.csv', index_col = 0)
uniques_sozs = hup_sozs[['pt_id','SOZ']].drop_duplicates()
# merge them
all_spikes = all_spikes.merge(uniques_sozs, on = 'pt_id', how='inner')
all_spikes = all_spikes.drop(columns = 'channel_label_y')

hup_all_spikes = all_spikes


#%% 
#combine them
all_spikes = pd.concat([hup_all_spikes, musc_all_spikes])

# %%
#concatenate mesial_temp_spikes_avg and non_mesial_temp_spikes_avg
def quantile_75(x):
    return np.percentile(x, 75)


all_spikes_avg = all_spikes.pivot_table(index=['pt_id','SOZ'], columns='channel_label_x', values="EI", aggfunc='median')
all_spikes_avg = all_spikes_avg.reindex(columns=['1','2','3','4','5','6','7','8','9','10','11','12'])

def remove_rows_by_index(pivot_table, index_values_to_remove):
    """
    Remove rows from a pivot table based on index values.

    Args:
    - pivot_table (DataFrame): The pivot table to filter.
    - index_values_to_remove (str or list): Index value(s) to remove.

    Returns:
    - DataFrame: The filtered pivot table.
    """
    if isinstance(index_values_to_remove, str):
        index_values_to_remove = [index_values_to_remove]

    mask = ~pivot_table.index.get_level_values('SOZ').isin(index_values_to_remove)
    return pivot_table[mask]

#look to move some SOZ's around:
soz_to_remove = ['temporal']
all_spikes_avg = remove_rows_by_index(all_spikes_avg, soz_to_remove)

#probably change the Frontal SOZ to other? 
if 'frontal' in all_spikes_avg.index.get_level_values(1):
    all_spikes_avg.rename(index={'frontal': 'other cortex'}, inplace=True, level = 'SOZ')

#reorder all_spikes_avg, so that is_mesial is decesending
all_spikes_avg = all_spikes_avg.sort_values(by=['SOZ', 'pt_id'], ascending=[True, True])

# %% PLOT

sns.set_style('ticks')
plt.figure(figsize=(20,20))

# sns.heatmap(all_spikes_avg, cmap='viridis', alpha = 1)
sns.heatmap(all_spikes_avg, cmap = 'viridis', alpha = 1)
plt.xlabel('Channel Number', fontsize=20)
plt.ylabel('Patient ID', fontsize=20)
plt.title(f'75% EI by Channel and Patient', fontsize=24)
#change y-tick labels to only be the first element in the index, making the first 25 red and the rest black
plt.yticks(np.arange(0.5, len(all_spikes_avg.index), 1), all_spikes_avg.index.get_level_values(0), fontsize=13)

#in all_spikes_avg, get the number of 'temporal neocortical' patients
temp_neocort_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 'temporal neocortical'])
#in all_spikes_avg, get the number of 'temporal' patients
temp_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 'temporal'])
#same for other cortex
other_cortex_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 'other cortex'])
#same for mesial temporal
mesial_temp_pts = len(all_spikes_avg[all_spikes_avg.index.get_level_values(1) == 'mesial temporal'])

plt.axhline(mesial_temp_pts, color='white', linewidth=2.5)
plt.axhline(mesial_temp_pts+other_cortex_pts, color='white', linewidth=1.5, linestyle = '--')
plt.axhline(mesial_temp_pts+other_cortex_pts+temp_pts, color='white', linewidth=1.5, linestyle = '--')
#create a list of 48 colors
colors = ['#E64B35FF']*mesial_temp_pts + ['#7E6148FF']*other_cortex_pts + ['#00A087FF']*temp_pts + ['#3C5488FF']*temp_neocort_pts
for ytick, color in zip(plt.gca().get_yticklabels(), colors):
    ytick.set_color(color)

#add a legend that has red == mesial temporal patients and black == non-mesial temporal patients
import matplotlib.patches as mpatches
mesial_patch = mpatches.Patch(color='#E64B35FF', label='Mesial Temporal Patients')
other_patch = mpatches.Patch(color='#7E6148FF', label='Other Cortex Patients')
# temporal_patch = mpatches.Patch(color='#00A087FF', label='Temporal Patients')
neocort_patch = mpatches.Patch(color='#3C5488FF', label='Temporal Neocortical Patients')

plt.legend(handles=[mesial_patch, other_patch, neocort_patch], loc='upper right')
sns.despine()

# %%
#find the spearman correlation of each row in all_spikes_avg
#initialize a list to store the spearman correlation
channel_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
channel_labels = [int(x) for x in channel_labels]
spearman_corr = []
label = []
for row in range(len(all_spikes_avg)):
    # #if the row has less than 8 channels, omit from analysis
    # if len(all_spikes_avg.iloc[row].dropna()) < 8:
    #     continue
    spearman_corr.append(stats.spearmanr(channel_labels,all_spikes_avg.iloc[row].to_list(), nan_policy='omit'))
    label.append(all_spikes_avg.index[row]) 

corr_df = pd.DataFrame(spearman_corr, columns=['correlation', 'p-value'])
corr_df['SOZ'] = [x[1] for x in label]
corr_df['pt_id'] = [x[0] for x in label]

# find the pearson correlation of each row in all_spikes_avg
# initialize a list to store the spearman correlation
pearson_corr = []
p_label = []
for row in range(len(all_spikes_avg)):
    # #if the row has less than 8 channels, omit from analysis
    # if len(all_spikes_avg.iloc[row].dropna()) < 8:
    #     continue
    gradient = all_spikes_avg.iloc[row].to_list()
    channel_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
    channel_labels = [int(x) for x in channel_labels]
    # for each nan in the graident list, remove the corresponding channel_labels
    list_to_remove = []
    for i in range(len(channel_labels)):
        if np.isnan(gradient[i]):
            list_to_remove.append(i)

    #remove list_to_remove from channel_labels and gradient
    channel_labels = [i for j, i in enumerate(channel_labels) if j not in list_to_remove]
    gradient = [i for j, i in enumerate(gradient) if j not in list_to_remove]

    pearson_corr.append(stats.pearsonr(channel_labels,gradient))
    p_label.append(all_spikes_avg.index[row])

pearson_df = pd.DataFrame(pearson_corr, columns=['correlation', 'p-value'])
pearson_df['SOZ'] = [x[1] for x in label]
pearson_df['pt_id'] = [x[0] for x in label]

### New METRIC
coeff_5 = []
coeff_10 = []
firstonly = []
m_label = []
for row in range(len(all_spikes_avg)):
    # #if the row has less than 8 channels, omit from analysis
    # if len(all_spikes_avg.iloc[row].dropna()) < 8:
    #     continue
    gradient = all_spikes_avg.iloc[row].to_list()
    channel_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
    channel_labels = [int(x) for x in channel_labels]
    # for each nan in the graident list, remove the corresponding channel_labels
    list_to_remove = []
    for i in range(len(channel_labels)):
        if np.isnan(gradient[i]):
            list_to_remove.append(i)

    #remove list_to_remove from channel_labels and gradient
    channel_labels = [i for j, i in enumerate(channel_labels) if j not in list_to_remove]
    gradient = [i for j, i in enumerate(gradient) if j not in list_to_remove]
    m_label.append(all_spikes_avg.index[row])

    # coeff_5.append((gradient[4]-gradient[0])/len(gradient[0:5]))
    coeff_10.append((gradient[-1]-gradient[0])/len(gradient))
    firstonly.append(gradient[0])

slope_df = pd.DataFrame(data = coeff_5, columns = ['coef5'])
slope_df['coef10'] = coeff_10
slope_df['first_value'] = firstonly
slope_df['SOZ'] = [x[1] for x in m_label]
slope_df['pt_id'] = [x[0] for x in m_label]

#remove the temporal patients for this plot (corr_df and pearson_df)
# corr_df = corr_df[corr_df['SOZ'] != 'temporal']
# pearson_df = pearson_df[pearson_df['SOZ'] != 'temporal']

def soz_assigner(row):
    if row['SOZ'] == 'temporal neocortical':
        return int(2)
    elif row['SOZ'] == 'other cortex':
        return int(2)
    elif row['SOZ'] == 'mesial temporal':
        return int(1)
    else:
        return None

corr_df['SOZ'] = corr_df.apply(soz_assigner, axis = 1)
pearson_df['SOZ'] = pearson_df.apply(soz_assigner, axis = 1)
slope_df['SOZ'] = slope_df.apply(soz_assigner, axis = 1)

#%%
#find the spearman correlation of each row in all_spikes_avg
#initialize a list to store the spearman correlation
channel_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
channel_labels = [int(x) for x in channel_labels]
spearman_corr = []
label = []
for row in range(len(all_spikes_avg)):
    # #if the row has less than 8 channels, omit from analysis
    # if len(all_spikes_avg.iloc[row].dropna()) < 8:
    #     continue
    spearman_corr.append(stats.spearmanr(channel_labels,all_spikes_avg.iloc[row].to_list(), nan_policy='omit'))
    label.append(all_spikes_avg.index[row]) 

corr_df = pd.DataFrame(spearman_corr, columns=['correlation', 'p-value'])
corr_df['SOZ'] = [x[1] for x in label]
corr_df['pt_id'] = [x[0] for x in label]

# find the pearson correlation of each row in all_spikes_avg
# initialize a list to store the spearman correlation
pearson_corr = []
p_label = []
for row in range(len(all_spikes_avg)):
    # #if the row has less than 8 channels, omit from analysis
    # if len(all_spikes_avg.iloc[row].dropna()) < 8:
    #     continue
    gradient = all_spikes_avg.iloc[row].to_list()
    channel_labels = ['1','2','3','4','5','6','7','8','9','10','11','12']
    channel_labels = [int(x) for x in channel_labels]
    # for each nan in the graident list, remove the corresponding channel_labels
    list_to_remove = []
    for i in range(len(channel_labels)):
        if np.isnan(gradient[i]):
            list_to_remove.append(i)

    #remove list_to_remove from channel_labels and gradient
    channel_labels = [i for j, i in enumerate(channel_labels) if j not in list_to_remove]
    gradient = [i for j, i in enumerate(gradient) if j not in list_to_remove]

    pearson_corr.append(stats.pearsonr(channel_labels,gradient))
    p_label.append(all_spikes_avg.index[row])

pearson_df = pd.DataFrame(pearson_corr, columns=['correlation', 'p-value'])
pearson_df['SOZ'] = [x[1] for x in label]
pearson_df['pt_id'] = [x[0] for x in label]

#remove the temporal patients for this plot (corr_df and pearson_df)
# corr_df = corr_df[corr_df['SOZ'] != 'temporal']
# pearson_df = pearson_df[pearson_df['SOZ'] != 'temporal']

def soz_assigner(row):
    if row['SOZ'] == 'temporal neocortical':
        return int(2)
    elif row['SOZ'] == 'other cortex':
        return int(3)
    elif row['SOZ'] == 'mesial temporal':
        return int(1)
    else:
        return None

corr_df['SOZ'] = corr_df.apply(soz_assigner, axis = 1)
pearson_df['SOZ'] = pearson_df.apply(soz_assigner, axis = 1)

#%%

#Pearson Correlation PLOTS
#create a boxplot comparing the distribution of correlation across SOZ types
plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'

try:
    my_palette = {1:'#E64B35FF', 3:'#7E6148FF', 2:'#3C5488FF'}
except: 
    my_palette= {'1':'#E64B35FF', '3':'#7E6148FF', '2':'#3C5488FF'}
# my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
pairs=[(1, 2),(1,3), (2,3)]
order = [1,2,3]

ax = sns.boxplot(x='SOZ', y='correlation', data=pearson_df, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="SOZ", y="correlation", data=pearson_df, color="black", alpha=0.5)
plt.axhline(y=0, color='k', linestyle='--')
annotator = Annotator(ax, pairs, data=pearson_df, x="SOZ", y="correlation", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction='BH', verbose = True,hide_non_significant=True)
annotator.apply_and_annotate()

plt.xlabel('SOZ Type', fontsize=12)
plt.ylabel('Pearson Correlation', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(3), ['Mesial Temporal', 'Neocortical', 'Other Cortex'], fontsize = 12)
plt.yticks(np.arange(-1, 1.1, 0.5), fontsize=12)  # This sets ticks at -1, -0.5, 0, 0.5, 1

#part to change
plt.title(f'HFER Directionality', fontsize=16)
sns.despine()
plt.ylim([-1,1])

plt.savefig(f'../../Results/Fig4-{type}_HFER.pdf')
plt.show()
all_effect_szs = []
for comparison in pairs:
    # print(comparison)
    soz1 = comparison[0]
    soz2 = comparison[1]
    group1 = pearson_df[pearson_df['SOZ'] == soz1]['correlation']
    group2 = pearson_df[pearson_df['SOZ'] == soz2]['correlation']

    all_effect_szs.append(['Pearson Corr EI', soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)
# %%
#######
#EI ANOVA
from scipy.stats import kruskal, shapiro, levene, f_oneway

ei = pearson_df[['SOZ','correlation']]

#change if you want anova, but really no different in results
print("Pearson Correlations")
print(kruskal(ei[ei['SOZ'] == 1]['correlation'], ei[ei['SOZ'] == 2]['correlation'],ei[ei['SOZ'] == 3]['correlation']))
