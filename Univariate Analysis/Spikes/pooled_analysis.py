import pandas as pd
import numpy as np
import scipy.stats as stats
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import warnings
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

list_of_feats = ['spike_rate','recruitment_latency_thresh','decay_amp','sharpness','linelen','slow_amp', 'rise_amp','spike_width','slow_width']

#%%
#Load in dataframe with all calculated features in our study
#This contains the mesial-to-lateral correlation across all patients in our study.

pearson_df = pd.read_csv('../../Dataset/spike-correlations.csv', index_col = 0)

#%%
#PEARSON PLOTS MORPHOLOGY
plt.rcParams['font.family'] = 'Arial'
pearson_df['SOZ'] = pearson_df['SOZ'].astype('category')

# Melt the dataframe to long format
melted_pearson_df = pearson_df.melt(id_vars='SOZ', 
                              value_vars=['rise_amp_corr', 'sharpness_corr', 'spike_width_corr'],
                              var_name='Metric', value_name='Value')

# Set up the matplotlib figure
fig, ax = plt.subplots(1,1, figsize=(10,6))
my_palette = {1:'#E64B35FF', 3:'#7E6148FF', 2:'#3C5488FF'}
fig_args = {'x':'Metric',
            'y':'Value',
            'hue':'SOZ',
            'data':melted_pearson_df,
            'order': ['rise_amp_corr','sharpness_corr','spike_width_corr'],
            'hue_order':[1,2,3]}

significanceComparisons = [
                        (('rise_amp_corr',1), ('rise_amp_corr',3)),
                        (('rise_amp_corr',1), ('rise_amp_corr',2)),
                        (('rise_amp_corr',2), ('rise_amp_corr',3)),
                        (('sharpness_corr',1), ('sharpness_corr',3)),
                        (('sharpness_corr',1), ('sharpness_corr',2)),
                        (('sharpness_corr',2), ('sharpness_corr',3)),
                        (('spike_width_corr',1),('spike_width_corr',3)),
                        (('spike_width_corr',1),('spike_width_corr',2)),
                        (('spike_width_corr',2),('spike_width_corr',3))
                        ]

sns.boxplot(ax=ax, showfliers = False, palette=my_palette, **fig_args)
sns.stripplot(ax =ax, color = 'k', alpha = 0.5, dodge=True, jitter=True, size=5, **fig_args)

annotator = Annotator(ax=ax, pairs=significanceComparisons,
                    **fig_args, plot='boxplot')

# Assign Mann-Whitney U test p-values to the annotator
test = 'Mann-Whitney'
comp = 'BH' #benjamani hochberg correction
configuration = {'test':test,
                    'comparisons_correction':comp,
                    'text_format':'star',
                    'loc':'inside',
                    'verbose':True}
annotator.configure(**configuration)
annotator.apply_and_annotate()

# Set plot title and labels
plt.title('Distribution of Pearson Correlation by SOZ Type', fontsize = 20)

new_labels = ['Rise Amplitude', 'Sharpness', 'Width']
ax.set_xticklabels(new_labels, fontsize=12)
plt.ylabel('Correlation Coef.', fontsize = 12)
ax.set(xlabel=None)

# Update the legend to prevent duplication
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:3], ['mTLE', 'Neo', 'Other'], loc='upper right', fontsize=12, bbox_to_anchor=(1.05, 1))

# Show the plot
sns.despine()
plt.axhline(y=0, color='k', linestyle='--')
plt.savefig('../../Results/FIG3-Morphology.pdf')
plt.show()

all_effect_szs = []
for comparison in significanceComparisons:
    metric, soz1 = comparison[0]
    _, soz2 = comparison[1]
    group1 = melted_pearson_df[(melted_pearson_df['Metric'] == metric) & (melted_pearson_df['SOZ'] == soz1)]['Value']
    group2 = melted_pearson_df[(melted_pearson_df['Metric'] == metric) & (melted_pearson_df['SOZ'] == soz2)]['Value']
    all_effect_szs.append([metric, soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)

#%%
#PEARSON PLOTS OTHER MORPHOLOGY
plt.rcParams['font.family'] = 'Arial'
pearson_df['SOZ'] = pearson_df['SOZ'].astype('category')

new_metrics = ['linelen_corr', 'decay_amp_corr', 'slow_width_corr', 'slow_amp_corr', 
               'rise_slope_corr', 'decay_slope_corr', 'average_amp_corr', 
               'rise_duration_corr', 'decay_duration_corr']

# Melt the dataframe to long format
melted_pearson_df = pearson_df.melt(id_vars='SOZ', 
                              value_vars=new_metrics,
                              var_name='Metric', value_name='Value')

# Set up the matplotlib figure
fig, ax = plt.subplots(1,1, figsize=(15,6))

my_palette = {1:'#E64B35FF', 3:'#7E6148FF', 2:'#3C5488FF'}
fig_args = {'x':'Metric',
            'y':'Value',
            'hue':'SOZ',
            'data':melted_pearson_df,
            'order': new_metrics,
            'hue_order':[1,2,3]}

# Generate significance comparisons for all pairs of SOZ values for each metric
significanceComparisons = []
for metric in new_metrics:
    significanceComparisons.extend([
        ((metric, 1), (metric, 3)),
        ((metric, 1), (metric, 2)),
        ((metric, 2), (metric, 3))
    ])

sns.boxplot(ax=ax, showfliers = False, palette=my_palette, **fig_args)
sns.stripplot(ax =ax, color = 'k', alpha = 0.5, dodge=True, jitter=True, size=5, **fig_args)

annotator = Annotator(ax=ax, pairs=significanceComparisons,
                    **fig_args, plot='boxplot')

# Assign Mann-Whitney U test p-values to the annotator
test = 'Mann-Whitney'
# comp = 'BH' #benjamani hochberg correction
configuration = {'test':test,
                    'comparisons_correction':None,
                    'text_format':'star',
                    'loc':'inside',
                    'verbose':True,
                    'hide_non_significant':True}
annotator.configure(**configuration)
annotator.apply_and_annotate()

# Set plot title and labels
plt.title('Distribution of Pearson Correlation by SOZ Type', fontsize = 28)

new_labels = ['Linelength', 'Decay Amplitude', 'Slow Wave Width', 'Slow Wave Amplitude', 
               'Rising Slope', 'Decay Slope', 'Average Amplitude', 
               'Rising Spike Width', 'Decay Spike Width']
ax.set_xticklabels(new_labels, fontsize=20, rotation = 45, ha = 'right')
plt.ylabel('Correlation Coef.', fontsize = 20)
ax.set(xlabel=None)

# Update the legend to prevent duplication
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:3], ['mTLE', 'Neo', 'Other'], loc='upper right', fontsize=20, bbox_to_anchor=(1.05, 1))

# Show the plot
sns.despine()
plt.axhline(y=0, color='k', linestyle='--')
plt.savefig(f'../../Results/Supplement-FigS5-Morphology.pdf')
plt.show()

all_effect_szs = []
for comparison in significanceComparisons:
    metric, soz1 = comparison[0]
    _, soz2 = comparison[1]
    group1 = melted_pearson_df[(melted_pearson_df['Metric'] == metric) & (melted_pearson_df['SOZ'] == soz1)]['Value']
    group2 = melted_pearson_df[(melted_pearson_df['Metric'] == metric) & (melted_pearson_df['SOZ'] == soz2)]['Value']
    all_effect_szs.append([metric, soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)


# %%
####################################
## NOW CREATE PLOTS FOR SPIKE RATE 
####################################

plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'
test = 'Mann-Whitney'
plt.axhline(y=0, color='k', linestyle='--')

my_palette = {1:'#E64B35FF', 3:'#7E6148FF', 2:'#3C5488FF'}
pairs=[(1, 2), (2,3), (1,3)]
order = [1,2,3]
ax = sns.boxplot(x='SOZ', y='spike_rate_corr', data=pearson_df, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="SOZ", y="spike_rate_corr", data=pearson_df, color="black", alpha=0.5)
annotator = Annotator(ax, pairs, data=pearson_df, x="SOZ", y="spike_rate_corr", order=order)
annotator.configure(test=test, text_format='star', loc='inside', verbose = True, comparisons_correction='Benjamini-Hochberg')
annotator.apply_and_annotate()

plt.xlabel('SOZ Type', fontsize=12)
plt.ylabel('Pearson Correlation', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(3), ['Mesial Temporal', 'Neo', 'Other'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title('Spike Rate Directionality', fontsize=16)
sns.despine()
plt.savefig(f'../../Results/Fig2-SpikeRate.pdf')
plt.show()

all_effect_szs = []
for comparison in pairs:
    # print(comparison)
    soz1 = comparison[0]
    soz2 = comparison[1]
    group1 = pearson_df[pearson_df['SOZ'] == soz1]['spike_rate_corr']
    group2 = pearson_df[pearson_df['SOZ'] == soz2]['spike_rate_corr']
    all_effect_szs.append([metric, soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)


#%%
#############################
# NOW CREATE PLOTS FOR TIMING
#############################

plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'
test = 'Mann-Whitney'
plt.axhline(y=0, color='k', linestyle='--')

my_palette = {1:'#E64B35FF', 3:'#7E6148FF', 2:'#3C5488FF'}
pairs=[(1, 2), (2,3), (1,3)]
order = [1,2,3]
ax = sns.boxplot(x='SOZ', y='recruitment_latency_thresh_corr', data=pearson_df, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="SOZ", y="recruitment_latency_thresh_corr", data=pearson_df, color="black", alpha=0.5)
annotator = Annotator(ax, pairs, data=pearson_df, x="SOZ", y="recruitment_latency_thresh_corr", order=order)
annotator.configure(test=test, text_format='star', loc='inside', verbose = True, comparisons_correction='Benjamini-Hochberg')
annotator.apply_and_annotate()

plt.xlabel('SOZ Type', fontsize=12)
plt.ylabel('Pearson Correlation', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(3), ['Mesial Temporal', 'Neo', 'Other'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title('Spike Timing Directionality', fontsize=16)
sns.despine()
plt.savefig(f'../../Results/Fig2-Timing.pdf')
plt.show()

all_effect_szs = []
for comparison in pairs:
    soz1 = comparison[0]
    soz2 = comparison[1]
    group1 = pearson_df[pearson_df['SOZ'] == soz1]['recruitment_latency_thresh_corr']
    group2 = pearson_df[pearson_df['SOZ'] == soz2]['recruitment_latency_thresh_corr']
    all_effect_szs.append([metric, soz1, soz2, cohend(group1, group2)])

print(all_effect_szs)


# %%
####################################
# look for grouped stats (high level)
####################################

#######
#Morphology

from scipy.stats import f_oneway, levene, shapiro, kruskal

melted_corr_df = melted_pearson_df

x = melted_corr_df[melted_corr_df['Metric'] == 'rise_amp_corr']
# _,p1 = shapiro(x[x['SOZ'] == 1]['Value'])
# _,p2 = shapiro(x[x['SOZ'] == 2]['Value'])
# _,p3 = shapiro(x[x['SOZ'] == 3]['Value'])
# print(f"Shapiro-Wilk test p-values: group1={p1}, group2={p2}, group3={p3}")
# _, p_levene = levene(x[x['SOZ'] == 1]['Value'], x[x['SOZ'] == 2]['Value'], x[x['SOZ'] == 3]['Value'])
# print(f"Levene's test p-value: {p_levene}")
k1,tp1 = kruskal(x[x['SOZ'] == 1]['Value'],x[x['SOZ'] == 2]['Value'],x[x['SOZ'] == 3]['Value'])
print('decay amp p:',tp1)
print('K = ', k1)

print('---------------------')

x = melted_corr_df[melted_corr_df['Metric'] == 'sharpness_corr']
# _,p1 = shapiro(x[x['SOZ'] == 1]['Value'])
# _,p2 = shapiro(x[x['SOZ'] == 2]['Value'])
# _,p3 = shapiro(x[x['SOZ'] == 3]['Value'])
# print(f"Shapiro-Wilk test p-values: group1={p1}, group2={p2}, group3={p3}")
# _, p_levene = levene(x[x['SOZ'] == 1]['Value'], x[x['SOZ'] == 2]['Value'], x[x['SOZ'] == 3]['Value'])
# print(f"Levene's test p-value: {p_levene}")
k3,tp3 = kruskal(x[x['SOZ'] == 1]['Value'],x[x['SOZ'] == 2]['Value'],x[x['SOZ'] == 3]['Value'])
print('sharpness corr',tp3)
print('K = ', k3)

print('---------------------')


x = melted_corr_df[melted_corr_df['Metric'] == 'spike_width_corr']
# _,p1 = shapiro(x[x['SOZ'] == 1]['Value'])
# _,p2 = shapiro(x[x['SOZ'] == 2]['Value'])
# _,p3 = shapiro(x[x['SOZ'] == 3]['Value'])
# print(f"Shapiro-Wilk test p-values: group1={p1}, group2={p2}, group3={p3}")
# _, p_levene = levene(x[x['SOZ'] == 1]['Value'], x[x['SOZ'] == 2]['Value'], x[x['SOZ'] == 3]['Value'])
# print(f"Levene's test p-value: {p_levene}")
k2,tp2 = kruskal(x[x['SOZ'] == 1]['Value'],x[x['SOZ'] == 2]['Value'],x[x['SOZ'] == 3]['Value'])
print('slow wave amp p',tp2)
print('K = ', k2)
print('---------------------')

# Collect all p-values
p_values = np.array([tp1, tp3, tp2])

from statsmodels.stats.multitest import multipletests
# Apply FDR correction
rejected, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print('rise_amp_corr,sharpness_corr,spike_width_corr')
print(rejected)
print(p_values_corrected)

print('---------------------')

#%%
#TIMING + RATE

rate = pearson_df[['spike_rate_corr','SOZ']]
latency = pearson_df[['SOZ','recruitment_latency_thresh_corr']]

#change if you want anova, but really no different in results
print("SPIKE RATE:")
print(kruskal(rate[rate['SOZ'] == 1]['spike_rate_corr'], rate[rate['SOZ'] == 2]['spike_rate_corr'],rate[rate['SOZ'] == 3]['spike_rate_corr']))
print("SPIKE TIMING:")
print(kruskal(latency[latency['SOZ'] == 1]['recruitment_latency_thresh_corr'], latency[latency['SOZ'] == 2]['recruitment_latency_thresh_corr'],latency[latency['SOZ'] == 3]['recruitment_latency_thresh_corr']))

# %%
#Supplementary figure X

morphology_df = pearson_df.drop(columns = {"spike_rate_corr","recruitment_latency_thresh_corr","SOZ","pt_id"})
morphology_df = morphology_df.rename(columns = {
    'slow_width_corr': 'Slow Wave Width',
    'spike_width_corr': 'Spike Width',
    'slow_amp_corr': 'Slow Wave Amplitude',
    'sharpness_corr': 'Spike Sharpness',
    'rise_amp_corr': 'Rising Amplitude',
    'decay_amp_corr': 'Decay Amplitude',
    'linelen_corr': 'Line Length'
})

plt.rcParams['font.family'] = 'Arial'

# Assuming morphology_df is your DataFrame with the feature data
corr_matrix = morphology_df.drop(columns = ['Spike Width','Slow Wave Amplitude']).corr()

# Set up the matplotlib figure
fig = plt.figure(figsize=(14, 10))

# Create the clustermap
g = sns.clustermap(corr_matrix,
                   cmap='viridis',
                   center=0,
                   vmin=-1,
                   vmax=1,
                   dendrogram_ratio=(0.2, 0),
                   cbar_pos=(0.02, 0.7, 0.05, 0.18),
                   tree_kws={'color': 'black'},
                   figsize=(14, 10))

# Adjust the layout
g.ax_row_dendrogram.set_visible(True)
g.ax_col_dendrogram.set_visible(False)

# Move the main axes (heatmap) to the right
g.ax_heatmap.set_position([0.3, 0.1, 0.6, 0.8])

# Move the row dendrogram to the left
g.ax_row_dendrogram.set_position([0.05, 0.1, 0.15, 0.8])

# Remove default y-axis labels
g.ax_heatmap.set_yticks([])

# Add centered y-axis labels
for i, label in enumerate(corr_matrix.index):
    g.ax_heatmap.text(-0.05, i + 0.5, label, 
                      va='center', ha='right',
                      transform=g.ax_heatmap.get_yaxis_transform())

# Rotate x-axis labels
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, ha='center')

# Adjust colorbar position
g.cax.set_position([0.92, 0.1, 0.02, 0.8])

# Add title
plt.suptitle('Correlation between Morphology Features', fontsize=30, fontweight='bold', y=1.02)

# Adjust layout
plt.tight_layout(rect=[0, 0, 0.95, 0.95])
plt.savefig('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/figures/MUSC+HUP/official/choosing_feats_noslow.pdf')

plt.show()

# %%
