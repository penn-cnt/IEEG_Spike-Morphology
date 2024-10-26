#%%
#Load data

import pandas as pd
import numpy as np
from scipy import stats

combined_preds = pd.read_csv('../Results/ML_results/LR_elastic/Combined_predictions.csv').rename(columns = {"Predicted_Probability":"combined_predprob"})
interictal_preds = pd.read_csv('../Results/ML_results/LR_elastic/Interictal_predictions.csv').rename(columns = {"Predicted_Probability":"interictal_predprob"})
ictal_preds = pd.read_csv('../Results/ML_results/LR_elastic/Ictal_predictions.csv').rename(columns = {"Predicted_Probability":"ictal_predprob"})

merged_preds_v1 = combined_preds.merge(interictal_preds[['Patient_ID','interictal_predprob']], on='Patient_ID', how = "left")
merged_preds = merged_preds_v1.merge(ictal_preds[['Patient_ID','ictal_predprob']], on = 'Patient_ID', how = "inner")

#Outcomes
pec_outcomes = pd.read_csv('../Dataset/abrv_outcomes.csv', index_col = 0)

#%%
#merge the outcomes w/ the predprob
combined_outcomes = combined_preds.merge(pec_outcomes, left_on = 'Patient_ID',right_on = 'hup_id', how = 'left')
interictal_outcomes = interictal_preds.merge(pec_outcomes, left_on = 'Patient_ID',right_on = 'hup_id', how = 'left').dropna(subset = 'interictal_predprob')
ictal_outcomes = ictal_preds.merge(pec_outcomes, left_on = 'Patient_ID',right_on = 'hup_id', how = 'left').dropna(subset = 'ictal_predprob')

# %%
#graph for combined_outcomes
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

code_path = os.path.dirname('../tools/functions/')
sys.path.append(code_path)
from ied_fx_v3 import *

#Pearson Correlation PLOTS
#create a boxplot comparing the distribution of correlation across SOZ types
plt.figure(figsize=(8,6))
#change font to arial
plt.rcParams['font.family'] = 'Arial'

my_palette = {0:'#E64B35FF', 3:'#7E6148FF', 1:'#3C5488FF'}
# my_palette = {1:'#E64B35FF', 2:'#3C5488FF'}
pairs=[(0, 1),(0,3),(1,3)]
order = [3,0,1]

ax = sns.boxplot(x='engel_outcomes_12m', y='combined_predprob', data=combined_outcomes, palette=my_palette, order=order, showfliers = False)
sns.stripplot(x="engel_outcomes_12m", y="combined_predprob", data=combined_outcomes, order = order, color="black", alpha=0.5)
annotator = Annotator(ax, pairs, data=combined_outcomes, x="engel_outcomes_12m", y="combined_predprob", order=order)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', comparisons_correction='BH', verbose = True,hide_non_significant=False)
annotator.apply_and_annotate()

plt.xlabel('Labels', fontsize=12)
plt.ylabel('mTLE Model Probability', fontsize=12)
#change the x-tick labels to be more readable
plt.xticks(np.arange(3), ['Non mTLE','Bad', 'Good'], fontsize = 12)
plt.yticks(fontsize = 12)

#part to change
plt.title(f'Outcome Analysis - 12m', fontsize=16)
sns.despine()
plt.savefig('../Results/Fig6-outcomes_12m.pdf')
plt.show()

all_effect_szs = []
for comparison in pairs:
    # print(comparison)
    soz1 = comparison[0]
    soz2 = comparison[1]
    group1 = combined_outcomes[combined_outcomes['engel_outcomes_12m'] == soz1]['combined_predprob']
    group2 = combined_outcomes[combined_outcomes['engel_outcomes_12m'] == soz2]['combined_predprob']

    all_effect_szs.append(['Outcome 12m cohens d:', soz1, soz2, cohend(group1, group2)])
print(all_effect_szs)

print("KRUSKAL WALLIS:")
from scipy.stats import f_oneway, levene, shapiro, kruskal
rate=combined_outcomes
print(kruskal(rate[rate['engel_outcomes_12m'] == 0]['combined_predprob'], rate[rate['engel_outcomes_12m'] == 1]['combined_predprob'],rate[rate['engel_outcomes_12m'] == 3]['combined_predprob']))