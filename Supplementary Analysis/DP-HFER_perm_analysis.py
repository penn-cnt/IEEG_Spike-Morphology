import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# permuted_data = np.load('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/permutation_test/1000_permuted_DP.npy')
# observed_statistic = 0.51 #DP median

permuted_data = np.load('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/5-propagation/dataset/consistency/permutation_test/1000_permuted_hfer_11s.npy')
observed_statistic = 0.61     
# Calculate and store the permuted statistic
permuted_stats= np.mean(permuted_data, axis =0)
num_permutations = permuted_data.shape[0]
    
# Calculate p-value
p_value = np.sum(permuted_stats >= observed_statistic) / num_permutations

print(f"Observed statistic: {observed_statistic}")
print(f"P-value: {p_value}")

plt.figure(figsize=(10, 6))
plt.hist(permuted_stats, bins=50, edgecolor='black')
plt.axvline(observed_statistic, color='red', linestyle='dashed', linewidth=2)
plt.title('Distribution of Permuted Statistics')
plt.xlabel('Mean Correlation')
plt.ylabel('Frequency')
plt.legend(['Observed Statistic', 'Permuted Statistics'])
plt.show()