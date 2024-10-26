#%%
#load predictions
import pandas as pd
import numpy as np
code_path = os.path.dirname('../tools/functions/')
sys.path.append(code_path)
from delongs_test import *

def main():
    combined_preds = pd.read_csv('../Results/LR_elastic_outcome/Combined_predictions-12m-outcomes.csv').rename(columns = {"Predicted_Probability":"combined_predprob"})
    interictal_preds = pd.read_csv('../Results/LR_elastic_outcome/Interictal_predictions-12m-outcomes.csv').rename(columns = {"Predicted_Probability":"interictal_predprob"})
    ictal_preds = pd.read_csv('../Results/LR_elastic_outcome/Ictal_predictions-12m-outcomes.csv').rename(columns = {"Predicted_Probability":"ictal_predprob"})

    merged_preds_v1 = combined_preds.merge(interictal_preds[['Patient_ID','interictal_predprob']], on='Patient_ID', how = "left")
    merged_preds = merged_preds_v1.merge(ictal_preds[['Patient_ID','ictal_predprob']], on = 'Patient_ID', how = "inner")


    #Run the delong test on the pairs
    print('\nDifference between COMBINED vs. ICTAL-ONLY:')
    auc1, cov1, p1 = (delong_roc_test(np.array(merged_preds['True_Label']), np.array(merged_preds['combined_predprob']), np.array(merged_preds['ictal_predprob'])))
    print("AUC:",auc1)
    print("COV:",cov1)
    print("p-value:",np.exp(np.log(10)*p1)) #needed extra math to calculate the ture p-val, this was implemented from the github

    print('\nDifference between INTER-ONLY vs. ICTAL-ONLY:')
    auc2, cov2, p2 = (delong_roc_test(np.array(merged_preds['True_Label']), np.array(merged_preds['interictal_predprob']), np.array(merged_preds['ictal_predprob'])))
    print("AUC:",auc2)
    print("COV:",cov2)
    print("p-value:",np.exp(np.log(10)*p2))

    print('\nDifference between INTER-ONLY vs. COMBINED:')
    auc3, cov3, p3 = (delong_roc_test(np.array(merged_preds['True_Label']), np.array(merged_preds['interictal_predprob']), np.array(merged_preds['combined_predprob'])))
    print("AUC:",auc3)
    print("COV:",cov3)
    print("p-value:",np.exp(np.log(10)*p3)) 

    p_values =  [p1,p2,p3]
    p_values = [np.exp(np.log(10)*p) for p in p_values]
    desired_fdr = 0.05

    def benjamini_hochberg(p_values, fdr):
        m = len(p_values)
        sorted_p_values = np.sort(p_values)
        sorted_index = np.argsort(p_values)
        bh_critical_values = np.arange(1, m + 1) / m * fdr

        # Determine the largest p-value that is less than or equal to its BH critical value
        significant = sorted_p_values <= bh_critical_values
        if significant.any():
            max_significant_index = np.where(significant)[0][-1]
            threshold_p_value = sorted_p_values[max_significant_index]
        else:
            threshold_p_value = None

        # Determine which p-values are significant
        significant_p_values = p_values <= threshold_p_value if threshold_p_value is not None else np.zeros_like(p_values, dtype=bool)

        return significant_p_values, threshold_p_value

    significant_p_values, threshold_p_value = benjamini_hochberg(p_values, desired_fdr)

    print("P-values:", p_values)
    print("Significant p-values:", significant_p_values)
    print("Threshold p-value:", threshold_p_value)

if __name__=="__main__":
    main()
