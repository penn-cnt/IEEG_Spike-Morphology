#%%
#load predictions
import pandas as pd
code_path = os.path.dirname('/mnt/leif/littlab/users/aguilac/Interictal_Spike_Analysis/HUMAN/working_feat_extract_code/functions/')
sys.path.append(code_path)
from delongs_test import *

def main():
    combined_preds = pd.read_csv('ML_results/LR_elastic/outcome/Combined_predictions-24m-outcomes.csv').rename(columns = {"Predicted_Probability":"combined_predprob"})
    interictal_preds = pd.read_csv('ML_results/LR_elastic/outcome/Interictal_predictions-24m-outcomes.csv').rename(columns = {"Predicted_Probability":"interictal_predprob"})
    ictal_preds = pd.read_csv('ML_results/LR_elastic/outcome/Ictal_predictions-24m-outcomes.csv').rename(columns = {"Predicted_Probability":"ictal_predprob"})

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

# %%
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    balanced_accuracy = (sensitivity + specificity) / 2
    
    return sensitivity, specificity, balanced_accuracy

def load_preds(name_to_load):
    # Load the Combined_predictions.csv file
    df = pd.read_csv(f'ML_results/LR_elastic/{name_to_load}.csv')
    # Extract true labels and predicted labels
    y_true = df['True_Label']
    y_pred = df['Predicted_Label']
    return y_true, y_pred

names = ['Combined_predictions','Interictal_predictions','Ictal_predictions']
for i in range(3):
    y_true,y_pred = load_preds(names[i])
    # Calculate metrics
    sensitivity, specificity, balanced_accuracy = calculate_metrics(y_true, y_pred)
    # Print results
    print("\nDataset:", names[i])
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.3f}")

    # Calculate confidence intervals using bootstrapping
    n_bootstraps = 2000
    n_samples = len(y_true)
    rng = np.random.RandomState(42)
    bootstrapped_metrics = []

    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for each bootstrap
            continue
        sens, spec, bal_acc = calculate_metrics(y_true[indices], y_pred[indices])
        bootstrapped_metrics.append((sens, spec, bal_acc))

    # Calculate 95% confidence intervals
    confidence_intervals = np.percentile(bootstrapped_metrics, [2.5, 97.5], axis=0)

    # Print results with confidence intervals
    print("Metrics with 95% Confidence Intervals:")
    print(f"Sensitivity: {sensitivity:.3f} (95% CI: {confidence_intervals[0][0]:.3f} - {confidence_intervals[1][0]:.3f})")
    print(f"Specificity: {specificity:.3f} (95% CI: {confidence_intervals[0][1]:.3f} - {confidence_intervals[1][1]:.3f})")
    print(f"Balanced Accuracy: {balanced_accuracy:.3f} (95% CI: {confidence_intervals[0][2]:.3f} - {confidence_intervals[1][2]:.3f})")
# %%
