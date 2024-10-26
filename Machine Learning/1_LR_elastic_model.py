import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample as sklearn_resample
from sklearn.preprocessing import StandardScaler
import os
import sys
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set default output directory to current working directory
default_output_dir = os.path.join('../Results/', "LR_elastic")

# Allow custom output directory via environment variable
output_dir = os.environ.get("ML_OUTPUT_DIR", default_output_dir)

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Create and open log file
log_file = os.path.join(output_dir, f"logistic_regression_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
try:
    sys.stdout = open(log_file, 'w')
except IOError as e:
    print(f"Error: Unable to create or write to log file. {e}")
    sys.exit(1)

print(f"Log file created at: {log_file}")

# Load and preprocess data
def load_data():
    pearson_df = pd.read_csv('../Dataset/pooled_pearson_all_norm.csv', index_col=0)
    pearson_df['SOZ'] = pearson_df['SOZ'].replace({2: 0, 3: 0})
    pearson_df['pt_id'] = pearson_df['pt_id'].astype(int)

    EI_df = pd.read_csv('../Dataset/EI_pearson_final1.csv', index_col=0)[['correlation', 'pt_id']]
    EI_df['pt_id'] = EI_df['pt_id'].str.replace('3T_MP0', '').str.replace('HUP', '').astype(int)

    combined_df = pearson_df.merge(EI_df, on='pt_id')
    combined_df = combined_df.dropna(subset='correlation')

    return combined_df, EI_df, pearson_df

# Bootstrap ROC curve
def bootstrap_roc(y_true, y_pred, n_bootstraps=2000, alpha=0.95):
    bootstrapped_tpr = []
    bootstrapped_fpr = np.linspace(0, 1, 100)
    boot_aucs = []
    
    for i in range(n_bootstraps):
        indices = sklearn_resample(np.arange(len(y_pred)), random_state=42 + i)
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        fpr_, tpr_, _ = roc_curve(y_true[indices], y_pred[indices])
        boot_aucs.append(auc(fpr_, tpr_))
        tpr_interp = np.interp(bootstrapped_fpr, fpr_, tpr_)
        bootstrapped_tpr.append(tpr_interp)

    bootstrapped_tpr = np.array(bootstrapped_tpr)
    sort_idx = np.argsort(boot_aucs)
    bootstrapped_tpr = bootstrapped_tpr[sort_idx]

    lower_percentile = ((1.0 - alpha) / 2.0)
    upper_percentile = (alpha + ((1.0 - alpha) / 2.0))

    tpr_lower = bootstrapped_tpr[int(lower_percentile*n_bootstraps)]
    tpr_upper = bootstrapped_tpr[int(upper_percentile*n_bootstraps)]
    mean_tpr = np.mean(bootstrapped_tpr, axis=0)

    return bootstrapped_fpr, mean_tpr, tpr_lower, tpr_upper

def calculate_auc_ci(y_true, y_pred, n_bootstraps=2000, alpha=0.95):
    bootstrapped_aucs = []
    
    for i in range(n_bootstraps):
        indices = sklearn_resample(np.arange(len(y_pred)), random_state=42 + i)
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        fpr, tpr, _ = roc_curve(y_true[indices], y_pred[indices])
        bootstrapped_aucs.append(auc(fpr, tpr))
    
    sorted_aucs = np.sort(bootstrapped_aucs)
    ci_lower = sorted_aucs[int((1.0 - alpha) / 2.0 * len(sorted_aucs))]
    ci_upper = sorted_aucs[int((alpha + (1.0 - alpha) / 2.0) * len(sorted_aucs))]
    
    return ci_lower, ci_upper

# Plot ROC curve with confidence interval
def plot_roc_with_ci(bootstrapped_fpr, tpr_lower, tpr_upper, color):
    plt.fill_between(bootstrapped_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2)

# Perform leave-one-out cross-validation with Logistic Regression Elastic Net
def leave_one_out_cv(X, y):
    unique_ids = X['pt_id'].unique()
    np.random.shuffle(unique_ids)
    
    loo = LeaveOneOut()
    y_true, y_pred, y_predprob = [], [], []
    feature_importances = []
    pt_ids = []
    
    scaler = StandardScaler()
    
    for train_ix, test_ix in loo.split(unique_ids):
        X_train = X[X['pt_id'].isin(unique_ids[train_ix])]
        X_test = X[X['pt_id'].isin(unique_ids[test_ix])]
        y_train = X_train['SOZ']
        y_test = X_test['SOZ']
        
        pt_ids.extend(X_test['pt_id'].values)
        
        X_train = X_train.drop(columns=['SOZ', 'pt_id'])
        X_test = X_test.drop(columns=['SOZ', 'pt_id'])
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize Logistic Regression with Elastic Net
        model = LogisticRegressionCV(
            cv=5,  # 5-fold cross-validation
            penalty='elasticnet',
            random_state=42,
            solver='saga',
            l1_ratios=np.linspace(0, 1, 10),  # This creates an array of 10 values from 0 to 1
            Cs=np.logspace(-4, 4, 20),  # This creates an array of 20 values from 10^-4 to 10^4
            max_iter=10000  # Increase max_iter to ensure convergence
        )
        
        model.fit(X_train_scaled, y_train)
        
        y_pred.append(model.predict(X_test_scaled))
        y_predprob.append(model.predict_proba(X_test_scaled)[:, 1])
        y_true.append(y_test.to_numpy())
        
        # For Logistic Regression, we use the coefficients as feature importances
        feature_importances.append(np.abs(model.coef_[0]))
    
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_predprob), np.mean(feature_importances, axis=0), pt_ids

# Main execution
def main():
    print("Starting training")
    plt.rcParams['font.family'] = 'Arial'

    combined_data, ictal_data, interictal_data = load_data()
    
    # Prepare datasets
    combined_features = combined_data.drop(columns=['SOZ', 'pt_id'])
    interictal_features = interictal_data.drop(columns=['SOZ', 'pt_id'])
    ictal_features = combined_data[['correlation', 'SOZ', 'pt_id']].drop(columns=['SOZ', 'pt_id'])
    
    datasets = [
        ("Combined", combined_features, '#E64B35FF'),
        ("Interictal", interictal_features, '#7E6148FF'),
        ("Ictal", ictal_features, '#00A087FF')
    ]
    
    plt.figure(figsize=(8, 8))
    
    for name, features, color in datasets:
        if (name == "Combined") or (name == "Ictal"):
            data = combined_data
        else: 
            data = interictal_data
        X = pd.concat([features, data[['SOZ', 'pt_id']]], axis=1)
        y_true, y_pred, y_predprob, feature_importance, pt_ids = leave_one_out_cv(X, data['SOZ'])
        
        fpr, tpr, _ = roc_curve(y_true, y_predprob)
        roc_auc = auc(fpr, tpr)
        
        # Calculate AUC confidence interval
        ci_lower, ci_upper = calculate_auc_ci(y_true, y_predprob)
        
        plt.plot(fpr, tpr, color=color, lw=3, label=f'{name} (AUC = {roc_auc:.2f})')
        
        bootstrapped_fpr, mean_tpr, tpr_lower, tpr_upper = bootstrap_roc(y_true, y_predprob)
        plot_roc_with_ci(bootstrapped_fpr, tpr_lower, tpr_upper, color)
        
        # Print AUC with confidence interval
        print(f"\nResults for {name}:")
        print(f"AUC: {roc_auc:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})")
        
        # Print and save feature importance
        print(f"\nFeature Importance for {name}:")
        importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importance})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        print(importance_df)
        importance_df.to_csv(os.path.join(output_dir, f"{name}_feature_importance.csv"), index=False)
        
        # Save predictions vs. truth labels
        predictions_df = pd.DataFrame({
            'Patient_ID': pt_ids,
            'True_Label': y_true,
            'Predicted_Label': y_pred,
            'Predicted_Probability': y_predprob
        })
        predictions_df.to_csv(os.path.join(output_dir, f"{name}_predictions.csv"), index=False)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curves for Classifying MTLE', fontsize=24, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=20, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=20, fontweight='bold')
    plt.legend(loc="lower right", prop={'size': 16, 'weight': 'bold'})
    plt.tick_params(axis='both', which='major', labelsize=16)

    # Manually set the weight to bold for tick labels
    for label in plt.gca().get_xticklabels():
        label.set_weight('bold')
    for label in plt.gca().get_yticklabels():
        label.set_weight('bold')

    sns.despine()
    plt.tight_layout()
    plt.savefig('../Results/Fig5-ROC_curves.pdf')
    plt.close()

if __name__ == "__main__":
    main()
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    print(f"Results saved to {output_dir}")

