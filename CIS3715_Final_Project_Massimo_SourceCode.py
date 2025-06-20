import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle
import random
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer, fbeta_score


random.seed(123)

df = pd.read_csv(r"C:\Users\Massimo Camuso\Desktop\Academics\Spring 2025\CIS 3715 (Principles of Data Science)\Final Project\preprocessed_loan_data.csv")

df.info()

counts = df['Default'].value_counts()
print(counts) # Target variable is heavily imbalanced 

X = df[['Age', 'Income', 'CreditScore', 'DTIRatio', 'LoanAmount', 'InterestRate', 'MonthsEmployed', 'NumCreditLines', 'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner', 
                   'LoanTerm_12', 'LoanTerm_24', 'LoanTerm_48', 'LoanTerm_60', 
                   'LoanPurpose_Auto', 'LoanPurpose_Business', 'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other']]  # AFTER MONTHS EMPlOYED, ALL CATEGORICAL
y = df['Default']




X_train_val, X_test, y_train_val, y_test = train_test_split(X, y,
                                                            train_size=0.05, 
                                                            stratify=y,
                                                            random_state=0)
#print("train_val: {}, test: {}".format(X_train_val.shape[0], X_test.shape[0]))

# Split data into majority (non-defaults) and minority (defaults)
df_train = pd.concat([X_train_val, y_train_val], axis=1)
df_majority = df_train[df_train['Default'] == 0]
df_minority = df_train[df_train['Default'] == 1]

# Upsample minority class by randomly duplicating samples
df_minority_upsampled = resample(
    df_minority, 
    replace=True,  # Sample with replacement (duplicates allowed)
    n_samples=len(df_majority)  # Match majority class size
)

# Combine upsampled minority with original majority
df_train_balanced = pd.concat([df_majority, df_minority_upsampled])


continuous_feat = ['Age', 'Income', 'CreditScore', 'DTIRatio', 'LoanAmount', 'InterestRate', 'MonthsEmployed']  # Replace with your actual names
cat_features = ['NumCreditLines', 'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner', 
                   'LoanTerm_12', 'LoanTerm_24', 'LoanTerm_48', 'LoanTerm_60', 
                   'LoanPurpose_Auto', 'LoanPurpose_Business', 'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other']
all_features = ['Age', 'Income', 'CreditScore', 'DTIRatio', 'LoanAmount', 'InterestRate', 'MonthsEmployed', 'NumCreditLines', 'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner', 
                   'LoanTerm_12', 'LoanTerm_24', 'LoanTerm_48', 'LoanTerm_60', 
                   'LoanPurpose_Auto', 'LoanPurpose_Business', 'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other']

X_train_balanced = df_train_balanced.drop('Default', axis=1)
y_train_balanced = df_train_balanced['Default']
scaler = StandardScaler()
X_train_balanced_scaled = scaler.fit_transform(X_train_balanced[continuous_feat])
X_test_scaled = scaler.transform(X_test[continuous_feat])

#add categorical features
X_train_cat = X_train_balanced[cat_features].reset_index(drop=True)
X_test_cat = X_test[cat_features].reset_index(drop=True)

X_train_all = pd.concat([
    pd.DataFrame(X_train_balanced_scaled, columns=continuous_feat),
    X_train_cat
], axis=1)

X_test_all = pd.concat([
    pd.DataFrame(X_test_scaled, columns=continuous_feat),
    X_test_cat
], axis=1)
# Reset indices before concatenation






#X_balanced_scaled_df = pd.DataFrame(X_balanced_scaled, columns=feature_names)
#^^ USE THESE VARIABLES FOR MODELING; NOT THE ORIGINAL VARIABLES
#print(X_train_val_scaled[alr_scaled_cols])
#print(X_test_scaled[alr_scaled_cols])

#imbalance_ratio = len(y_train_val[y_train_val == 0]) / len(y_train_val[y_train_val == 1])

#---------------------------TRAINING WITH CROSS VALIDATION - LOGISTIC REGRESSION---------------
'''
folds = 5

num_train_val = X_train_balanced_scaled.shape[0]

index_of_samples = np.arange(num_train_val)
shuffle(index_of_samples)

index_of_folds = np.array_split(index_of_samples, folds)
print(index_of_folds)

regularization_coefficient = [0.1, 1, 10, 100, 1000]

#best_acc = 0.0
best_prec = -1000
best_recall = -1000
best_f1 = -1000

for reg in regularization_coefficient:
    #10-fold cross validation
    #sum_acc = 0.0
    sum_prec = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    for fold in range(folds):
        index_of_folds_temp = index_of_folds.copy()
        
        valid_index = index_of_folds_temp[fold] #get the index of the validation set
        train_index = np.hstack([index_of_folds_temp[i] for i in range(folds) if i != fold]) #get the index of the training set
        
        # training set
        X_train = X_train_all.iloc[train_index]
        y_train = y_train_balanced.iloc[train_index]
        
        # validation set
        X_valid = X_train_all.iloc[valid_index]
        y_valid = y_train_balanced.iloc[valid_index]
                
        # build the model with different hyperparameters
        clf = LogisticRegression(penalty='elasticnet', 
                                 l1_ratio=0.7, 
                                 C=reg, 
                                 solver='saga', 
                                 max_iter=10000,  
                                 class_weight='balanced')
        
        #train the model with the training set
        clf.fit(X_train, y_train)
        
        y_valid_pred = clf.predict(X_valid)
        #acc = accuracy_score(y_valid, y_valid_pred)
        #sum_acc += acc
        
        prec = precision_score(y_valid, y_valid_pred)
        sum_prec += prec

        recall = recall_score(y_valid, y_valid_pred)
        sum_recall += recall

        f1 = f1_score(y_valid, y_valid_pred)
        sum_f1 += f1
        
    
    #cur_acc = sum_acc / folds
    curr_prec = sum_prec / folds
    curr_recall = sum_recall / folds
    curr_f1 = sum_f1 / folds
    
    print(f"C={reg:.3f}, Avg Precision={curr_prec:.3f}, Avg Recall={curr_recall:.3f}, Avg F1={curr_f1:.3f}")
    print(f"C={reg}, Coefficients: {clf.coef_}, Intercept: {clf.intercept_}")    
    # store the best hyperparameter
    #if cur_acc > best_acc:
        #best_acc = cur_acc
        #best_reg = reg

    if curr_prec > best_prec:
        best_prec = curr_prec
        best_reg_prec = reg

    if curr_recall > best_recall:
        best_recall = curr_recall
        best_reg_recall = reg

    if curr_f1 > best_f1:
        best_f1 = curr_f1
        best_reg_f1 = reg
    
    print(f"Best Reg for Precision={best_reg_prec}")
    print(f"Best Reg for Recall={best_reg_recall}")
    print(f"Best Reg for F1={best_reg_f1}")

'''

final_clf = LogisticRegression(
    penalty='elasticnet',
    l1_ratio=0.2,
    C=0.1,
    solver='saga',
    class_weight='balanced',
    max_iter=10000
)
final_clf.fit(X_train_all, y_train_balanced)
y_test_probs = final_clf.predict_proba(X_test_all)[:, 1]
y_test_pred = (y_test_probs >= 0.6).astype(int)


print(f"Coefficients: {final_clf.coef_}, Intercept: {final_clf.intercept_}")    

print("Test Precision:", precision_score(y_test, y_test_pred))
print("Test Recall:", recall_score(y_test, y_test_pred))
print("Test F1:", f1_score(y_test, y_test_pred))







#----TRAINING AND TESTING RANDOM FOREST-----#
'''
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(class_weight='balanced', random_state=123)

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_balanced_scaled, y_train_balanced)

print("Best Params:", grid_search.best_params_)
best_rf = grid_search.best_estimator_
'''
'''
best_rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=3,
    max_features=0.8,
    class_weight={0: 1, 1: 7},
    random_state=123
)

calibrated_rf = CalibratedClassifierCV(best_rf, cv=5, method='isotonic')
calibrated_rf.fit(X_train_val, y_train_val)


best_rf.fit(X_train_val, y_train_val)
y_probs_rf = best_rf.predict_proba(X_test)[:, 1]  # Probabilities for class 1

precision, recall, thresholds = precision_recall_curve(y_test, y_probs_rf)

f1_5_scores = (2.25 * precision * recall) / (1.25 * precision + recall + 1e-9)
best_thresh = thresholds[np.argmax(f1_5_scores)]
print("Best Thresh:", best_thresh)


y_pred_rf = (y_probs_rf >= best_thresh).astype(int)
print("RF Precision:", precision_score(y_test, y_pred_rf))
print("RF Recall:", recall_score(y_test, y_pred_rf))
print("RF F1.5:", fbeta_score(y_test, y_pred_rf, beta=1.5))
'''

'''
#PRECISION RECALL CURVE FOR RANDOM FOREST
precision, recall, thresholds = precision_recall_curve(y_test, y_probs_rf)
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.legend()
plt.show()


# Plot feature importances
import matplotlib.pyplot as plt
plt.barh(X_train_all.columns, best_rf.feature_importances_)
plt.show()
'''
'''


#------TRAINING AND TESTING XGBOOST-----
param_grid = {
    'max_depth': [3, 5, 7],          # Control tree complexity
    'learning_rate': [0.01, 0.1],     # Shrinkage to prevent overfitting
    'subsample': [0.8, 1.0],          # Fraction of samples per tree
    'colsample_bytree': [0.8, 1.0],   # Fraction of features per tree
    'scale_pos_weight': [5, 7, 10],   # Adjust class imbalance weighting
    'min_child_weight': [1, 5]        # Regularization (higher = more conservative)
}
xgb = XGBClassifier(
    scale_pos_weight=7,  
    eval_metric='aucpr',  # Optimizes for precision-recall (better for imbalance)
    random_state=42
)
grid = GridSearchCV(xgb, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train_all, y_train_balanced)
print("Best parameters:", grid.best_params_)
'''
'''
best_xgb = XGBClassifier(
    scale_pos_weight=13,  
    eval_metric='aucpr',
    colsample_bytree = 1.0,
    learning_rate = 0.1,
    max_depth = 10,
    min_child_weight = 1,
    subsample = 0.8,
    random_state=42,
    n_estimators = 200,
    reg_alpha = 5,
    reg_lambda = 5
)


calibrated = CalibratedClassifierCV(best_xgb, cv=5, method='isotonic')
calibrated.fit(X_train_val, y_train_val)


# Get predicted probabilities for defaults
y_probs_calibrated = calibrated.predict_proba(X_test)[:, 1]

# Calculate F1.5 at different thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_probs_calibrated)
f2_scores = (3.5 * precision * recall) / (2.5 * precision + recall + 1e-9)  # F2 formula

# Find the best threshold
best_thresh = thresholds[np.argmax(f2_scores)]
print("Best threshold for F2:", best_thresh)


y_pred = (y_probs_calibrated >= best_thresh).astype(int)
# Apply to predictions
y_pred = (y_probs_calibrated >= best_thresh).astype(int)
print("F2-Score:", fbeta_score(y_test, y_pred, beta=1.5))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
'''

'''
#PRECISION RECALL CURVE
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.legend()
plt.show()
'''
'''
from xgboost import plot_importance
plot_importance(best_xgb)
plt.show()
'''



'''
#FINAL PLOT OF RECALL/PRECISION
metrics = {
    'Logistic Regression': {
        'Precision': 0.26,  # Replace with your values
        'Recall': 0.53,

    },
    'Random Forest': {
        'Precision': 0.25,  # From your RF results
        'Recall': 0.56,
    },
    'XGBoost': {
        'Precision': 0.21,  # From your XGBoost results
        'Recall': 0.65,
    }
}



fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.25
index = np.arange(2)  # Only 2 metrics now: Precision and Recall


colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Added third color for XGBoost

# Plot bars for each metric
for i, (model, values) in enumerate(metrics.items()):
    ax.bar(
        index + i * bar_width,
        [values['Precision'], values['Recall']],
        width=bar_width,
        label=model,
        color=colors[i]
    )

# Customization
ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Model Comparison: Precision and Recall')
ax.set_xticks(index + bar_width * (len(metrics)-1)/2)  # Center labels
ax.set_xticklabels(['Precision', 'Recall'])
ax.legend(loc='upper right')
ax.set_ylim(0, 1.0)

# Add value labels
for i, (model, values) in enumerate(metrics.items()):
    for j, metric in enumerate(['Precision', 'Recall']):
        ax.text(
            index[j] + i * bar_width,
            values[metric] + 0.02,
            f'{values[metric]:.2f}',
            ha='center',
            va='bottom'
        )

plt.tight_layout()
plt.show()
'''