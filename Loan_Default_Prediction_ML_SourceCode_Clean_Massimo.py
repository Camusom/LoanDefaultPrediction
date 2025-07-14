import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, classification_report, confusion_matrix, make_scorer, fbeta_score, roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as pipeline 
import joblib
import seaborn as sns
import xgboost as xgb 

SEED = 456

df = pd.read_csv(r"C:\Users\Massimo Camuso\Desktop\Academics\Spring 2025\CIS 3715 (Principles of Data Science)\Final Project\preprocessed_loan_data.csv")

df.info()
df = df.dropna()
df['Age'] = pd.to_numeric(df['Age'])
df['Income'] = pd.to_numeric(df['Income'])
df['LoanAmount'] = pd.to_numeric(df['LoanAmount'])

y = df['Default']
X = df.drop(['Default', 'LoanID'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.80,
                                                    stratify=y,
                                                    random_state=SEED)

continuous_feat = ['Age', 'Income', 'CreditScore', 'DTIRatio', 'LoanAmount', 'InterestRate', 'MonthsEmployed']  # Replace with your actual names
cat_features = ['NumCreditLines', 'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage', 'HasDependents', 'HasCoSigner', 
                   'LoanTerm_12', 'LoanTerm_24', 'LoanTerm_48', 'LoanTerm_60', 
                   'LoanPurpose_Auto', 'LoanPurpose_Business', 'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other']

preprocessor = ColumnTransformer(
    transformers=[
        ('cont', StandardScaler(), continuous_feat),
        ('cat', 'passthrough', cat_features)
    ],
    remainder='passthrough'
)

full_pipeline = pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=SEED)),
    ('classifier', LogisticRegression(solver='saga', max_iter=10000, penalty='elasticnet'))
])

param_grid = {
    'classifier__C': [0.005, 0.01, 0.1, 0.5, 5.0],
    'classifier__l1_ratio': [0.4, 0.5, 0.6]
}

param_grid_v2 = {
    'classifier__C': [0.001, 0.003, 0.005, 0.007, 0.01],
    'classifier__l1_ratio': [0.4, 0.5, 0.6]
}

param_grid_v3 = {
    'classifier__C': [0.0001, 0.0005, 0.001, 0.002],
    'classifier__l1_ratio': [0.5]
}

f1_5_scorer = make_scorer(fbeta_score, beta=1.5, greater_is_better=True)

scoring = {
    'AUC': 'roc_auc',
    'F1.5': f1_5_scorer,
    'Precision': 'precision',
    'Recall': 'recall'
}

grid_search = GridSearchCV(
    estimator=full_pipeline,
    param_grid=param_grid_v3,
    scoring=scoring,
    refit='F1.5',
    cv=5,
    n_jobs=-1,
    verbose=2
)
'''
print("starting gridsearch...")
grid_search.fit(X_train, y_train)

print("gridsearchCV completed")

pd.set_option('display.max_columns', None)
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df[['param_classifier__C', 'param_classifier__l1_ratio', 'mean_test_AUC', 'mean_test_F1.5', 'mean_test_Recall', 'mean_test_Precision']])

best_model = grid_search.best_estimator_
print(f"Best Hyperparameters (for F1.5 score): {grid_search.best_params_}")

joblib.dump(best_model, 'loan_default_model_total.pkl')
print("best model has been saved")

#print("loading pre-trained model...")
best_model = joblib.load('loan_default_model_total.pkl')
#print("successfully loaded model:", best_model)
'''



#Threshold tuning
'''
y_probas_train = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=5,
    method='predict_proba',
    n_jobs=-1
)

y_scores = y_probas_train[:, 1]

auc_score = roc_auc_score(y_train, y_scores)

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

f1_5_scores = (1 + 1.5**2) * (precisions[:-1] * recalls[:-1]) / ((1.5**2 * precisions[:-1]) + recalls[:-1])

optimal_idx = np.argmax(f1_5_scores)
optimal_threshold = thresholds[optimal_idx]
best_f1_5_score = f1_5_scores[optimal_idx]

print(f"\nBest F-1.5 score on CV data: {best_f1_5_score:.4f}")
print(f"Optimal threshold found: {optimal_threshold:.4f}")
print(f"At this threshold: Precision = {precisions[optimal_idx]:.4f}, Recall = {recalls[optimal_idx]:.4f}")
print(f"AUC score: {auc_score}")



print("\nGenerating Precision-Recall-F1.5 vs. Threshold plot...")
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
plt.plot(thresholds, f1_5_scores, 'r-', lw=2, label='F-1.5 Score')
plt.axvline(x=optimal_threshold, color='purple', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.2f})')
plt.title('Precision, Recall, and F-1.5 Score vs. Decision Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend(loc='best')
plt.ylim([0, 1.05])
plt.grid(True)
plt.show()


#testing logistic regression
optimal_threshold = 0.56
y_probas_test = best_model.predict_proba(X_test)[:, 1]

y_pred_test_final = (y_probas_test >= optimal_threshold).astype(int)

print("Final Results:")
print(classification_report(y_test, y_pred_test_final, target_names=['Non-Default (0)', 'Default (1)']))
final_f1_5 = fbeta_score(y_test, y_pred_test_final, beta=1.5)
final_auc = roc_auc_score(y_test, y_probas_test)

print(f"Final F-1.5 Score: {final_f1_5:.4f}")
print(f"Final AUC score: {final_auc:.4f}")

print("\nConfusion Matrix on Test Set:")
cm = confusion_matrix(y_test, y_pred_test_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Non-Default', 'Predicted Default'], yticklabels=['Actual Non-Default', 'Actual Default'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
'''

'''
#Training RF model
pipeline_rf = pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=SEED))
])

param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__class_weight': ['balanced', None] 
}

grid_search_rf = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid_rf,
    scoring=scoring,
    refit='F1.5',
    cv=5,
    n_jobs=-1,
    verbose=2
)
print("starting RF gridsearch...")
grid_search_rf.fit(X_train, y_train)

print("gridsearchCV for RF completed")

results_df_rf = pd.DataFrame(grid_search_rf.cv_results_)
display_columns = [
    'param_classifier__n_estimators',
    'param_classifier__max_depth',
    'param_classifier__min_samples_split',
    'param_classifier__min_samples_leaf',
    'param_classifier__max_features',
    'param_classifier__class_weight',
    'mean_test_AUC',
    'mean_test_F1.5',
    'mean_test_Recall',
    'mean_test_Precision'
]

print(results_df_rf[display_columns].sort_values(by='mean_test_F1.5', ascending=False).head(10))
print(f"Best RF Hyperparameters (for F1.5 score): {grid_search_rf.best_params_}")

best_rf_params = grid_search_rf.best_params_
best_rf = RandomForestClassifier(**best_rf_params, random_state=SEED)
rf_model = grid_search_rf.best_estimator_


#feature importance graph function:
def plot_feature_importance(pipeline, top_n=20):
    
    model = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']

    feature_names = preprocessor.get_feature_names_out()

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance_df)
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.grid(True)
    plt.show()


plot_feature_importance(rf_model, top_n=20)


#Threshold tuning for RF

calibrated_rf = CalibratedClassifierCV(
    base_estimator=best_rf,
    method='isotonic',
    cv=5
)

y_probas_train_rf = cross_val_predict(
    calibrated_rf,
    X_train,
    y_train,
    cv=5,   
    method='predict_proba'
)
y_scores_rf = y_probas_train_rf[:, 1]

auc_score_rf = roc_auc_score(y_train, y_scores_rf)

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores_rf)

f1_5_scores = (1 + 1.5**2) * (precisions[:-1] * recalls[:-1]) / ((1.5**2 * precisions[:-1]) + recalls[:-1])

optimal_idx = np.argmax(f1_5_scores)
optimal_threshold = thresholds[optimal_idx]
best_f1_5_score = f1_5_scores[optimal_idx]

print(f"\nBest F-1.5 score on CV data for RF: {best_f1_5_score:.4f}")
print(f"Optimal threshold found for RF: {optimal_threshold:.4f}")
print(f"At this threshold: Precision = {precisions[optimal_idx]:.4f}, Recall = {recalls[optimal_idx]:.4f}")
print(f"AUC score: {auc_score_rf}")



print("\nGenerating Precision-Recall-F1.5 vs. Threshold plot...")
plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
plt.plot(thresholds, f1_5_scores, 'r-', lw=2, label='F-1.5 Score')
plt.axvline(x=optimal_threshold, color='purple', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.2f})')
plt.title('Precision, Recall, and F-1.5 Score vs. Decision Threshold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend(loc='best')
plt.ylim([0, 1.05])
plt.grid(True)
plt.show()


#testing random forest 
calibrated_rf.fit(X_train, y_train)
y_probas_test_rf = calibrated_rf.predict_proba(X_test)[:, 1]

y_pred_test_rf = (y_probas_test_rf >= optimal_threshold).astype(int)

print("Final Results:")
print(classification_report(y_test, y_pred_test_rf, target_names=['Non-Default (0)', 'Default (1)']))
final_f1_5 = fbeta_score(y_test, y_pred_test_rf, beta=1.5)
final_auc = roc_auc_score(y_test, y_probas_test_rf)

print(f"Final F-1.5 Score: {final_f1_5:.4f}")
print(f"Final AUC score: {final_auc:.4f}")

print("\nConfusion Matrix on Test Set:")
cm = confusion_matrix(y_test, y_pred_test_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Non-Default', 'Predicted Default'], yticklabels=['Actual Non-Default', 'Actual Default'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
'''

'''
#Training XGBOOST

pipeline_xgb = pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=SEED
    ))
])

scale_pos_weight_value = np.sum(y_train == 0) / np.sum(y_train == 1)
print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight_value:.2f}")

param_grid_xgb = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__subsample': [0.7, 1.0],
    'classifier__colsample_bytree': [0.7, 1.0],
    'classifier__scale_pos_weight': [scale_pos_weight_value, 1] 
}

grid_search_xgb = GridSearchCV(
    estimator=pipeline_xgb,
    param_grid=param_grid_xgb, # Use the new XGBoost grid
    scoring=scoring,
    refit='F1.5',
    cv=5,
    n_jobs=-1,
    verbose=2
)
grid_search_xgb.fit(X_train, y_train)
print("GridSearchCV for XGBoost completed.")
print(f"Best XGBoost Hyperparameters (for F1.5 score): {grid_search_xgb.best_params_}")
xgb_model = grid_search.xgb.best_estimator_
plot_feature_importance(xgb_model, top_n=20)


#Threshold Tuning

best_xgb_params = grid_search_xgb.best_params_
best_xgb_params_clean = {k.replace('classifier__', ''): v for k, v in best_xgb_params.items()}

best_xgb = xgb.XGBClassifier(
    **best_xgb_params_clean,
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=SEED
)

calibrated_xgb = CalibratedClassifierCV(
    base_estimator=best_xgb, 
    method='isotonic',
    cv=5
)

y_probas_train_xgb = cross_val_predict(
    calibrated_xgb,
    X_train,
    y_train,
    cv=5,
    method='predict_proba'
)
y_scores_xgb = y_probas_train_xgb[:, 1]


precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores_xgb)

f1_5_scores = (1 + 1.5**2) * (precisions * recalls) / ((1.5**2 * precisions) + recalls)
f1_5_scores = np.nan_to_num(f1_5_scores)

optimal_idx = np.argmax(f1_5_scores[:-1])
optimal_threshold = thresholds[optimal_idx]
best_f1_5_score = f1_5_scores[optimal_idx]

print(f"\nBest F-1.5 score on CV data for XGBoost: {best_f1_5_score:.4f}")
print(f"Optimal threshold found for XGBoost: {optimal_threshold:.4f}")


# Testing XGB

# Fit the final calibrated model on the ENTIRE training set
calibrated_xgb.fit(X_train, y_train)

# Get predictions for the unseen test set
y_probas_test_xgb = calibrated_xgb.predict_proba(X_test)[:, 1]

# Apply the optimal threshold found from the training data
y_pred_test_xgb = (y_probas_test_xgb >= optimal_threshold).astype(int)

# Report the final, unbiased performance
print("\nFinal XGBoost Model Performance on Test Set")
print(classification_report(y_test, y_pred_test_xgb, target_names=['Non-Default (0)', 'Default (1)']))
final_f1_5 = fbeta_score(y_test, y_pred_test_xgb, beta=1.5)
final_auc = roc_auc_score(y_test, y_probas_test_xgb) 

print(f"Final F-1.5 Score: {final_f1_5:.4f}")
print(f"Final AUC score: {final_auc:.4f}")

# confusion matrix
print("\nConfusion Matrix on Test Set:")
cm = confusion_matrix(y_test, y_pred_test_xgb)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Non-Default', 'Predicted Default'], yticklabels=['Actual Non-Default', 'Actual Default'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('XGBoost Final Confusion Matrix')
plt.show()
'''