# K-Fold Ensemble Implementation for Fertilizer Prediction
# This script contains the K-fold cross-validation implementation that can be integrated into the existing notebook

# Cell 1: Import additional libraries for K-fold
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
import joblib
import os
import warnings

warnings.filterwarnings('ignore')
"""

# Cell 2: Utility functions for MAP@k
"""
def average_precision_at_k(y_true, y_pred_topk):
    for i, pred in enumerate(y_pred_topk):
        if pred == y_true:
            return 1.0 / (i + 1)
    return 0.0

def map_at_k(probs, true_labels, k=3):
    top_k_preds = np.argsort(probs, axis=1)[:, ::-1][:, :k]
    
    ap_scores = []
    for i in range(len(true_labels)):
        ap = average_precision_at_k(true_labels[i], top_k_preds[i])
        ap_scores.append(ap)
    return np.mean(ap_scores)
"""

# Cell 3: Load test data early (before K-fold)
"""
# Load test data at the beginning
try:
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    test_df_original = test_df.copy()
    print("Test data loaded successfully.")
    print(f"Test data shape: {test_df.shape}")
except FileNotFoundError:
    print("Error: Test data not found. Make sure the path is correct.")
    test_df = pd.DataFrame()
    test_df_original = pd.DataFrame()

# Load sample submission for format reference
try:
    submission_df = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
except FileNotFoundError:
    print("Error: Sample submission not found. Creating empty submission.")
    submission_df = pd.DataFrame()
"""

# Cell 4: K-fold training functions
"""
def train_lgbm_kfold(params, X, y, test, folds=5, verbose=True, random_state=42):
    model = lgb.LGBMClassifier(**params)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    oof_preds = np.zeros((len(X), len(np.unique(y))))
    test_preds = np.zeros((len(test), len(np.unique(y))))
    map3_scores = np.zeros(folds)

    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    models = []
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
        model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)], 
            categorical_feature=cat_features,
            verbose=0
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(test)
        
        map3 = map_at_k(oof_preds[val_idx], np.array(y_val), k=3)
        map3_scores[i] = map3
        models.append(model)
        
        if verbose:
            print(f'Fold: {i+1} | MAP @ 3: {map3:.5f}')

    test_preds /= folds
    return models, oof_preds, map3_scores, test_preds

def train_catboost_kfold(params, X, y, test, folds=5, verbose=True, random_state=42):
    model = CatBoostClassifier(**params)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    oof_preds = np.zeros((len(X), len(np.unique(y))))
    test_preds = np.zeros((len(test), len(np.unique(y))))
    map3_scores = np.zeros(folds)

    models = []
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        oof_preds[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(test)
        
        map3 = map_at_k(oof_preds[val_idx], np.array(y_val), k=3)
        map3_scores[i] = map3
        models.append(model)
        
        if verbose:
            print(f'Fold: {i+1} | MAP @ 3: {map3:.5f}')

    test_preds /= folds
    return models, oof_preds, map3_scores, test_preds

def train_xgboost_kfold(params, X, y, test, folds=5, verbose=True, random_state=42):
    model = xgb.XGBClassifier(**params)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    oof_preds = np.zeros((len(X), len(np.unique(y))))
    test_preds = np.zeros((len(test), len(np.unique(y))))
    map3_scores = np.zeros(folds)

    models = []
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        oof_preds[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(test)
        
        map3 = map_at_k(oof_preds[val_idx], np.array(y_val), k=3)
        map3_scores[i] = map3
        models.append(model)
        
        if verbose:
            print(f'Fold: {i+1} | MAP @ 3: {map3:.5f}')

    test_preds /= folds
    return models, oof_preds, map3_scores, test_preds

def train_logistic_regression_kfold(params, X, y, test, folds=5, verbose=True, random_state=42):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    n_classes = len(np.unique(y))
    oof_preds = np.zeros((len(X), n_classes))
    test_preds = np.zeros((len(test), n_classes))
    map3_scores = np.zeros(folds)
    
    models = []

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(**params))
        ])

        model.fit(X_train, y_train)
        oof_preds[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(test)
        
        map3 = map_at_k(oof_preds[val_idx], np.array(y_val), k=3)
        map3_scores[i] = map3
        models.append(model)
        
        if verbose:
            print(f'Fold: {i+1} | MAP @ 3: {map3:.5f}')

    test_preds /= folds
    return models, oof_preds, map3_scores, test_preds
"""

# Cell 5: Prepare data for K-fold (after preprocessing and feature engineering)
"""
# After all preprocessing and feature engineering is done, prepare data for K-fold
if not df_processed.empty and not test_df.empty:
    # Prepare training data
    X = df_processed.drop('Fertilizer Name_Encoded', axis=1)
    y = df_processed['Fertilizer Name_Encoded']
    
    # Prepare test data (apply same preprocessing)
    test_df_processed = test_df.copy()
    
    # Apply same feature engineering to test data
    # (This should include all the same transformations applied to training data)
    
    # For now, assuming test_df_processed is ready
    X_test = test_df_processed  # Adjust based on your preprocessing pipeline
    
    print("Data prepared for K-fold training:")
    print(f"Training data shape: {X.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target classes: {len(np.unique(y))}")
else:
    print("Data not ready for K-fold training.")
"""

# Cell 6: Model parameters for K-fold
"""
# Model parameters optimized for K-fold training
lgbm_kfold_params = {
    'boosting_type': 'gbdt',
    'n_estimators': 1000,
    'learning_rate': 0.065,
    'num_leaves': 170,
    'max_depth': 10,
    'min_child_samples': 19,
    'subsample': 0.65,
    'colsample_bytree': 0.43,
    'reg_alpha': 6.3,
    'reg_lambda': 5.56,
    'random_state': 42,
    'verbosity': -1,
}

catboost_kfold_params = {
    'iterations': 1000,
    'learning_rate': 0.02,
    'depth': 8,
    'l2_leaf_reg': 3,
    'subsample': 0.8, 
    'random_seed': 42,
    'verbose': 0, 
    'early_stopping_rounds': 50,
    'bootstrap_type': 'Bernoulli' 
}

xgb_kfold_params = {
    'max_depth': 12,
    'colsample_bytree': 0.467,
    'subsample': 0.86,
    'n_estimators': 1000,
    'learning_rate': 0.015,
    'gamma': 0.25,
    'max_delta_step': 4,
    'reg_alpha': 2.7,
    'reg_lambda': 1.4,
    'early_stopping_rounds': 50,
    'objective': 'multi:softprob',
    'random_state': 13,
    'enable_categorical': True,
    'tree_method': 'hist',
}

lr_kfold_params = {
    'penalty': 'l2',
    'solver': 'lbfgs',
    'C': 0.00459,
    'max_iter': 5000
}
"""

# Cell 7: Train LightGBM with K-fold
"""
print("="*50)
print("Training LightGBM with K-fold...")
lgbm_models, lgbm_oof_preds, lgbm_map3_scores, lgbm_test_preds = train_lgbm_kfold(
    lgbm_kfold_params, X, y, X_test
)
print(f"LightGBM Mean MAP@3: {lgbm_map3_scores.mean():.5f} (+/- {lgbm_map3_scores.std() * 2:.5f})")
"""

# Cell 8: Train CatBoost with K-fold
"""
print("="*50)
print("Training CatBoost with K-fold...")
catboost_models, catboost_oof_preds, catboost_map3_scores, catboost_test_preds = train_catboost_kfold(
    catboost_kfold_params, X, y, X_test
)
print(f"CatBoost Mean MAP@3: {catboost_map3_scores.mean():.5f} (+/- {catboost_map3_scores.std() * 2:.5f})")
"""

# Cell 9: Train XGBoost with K-fold
"""
print("="*50)
print("Training XGBoost with K-fold...")
xgb_models, xgb_oof_preds, xgb_map3_scores, xgb_test_preds = train_xgboost_kfold(
    xgb_kfold_params, X, y, X_test
)
print(f"XGBoost Mean MAP@3: {xgb_map3_scores.mean():.5f} (+/- {xgb_map3_scores.std() * 2:.5f})")
"""

# Cell 10: Stack results with Logistic Regression
"""
print("="*50)
print("Stacking results with Logistic Regression...")

stacked_oof = np.concatenate([lgbm_oof_preds, catboost_oof_preds, xgb_oof_preds], axis=1)
stacked_test_preds = np.concatenate([lgbm_test_preds, catboost_test_preds, xgb_test_preds], axis=1)

X_oof = pd.DataFrame(stacked_oof)
X_test_preds = pd.DataFrame(stacked_test_preds)

_, lr_oof_preds, lr_map3_scores, lr_test_preds = train_logistic_regression_kfold(
    lr_kfold_params, X_oof, y, X_test_preds
)
print(f"Logistic Regression Mean MAP@3: {lr_map3_scores.mean():.5f} (+/- {lr_map3_scores.std() * 2:.5f})")
"""

# Cell 11: Generate submission
"""
print("="*50)
print("Generating submission...")

top_3_preds = np.argsort(lr_test_preds, axis=1)[:, -3:][:, ::-1]
top_3_labels = target_encoder.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

submission_fertilizer_names = [" ".join(label) for label in top_3_labels]

submission_df = pd.DataFrame({
    'id': test_df_original['id'],
    'Fertilizer Name': submission_fertilizer_names
})

submission_file_path = 'submission_kfold.csv'
submission_df.to_csv(submission_file_path, index=False)
print(f"Submission file created successfully at: {submission_file_path}")
print(submission_df.head())
"""

# Cell 12: Save K-fold models and artifacts
"""
print("="*50)
print("Saving K-fold models and artifacts...")

# Save K-fold models
joblib.dump(lgbm_models, os.path.join(model_dir, 'lgbm_kfold_models.joblib'))
joblib.dump(catboost_models, os.path.join(model_dir, 'catboost_kfold_models.joblib'))
joblib.dump(xgb_models, os.path.join(model_dir, 'xgboost_kfold_models.joblib'))

print("K-fold models saved successfully!")
"""

# Cell 13: Print final results
"""
print("="*50)
print("FINAL K-FOLD RESULTS:")
print(f"LightGBM MAP@3: {lgbm_map3_scores.mean():.5f} (+/- {lgbm_map3_scores.std() * 2:.5f})")
print(f"CatBoost MAP@3: {catboost_map3_scores.mean():.5f} (+/- {catboost_map3_scores.std() * 2:.5f})")
print(f"XGBoost MAP@3: {xgb_map3_scores.mean():.5f} (+/- {xgb_map3_scores.std() * 2:.5f})")
print(f"Ensemble (LR) MAP@3: {lr_map3_scores.mean():.5f} (+/- {lr_map3_scores.std() * 2:.5f})")
""" 