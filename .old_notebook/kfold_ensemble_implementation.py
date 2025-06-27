"""
Enhanced Fertilizer Prediction with Ensemble Modeling and K-Fold Cross-Validation

This script implements K-fold cross-validation for LightGBM, CatBoost, and XGBoost classifiers,
then stacks the results using Logistic Regression, similar to the sample_KFOLD.ipynb approach.
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

def average_precision_at_k(y_true, y_pred_topk):
    """Calculate Average Precision at k for a single sample."""
    for i, pred in enumerate(y_pred_topk):
        if pred == y_true:
            return 1.0 / (i + 1)
    return 0.0

def map_at_k(probs, true_labels, k=3):
    """Calculate Mean Average Precision at k."""
    top_k_preds = np.argsort(probs, axis=1)[:, ::-1][:, :k]
    
    ap_scores = []
    for i in range(len(true_labels)):
        ap = average_precision_at_k(true_labels[i], top_k_preds[i])
        ap_scores.append(ap)
    return np.mean(ap_scores)

class DataFrameColumnEncoder:
    """Custom encoder for categorical columns."""
    def __init__(self, columns, handle_unknown='use_encoded_value', unknown_value=-1):
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoders_ = {}

    def fit(self, X, y=None):
        X = X[self.columns].copy()
        for col in X.columns:
            enc = LabelEncoder()
            enc.fit(X[col])
            self.encoders_[col] = enc
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            enc = self.encoders_[col]
            try:
                encoded = enc.transform(X[col])
            except ValueError:
                # Handle unknown values
                encoded = np.full(len(X[col]), self.unknown_value)
            X[col] = pd.Series(encoded, index=X.index).astype('int')
        return X.drop(['id'], axis=1, errors='ignore')

def train_lgbm_kfold(params, X, y, test, folds=5, verbose=True, random_state=42):
    """Train LightGBM with K-fold cross-validation."""
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
    """Train CatBoost with K-fold cross-validation."""
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
    """Train XGBoost with K-fold cross-validation."""
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
    """Train Logistic Regression with K-fold cross-validation for stacking."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    
    n_classes = len(np.unique(y))
    oof_preds = np.zeros((len(X), n_classes))
    test_preds = np.zeros((len(test), n_classes))
    map3_scores = np.zeros(folds)
    
    models = []

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Optional scaler + model pipeline
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

def main():
    """Main function to run the K-fold ensemble training."""
    
    # Load data
    print("Loading data...")
    try:
        # Load training data
        df = pd.read_csv('model/data.csv')
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Load test data
        test_df = pd.read_csv('model/test.csv')
        test_df_original = test_df.copy()
        
        print(f"Training data shape: {df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Basic preprocessing
    print("\nPreprocessing data...")
    
    # Encode categorical features
    encoder = DataFrameColumnEncoder(columns=['Soil Type', 'Crop Type'])
    encoder.fit(df.drop('Fertilizer Name', axis=1))
    X = encoder.transform(df.drop('Fertilizer Name', axis=1))
    
    # Label encode target
    le = LabelEncoder()
    y = le.fit_transform(df['Fertilizer Name'])
    y = pd.Series(y)
    
    # Transform test data
    X_test = encoder.transform(test_df)
    
    print(f"Processed training data shape: {X.shape}")
    print(f"Processed test data shape: {X_test.shape}")
    
    # Model parameters
    lgbm_params = {
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

    catboost_params = {
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

    xgb_params = {
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

    lr_params = {
        'penalty': 'l2',
        'solver': 'lbfgs',
        'C': 0.00459,
        'max_iter': 5000
    }
    
    # Train models with K-fold
    print("\n" + "="*50)
    print("Training LightGBM with K-fold...")
    lgbm_models, lgbm_oof_preds, lgbm_map3_scores, lgbm_test_preds = train_lgbm_kfold(
        lgbm_params, X, y, X_test
    )
    print(f"LightGBM Mean MAP@3: {lgbm_map3_scores.mean():.5f} (+/- {lgbm_map3_scores.std() * 2:.5f})")
    
    print("\n" + "="*50)
    print("Training CatBoost with K-fold...")
    catboost_models, catboost_oof_preds, catboost_map3_scores, catboost_test_preds = train_catboost_kfold(
        catboost_params, X, y, X_test
    )
    print(f"CatBoost Mean MAP@3: {catboost_map3_scores.mean():.5f} (+/- {catboost_map3_scores.std() * 2:.5f})")
    
    print("\n" + "="*50)
    print("Training XGBoost with K-fold...")
    xgb_models, xgb_oof_preds, xgb_map3_scores, xgb_test_preds = train_xgboost_kfold(
        xgb_params, X, y, X_test
    )
    print(f"XGBoost Mean MAP@3: {xgb_map3_scores.mean():.5f} (+/- {xgb_map3_scores.std() * 2:.5f})")
    
    # Stack results
    print("\n" + "="*50)
    print("Stacking results with Logistic Regression...")
    
    stacked_oof = np.concatenate([lgbm_oof_preds, catboost_oof_preds, xgb_oof_preds], axis=1)
    stacked_test_preds = np.concatenate([lgbm_test_preds, catboost_test_preds, xgb_test_preds], axis=1)

    X_oof = pd.DataFrame(stacked_oof)
    X_test_preds = pd.DataFrame(stacked_test_preds)
    
    _, lr_oof_preds, lr_map3_scores, lr_test_preds = train_logistic_regression_kfold(
        lr_params, X_oof, y, X_test_preds
    )
    print(f"Logistic Regression Mean MAP@3: {lr_map3_scores.mean():.5f} (+/- {lr_map3_scores.std() * 2:.5f})")
    
    # Generate submission
    print("\n" + "="*50)
    print("Generating submission...")
    
    top_3_preds = np.argsort(lr_test_preds, axis=1)[:, -3:][:, ::-1]
    top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
    
    submission_fertilizer_names = [" ".join(label) for label in top_3_labels]
    
    submission_df = pd.DataFrame({
        'id': test_df_original['id'],
        'Fertilizer Name': submission_fertilizer_names
    })
    
    submission_file_path = 'submission_kfold.csv'
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file created successfully at: {submission_file_path}")
    print(submission_df.head())
    
    # Save models and artifacts
    print("\n" + "="*50)
    print("Saving models and artifacts...")
    
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save models
    joblib.dump(lgbm_models, os.path.join(model_dir, 'lgbm_kfold_models.joblib'))
    joblib.dump(catboost_models, os.path.join(model_dir, 'catboost_kfold_models.joblib'))
    joblib.dump(xgb_models, os.path.join(model_dir, 'xgboost_kfold_models.joblib'))
    
    # Save encoders and mappings
    joblib.dump(encoder, os.path.join(model_dir, 'column_encoder.joblib'))
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.joblib'))
    
    # Save feature names
    joblib.dump(X.columns.tolist(), os.path.join(model_dir, 'feature_names.joblib'))
    
    print("Models and artifacts saved successfully!")
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"LightGBM MAP@3: {lgbm_map3_scores.mean():.5f} (+/- {lgbm_map3_scores.std() * 2:.5f})")
    print(f"CatBoost MAP@3: {catboost_map3_scores.mean():.5f} (+/- {catboost_map3_scores.std() * 2:.5f})")
    print(f"XGBoost MAP@3: {xgb_map3_scores.mean():.5f} (+/- {xgb_map3_scores.std() * 2:.5f})")
    print(f"Ensemble (LR) MAP@3: {lr_map3_scores.mean():.5f} (+/- {lr_map3_scores.std() * 2:.5f})")

if __name__ == "__main__":
    main() 