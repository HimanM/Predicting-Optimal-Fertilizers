import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from collections import Counter

warnings.filterwarnings('ignore')

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

def feature_engineering(df):
    """Apply feature engineering to the dataset."""
    df = df.copy()
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-6
    
    # Nutrient ratios
    if 'Nitrogen' in df.columns and 'Phosphorous' in df.columns:
        df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + epsilon)
    
    if 'Nitrogen' in df.columns and 'Potassium' in df.columns:
        df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + epsilon)
        
    if 'Phosphorous' in df.columns and 'Potassium' in df.columns:
        df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + epsilon)
    
    # Logarithmic transformations
    numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    for col in numerical_features:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
    
    # Square transformations
    for col in numerical_features:
        if col in df.columns:
            df[f'sq_{col}'] = df[col]**2
    
    # Interaction features
    if 'Temparature' in df.columns and 'Humidity' in df.columns:
        df['Temp_Hum_Interaction'] = df['Temparature'] * df['Humidity']
    
    if 'Moisture' in df.columns and 'Temparature' in df.columns:
        df['Moisture_Temp_Interaction'] = df['Moisture'] * df['Temparature']
    
    return df

class StackingEnsemble:
    """
    A stacking ensemble model for fertilizer prediction.
    
    This class implements a two-level stacking ensemble:
    - Level 1: Base models (LightGBM, CatBoost, XGBoost)
    - Level 2: Meta-learner (Logistic Regression or Random Forest)
    """
    
    def __init__(self, n_folds=5, random_state=42):
        """
        Initialize the stacking ensemble.
        
        Args:
            n_folds (int): Number of folds for cross-validation
            random_state (int): Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.base_models = []
        self.meta_learner = None
        self.column_encoder = None
        self.label_encoder = None
        self.feature_names = None
        self.is_fitted = False
        
        # Base model parameters
        self.lgbm_params = {
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
            'random_state': random_state,
            'verbosity': -1,
        }
        
        self.catboost_params = {
            'iterations': 1000,
            'learning_rate': 0.065,
            'depth': 10,
            'l2_leaf_reg': 5.56,
            'random_strength': 6.3,
            'random_state': random_state,
            'verbose': False,
        }
        
        self.xgboost_params = {
            'n_estimators': 1000,
            'learning_rate': 0.065,
            'max_depth': 10,
            'min_child_weight': 19,
            'subsample': 0.65,
            'colsample_bytree': 0.43,
            'reg_alpha': 6.3,
            'reg_lambda': 5.56,
            'random_state': random_state,
            'verbosity': 0,
        }
    
    def _create_base_models(self):
        """Create base models for the ensemble."""
        self.base_models = [
            ('lightgbm', lgb.LGBMClassifier(**self.lgbm_params)),
            ('catboost', CatBoostClassifier(**self.catboost_params)),
            ('xgboost', xgb.XGBClassifier(**self.xgboost_params))
        ]
    
    def _create_meta_learner(self, meta_learner_type='logistic'):
        """Create the meta-learner."""
        if meta_learner_type == 'logistic':
            self.meta_learner = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear'
            )
        elif meta_learner_type == 'random_forest':
            self.meta_learner = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown meta-learner type: {meta_learner_type}")
    
    def fit(self, X, y, meta_learner_type='logistic'):
        """
        Fit the stacking ensemble.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target variable
            meta_learner_type (str): Type of meta-learner ('logistic' or 'random_forest')
        """
        print("Fitting Stacking Ensemble...")
        
        # Create base models and meta-learner
        self._create_base_models()
        self._create_meta_learner(meta_learner_type)
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        n_classes = len(np.unique(y))
        
        # Generate out-of-fold predictions for meta-features
        meta_features = np.zeros((len(X), len(self.base_models) * n_classes))
        
        print("Training base models with cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{self.n_folds}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train each base model
            for i, (name, model) in enumerate(self.base_models):
                print(f"  Training {name}...")
                
                if name == 'lightgbm':
                    cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        categorical_feature=cat_features,
                        callbacks=[lgb.log_evaluation(period=0)]
                    )
                elif name == 'catboost':
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                else:  # xgboost
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                
                # Get predictions for meta-features
                start_idx = i * n_classes
                end_idx = (i + 1) * n_classes
                meta_features[val_idx, start_idx:end_idx] = model.predict_proba(X_val)
        
        print("Training meta-learner...")
        # Train meta-learner on meta-features
        self.meta_learner.fit(meta_features, y)
        
        # Retrain base models on full dataset
        print("Retraining base models on full dataset...")
        for name, model in self.base_models:
            print(f"  Retraining {name}...")
            if name == 'lightgbm':
                cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
                model.fit(X, y, categorical_feature=cat_features, callbacks=[lgb.log_evaluation(period=0)])
            elif name == 'catboost':
                model.fit(X, y, verbose=0)
            else:  # xgboost
                model.fit(X, y, verbose=0)
        
        self.is_fitted = True
        print("Stacking ensemble training completed!")
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get predictions from base models
        meta_features = np.zeros((len(X), len(self.base_models) * len(self.base_models[0][1].classes_)))
        
        for i, (name, model) in enumerate(self.base_models):
            start_idx = i * len(model.classes_)
            end_idx = (i + 1) * len(model.classes_)
            meta_features[:, start_idx:end_idx] = model.predict_proba(X)
        
        # Get final predictions from meta-learner
        return self.meta_learner.predict_proba(meta_features)
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def save(self, filepath):
        """
        Save the stacking ensemble model.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        model_data = {
            'base_models': self.base_models,
            'meta_learner': self.meta_learner,
            'column_encoder': self.column_encoder,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'n_folds': self.n_folds,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"Stacking ensemble saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a saved stacking ensemble model.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            StackingEnsemble: Loaded model
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(n_folds=model_data['n_folds'], random_state=model_data['random_state'])
        
        # Restore attributes
        instance.base_models = model_data['base_models']
        instance.meta_learner = model_data['meta_learner']
        instance.column_encoder = model_data['column_encoder']
        instance.label_encoder = model_data['label_encoder']
        instance.feature_names = model_data['feature_names']
        instance.is_fitted = model_data['is_fitted']
        
        print(f"Stacking ensemble loaded from: {filepath}")
        return instance

def train_and_save_stacking_model(data_path, model_save_path='model/stacking_model.joblib'):
    """
    Train and save a stacking ensemble model.
    
    Args:
        data_path (str): Path to the training data CSV file
        model_save_path (str): Path to save the trained model
    """
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    print("Preprocessing data...")
    # Encode categorical features
    encoder = DataFrameColumnEncoder(columns=['Soil Type', 'Crop Type'])
    encoder.fit(df.drop('Fertilizer Name', axis=1))
    X = encoder.transform(df.drop('Fertilizer Name', axis=1))
    
    # Label encode target
    le = LabelEncoder()
    y = le.fit_transform(df['Fertilizer Name'])
    y = pd.Series(y)
    
    print("Applying feature engineering...")
    # Apply feature engineering
    X_engineered = feature_engineering(X)
    
    print("Creating and training stacking ensemble...")
    # Create and train stacking ensemble
    stacking_model = StackingEnsemble(n_folds=5, random_state=42)
    stacking_model.column_encoder = encoder
    stacking_model.label_encoder = le
    stacking_model.feature_names = X_engineered.columns.tolist()
    
    # Train the model
    stacking_model.fit(X_engineered, y, meta_learner_type='logistic')
    
    print("Saving model...")
    # Save the model
    stacking_model.save(model_save_path)
    
    # Also save individual components for compatibility
    model_dir = os.path.dirname(model_save_path)
    
    # Save encoders and feature names
    joblib.dump(encoder, os.path.join(model_dir, 'column_encoder.joblib'))
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.joblib'))
    joblib.dump(X_engineered.columns.tolist(), os.path.join(model_dir, 'feature_names.joblib'))
    
    # Create label map for compatibility
    label_map = {i: name for i, name in enumerate(le.classes_)}
    joblib.dump(label_map, os.path.join(model_dir, 'label_map.joblib'))
    
    print("Training completed and model saved!")
    return stacking_model

if __name__ == "__main__":
    # Example usage
    data_path = "model/data.csv"  # Update this path to your data file
    model_save_path = "model/stacking_model.joblib"
    
    try:
        model = train_and_save_stacking_model(data_path, model_save_path)
        print("Stacking model training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}") 