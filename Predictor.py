import pandas as pd
import numpy as np
import joblib
import os
from scipy.stats import mode
import catboost
import lightgbm
import xgboost
from sklearn.preprocessing import LabelEncoder
import warnings
import importlib
import sys
import __main__

warnings.filterwarnings('ignore')
np.random.seed(42)

class DataFrameColumnEncoder:
    """Custom encoder for categorical columns using LabelEncoder."""
    def __init__(self, columns_to_encode):
        self.columns_to_encode = columns_to_encode
        self.encoders_ = {col: LabelEncoder() for col in self.columns_to_encode}
        self.fitted_columns_ = []

    def fit(self, X, y=None):
        self.fitted_columns_ = [col for col in self.columns_to_encode if col in X.columns]
        for col in self.fitted_columns_:
            self.encoders_[col].fit(X[col].astype(str))
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.fitted_columns_:
            try:
                X_transformed[col] = self.encoders_[col].transform(X_transformed[col].astype(str))
            except ValueError as e:
                print(f"Warning: Encountered new value in column '{col}' during transform. This may lead to errors or unexpected behavior if models were not trained to handle unknowns. Error: {e}")
                pass
        return X_transformed
    
    def get_params(self, deep=True):
        return {"columns_to_encode": self.columns_to_encode}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# Register DataFrameColumnEncoder with joblib if needed
setattr(__main__, "DataFrameColumnEncoder", DataFrameColumnEncoder)

def feature_engineering(df_input):
    df = df_input.copy()
    epsilon = 1e-6
    numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    
    if 'Nitrogen' in df.columns and 'Phosphorous' in df.columns:
        df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + epsilon)
    if 'Nitrogen' in df.columns and 'Potassium' in df.columns:
        df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + epsilon)
    if 'Phosphorous' in df.columns and 'Potassium' in df.columns:
        df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + epsilon)
    
    for col in numerical_cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
            df[f'sq_{col}'] = df[col]**2
    
    if 'Temparature' in df.columns and 'Humidity' in df.columns:
        df['Temp_Hum_Interaction'] = df['Temparature'] * df['Humidity']
    if 'Moisture' in df.columns and 'Temparature' in df.columns:
        df['Moisture_Temp_Interaction'] = df['Moisture'] * df['Temparature']
        
    return df

class FertilizerRecommender:
    def __init__(self, model_dir='notebook/Kaggle Notebook and Outputs/trainer output/trained_models_kfold'):
        self.model_dir = model_dir
        self.lgbm_kfold_models = []
        self.catboost_kfold_models = []
        self.xgb_kfold_models = []
        self.meta_model = None
        self.stacking_scaler = None
        self.column_encoder = None
        self.target_encoder = None
        self.feature_names = None
        self.is_loaded = False
        self._load_artifacts()

    def _load_artifacts(self):
        print(f"Loading artifacts from: {self.model_dir}")
        try:
            self.lgbm_kfold_models = joblib.load(os.path.join(self.model_dir, 'lgbm_kfold_models.joblib'))
            self.catboost_kfold_models = joblib.load(os.path.join(self.model_dir, 'catboost_kfold_models.joblib'))
            self.xgb_kfold_models = joblib.load(os.path.join(self.model_dir, 'xgboost_kfold_models.joblib'))
            self.meta_model = joblib.load(os.path.join(self.model_dir, 'meta_model_logistic_regression.joblib'))
            self.stacking_scaler = joblib.load(os.path.join(self.model_dir, 'stacking_scaler.joblib'))
            self.column_encoder = joblib.load(os.path.join(self.model_dir, 'dataframe_column_encoder.joblib'))
            self.target_encoder = joblib.load(os.path.join(self.model_dir, 'target_label_encoder.joblib'))
            self.feature_names = joblib.load(os.path.join(self.model_dir, 'feature_names.joblib'))
            self.is_loaded = True
            print("All models and artifacts loaded successfully.")
            print(f"  Loaded {len(self.lgbm_kfold_models)} LGBM models.")
            print(f"  Loaded {len(self.catboost_kfold_models)} CatBoost models.")
            print(f"  Loaded {len(self.xgb_kfold_models)} XGBoost models.")
            print(f"  Loaded Meta-Model (Logistic Regression).")
        except FileNotFoundError as e:
            print(f"Error loading artifacts: {e}. Ensure all artifacts are in '{self.model_dir}'.")
            self.is_loaded = False
        except Exception as e:
            print(f"An unexpected error occurred during loading: {e}")
            self.is_loaded = False

    def _preprocess_input(self, input_data_raw):
        if not self.is_loaded or self.column_encoder is None:
            raise RuntimeError("Predictor is not loaded or column encoder missing. Call _load_artifacts() or ensure it ran successfully in __init__.")
        
        if isinstance(input_data_raw, dict):
            input_df = pd.DataFrame([input_data_raw])
        elif isinstance(input_data_raw, pd.DataFrame):
            input_df = input_data_raw.copy()
        else:
            raise ValueError("Input data must be a dictionary (for single instance) or a Pandas DataFrame.")

        if 'id' in input_df.columns:
            input_df = input_df.drop('id', axis=1)
        if 'Humidity ' in input_df.columns:
            input_df.rename(columns={'Humidity ': 'Humidity'}, inplace=True)

        encoded_df = self.column_encoder.transform(input_df)
        engineered_df = feature_engineering(encoded_df)
        if self.feature_names is None:
            raise RuntimeError("Feature names not loaded. Cannot preprocess input.")
        final_df = pd.DataFrame(columns=self.feature_names)
        for col in self.feature_names:
            if col in engineered_df.columns:
                final_df[col] = engineered_df[col]
            else:
                final_df[col] = 0
        return final_df[self.feature_names]

    def predict_proba_base_models(self, processed_input_df):
        if self.target_encoder is None:
            raise RuntimeError("Target encoder not loaded. Cannot predict.")
        base_model_avg_probas = {}
        num_classes = len(self.target_encoder.classes_)

        lgbm_agg_proba = np.zeros((len(processed_input_df), num_classes))
        for model in self.lgbm_kfold_models:
            lgbm_agg_proba += model.predict_proba(processed_input_df)
        base_model_avg_probas['lgbm'] = lgbm_agg_proba / len(self.lgbm_kfold_models)

        catboost_agg_proba = np.zeros((len(processed_input_df), num_classes))
        for model in self.catboost_kfold_models:
            catboost_agg_proba += model.predict_proba(processed_input_df)
        base_model_avg_probas['catboost'] = catboost_agg_proba / len(self.catboost_kfold_models)

        xgb_agg_proba = np.zeros((len(processed_input_df), num_classes))
        for model in self.xgb_kfold_models:
            xgb_agg_proba += model.predict_proba(processed_input_df)
        base_model_avg_probas['xgboost'] = xgb_agg_proba / len(self.xgb_kfold_models)
        
        return base_model_avg_probas

    def predict(self, input_data_raw, top_k=1):
        if not self.is_loaded or self.stacking_scaler is None or self.meta_model is None or self.target_encoder is None:
            print("Models or required artifacts not loaded. Cannot predict.")
            return None

        processed_df = self._preprocess_input(input_data_raw)
        base_model_avg_probas = self.predict_proba_base_models(processed_df)
        stacked_features_for_meta = np.concatenate([
            base_model_avg_probas['lgbm'],
            base_model_avg_probas['catboost'],
            base_model_avg_probas['xgboost']
        ], axis=1)
        stacked_features_scaled = self.stacking_scaler.transform(stacked_features_for_meta)
        final_pred_proba = self.meta_model.predict_proba(stacked_features_scaled)
        if final_pred_proba.ndim == 1:
            final_pred_proba = final_pred_proba.reshape(1, -1)
        top_k_indices = np.argsort(final_pred_proba, axis=1)[:, -top_k:][:, ::-1]
        top_k_probabilities = np.array([final_pred_proba[i, top_k_indices[i]] for i in range(len(top_k_indices))])
        top_k_labels_encoded = top_k_indices
        results = []
        for i in range(len(top_k_labels_encoded)):
            labels = self.target_encoder.inverse_transform(top_k_labels_encoded[i])
            probs = top_k_probabilities[i]
            results.append(list(zip(labels, probs)))
        if isinstance(input_data_raw, dict):
            return results[0]
        return results

    def predict_top3(self, input_data):
        """Return the top 3 fertilizer names as a space-separated string (single input) or list of strings (batch)."""
        results = self.predict(input_data, top_k=3)
        if results is None:
            return ""
        # Single input (dict): results is a list of (fertilizer, prob)
        if isinstance(input_data, dict):
            return " ".join([fert for fert, _ in results])
        # Batch input (DataFrame): results is a list of lists
        print(results)
        return [" ".join([fert for fert, _ in row]) for row in results]

# Example usage
# if __name__ == "__main__":
#     recommender = FertilizerRecommender(model_dir='trained_models_kfold')
#     input_data_single = {
#         'Temparature': 30,
#         'Humidity': 80,
#         'Moisture': 20,
#         'Soil Type': 'Sandy',
#         'Crop Type': 'Wheat',
#         'Nitrogen': 50, 
#         'Potassium': 30, 
#         'Phosphorous': 70
#     }
#     if recommender.is_loaded:
#         print("--- Single Input Prediction (Top 3) ---")
#         top3 = recommender.predict(input_data_single, top_k=3)
#         print(f"Top 3 Fertilizers and Probabilities: {top3}")
#         print("--- Single Input Prediction (Top 1) ---")
#         top1 = recommender.predict(input_data_single, top_k=1)
#         print(f"Top 1 Fertilizer and Probability: {top1}")
#     else:
#         print("Models did not load. Cannot run example usage.")