import pandas as pd
import numpy as np
import joblib
import os
from scipy.stats import mode
import catboost
import lightgbm
import xgboost

class FertilizerPredictor:
    def __init__(self, model_dir='model', # Adjusted default model_dir
                 model_filenames=['lightgbm_model.joblib', 
                                  'catboost_model.joblib', 
                                  'xgboost_ensemble_model.joblib']):
        
        self.model_paths = [os.path.join(model_dir, fname) for fname in model_filenames]
        # Artifacts like label_map and feature_names are assumed to be in the same model_dir
        self.label_map_path = os.path.join(model_dir, 'label_map.joblib') 
        self.feature_names_path = os.path.join(model_dir, 'feature_names.joblib')
        
        self.models = []
        self.original_numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
        self.original_categorical_features = ['Soil Type', 'Crop Type']
        
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            for model_path in self.model_paths:
                self.models.append(joblib.load(model_path))
            self.index_to_name_map = joblib.load(self.label_map_path)
            self.trained_feature_names = joblib.load(self.feature_names_path)
            print(f"{len(self.models)} models and other artifacts loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading artifacts: {e}. Ensure models and artifacts are in specified paths.")
            self.models = []
            self.index_to_name_map = None
            self.trained_feature_names = None
        except Exception as e:
            print(f"An unexpected error occurred while loading artifacts: {e}")
            self.models = []
            self.index_to_name_map = None
            self.trained_feature_names = None

    def _preprocess_input(self, input_data):
        if not isinstance(input_data, pd.DataFrame):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        if 'Humidity ' in input_df.columns:
            input_df.rename(columns={'Humidity ': 'Humidity'}, inplace=True)

        for col in self.original_categorical_features:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype('category')
        
        input_df_encoded = pd.get_dummies(input_df, columns=self.original_categorical_features, prefix=self.original_categorical_features, dtype=int)
        return input_df_encoded

    def _engineer_features(self, input_df):
        df_eng = input_df.copy()
        epsilon = 1e-6
        if 'Nitrogen' in df_eng.columns and 'Phosphorous' in df_eng.columns:
            df_eng['N_P_ratio'] = df_eng['Nitrogen'] / (df_eng['Phosphorous'] + epsilon)
        if 'Nitrogen' in df_eng.columns and 'Potassium' in df_eng.columns:
            df_eng['N_K_ratio'] = df_eng['Nitrogen'] / (df_eng['Potassium'] + epsilon)
        if 'Phosphorous' in df_eng.columns and 'Potassium' in df_eng.columns:
            df_eng['P_K_ratio'] = df_eng['Phosphorous'] / (df_eng['Potassium'] + epsilon)
        if 'Temparature' in df_eng.columns and 'Humidity' in df_eng.columns:
            df_eng['Temp_Hum_Interaction'] = df_eng['Temparature'] * df_eng['Humidity']
        if 'Temparature' in df_eng.columns and 'Moisture' in df_eng.columns:
            df_eng['Moisture_Temp_Interaction'] = df_eng['Moisture'] * df_eng['Temparature']
        for col in self.original_numerical_features:
            if col in df_eng.columns:
                df_eng[f'log_{col}'] = np.log1p(df_eng[col])
                df_eng[f'sq_{col}'] = df_eng[col]**2
        return df_eng

    def _align_features(self, input_df_engineered):
        aligned_df = pd.DataFrame(columns=self.trained_feature_names)
        for col in self.trained_feature_names:
            if col in input_df_engineered.columns:
                aligned_df[col] = input_df_engineered[col]
            else:
                aligned_df[col] = 0 
        return aligned_df[self.trained_feature_names]

    def predict(self, input_data, strategy='hard_voting'):
        if not self.models or self.index_to_name_map is None or self.trained_feature_names is None:
            print("Predictor not initialized properly or no models loaded. Cannot predict.")
            return None

        processed_df = self._preprocess_input(input_data)
        engineered_df = self._engineer_features(processed_df)
        final_df = self._align_features(engineered_df)

        try:
            if strategy == 'hard_voting':
                all_predictions = []
                for model in self.models:
                    # Ensure predictions are always 1D for individual samples or 2D for batch
                    pred = model.predict(final_df)
                    if pred.ndim == 0: # Handle scalar prediction for single input
                        pred = np.array([pred])
                    elif pred.ndim == 2 and pred.shape[1] == 1: # Handle (N, 1) predictions
                         pred = pred.flatten()
                    all_predictions.append(pred)

                # Stack predictions. Transpose if needed so that each row is a sample and columns are model predictions.
                # Then take mode along axis 0 to get the most frequent prediction for each sample.
                all_predictions_stacked = np.array(all_predictions)
                
                # If there's only one sample, mode might return a scalar. Ensure it's an array.
                ensemble_prediction_encoded, _ = mode(all_predictions_stacked, axis=0, keepdims=False)

                if not isinstance(ensemble_prediction_encoded, np.ndarray):
                    ensemble_prediction_encoded = np.array([ensemble_prediction_encoded])

            elif strategy == 'soft_voting':
                all_proba_predictions = []
                for model in self.models:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(final_df)
                        all_proba_predictions.append(proba)
                    else:
                        # Fallback for models without predict_proba: one-hot encode predictions
                        num_classes = len(self.index_to_name_map)
                        predictions = model.predict(final_df)
                        # Ensure predictions is an array for consistent indexing
                        if predictions.ndim == 0:
                            predictions = np.array([predictions])
                        elif predictions.ndim == 2 and predictions.shape[1] == 1:
                            predictions = predictions.flatten()

                        proba_like = np.zeros((final_df.shape[0], num_classes))
                        for i, pred_class_val in enumerate(predictions):
                            # Ensure pred_class_val is an integer and valid index
                            if isinstance(pred_class_val, (np.integer, int)) and 0 <= pred_class_val < num_classes:
                                proba_like[i, pred_class_val] = 1.0
                            else:
                                print(f"Warning: Model predicted {pred_class_val} which is not a valid class index. Skipping.")
                        all_proba_predictions.append(proba_like)


                avg_proba = np.mean(np.array(all_proba_predictions), axis=0)
                ensemble_prediction_encoded = np.argmax(avg_proba, axis=1)

            else:
                raise ValueError("Unsupported voting strategy. Choose 'hard_voting' or 'soft_voting'.")

            predicted_fertilizer_names = [self.index_to_name_map.get(pred_code, "Unknown") for pred_code in ensemble_prediction_encoded]

            return predicted_fertilizer_names[0] if len(predicted_fertilizer_names) == 1 and not isinstance(input_data, pd.DataFrame) else predicted_fertilizer_names
        except Exception as e:
            print(f"Error during ensemble prediction: {e}")
            print("Input data after processing and engineering leading to error:")
            print(final_df.head())
            print(f"Expected columns: {self.trained_feature_names}")
            return None
        
    def predict_proba(self, input_data):
        if not self.models or self.trained_feature_names is None:
            print("Predictor not initialized properly or no models loaded. Cannot predict probabilities.")
            return None
        
        processed_df = self._preprocess_input(input_data)
        engineered_df = self._engineer_features(processed_df)
        final_df = self._align_features(engineered_df)
        
        try:
            all_proba_predictions = []
            for model in self.models:
                if hasattr(model, 'predict_proba'):
                    all_proba_predictions.append(model.predict_proba(final_df))
                else:
                    # Fallback for models without predict_proba: one-hot encode predictions
                    # This is a simplification; proper handling might require calibration or different ensemble strategy
                    num_classes = len(self.index_to_name_map)
                    predictions = model.predict(final_df)
                    proba_like = np.zeros((final_df.shape[0], num_classes))
                    for i, pred_class in enumerate(predictions):
                        proba_like[i, pred_class] = 1.0
                    all_proba_predictions.append(proba_like)
            
            avg_proba = np.mean(np.array(all_proba_predictions), axis=0)
            return avg_proba
        except Exception as e:
            print(f"Error during ensemble probability prediction: {e}")
            return None
            
    def predict_top_n(self, input_data, n=3):
        probabilities = self.predict_proba(input_data)
        if probabilities is None:
            return None
        
        # Ensure probabilities is a 2D array
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(1, -1)
            
        top_n_indices = np.argsort(probabilities, axis=1)[:, -n:][:, ::-1]
        top_n_fertilizers = []
        for indices_row in top_n_indices:
            fertilizers_row = [self.index_to_name_map.get(idx, "Unknown") for idx in indices_row]
            top_n_fertilizers.append(fertilizers_row)
            
        if not isinstance(input_data, pd.DataFrame) or (isinstance(input_data, pd.DataFrame) and len(input_data) == 1):
            return top_n_fertilizers[0] if top_n_fertilizers else [] 
        return top_n_fertilizers
