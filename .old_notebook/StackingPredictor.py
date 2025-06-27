import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from stacking_model import StackingEnsemble, feature_engineering

class StackingFertilizerPredictor:
    """
    A fertilizer predictor that uses a stacking ensemble model.
    
    This class provides functionality to:
    - Load trained stacking ensemble models
    - Make predictions on new datasets
    - Evaluate model performance
    - Generate detailed prediction reports
    """
    
    def __init__(self, model_dir='model'):
        """
        Initialize the predictor with model directory.
        
        Args:
            model_dir (str): Directory containing saved models and artifacts
        """
        self.model_dir = model_dir
        self.stacking_model = None
        self.column_encoder = None
        self.label_encoder = None
        self.feature_names = None
        self.label_map = None
        self.is_loaded = False
    
    def load_models(self):
        """
        Load the trained stacking ensemble model and preprocessing artifacts.
        
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            print("Loading stacking ensemble model and artifacts...")
            
            # Load stacking model
            stacking_model_path = os.path.join(self.model_dir, 'stacking_model.joblib')
            if os.path.exists(stacking_model_path):
                self.stacking_model = StackingEnsemble.load(stacking_model_path)
                self.column_encoder = self.stacking_model.column_encoder
                self.label_encoder = self.stacking_model.label_encoder
                self.feature_names = self.stacking_model.feature_names
            else:
                # Try loading individual components
                self.column_encoder = joblib.load(os.path.join(self.model_dir, 'column_encoder.joblib'))
                self.label_encoder = joblib.load(os.path.join(self.model_dir, 'label_encoder.joblib'))
                self.feature_names = joblib.load(os.path.join(self.model_dir, 'feature_names.joblib'))
                self.label_map = joblib.load(os.path.join(self.model_dir, 'label_map.joblib'))
                
                print("Warning: Stacking model not found, using individual components only.")
            
            self.is_loaded = True
            print("✓ Stacking ensemble model and artifacts loaded successfully!")
            
            if self.stacking_model:
                print(f"  - Stacking model: {type(self.stacking_model).__name__}")
                print(f"  - Base models: {len(self.stacking_model.base_models)}")
                print(f"  - Meta-learner: {type(self.stacking_model.meta_learner).__name__}")
            
            print(f"  - Feature names: {len(self.feature_names) if self.feature_names else 0}")
            print(f"  - Number of classes: {len(self.label_encoder.classes_) if self.label_encoder else 0}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            print("Please ensure all model files exist in the model directory.")
            return False
    
    def preprocess_data(self, df, is_training_data=False):
        """
        Preprocess input data using the same pipeline as training.
        
        Args:
            df (pd.DataFrame): Input data
            is_training_data (bool): Whether this is training data (has target column)
            
        Returns:
            tuple: (processed_features, target_series) if training data, else processed_features
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        # Make a copy to avoid modifying original data
        df = df.copy()
        
        # Remove 'id' column if present
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        
        # Separate target if training data
        target = None
        if is_training_data and 'Fertilizer Name' in df.columns:
            target = df['Fertilizer Name'].copy()
            df = df.drop('Fertilizer Name', axis=1)
        
        # Encode categorical features
        X_encoded = self.column_encoder.transform(df)
        
        # Apply feature engineering
        X_engineered = feature_engineering(X_encoded)
        
        # Ensure all expected features are present
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(X_engineered.columns)
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
                for feature in missing_features:
                    X_engineered[feature] = 0
            
            # Ensure correct order of features
            X_engineered = X_engineered[self.feature_names]
        
        if is_training_data:
            return X_engineered, target
        else:
            return X_engineered
    
    def predict_proba(self, input_data):
        """
        Predict class probabilities using the stacking ensemble.
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted probabilities for each class
        """
        if not self.is_loaded or self.stacking_model is None:
            raise ValueError("Stacking model not loaded. Call load_models() first.")
        
        # Preprocess input data
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        X_processed = self.preprocess_data(input_df)
        
        # Get predictions from stacking model
        probabilities = self.stacking_model.predict_proba(X_processed)
        
        return probabilities
    
    def predict_top3(self, input_data):
        """
        Predict top 3 fertilizer recommendations.
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            
        Returns:
            str: Space-separated top 3 fertilizer names
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        probabilities = self.predict_proba(input_data)
        
        # Get indices of top 3 predictions
        top_3_indices = np.argsort(probabilities[0])[-3:][::-1]
        
        # Get fertilizer names
        if self.label_encoder:
            fertilizer_names = [self.label_encoder.classes_[idx] for idx in top_3_indices]
        elif self.label_map:
            fertilizer_names = [self.label_map[idx] for idx in top_3_indices]
        else:
            fertilizer_names = [f"Fertilizer_{idx}" for idx in top_3_indices]
        
        return " ".join(fertilizer_names)
    
    def predict(self, input_data):
        """
        Predict the most likely fertilizer.
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            
        Returns:
            str: Predicted fertilizer name
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        probabilities = self.predict_proba(input_data)
        
        # Get the most likely class
        predicted_idx = np.argmax(probabilities[0])
        
        # Get fertilizer name
        if self.label_encoder:
            fertilizer_name = self.label_encoder.classes_[predicted_idx]
        elif self.label_map:
            fertilizer_name = self.label_map[predicted_idx]
        else:
            fertilizer_name = f"Fertilizer_{predicted_idx}"
        
        return fertilizer_name
    
    def evaluate(self, df, top_k=3):
        """
        Evaluate the model on a test dataset.
        
        Args:
            df (pd.DataFrame): Test data with 'Fertilizer Name' column
            top_k (int): Number of top predictions to consider
            
        Returns:
            dict: Evaluation results
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        if 'Fertilizer Name' not in df.columns:
            raise ValueError("Test data must contain 'Fertilizer Name' column")
        
        # Preprocess data
        X_processed, y_true = self.preprocess_data(df, is_training_data=True)
        
        # Get predictions
        probabilities = self.stacking_model.predict_proba(X_processed)
        
        # Get top-k predictions
        top_k_indices = np.argsort(probabilities, axis=1)[:, -top_k:][:, ::-1]
        top_k_probs = np.take_along_axis(probabilities, top_k_indices, axis=1)
        
        # Convert indices to labels
        if self.label_encoder:
            top_k_labels = self.label_encoder.inverse_transform(top_k_indices.flatten()).reshape(top_k_indices.shape)
        elif self.label_map:
            top_k_labels = np.array([[self.label_map[idx] for idx in row] for row in top_k_indices])
        else:
            top_k_labels = np.array([[f"Fertilizer_{idx}" for idx in row] for row in top_k_indices])
        
        # Calculate metrics
        accuracy_at_1 = np.mean(top_k_labels[:, 0] == y_true)
        hit_rate = np.mean([y_true.iloc[i] in top_k_labels[i] for i in range(len(y_true))])
        
        # Calculate MAP@k
        y_true_encoded = self.label_encoder.transform(y_true) if self.label_encoder else y_true
        map_at_k_score = self._map_at_k(top_k_probs, y_true_encoded, k=top_k)
        
        # Detailed results
        detailed_results = []
        for i in range(len(df)):
            result = {
                'id': df.iloc[i].get('id', i),
                'true_fertilizer': y_true.iloc[i],
                'predicted_fertilizers': list(top_k_labels[i]),
                'predicted_probabilities': list(top_k_probs[i]),
                'is_correct_at_1': top_k_labels[i, 0] == y_true.iloc[i],
                'is_correct_in_top_k': y_true.iloc[i] in top_k_labels[i],
                'true_position': np.where(top_k_labels[i] == y_true.iloc[i])[0][0] + 1 if y_true.iloc[i] in top_k_labels[i] else None
            }
            detailed_results.append(result)
        
        # Confusion matrix for top-1 predictions
        cm = confusion_matrix(y_true, top_k_labels[:, 0])
        
        # Classification report for top-1 predictions
        class_report = classification_report(y_true, top_k_labels[:, 0], output_dict=True)
        
        evaluation_results = {
            'map_at_k': map_at_k_score,
            'accuracy_at_1': accuracy_at_1,
            'hit_rate': hit_rate,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'detailed_results': detailed_results,
            'top_k': top_k,
            'n_samples': len(df)
        }
        
        return evaluation_results
    
    def _map_at_k(self, probs, true_labels, k=3):
        """Calculate Mean Average Precision at k."""
        def average_precision_at_k(y_true, y_pred_topk):
            """Calculate Average Precision at k for a single sample."""
            for i, pred in enumerate(y_pred_topk):
                if pred == y_true:
                    return 1.0 / (i + 1)
            return 0.0
        
        top_k_preds = np.argsort(probs, axis=1)[:, ::-1][:, :k]
        
        ap_scores = []
        for i in range(len(true_labels)):
            ap = average_precision_at_k(true_labels[i], top_k_preds[i])
            ap_scores.append(ap)
        return np.mean(ap_scores)
    
    def print_evaluation_summary(self, evaluation_results):
        """
        Print a summary of evaluation results.
        
        Args:
            evaluation_results (dict): Results from evaluate() method
        """
        print("\n" + "="*60)
        print("STACKING ENSEMBLE EVALUATION SUMMARY")
        print("="*60)
        print(f"Number of samples: {evaluation_results['n_samples']}")
        print(f"Top-k predictions: {evaluation_results['top_k']}")
        print(f"MAP@{evaluation_results['top_k']}: {evaluation_results['map_at_k']:.5f}")
        print(f"Accuracy at position 1: {evaluation_results['accuracy_at_1']:.5f}")
        print(f"Hit rate (correct in top-{evaluation_results['top_k']}): {evaluation_results['hit_rate']:.5f}")
        
        # Print confusion matrix
        print(f"\nConfusion Matrix (Top-1 predictions):")
        print(evaluation_results['confusion_matrix'])
        
        # Print classification report
        print(f"\nClassification Report (Top-1 predictions):")
        print(classification_report(
            [r['true_fertilizer'] for r in evaluation_results['detailed_results']],
            [r['predicted_fertilizers'][0] for r in evaluation_results['detailed_results']]
        ))
        
        # Print some examples
        print(f"\nSample Predictions:")
        print("-" * 80)
        for i, result in enumerate(evaluation_results['detailed_results'][:5]):
            print(f"Sample {i+1}:")
            print(f"  True: {result['true_fertilizer']}")
            print(f"  Predicted: {result['predicted_fertilizers']}")
            print(f"  Probabilities: {[f'{p:.3f}' for p in result['predicted_probabilities']]}")
            print(f"  Correct at 1: {result['is_correct_at_1']}")
            print(f"  Correct in top-{evaluation_results['top_k']}: {result['is_correct_in_top_k']}")
            if result['true_position']:
                print(f"  True position: {result['true_position']}")
            print()
    
    def save_predictions(self, predictions, filepath):
        """
        Save predictions to a CSV file.
        
        Args:
            predictions (pd.DataFrame): Predictions DataFrame
            filepath (str): Path to save the file
        """
        predictions.to_csv(filepath, index=False)
        print(f"Predictions saved to: {filepath}")

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = StackingFertilizerPredictor(model_dir='model')
    
    # Load models
    if predictor.load_models():
        # Example prediction
        sample_input = {
            'Temparature': 25.0,
            'Humidity': 60.0,
            'Moisture': 0.5,
            'Soil Type': 'Sandy',
            'Crop Type': 'Wheat',
            'Nitrogen': 50.0,
            'Potassium': 30.0,
            'Phosphorous': 20.0
        }
        
        # Make predictions
        top3_prediction = predictor.predict_top3(sample_input)
        single_prediction = predictor.predict(sample_input)
        probabilities = predictor.predict_proba(sample_input)
        
        print(f"Top 3 predictions: {top3_prediction}")
        print(f"Single prediction: {single_prediction}")
        print(f"Probabilities: {probabilities[0]}")
    else:
        print("Failed to load models. Please ensure the stacking model is trained and saved.") 