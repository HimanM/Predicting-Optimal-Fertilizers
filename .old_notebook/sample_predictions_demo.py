#!/usr/bin/env python3
"""
Sample Predictions Demo using FertilizerPredictor Class

This script demonstrates how to use the FertilizerPredictor class to make
predictions on a sample of 10 rows from the training dataset.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Import the FertilizerPredictor class
from complete_kfold_fertilizer_prediction import FertilizerPredictor

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

def load_training_data():
    """
    Load the training dataset.
    
    Returns:
        pd.DataFrame: Training dataset
    """
    try:
        # Try to load from model directory first
        if os.path.exists('model/data.csv'):
            df = pd.read_csv('model/data.csv')
            print("✓ Training data loaded from model/data.csv")
        elif os.path.exists('data.csv'):
            df = pd.read_csv('data.csv')
            print("✓ Training data loaded from data.csv")
        else:
            print("✗ Training data not found. Please ensure data.csv exists.")
            return None
            
        print(f"  - Shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"✗ Error loading training data: {e}")
        return None

def create_sample_confusion_matrix(y_true, y_pred, class_names, save_path='sample_confusion_matrix.png'):
    """
    Create and save a confusion matrix visualization for sample predictions.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list): Names of the classes
        save_path (str): Path to save the confusion matrix plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Sample Predictions: Confusion Matrix (10 Rows)', fontsize=14, pad=20)
    plt.xlabel('Predicted Fertilizer Name', fontsize=12)
    plt.ylabel('True Fertilizer Name', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sample confusion matrix saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    return cm

def main():
    """
    Main function to demonstrate sample predictions.
    """
    print("="*60)
    print("SAMPLE PREDICTIONS DEMO USING FERTILIZER PREDICTOR")
    print("="*60)
    
    # Step 1: Load training data
    print("\n1. Loading training data...")
    train_df = load_training_data()
    
    if train_df is None:
        print("✗ Failed to load training data. Exiting.")
        return
    
    # Step 2: Select 10 random samples
    print("\n2. Selecting 10 random samples...")
    sample_df = train_df.sample(n=10, random_state=42).reset_index(drop=True)
    
    print(f"✓ Selected 10 random samples")
    print("\nSample data:")
    print(sample_df[['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Fertilizer Name']].head())
    
    # Step 3: Initialize and load the predictor
    print("\n3. Loading trained models...")
    predictor = FertilizerPredictor()
    
    if not predictor.load_models():
        print("✗ Failed to load models. Please run the training script first.")
        return
    
    # Step 4: Make predictions on the sample
    print("\n4. Making predictions on sample data...")
    try:
        # Get predictions with top-3 recommendations
        predictions = predictor.predict(sample_df, top_k=3)
        
        print(f"✓ Predictions completed for {len(sample_df)} samples")
        
        # Get the predicted fertilizer names
        predicted_fertilizers = predictions['Fertilizer_1'].values
        true_fertilizers = sample_df['Fertilizer Name'].values
        
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        return
    
    # Step 5: Display detailed predictions
    print("\n5. Detailed Sample Predictions:")
    print("="*80)
    
    for i in range(len(sample_df)):
        print(f"\nSample {i+1}:")
        print(f"  Input Features:")
        print(f"    - Temperature: {sample_df.iloc[i]['Temperature']}")
        print(f"    - Humidity: {sample_df.iloc[i]['Humidity']}")
        print(f"    - Moisture: {sample_df.iloc[i]['Moisture']}")
        print(f"    - Soil Type: {sample_df.iloc[i]['Soil Type']}")
        print(f"    - Crop Type: {sample_df.iloc[i]['Crop Type']}")
        print(f"    - Nitrogen: {sample_df.iloc[i]['Nitrogen']}")
        print(f"    - Phosphorous: {sample_df.iloc[i]['Phosphorous']}")
        print(f"    - Potassium: {sample_df.iloc[i]['Potassium']}")
        print(f"    - pH: {sample_df.iloc[i]['pH']}")
        
        print(f"  True Fertilizer: {true_fertilizers[i]}")
        print(f"  Predictions:")
        print(f"    1st: {predictions.iloc[i]['Fertilizer_1']} (prob: {predictions.iloc[i]['Probability_1']:.3f})")
        print(f"    2nd: {predictions.iloc[i]['Fertilizer_2']} (prob: {predictions.iloc[i]['Probability_2']:.3f})")
        print(f"    3rd: {predictions.iloc[i]['Fertilizer_3']} (prob: {predictions.iloc[i]['Probability_3']:.3f})")
        
        is_correct = predicted_fertilizers[i] == true_fertilizers[i]
        print(f"  Correct at 1st: {'✓' if is_correct else '✗'}")
        
        # Check if true fertilizer is in top-3
        top_3_preds = [predictions.iloc[i]['Fertilizer_1'], 
                      predictions.iloc[i]['Fertilizer_2'], 
                      predictions.iloc[i]['Fertilizer_3']]
        in_top_3 = true_fertilizers[i] in top_3_preds
        print(f"  In top-3: {'✓' if in_top_3 else '✗'}")
        
        if in_top_3:
            position = top_3_preds.index(true_fertilizers[i]) + 1
            print(f"  True position: {position}")
    
    # Step 6: Evaluate the sample predictions
    print("\n6. Evaluating sample predictions...")
    try:
        evaluation_results = predictor.evaluate(sample_df, top_k=3)
        
        print(f"\nSample Evaluation Results:")
        print(f"  - Accuracy at 1st position: {evaluation_results['accuracy_at_1']:.3f}")
        print(f"  - Hit rate (correct in top-3): {evaluation_results['hit_rate']:.3f}")
        print(f"  - MAP@3: {evaluation_results['map_at_k']:.5f}")
        
    except Exception as e:
        print(f"✗ Error evaluating predictions: {e}")
    
    # Step 7: Create confusion matrix for sample
    print("\n7. Creating confusion matrix for sample...")
    try:
        # Get unique class names
        class_names = sorted(list(set(true_fertilizers) | set(predicted_fertilizers)))
        
        # Create confusion matrix
        cm = create_confusion_matrix(
            true_fertilizers, 
            predicted_fertilizers, 
            class_names,
            'sample_confusion_matrix.png'
        )
        
        print(f"✓ Confusion matrix created with {len(class_names)} classes")
        
    except Exception as e:
        print(f"✗ Error creating confusion matrix: {e}")
    
    # Step 8: Save detailed results
    print("\n8. Saving detailed results...")
    try:
        # Create detailed results DataFrame
        detailed_results = []
        for i in range(len(sample_df)):
            result = {
                'sample_id': i + 1,
                'temperature': sample_df.iloc[i]['Temperature'],
                'humidity': sample_df.iloc[i]['Humidity'],
                'moisture': sample_df.iloc[i]['Moisture'],
                'soil_type': sample_df.iloc[i]['Soil Type'],
                'crop_type': sample_df.iloc[i]['Crop Type'],
                'nitrogen': sample_df.iloc[i]['Nitrogen'],
                'phosphorous': sample_df.iloc[i]['Phosphorous'],
                'potassium': sample_df.iloc[i]['Potassium'],
                'ph': sample_df.iloc[i]['pH'],
                'true_fertilizer': true_fertilizers[i],
                'predicted_1st': predictions.iloc[i]['Fertilizer_1'],
                'predicted_2nd': predictions.iloc[i]['Fertilizer_2'],
                'predicted_3rd': predictions.iloc[i]['Fertilizer_3'],
                'probability_1st': predictions.iloc[i]['Probability_1'],
                'probability_2nd': predictions.iloc[i]['Probability_2'],
                'probability_3rd': predictions.iloc[i]['Probability_3'],
                'is_correct_at_1': predicted_fertilizers[i] == true_fertilizers[i],
                'is_correct_in_top_3': true_fertilizers[i] in [predictions.iloc[i]['Fertilizer_1'], 
                                                              predictions.iloc[i]['Fertilizer_2'], 
                                                              predictions.iloc[i]['Fertilizer_3']]
            }
            detailed_results.append(result)
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv('sample_predictions_detailed.csv', index=False)
        print("✓ Detailed results saved to: sample_predictions_detailed.csv")
        
    except Exception as e:
        print(f"✗ Error saving results: {e}")
    
    # Step 9: Summary statistics
    print("\n9. Summary Statistics:")
    print("="*40)
    
    correct_at_1 = sum(predicted_fertilizers == true_fertilizers)
    correct_in_top_3 = sum([true_fertilizers[i] in [predictions.iloc[i]['Fertilizer_1'], 
                                                   predictions.iloc[i]['Fertilizer_2'], 
                                                   predictions.iloc[i]['Fertilizer_3']] 
                           for i in range(len(sample_df))])
    
    print(f"Total samples: {len(sample_df)}")
    print(f"Correct at 1st position: {correct_at_1}/{len(sample_df)} ({correct_at_1/len(sample_df):.1%})")
    print(f"Correct in top-3: {correct_in_top_3}/{len(sample_df)} ({correct_in_top_3/len(sample_df):.1%})")
    
    # Show fertilizer distribution
    print(f"\nFertilizer Distribution in Sample:")
    true_dist = pd.Series(true_fertilizers).value_counts()
    for fert, count in true_dist.items():
        print(f"  {fert}: {count} samples")
    
    # Step 10: Final summary
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS DEMO COMPLETED!")
    print("="*60)
    print(f"✓ Sample size: {len(sample_df)} rows")
    print(f"✓ Accuracy at 1st: {correct_at_1/len(sample_df):.1%}")
    print(f"✓ Hit rate (top-3): {correct_in_top_3/len(sample_df):.1%}")
    print(f"✓ Confusion matrix: sample_confusion_matrix.png")
    print(f"✓ Detailed results: sample_predictions_detailed.csv")
    print("="*60)

if __name__ == "__main__":
    main() 