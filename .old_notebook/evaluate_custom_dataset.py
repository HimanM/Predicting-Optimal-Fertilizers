#!/usr/bin/env python3
"""
Evaluate Custom Dataset with Trained K-Fold Ensemble Models

This script loads a custom dataset from Kaggle, makes predictions using the trained
K-fold ensemble models, and creates a confusion matrix comparing truth vs predicted values.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys

# Add the current directory to path to import from complete_kfold_fertilizer_prediction
sys.path.append('.')

# Import the FertilizerPredictor class from the main script
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

def load_custom_dataset(dataset_path='/kaggle/input/fertilizer-prediction'):
    """
    Load the custom dataset from the specified path.
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading dataset from: {dataset_path}")
    
    # Try to find CSV files in the directory
    csv_files = []
    for file in os.listdir(dataset_path):
        if file.endswith('.csv'):
            csv_files.append(file)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_path}")
    
    print(f"Found CSV files: {csv_files}")
    
    # Load the first CSV file (assuming it's the main dataset)
    dataset_file = os.path.join(dataset_path, csv_files[0])
    df = pd.read_csv(dataset_file)
    
    print(f"✓ Dataset loaded successfully!")
    print(f"  - Shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
    
    return df

def create_confusion_matrix(y_true, y_pred, class_names, save_path='custom_dataset_confusion_matrix.png'):
    """
    Create and save a confusion matrix visualization.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list): Names of the classes
        save_path (str): Path to save the confusion matrix plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix: Truth vs Predicted Fertilizer Names', fontsize=16, pad=20)
    plt.xlabel('Predicted Fertilizer Name', fontsize=12)
    plt.ylabel('True Fertilizer Name', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {save_path}")
    
    # Show the plot
    plt.show()
    
    return cm

def analyze_predictions(evaluation_results, save_path='custom_dataset_analysis.csv'):
    """
    Analyze and save detailed prediction results.
    
    Args:
        evaluation_results (dict): Results from predictor.evaluate()
        save_path (str): Path to save the analysis CSV
    """
    # Create detailed analysis DataFrame
    analysis_data = []
    
    for result in evaluation_results['detailed_results']:
        analysis_data.append({
            'id': result['id'],
            'true_fertilizer': result['true_fertilizer'],
            'predicted_1st': result['predicted_fertilizers'][0],
            'predicted_2nd': result['predicted_fertilizers'][1],
            'predicted_3rd': result['predicted_fertilizers'][2],
            'probability_1st': result['predicted_probabilities'][0],
            'probability_2nd': result['predicted_probabilities'][1],
            'probability_3rd': result['predicted_probabilities'][2],
            'is_correct_at_1': result['is_correct_at_1'],
            'is_correct_in_top_3': result['is_correct_in_top_k'],
            'true_position': result['true_position']
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    
    # Save to CSV
    analysis_df.to_csv(save_path, index=False)
    print(f"✓ Detailed analysis saved to: {save_path}")
    
    return analysis_df

def main():
    """
    Main function to evaluate the custom dataset.
    """
    print("="*60)
    print("EVALUATING CUSTOM DATASET WITH K-FOLD ENSEMBLE MODELS")
    print("="*60)
    
    # Step 1: Load the custom dataset
    print("\n1. Loading custom dataset...")
    try:
        custom_df = load_custom_dataset('/kaggle/input/fertilizer-prediction')
        
        # Check if 'Fertilizer Name' column exists
        if 'Fertilizer Name' not in custom_df.columns:
            print("✗ Error: 'Fertilizer Name' column not found in the dataset")
            print(f"Available columns: {list(custom_df.columns)}")
            return
            
        print(f"✓ Dataset contains {len(custom_df)} samples with fertilizer names")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return
    
    # Step 2: Initialize and load the predictor
    print("\n2. Loading trained models...")
    predictor = FertilizerPredictor()
    
    if not predictor.load_models():
        print("✗ Failed to load models. Please run the training script first.")
        return
    
    # Step 3: Make predictions
    print("\n3. Making predictions...")
    try:
        # Get predictions (top-1 for confusion matrix)
        predictions = predictor.predict(custom_df, top_k=1)
        
        # Get the predicted fertilizer names
        predicted_fertilizers = predictions['Fertilizer_1'].values
        true_fertilizers = custom_df['Fertilizer Name'].values
        
        print(f"✓ Predictions completed for {len(custom_df)} samples")
        
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        return
    
    # Step 4: Evaluate predictions
    print("\n4. Evaluating predictions...")
    try:
        evaluation_results = predictor.evaluate(custom_df, top_k=3)
        predictor.print_evaluation_summary(evaluation_results)
        
    except Exception as e:
        print(f"✗ Error evaluating predictions: {e}")
        return
    
    # Step 5: Create confusion matrix
    print("\n5. Creating confusion matrix...")
    try:
        # Get unique class names
        class_names = sorted(list(set(true_fertilizers) | set(predicted_fertilizers)))
        
        # Create confusion matrix
        cm = create_confusion_matrix(
            true_fertilizers, 
            predicted_fertilizers, 
            class_names,
            'custom_dataset_confusion_matrix.png'
        )
        
        print(f"✓ Confusion matrix created with {len(class_names)} classes")
        
    except Exception as e:
        print(f"✗ Error creating confusion matrix: {e}")
        return
    
    # Step 6: Analyze and save detailed results
    print("\n6. Analyzing and saving detailed results...")
    try:
        analysis_df = analyze_predictions(evaluation_results, 'custom_dataset_analysis.csv')
        
        # Print some statistics
        print(f"\nPrediction Statistics:")
        print(f"  - Total samples: {len(analysis_df)}")
        print(f"  - Correct at 1st position: {analysis_df['is_correct_at_1'].sum()} ({analysis_df['is_correct_at_1'].mean():.3f})")
        print(f"  - Correct in top 3: {analysis_df['is_correct_in_top_3'].sum()} ({analysis_df['is_correct_in_top_3'].mean():.3f})")
        
        # Show distribution of true positions
        position_counts = analysis_df['true_position'].value_counts().sort_index()
        print(f"\nTrue Position Distribution:")
        for pos, count in position_counts.items():
            print(f"  - Position {pos}: {count} samples ({count/len(analysis_df):.3f})")
        
    except Exception as e:
        print(f"✗ Error analyzing results: {e}")
        return
    
    # Step 7: Print sample predictions
    print("\n7. Sample Predictions:")
    print("-" * 80)
    
    # Show first 10 predictions
    for i in range(min(10, len(analysis_df))):
        row = analysis_df.iloc[i]
        print(f"Sample {i+1}:")
        print(f"  True: {row['true_fertilizer']}")
        print(f"  Predicted: {row['predicted_1st']} (prob: {row['probability_1st']:.3f})")
        print(f"  Correct: {row['is_correct_at_1']}")
        if row['true_position']:
            print(f"  True position: {row['true_position']}")
        print()
    
    # Step 8: Final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"✓ Dataset evaluated: {len(custom_df)} samples")
    print(f"✓ Accuracy at 1st position: {evaluation_results['accuracy_at_1']:.5f}")
    print(f"✓ MAP@3: {evaluation_results['map_at_k']:.5f}")
    print(f"✓ Hit rate (top 3): {evaluation_results['hit_rate']:.5f}")
    print(f"✓ Confusion matrix: custom_dataset_confusion_matrix.png")
    print(f"✓ Detailed analysis: custom_dataset_analysis.csv")
    print("="*60)

if __name__ == "__main__":
    main() 