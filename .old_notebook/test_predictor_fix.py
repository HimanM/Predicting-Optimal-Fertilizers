#!/usr/bin/env python3
"""
Test script to verify the FertilizerPredictor class fix
"""

import pandas as pd
import numpy as np
import os

# Import the predictor class
from complete_kfold_fertilizer_prediction import FertilizerPredictor

def test_predictor():
    """
    Test the FertilizerPredictor class with a small sample.
    """
    print("Testing FertilizerPredictor class...")
    
    # Check if training data exists
    if not os.path.exists('model/data.csv'):
        print("✗ Training data not found at model/data.csv")
        return False
    
    # Load a small sample of training data
    try:
        train_df = pd.read_csv('model/data.csv')
        print(f"✓ Loaded training data: {train_df.shape}")
        
        # Take first 5 rows for testing
        test_sample = train_df.head(5).copy()
        print(f"✓ Using {len(test_sample)} samples for testing")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return False
    
    # Initialize predictor
    predictor = FertilizerPredictor()
    
    # Try to load models
    if not predictor.load_models():
        print("✗ Failed to load models")
        return False
    
    # Test prediction
    try:
        print("\nTesting predictions...")
        predictions = predictor.predict(test_sample, top_k=3)
        print(f"✓ Predictions successful: {len(predictions)} samples")
        print("Sample predictions:")
        print(predictions[['Fertilizer_1', 'Probability_1']].head())
        
    except Exception as e:
        print(f"✗ Error in predictions: {e}")
        return False
    
    # Test evaluation
    try:
        print("\nTesting evaluation...")
        evaluation_results = predictor.evaluate(test_sample, top_k=3)
        print(f"✓ Evaluation successful!")
        print(f"  - MAP@3: {evaluation_results['map_at_k']:.5f}")
        print(f"  - Accuracy at 1: {evaluation_results['accuracy_at_1']:.5f}")
        print(f"  - Hit rate: {evaluation_results['hit_rate']:.5f}")
        
    except Exception as e:
        print(f"✗ Error in evaluation: {e}")
        return False
    
    print("\n✓ All tests passed! The predictor class is working correctly.")
    return True

if __name__ == "__main__":
    test_predictor() 