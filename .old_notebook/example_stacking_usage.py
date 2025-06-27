#!/usr/bin/env python3
"""
Example script demonstrating how to use the stacking ensemble model
for fertilizer prediction.
"""

import pandas as pd
import numpy as np
from StackingPredictor import StackingFertilizerPredictor

def main():
    """Main function demonstrating stacking model usage."""
    
    print("="*60)
    print("STACKING ENSEMBLE MODEL USAGE EXAMPLE")
    print("="*60)
    
    # Initialize the predictor
    predictor = StackingFertilizerPredictor(model_dir='model')
    
    # Load the trained model
    print("Loading stacking ensemble model...")
    if not predictor.load_models():
        print("Failed to load models. Please ensure the stacking model is trained first.")
        print("Run: python train_stacking_model.py")
        return
    
    print("✓ Model loaded successfully!")
    
    # Example 1: Single prediction
    print("\n" + "="*40)
    print("EXAMPLE 1: SINGLE PREDICTION")
    print("="*40)
    
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
    
    print("Input data:")
    for key, value in sample_input.items():
        print(f"  {key}: {value}")
    
    # Make predictions
    top3_prediction = predictor.predict_top3(sample_input)
    single_prediction = predictor.predict(sample_input)
    probabilities = predictor.predict_proba(sample_input)
    
    print(f"\nPredictions:")
    print(f"  Top 3 recommendations: {top3_prediction}")
    print(f"  Best recommendation: {single_prediction}")
    print(f"  Probabilities: {[f'{p:.3f}' for p in probabilities[0]]}")
    
    # Example 2: Multiple predictions
    print("\n" + "="*40)
    print("EXAMPLE 2: MULTIPLE PREDICTIONS")
    print("="*40)
    
    multiple_inputs = [
        {
            'Temparature': 30.0,
            'Humidity': 70.0,
            'Moisture': 0.8,
            'Soil Type': 'Clay',
            'Crop Type': 'Rice',
            'Nitrogen': 40.0,
            'Potassium': 25.0,
            'Phosphorous': 15.0
        },
        {
            'Temparature': 20.0,
            'Humidity': 50.0,
            'Moisture': 0.3,
            'Soil Type': 'Loamy',
            'Crop Type': 'Corn',
            'Nitrogen': 60.0,
            'Potassium': 35.0,
            'Phosphorous': 25.0
        },
        {
            'Temparature': 35.0,
            'Humidity': 80.0,
            'Moisture': 0.9,
            'Soil Type': 'Black',
            'Crop Type': 'Cotton',
            'Nitrogen': 45.0,
            'Potassium': 20.0,
            'Phosphorous': 30.0
        }
    ]
    
    print("Multiple predictions:")
    for i, input_data in enumerate(multiple_inputs, 1):
        top3 = predictor.predict_top3(input_data)
        single = predictor.predict(input_data)
        print(f"  Sample {i}: {single} (Top 3: {top3})")
    
    # Example 3: Load test data and evaluate (if available)
    print("\n" + "="*40)
    print("EXAMPLE 3: MODEL EVALUATION")
    print("="*40)
    
    # Try to load test data
    test_data_paths = [
        "model/data.csv",  # Use training data as test for demo
        "test_data.csv",
        "validation_data.csv"
    ]
    
    test_df = None
    for path in test_data_paths:
        try:
            if os.path.exists(path):
                test_df = pd.read_csv(path)
                print(f"Using test data from: {path}")
                break
        except:
            continue
    
    if test_df is not None and 'Fertilizer Name' in test_df.columns:
        # Use a small sample for evaluation
        if len(test_df) > 50:
            test_df = test_df.sample(n=50, random_state=42)
        
        print(f"Evaluating on {len(test_df)} samples...")
        
        # Evaluate the model
        evaluation_results = predictor.evaluate(test_df, top_k=3)
        
        # Print evaluation summary
        predictor.print_evaluation_summary(evaluation_results)
        
        # Save detailed results
        detailed_results = pd.DataFrame(evaluation_results['detailed_results'])
        detailed_results.to_csv('stacking_evaluation_results.csv', index=False)
        print(f"\nDetailed evaluation results saved to: stacking_evaluation_results.csv")
    else:
        print("No test data found for evaluation.")
        print("To evaluate the model, provide a CSV file with 'Fertilizer Name' column.")
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("The stacking ensemble model is ready for use in your application.")
    print("You can integrate it into your web app or other prediction systems.")

if __name__ == "__main__":
    import os
    main() 