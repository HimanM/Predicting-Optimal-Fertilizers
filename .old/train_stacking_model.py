#!/usr/bin/env python3
"""
Script to train and save a stacking ensemble model for fertilizer prediction.
"""

import os
import sys
import pandas as pd
import numpy as np
from stacking_model import train_and_save_stacking_model

def main():
    """Main function to train and save the stacking model."""
    
    print("="*60)
    print("STACKING ENSEMBLE MODEL TRAINING")
    print("="*60)
    
    # Check if data file exists
    data_path = "model/data.csv"
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file exists in the model directory.")
        return False
    
    # Model save path
    model_save_path = "model/stacking_model.joblib"
    
    try:
        print(f"Loading data from: {data_path}")
        
        # Load and check data
        df = pd.read_csv(data_path)
        print(f"Data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        if 'Fertilizer Name' not in df.columns:
            print("Error: 'Fertilizer Name' column not found in the dataset")
            return False
        
        # Check for required columns
        required_columns = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 
                          'Nitrogen', 'Potassium', 'Phosphorous', 'Fertilizer Name']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
        
        print("✓ Data validation passed")
        
        # Train and save the stacking model
        print(f"\nTraining stacking ensemble model...")
        model = train_and_save_stacking_model(data_path, model_save_path)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✓ Stacking model saved to: {model_save_path}")
        print(f"✓ Supporting files saved to: model/")
        print("\nYou can now use the StackingFertilizerPredictor class to make predictions.")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 