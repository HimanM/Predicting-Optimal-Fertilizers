import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, model_name, class_names=None, save_path=None):
    """
    Plot confusion matrix for a given model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for the title
        class_names: List of class names (optional)
        save_path: Path to save the plot (optional)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    # Print classification report
    print(f"\nClassification Report - {model_name}:")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return cm

def plot_confusion_matrix_from_oof(oof_preds, y_true, model_name, label_encoder, save_path=None):
    """
    Plot confusion matrix using out-of-fold predictions.
    
    Args:
        oof_preds: Out-of-fold predictions (probabilities)
        y_true: True labels
        model_name: Name of the model
        label_encoder: Label encoder to get class names
        save_path: Path to save the plot (optional)
    """
    # Get predicted labels from probabilities
    y_pred = np.argmax(oof_preds, axis=1)
    
    # Get class names
    class_names = label_encoder.classes_
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred, model_name, class_names, save_path)
    
    return cm 