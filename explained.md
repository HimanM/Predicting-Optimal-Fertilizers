# FertiliZen Notebooks Explained

This document provides a beginner-friendly, in-depth explanation of the three main Jupyter notebooks that power the FertiliZen web app. The project is inspired by the Kaggle competition [Playground Series S5E6](https://www.kaggle.com/competitions/playground-series-s5e6), and the web app is built on top of the competition solution.

---

## 1. Exploratory Data Analysis (EDA) Notebook

**Purpose:**
- To understand the dataset, its features, and the relationships between them before building any models.

**Key Steps:**
- **Loading the Data:** The dataset includes soil and environmental parameters (e.g., temperature, humidity, moisture, soil type, crop type, and nutrient levels).
- **Visualizations:**
  - Distribution of each feature (histograms, box plots)
  - Correlation matrix to see how features relate to each other
  - Distribution of the target variable (fertilizer name)
  - Fertilizer distribution by crop and soil type
- **Insights:**
  - Some fertilizers are much more common than others (class imbalance)
  - Certain crops and soil types are associated with specific fertilizers
  - Some features are highly correlated, which can inform feature engineering

**Why EDA Matters:**
- EDA helps identify patterns, outliers, and potential issues in the data. It guides feature engineering and model selection.

---

## 2. Training Notebook: K-Fold Cross-Validation & Stacking Ensemble

**Purpose:**
- To build a robust machine learning model using advanced techniques like k-fold cross-validation and stacking.

**Key Concepts:**
- **K-Fold Cross-Validation:**
  - The data is split into k parts (folds). Each model is trained k times, each time using a different fold as the validation set and the rest as training data. This helps ensure the model generalizes well and doesn't just memorize the training data.
- **Base Models:**
  - Three powerful models are used: LightGBM, CatBoost, and XGBoost. Each is a type of gradient boosting machine, known for high accuracy on structured/tabular data.
- **Feature Engineering:**
  - New features are created, such as nutrient ratios (e.g., Nitrogen/Phosphorous), log and square transformations, and interaction terms (e.g., Temperature × Humidity).
- **Custom Encoders:**
  - Categorical features (like soil and crop type) are converted to numbers using a custom label encoder.
- **Stacking Ensemble:**
  - After k-fold training, each base model makes predictions on the validation data (out-of-fold predictions). These predictions are used as new features to train a meta-model (Logistic Regression), which learns to combine the strengths of all base models.
- **Evaluation:**
  - Model performance is measured using metrics like accuracy, confusion matrices, and MAP@3 (how often the correct answer is in the top 3 predictions).
- **Artifact Saving:**
  - All trained models, encoders, and feature lists are saved to disk for use in the prediction pipeline.

**Why This Approach?**
- K-fold and stacking make the model more robust and less likely to overfit. The meta-model can learn which base model to trust more for different types of data.

---

## 3. Predictor Notebook: Inference Pipeline

**Purpose:**
- To provide a production-ready pipeline for making predictions on new data using the trained ensemble.

**Key Steps:**
- **Loading Artifacts:**
  - Loads all saved models, encoders, and feature lists from the training step.
- **Preprocessing:**
  - New input data is processed in the same way as the training data (encoding, feature engineering, etc.).
- **Prediction:**
  - Each base model (with its k-fold ensemble) predicts probabilities for each fertilizer type.
  - These probabilities are averaged and passed to the meta-model, which makes the final prediction.
  - The system can return the top-N fertilizer recommendations, each with a probability score.
- **Web App Integration:**
  - The predictor class is used by the Flask backend to serve predictions to the web app frontend.

**Why This Matters:**
- Ensures that predictions are consistent, robust, and fast. The same pipeline can be used for both batch and single-instance predictions.

---

## How This All Connects to the Web App
- The web app is a user-friendly interface built with React and Flask.
- Users enter soil and environmental data; the backend uses the predictor pipeline to return the top fertilizer recommendations.
- The app demonstrates how advanced ML models can be deployed in real-world applications.

---

## Glossary / Dictionary
- **Ensemble:** Combining multiple models to improve prediction accuracy.
- **K-Fold Cross-Validation:** Splitting data into k parts, training on k-1, and validating on the remaining part, repeated k times.
- **Stacking:** Using predictions from several models as input features for a final model (meta-model).
- **Meta-Model:** The model that learns to combine the outputs of base models in stacking.
- **OOF (Out-of-Fold) Predictions:** Predictions made on validation folds during k-fold training, used for stacking.
- **MAP@k (Mean Average Precision at k):** A metric for evaluating how often the correct answer is in the top-k predictions.
- **Label Encoding:** Converting categorical variables into numeric codes.
- **Feature Engineering:** Creating new features from raw data to improve model performance.
- **Confusion Matrix:** A table showing correct and incorrect predictions for each class.
- **LightGBM, CatBoost, XGBoost:** Popular gradient boosting machine learning libraries.
- **Logistic Regression:** A simple, interpretable model often used as a meta-model in stacking.
- **EDA:** Exploratory Data Analysis, the process of analyzing and visualizing data before modeling.
- **Artifact:** Any saved file from the ML pipeline (model, encoder, feature list, etc.) used for inference.

---

*For more details, see the actual notebooks and the README. This document is designed to help anyone, even with little ML background, understand the workflow and concepts behind FertiliZen.* 