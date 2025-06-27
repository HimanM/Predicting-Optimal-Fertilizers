import React from 'react';
import TrainerArchitectureDiagram from '../components/TrainerArchitectureDiagram';
import 'reactflow/dist/style.css';

const DocumentationPage = () => {
  return (
    <div className="container mx-auto px-6 py-8 pt-24 min-h-screen">
      <h1 className="text-4xl font-bold text-green-700 mb-6">Documentation</h1>
      <div className="bg-white p-8 rounded-xl shadow-lg text-gray-700">
        <p className="mb-4">
          Welcome to the FertiliZen documentation page. Here you'll find information about how the application works,
          the data it uses, and how to interpret the results.
        </p>
        <h2 className="text-2xl font-semibold text-green-600 mt-6 mb-3">How It Works</h2>
        <p className="mb-4">
          FertiliZen uses a machine learning model to predict the most suitable fertilizer for your crops based on several input parameters:
          temperature, humidity, soil moisture, soil type, crop type, and levels of Nitrogen, Phosphorous, and Potassium in the soil.
        </p>
        
        <h2 className="text-2xl font-semibold text-green-600 mt-6 mb-3">Input Parameters</h2>
        <p className="mb-4">
          Each input field on the form has a small question mark icon (?). Clicking this icon will provide a brief description of the parameter
          and a "Learn More" link that directs you to a page with more detailed information, including how to measure or determine that parameter.
        </p>
        <h2 className="text-2xl font-semibold text-green-600 mt-6 mb-3">Interpreting Results</h2>
        <p className="mb-4">
          The application provides the top three fertilizer recommendations. These are based on the model's prediction for maximizing crop health and yield
          given your specific inputs. It's always a good idea to consult with local agricultural experts for specific regional advice.
        </p>
        <h2 className="text-2xl font-semibold text-green-600 mt-6 mb-3">API Usage</h2>
        <p className="mb-4">
          The backend API endpoint <code>/api/predict</code> accepts a POST request with a JSON body containing all the input parameters.
          It returns a JSON response with the predictions.
        </p>
        <pre className="bg-gray-100 p-4 rounded-md text-sm overflow-x-auto">
          <code>
{`Request Body Example:
{
  "Temparature": 30,
  "Humidity": 60,
  "Moisture": 40,
  "Soil Type": "Loamy",
  "Crop Type": "Wheat",
  "Nitrogen": 30,
  "Potassium": 20,
  "Phosphorous": 40
}

Response Body Example:
{
  "predictions": "10-26-26,14-35-14,20-20"
}`}
          </code>
        </pre>

        <h2 className="text-2xl font-semibold text-green-600 mt-8 mb-3">FertiliZen Trainer: Architectural Overview</h2>
        <div className="mb-6">
          <ol className="list-decimal list-inside space-y-2 text-base">
            <li>
              <span className="font-semibold mb-4">Input Layer:</span> Raw Data: Soil, environmental, and crop features, plus the target (fertilizer name).
            </li>
            <li>
              <span className="font-semibold mb-4">Preprocessing & Feature Engineering:</span> Data Cleaning (handle missing values, remove unnecessary columns), Encoding (convert categorical features to numeric using custom label encoders), Feature Engineering (create new features: ratios, interactions, transformations).
            </li>
            <li>
            <span className="font-semibold mb-4">K-Fold Cross-Validation (for each Base Model):</span> Stratified k-fold splits (e.g., k=5). For each fold: train on k-1 folds, validate on the remaining fold, save Out-of-Fold (OOF) Predictions. Repeat for each base model: LightGBM, CatBoost, XGBoost. Result: OOF predictions for every sample from every base model.
            </li>
            <li>
              <span className="font-semibold mb-4">Stacking Layer (Meta-Model Training):</span> Concatenate OOF probabilities from all base models for each sample. Train a Logistic Regression (or similar) meta-model on these stacked features to predict the target.
            </li>
            <li>
              <span className="font-semibold mb-4">Artifact Saving:</span> Save trained base models (all k-folds for each), meta-model, encoders, feature list, and scaler (for stacking features). Enables consistent inference in production.
            </li>
          </ol>
        </div>
        <div className="my-8">
          <TrainerArchitectureDiagram />
        </div>
        <p className="mt-6">
          This documentation is currently a brief overview. More detailed sections will be added in the future.
        </p>
      </div>
    </div>
  );
};

export default DocumentationPage;
