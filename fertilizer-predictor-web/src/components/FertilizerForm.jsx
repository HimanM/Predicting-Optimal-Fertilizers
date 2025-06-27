import React, { useState } from 'react';
import FormField from './FormField';
import axios from 'axios'; // Make sure to install axios: npm install axios

const soilTypeOptions = [
  { value: 'Clayey', label: 'Clayey' },
  { value: 'Sandy', label: 'Sandy' },
  { value: 'Red', label: 'Red' },
  { value: 'Loamy', label: 'Loamy' },
  { value: 'Black', label: 'Black' },
];

const cropTypeOptions = [
  { value: 'Sugarcane', label: 'Sugarcane' },
  { value: 'Millets', label: 'Millets' },
  { value: 'Barley', label: 'Barley' },
  { value: 'Paddy', label: 'Paddy' },
  { value: 'Pulses', label: 'Pulses' },
  { value: 'Tobacco', label: 'Tobacco' },
  { value: 'Ground Nuts', label: 'Ground Nuts' },
  { value: 'Maize', label: 'Maize' },
  { value: 'Cotton', label: 'Cotton' },
  { value: 'Wheat', label: 'Wheat' },
  { value: 'Oil seeds', label: 'Oil seeds' },
];

// As per MVP.md
const initialFormData = {
  Temparature: '', // 20-40
  Humidity: '',    // 45-75
  Moisture: '',    // 20-70
  'Soil Type': '', // Clayey, Sandy, Red, Loamy, Black
  'Crop Type': '', // Sugarcane, Millets, Barley, Paddy, Pulses, Tobacco, Ground Nuts, Maize, Cotton, Wheat, Oil seeds
  Nitrogen: '',    // 0-50
  Potassium: '',   // 0-25
  Phosphorous: '', // 0-50
};

const formFieldInfo = {
  Temparature: { title: 'Temperature', description: 'Ambient temperature in Celsius.', learnMoreLink: '/how-to/temperature' },
  Humidity: { title: 'Humidity', description: 'Relative humidity percentage.', learnMoreLink: '/how-to/humidity' },
  Moisture: { title: 'Moisture', description: 'Soil moisture content percentage.', learnMoreLink: '/how-to/moisture' },
  'Soil Type': { title: 'Soil Type', description: 'The type of soil.', learnMoreLink: '/how-to/soil-type' }, // Assuming a general page for soil types
  'Crop Type': { title: 'Crop Type', description: 'The type of crop being cultivated.', learnMoreLink: '/how-to/crop-type' }, // Assuming a general page for crop types
  Nitrogen: { title: 'Nitrogen (N)', description: 'Nitrogen content in soil (mg/kg).', learnMoreLink: '/how-to/nitrogen' },
  Potassium: { title: 'Potassium (K)', description: 'Potassium content in soil (mg/kg).', learnMoreLink: '/how-to/potassium' },
  Phosphorous: { title: 'Phosphorous (P)', description: 'Phosphorous content in soil (mg/kg).', learnMoreLink: '/how-to/phosphorous' },
};


const FertilizerForm = () => {
  const [formData, setFormData] = useState(initialFormData);
  const [predictions, setPredictions] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setPredictions('');

    // Basic validation for empty fields
    for (const key in formData) {
        if (formData[key] === '') {
            setError(`Please fill in all fields. '${key}' is missing.`);
            setIsLoading(false);
            return;
        }
    }
    
    // Convert numeric fields from string to number
    const numericFields = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous'];
    const processedData = { ...formData };
    numericFields.forEach(field => {
        if (processedData[field] !== '') {
            processedData[field] = parseFloat(processedData[field]);
        }
    });


    try {
      // API call
      const response = await axios.post('/api/predict', processedData);
      if (response.data && response.data.predictions) {
        setPredictions(response.data.predictions);
      } else {
        setError('Prediction data not found in response.');
      }
    } catch (err) {
      if (err.response && err.response.data && err.response.data.error) {
        setError(`Error: ${err.response.data.error}`);
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
      console.error("Prediction API error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-8 bg-white rounded-2xl shadow-xl mt-10">
      <h2 className="text-3xl font-bold text-green-700 mb-8 text-center">Find the Best Fertilizer</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <FormField
            id="Temparature"
            label="Temperature (°C)"
            type="number"
            value={formData.Temparature}
            onChange={handleChange}
            placeholder="e.g., 30"
            info={formFieldInfo.Temparature}
            min="20" max="40"
          />
          <FormField
            id="Humidity"
            label="Humidity (%)"
            type="number"
            value={formData.Humidity}
            onChange={handleChange}
            placeholder="e.g., 60"
            info={formFieldInfo.Humidity}
            min="45" max="75"
          />
          <FormField
            id="Moisture"
            label="Moisture (%)"
            type="number"
            value={formData.Moisture}
            onChange={handleChange}
            placeholder="e.g., 45"
            info={formFieldInfo.Moisture}
            min="20" max="70"
          />
          <FormField
            id="Soil Type"
            label="Soil Type"
            type="select"
            value={formData['Soil Type']}
            onChange={handleChange}
            options={soilTypeOptions}
            placeholder="Select Soil Type"
            info={formFieldInfo['Soil Type']}
          />
          <FormField
            id="Crop Type"
            label="Crop Type"
            type="select"
            value={formData['Crop Type']}
            onChange={handleChange}
            options={cropTypeOptions}
            placeholder="Select Crop Type"
            info={formFieldInfo['Crop Type']}
          />
          <FormField
            id="Nitrogen"
            label="Nitrogen (mg/kg)"
            type="number"
            value={formData.Nitrogen}
            onChange={handleChange}
            placeholder="e.g., 30"
            info={formFieldInfo.Nitrogen}
            min="0" max="50"
          />
          <FormField
            id="Potassium"
            label="Potassium (mg/kg)"
            type="number"
            value={formData.Potassium}
            onChange={handleChange}
            placeholder="e.g., 15"
            info={formFieldInfo.Potassium}
            min="0" max="25"
          />
          <FormField
            id="Phosphorous"
            label="Phosphorous (mg/kg)"
            type="number"
            value={formData.Phosphorous}
            onChange={handleChange}
            placeholder="e.g., 25"
            info={formFieldInfo.Phosphorous}
            min="0" max="50"
          />
        </div>
        <button
          type="submit"
          disabled={isLoading}
          className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-4 rounded-lg shadow-md focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-75 disabled:opacity-50 transition duration-150 ease-in-out"
        >
          {isLoading ? 'Getting Recommendations...' : 'Get Recommendations'}
        </button>
      </form>

      {error && (
        <div className="mt-6 p-4 bg-red-100 border border-red-400 text-red-700 rounded-md">
          <p className="font-semibold">Error</p>
          <p>{error}</p>
        </div>
      )}

      {predictions && (
      <div className="mt-8 p-6 bg-green-50 border border-green-300 rounded-lg shadow-sm">
        <h3 className="text-xl font-semibold text-green-700 mb-4">
          Top Fertilizer Recommendations:
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
          {predictions.split(',').map((prediction, index) => (
            <div
              key={index}
              className="bg-green-200 text-green-800 px-6 py-4 rounded-xl shadow text-lg font-semibold"
            >
              {prediction.trim()}
            </div>
          ))}
        </div>
      </div>
    )}
    </div>
  );
};

export default FertilizerForm;
