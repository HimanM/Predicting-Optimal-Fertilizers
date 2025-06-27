import React from 'react';
import { FaGithub, FaReact } from 'react-icons/fa';
import { SiTailwindcss, SiFlask, SiScikitlearn, SiVite, SiPython } from 'react-icons/si';
import { SiKaggle } from 'react-icons/si';

const AboutMePage = () => {
  return (
    <div className="container mx-auto px-6 py-8 pt-24 min-h-screen">
      <h1 className="text-4xl font-bold text-green-700 mb-6">About FertiliZen</h1>
      <div className="bg-white p-8 rounded-xl shadow-lg text-gray-700 space-y-6">
        <p>
          <strong>FertiliZen</strong> is a personal project built on top of a model developed for the{' '}
          <a 
            href="https://www.kaggle.com/competitions/playground-series-s5e6" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="text-green-600 hover:text-green-500 underline inline-flex items-center"
          >
            Kaggle Fertilizer Prediction Challenge <SiKaggle className="ml-2" />
          </a>
        </p>
        <p>
          The goal of this project is to bridge the gap between agricultural data science and practical farming. 
          FertiliZen helps users — especially farmers — make informed decisions about fertilizer use, leading to 
          improved crop yields and more sustainable agricultural practices.
        </p>
        <p>
          The core of FertiliZen is a machine learning model trained on synthetic data from Kaggle that maps 
          environmental conditions and crop types to optimal fertilizers.
        </p>
        
        <h2 className="text-2xl font-semibold text-green-600 pt-4">Our Mission</h2>
        <p>
          To empower farmers with data-driven insights for efficient and effective crop management. We believe that technology
          can play a crucial role in modernizing agriculture and ensuring food security.
        </p>

        <h2 className="text-2xl font-semibold text-green-600 pt-4">The Technology</h2>
        <ul className="list-inside pl-2 space-y-2">
          <li className="flex items-center space-x-2">
            <FaReact className="text-blue-500" />
            <span><strong>Frontend:</strong> React (Vite)</span>
          </li>
          <li className="flex items-center space-x-2">
            <SiTailwindcss className="text-cyan-500" />
            <span><strong>Styling:</strong> Tailwind CSS</span>
          </li>
          <li className="flex items-center space-x-2">
            <SiFlask className="text-gray-800" />
            <span><strong>Backend:</strong> Flask (Python)</span>
          </li>
          <li className="flex items-center space-x-2">
            <SiScikitlearn className="text-yellow-600" />
            <span><strong>ML Model:</strong> Scikit-learn (.joblib format)</span>
          </li>
          <li className="flex items-center space-x-2">
            <SiPython className="text-blue-600" />
            <span><strong>Language:</strong> Python</span>
          </li>
        </ul>

        <h2 className="text-2xl font-semibold text-green-600 pt-4">Future Development</h2>
        <ul className="list-disc list-inside pl-4 space-y-1">
          <li>Expanding the dataset to include more crop types and geographical regions.</li>
          <li>Integrating real-time weather data.</li>
          <li>Providing more detailed information on fertilizer application techniques.</li>
          <li>Developing a mobile application.</li>
        </ul>

        <h2 className="text-2xl font-semibold text-green-600 pt-4">Contact & Feedback</h2>
        <p>
          This project is open source. You can find the code and contribute on{' '}
          <a 
            href="https://github.com/HimanM/Predicting-Optimal-Fertilizers.git" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="text-green-600 hover:text-green-500 underline inline-flex items-center"
          >
            GitHub <FaGithub className="ml-1" />
          </a>. 
          Feedback and suggestions are always welcome!
        </p>
        <p className="text-sm text-gray-500 pt-4">
          Note: FertiliZen provides fertilizer suggestions based on a predictive model trained on a synthetic dataset. 
          Always consult local agricultural experts for critical decisions.
        </p>
      </div>
    </div>
  );
};

export default AboutMePage;
