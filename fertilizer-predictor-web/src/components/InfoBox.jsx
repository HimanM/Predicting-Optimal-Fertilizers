import React from 'react';
import { Link } from 'react-router-dom';

const InfoBox = ({ title, description, learnMoreLink, position = 'bottom-full left-1/2', isVisible, onClose }) => {
  if (!isVisible) return null;

  const basePositionClasses = 'absolute z-10 mt-2 p-4 bg-white border border-green-200 rounded-lg shadow-xl';
  const transformClass = position.includes('left-1/2') || position.includes('right-1/2') ? '-translate-x-1/2' : '';
  
  return (
    <div className={`${basePositionClasses} ${position} ${transformClass} w-64 text-sm text-gray-700`}>
      <button 
        onClick={onClose} 
        className="absolute top-2 right-2 text-gray-500 hover:text-gray-700"
        aria-label="Close info box"
      >
        &times;
      </button>
      <h3 className="font-semibold text-green-700 mb-1">{title}</h3>
      <p className="mb-2">{description}</p>
      {learnMoreLink && (
        <Link to={learnMoreLink} className="text-green-600 hover:text-green-500 font-medium hover:underline">
          Learn More &rarr;
        </Link>
      )}
    </div>
  );
};

export default InfoBox;
