import React, { useState, useRef, useEffect } from 'react';
import InfoBox from './InfoBox';
import { FaQuestionCircle } from 'react-icons/fa';

const FormField = ({
  id,
  label,
  type = 'number', 
  value,
  onChange,
  placeholder,
  options = [], 
  info, 
  min,
  max,
  step = "any"
}) => {
  const [isInfoBoxVisible, setIsInfoBoxVisible] = useState(false);
  const infoIconRef = useRef(null);
  const infoBoxRef = useRef(null);

  const toggleInfoBox = (e) => {
    e.stopPropagation(); 
    setIsInfoBoxVisible(prev => !prev);
  };

  const closeInfoBox = () => {
    setIsInfoBoxVisible(false);
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        isInfoBoxVisible &&
        infoIconRef.current && !infoIconRef.current.contains(event.target) &&
        infoBoxRef.current && !infoBoxRef.current.contains(event.target)
      ) {
        closeInfoBox();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isInfoBoxVisible]);

  return (
    <div className="mb-6 relative">
      <label htmlFor={id} className="block text-sm font-medium text-green-700 mb-1">
        {label}
        {info && (
          <button
            ref={infoIconRef}
            type="button"
            onClick={toggleInfoBox}
            className="ml-2 text-green-500 hover:text-green-400 focus:outline-none"
            aria-label={`More info about ${label}`}
          >
            <FaQuestionCircle />
          </button>
        )}
      </label>
      {type === 'select' ? (
        <select
          id={id}
          name={id}
          value={value}
          onChange={onChange}
          className="mt-1 block w-full px-3 py-2 bg-white border border-green-300 rounded-md shadow-sm focus:outline-none focus:ring-green-500 focus:border-green-500 sm:text-sm hover:border-green-400"
        >
          {placeholder && <option value="">{placeholder}</option>}
          {options.map(option => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          type={type}
          id={id}
          name={id}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          min={min}
          max={max}
          step={step}
          className="mt-1 block w-full px-3 py-2 bg-white border border-green-300 rounded-md shadow-sm focus:outline-none focus:ring-green-500 focus:border-green-500 sm:text-sm hover:border-green-400"
        />
      )}
      {info && isInfoBoxVisible && (
        <div ref={infoBoxRef}>
          <InfoBox
            title={info.title}
            description={info.description}
            learnMoreLink={info.learnMoreLink}
            isVisible={isInfoBoxVisible}
            onClose={closeInfoBox}
            position="top-full left-1/2 transform -translate-x-1/2 mt-2"
          />
        </div>
      )}
    </div>
  );
};

export default FormField;
