import React from 'react';
import { useParams, Link } from 'react-router-dom';

// Placeholder content for each parameter
const howToContent = {
  temperature: {
    title: 'Understanding & Measuring Temperature',
    whatIsIt: 'Temperature refers to the ambient air temperature, which significantly affects plant growth and nutrient uptake.',
    howToMeasure: [
      'Use a digital or analog thermometer placed in a shaded, well-ventilated area near your crops.',
      'Check local weather station data or reliable weather apps for current and forecasted temperatures.',
    ],
    visualAids: [
      { text: 'Example of a Min-Max Thermometer', type: 'text'}, // Placeholder for image link
    ],
    youtubeGuide: 'https://www.youtube.com/watch?v=kC6t3Tqk3w0', // From MVP.md
    additionalInfo: 'Optimal temperature ranges vary by crop. Extreme temperatures can stress plants and reduce fertilizer effectiveness.'
  },
  humidity: {
    title: 'Understanding & Measuring Humidity',
    whatIsIt: 'Relative humidity is the amount of water vapor in the air, expressed as a percentage of the maximum amount the air can hold at that temperature.',
    howToMeasure: [
      'Use a hygrometer (digital or analog) to measure relative humidity.',
      'Some weather stations and indoor climate monitors also provide humidity readings.',
    ],
    visualAids: [
      { text: 'Digital Hygrometer', type: 'text'},
    ],
    youtubeGuide: 'https://www.youtube.com/watch?v=ZbN5XOtJodk', // From MVP.md
    additionalInfo: 'High humidity can promote fungal diseases, while very low humidity can cause plants to lose water too quickly.'
  },
  moisture: {
    title: 'Understanding & Measuring Soil Moisture',
    whatIsIt: 'Soil moisture is the water content in the soil, crucial for nutrient transport to plant roots.',
    howToMeasure: [
      'Use a soil moisture meter (capacitive or resistance-based probes). Insert the probe into the soil at root depth.',
      'The "feel test": Squeeze a handful of soil. If it forms a weak ball, moisture is likely adequate for many crops. If it crumbles, it\'s too dry. If water drips, it\'s too wet.',
      'Tensiometers can provide a more precise measurement of soil water tension.',
    ],
    visualAids: [
      { text: 'Soil Moisture Meter Probe', type: 'text'},
    ],
    youtubeGuide: 'https://www.youtube.com/watch?v=TZxAXu3hP94', // From MVP.md
    additionalInfo: 'Overwatering can lead to root rot and nutrient leaching, while underwatering stresses plants.'
  },
  nitrogen: {
    title: 'Understanding & Measuring Soil Nitrogen (N)',
    whatIsIt: 'Nitrogen is a primary macronutrient essential for leafy vegetative growth and overall plant vigor.',
    howToMeasure: [
      'Home soil test kits (colorimetric or strip tests) can provide an estimate.',
      'For accurate results, send soil samples to a professional agricultural laboratory.',
      'Digital NPK sensors are becoming available but verify their accuracy for your soil type.',
    ],
    visualAids: [
      { text: 'Soil Test Kit for NPK', type: 'text'},
    ],
    youtubeGuide: 'https://www.youtube.com/watch?v=2ZPqgPkhkE8', // From MVP.md
    additionalInfo: 'Nitrogen is highly mobile in soil. Levels can change rapidly due to rain or irrigation.'
  },
  phosphorous: {
    title: 'Understanding & Measuring Soil Phosphorous (P)',
    whatIsIt: 'Phosphorous is a primary macronutrient vital for root development, flowering, fruiting, and seed formation.',
    howToMeasure: [
      'Home soil test kits can estimate available phosphorous.',
      'Professional lab testing is recommended for accuracy.',
    ],
    visualAids: [
      { text: 'Example Soil Test Report Highlighting P levels', type: 'text'},
    ],
    youtubeGuide: 'https://www.youtube.com/watch?v=Ep7uR8SJmm4', // From MVP.md
    additionalInfo: 'Phosphorous availability is pH-dependent. It is less mobile in soil than nitrogen.'
  },
  potassium: {
    title: 'Understanding & Measuring Soil Potassium (K)',
    whatIsIt: 'Potassium is a primary macronutrient important for overall plant health, disease resistance, water regulation, and fruit quality.',
    howToMeasure: [
      'Home soil test kits are available.',
      'Professional lab analysis provides the most reliable measurements.',
    ],
    visualAids: [
      { text: 'Potassium Deficiency Symptoms in Leaves (e.g., yellowing edges)', type: 'text'},
    ],
    youtubeGuide: 'https://www.youtube.com/watch?v=kGiPZ5IgeYk', 
    additionalInfo: 'Potassium plays a key role in enzyme activation and photosynthesis.'
  },
  'soil-type': { 
    title: 'Understanding Soil Types',
    whatIsIt: 'Soil type refers to the texture of the soil, determined by the proportion of sand, silt, and clay particles. It affects drainage, nutrient retention, and aeration.',
    howToMeasure: [
        '**Visual Inspection & Feel Test:**',
        '   **Sandy:** Gritty feel, doesn\'t hold shape, drains quickly.',
        '   **Clayey:** Sticky when wet, forms a hard ball when dry, poor drainage.',
        '   **Loamy:** Even mix, feels smooth, holds shape, good drainage and nutrient retention (ideal for many crops).',
        '   **Red Soil:** Typically derives its color from iron oxides, can vary in texture.',
        '   **Black Soil (e.g., Chernozem):** Rich in organic matter, often very fertile, can be clay-like.',
        '**Jar Test:** A simple way to separate particles. Mix soil with water in a jar, let it settle, and observe layers.',
        'Professional soil analysis can provide a detailed textural analysis.'
    ],
    visualAids: [
        { text: 'Soil Texture Triangle Diagram', type: 'text'},
        { text: 'Soil Jar Test Example', type: 'text'},
    ],
    youtubeGuide: 'https://www.youtube.com/watch?v=AUfBAbx2L0E', 
    additionalInfo: 'Understanding your soil type is fundamental to choosing appropriate crops and soil management practices.'
  },
  'crop-type': { 
    title: 'Understanding Crop Types & Needs',
    whatIsIt: 'Different crops have varying nutrient requirements, soil preferences, and climate tolerances.',
    howToMeasure: [
        'This isn\'t something you "measure" in the same way as soil parameters.',
        'It involves identifying the crop you intend to grow or are currently cultivating.',
        'Research specific needs for your chosen crop (e.g., from agricultural extensions, university resources, or seed suppliers).',
    ],
    visualAids: [
        { text: 'Chart of Nutrient Needs for Common Crops', type: 'text'},
    ],
    youtubeGuide: 'https://www.youtube.com/watch?v=G4KIA72jgMg', 
    additionalInfo: 'The choice of crop type is a primary factor in determining fertilizer needs. The recommendations from this tool are tailored based on the selected crop.'
  }
};

const HowToPage = () => {
  const { parameterName } = useParams();
  const content = howToContent[parameterName] || { title: 'Information Not Found', whatIsIt: 'Detailed information for this parameter is not yet available.' };

  return (
    <div className="container mx-auto px-6 py-8 pt-24 min-h-screen"> 
      <Link to="/" className="text-green-600 hover:text-green-500 mb-6 inline-block">&larr; Back to Form</Link>
      <h1 className="text-3xl md:text-4xl font-bold text-green-700 mb-6">{content.title}</h1>
      
      <div className="bg-white p-6 md:p-8 rounded-xl shadow-lg text-gray-700 space-y-6">
        {content.whatIsIt && (
          <section>
            <h2 className="text-2xl font-semibold text-green-600 mb-3">What is it?</h2>
            <p className="leading-relaxed">{content.whatIsIt}</p>
          </section>
        )}

        {content.howToMeasure && content.howToMeasure.length > 0 && (
          <section>
            <h2 className="text-2xl font-semibold text-green-600 mb-3">How to Measure / Determine</h2>
            <ul className="list-disc list-inside pl-4 space-y-2 leading-relaxed">
              {content.howToMeasure.map((item, index) => (
                <li key={index} dangerouslySetInnerHTML={{ __html: item.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }}></li>
              ))}
            </ul>
          </section>
        )}

        {content.visualAids && content.visualAids.length > 0 && (
          <section>
            <h2 className="text-2xl font-semibold text-green-600 mb-3">Visual Aids / Examples</h2>
            <ul className="list-disc list-inside pl-4 space-y-2">
              {content.visualAids.map((item, index) => (
                <li key={index}>
                  {item.type === 'link' ? (
                    <a href={item.url} target="_blank" rel="noopener noreferrer" className="text-green-600 hover:underline">{item.text}</a>
                  ) : (
                    item.text 
                  )}
                </li>
              ))}
            </ul>
          </section>
        )}

        {content.youtubeGuide && (
          <section>
            <h2 className="text-2xl font-semibold text-green-600 mb-3">Video Guide</h2>
            <div className="aspect-video rounded-lg overflow-hidden shadow-lg">
              <iframe
                src={`https://www.youtube.com/embed/${content.youtubeGuide.split('v=')[1].split('&')[0]}`}
                title="YouTube video player"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen
                className="w-full h-full"
              ></iframe>
            </div>
          </section>
        )}
        
        {content.additionalInfo && (
          <section>
            <h2 className="text-2xl font-semibold text-green-600 mb-3">Additional Information</h2>
            <p className="leading-relaxed">{content.additionalInfo}</p>
          </section>
        )}

        {content.title === 'Information Not Found' && (
            <p className="text-red-500">The content for the parameter "{parameterName}" could not be found. Please check the link or try again later.</p>
        )}
      </div>
    </div>
  );
};

export default HowToPage;
