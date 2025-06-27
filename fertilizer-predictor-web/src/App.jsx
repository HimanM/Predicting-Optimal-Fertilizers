import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import DocumentationPage from './pages/DocumentationPage';
import AboutMePage from './pages/AboutMePage';
import HowToPage from './pages/HowToPage';
import './App.css'; // Contains global styles like background

function App() {
  return (
    <Router>
      <div className="flex flex-col min-h-screen bg-gradient-to-br from-green-100 via-emerald-50 to-teal-100">
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/documentation" element={<DocumentationPage />} />
            <Route path="/about" element={<AboutMePage />} />
            <Route path="/how-to/:parameterName" element={<HowToPage />} />
            {/* Add other routes here if needed */}
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
