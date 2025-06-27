import React from 'react';
import { Link } from 'react-router-dom';
import { FaGithub } from 'react-icons/fa';

const Navbar = () => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 shadow-md backdrop-blur-lg bg-white/70">
      <div className="container mx-auto px-6 py-3 flex justify-between items-center">
        <Link to="/" className="text-2xl font-bold text-green-700 hover:text-green-600">
          FertiliZen
        </Link>
        <div className="flex items-center space-x-4">
          <Link to="/" className="text-green-700 hover:text-green-500 px-3 py-2 rounded-md text-sm font-medium">
            Home
          </Link>
          <Link to="/documentation" className="text-green-700 hover:text-green-500 px-3 py-2 rounded-md text-sm font-medium">
            Documentation
          </Link>
          <Link to="/about" className="text-green-700 hover:text-green-500 px-3 py-2 rounded-md text-sm font-medium">
            About
          </Link>
          <a
            href="https://github.com/HimanM/Predicting-Optimal-Fertilizers.git" 
            target="_blank"
            rel="noopener noreferrer"
            className="text-green-700 hover:text-green-500"
            aria-label="GitHub Repository"
          >
            <FaGithub size={24} />
          </a>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
