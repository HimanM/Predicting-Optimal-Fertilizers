import React from 'react';

const Footer = () => {
  return (
    <footer className="bg-green-50 py-8 mt-16 text-center text-green-700">
      <div className="container mx-auto px-6">
        <p>&copy; {new Date().getFullYear()} FertiliZen. All rights reserved.</p>
        <p className="text-sm mt-1">
          Helping you grow smarter.
        </p>
      </div>
    </footer>
  );
};

export default Footer;
