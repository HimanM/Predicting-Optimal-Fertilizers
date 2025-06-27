import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Proxying API requests to Flask backend
      '/api': {
        target: 'http://127.0.0.1:5000', // Flask default development server
        changeOrigin: true, // Recommended for most cases
        secure: false,      // Set to true if your backend uses HTTPS
        // You might not need rewrite if your Flask routes already include /api
        // rewrite: (path) => path.replace(/^\/api/, ''), 
      },
    },
  },
})
