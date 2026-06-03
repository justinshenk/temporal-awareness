import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'

// Plugin to handle SPA routing - serve index.html for non-asset paths
function spaFallback(): Plugin {
  return {
    name: 'spa-fallback',
    configureServer(server) {
      // Run AFTER vite's middleware (including proxy)
      return () => {
        server.middlewares.use((req, res, next) => {
          // Skip API requests, static assets, and root
          const url = req.url || ''
          if (url.startsWith('/api') || url.includes('.') || url === '/') {
            return next()
          }
          // Serve index.html for SPA routes like /investment, /another_exp
          req.url = '/'
          next()
        })
      }
    },
  }
}

export default defineConfig({
  plugins: [react(), spaFallback()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
