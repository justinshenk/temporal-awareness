import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import './index.css'

// Extract dataset from URL path
const dataset = window.location.pathname.split('/')[1] || 'geometry';

// Startup logging
console.log('='.repeat(60));
console.log('[CLIENT] 🚀 GeoViz Explorer - React App Starting');
console.log(`[CLIENT] 📊 Dataset: ${dataset}`);
console.log(`[CLIENT] 📡 Will connect to API at /api/${dataset}`);
console.log('='.repeat(60));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 0, // Always refetch when query key changes
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
)
