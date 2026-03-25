/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          coral: '#D97757',    // Anthropic terracotta/coral
          teal: '#348296',     // Deep teal accent
          purple: '#8F70DB',   // Purple accent
          // Legacy aliases for gradual migration
          pink: '#D97757',
          cyan: '#348296',
        },
        text: {
          dark: '#1a1613',     // Anthropic dark brown
          muted: '#5c534a',    // Warm gray-brown
        },
        bg: {
          cream: '#faf8f5',    // Anthropic cream
          warm: '#f5f0eb',     // Warm beige
          start: '#faf8f5',
          mid: '#f5f0eb',
          end: '#f0ebe5',
        }
      },
      backgroundImage: {
        'gradient-main': 'linear-gradient(135deg, #faf8f5 0%, #f5f0eb 50%, #f0ebe5 100%)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
