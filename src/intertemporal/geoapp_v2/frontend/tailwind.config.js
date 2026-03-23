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
          pink: '#FF6B9D',
          purple: '#C678DD',
          cyan: '#56B6C2',
        },
        text: {
          dark: '#4a3f5c',
          muted: '#7a6b8a',
        },
        bg: {
          start: '#ffeef5',
          mid: '#f0e6ff',
          end: '#e6f5ff',
        }
      },
      backgroundImage: {
        'gradient-main': 'linear-gradient(135deg, #ffeef5 0%, #f0e6ff 50%, #e6f5ff 100%)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
