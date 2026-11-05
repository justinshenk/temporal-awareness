/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,ts,tsx}"],
  // Class strategy: the whole theme is driven by a single <html class="dark">,
  // set before paint by the inline script in BaseLayout.
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        // New York Times-style stack: Franklin (open-sourced by NYT) for UI/kickers,
        // Caslon for display headlines, Georgia (NYT's web body) for running text.
        sans: ['"Libre Franklin"', "ui-sans-serif", "system-ui", "sans-serif"],
        serif: ['"Libre Caslon Text"', "Georgia", "Cambria", "Times New Roman", "serif"],
        body: ["Georgia", "Cambria", "Times New Roman", "serif"],
      },
      colors: {
        // Semantic tokens ride CSS variables (set in global.css), so swapping the
        // variables under .dark re-themes every `bg-paper`/`text-ink`/etc. at once.
        paper: "rgb(var(--color-paper) / <alpha-value>)", // page background
        ink: "rgb(var(--color-ink) / <alpha-value>)", // headline/text
        "ink-soft": "rgb(var(--color-ink-soft) / <alpha-value>)", // body text
        rule: "rgb(var(--color-rule) / <alpha-value>)", // hairline rules
        surface: "rgb(var(--color-surface) / <alpha-value>)", // raised cards/panels
        "series-read": "rgb(var(--color-series-read) / <alpha-value>)", // probe accuracy marks
        "series-causal": "rgb(var(--color-series-causal) / <alpha-value>)", // patching marks
        "series-steer": "rgb(var(--color-series-steer) / <alpha-value>)", // steering marks
        accent: {
          50: "#ecfdf5",
          100: "#d1fae5",
          200: "#a7f3d0",
          300: "#6ee7b7",
          400: "#34d399",
          500: "#10b981",
          600: "#059669",
          700: "#047857",
          800: "#065f46",
          900: "#064e3b",
        },
      },
      maxWidth: {
        prose: "68ch",
      },
    },
  },
  plugins: [],
};
