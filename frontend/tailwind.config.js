/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        'oil-dark': '#0a0e1a',
        'oil-surface': '#111827',
        'oil-card': '#1a2332',
        'oil-border': '#2a3a4e',
        'oil-accent': '#00d4aa',
        'oil-warning': '#f59e0b',
        'oil-danger': '#ef4444',
        'oil-blue': '#3b82f6',
        'oil-purple': '#8b5cf6',
      }
    }
  },
  plugins: []
}
