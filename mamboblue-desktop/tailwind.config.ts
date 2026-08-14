/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#F9F9F8",
        surface: "#FFFFFF",
        primary: "#111111",
        secondary: "#68645C",
        border: "#E2DFD5",
        accent: "#EFEFEF",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
