/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#11203e",
        mist: "#e9f0ff",
        frost: "#f6f9ff",
        aqua: "#38bdf8",
        mint: "#35d8a6",
        coral: "#ff6f61",
      },
      boxShadow: {
        glass: "0 20px 50px -28px rgba(17, 32, 62, 0.45)",
      },
      keyframes: {
        reveal: {
          "0%": { opacity: 0, transform: "translateY(8px)" },
          "100%": { opacity: 1, transform: "translateY(0)" },
        },
      },
      animation: {
        reveal: "reveal 420ms ease-out both",
      },
    },
  },
  plugins: [],
};