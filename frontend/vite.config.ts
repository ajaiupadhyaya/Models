import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

// Vite config for the Bloomberg-style terminal SPA
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Local dev only: proxy /api to backend. Production build uses same origin (no proxy).
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true
      }
    }
  },
  build: {
    sourcemap: true
  }
});

