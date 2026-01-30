import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

// Vite config for the Bloomberg-style terminal SPA
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        ws: true
      }
    }
  },
  build: {
    sourcemap: true
  }
});

