import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the frontend and API run on separate ports, so proxy /api to the
// FastAPI server. In shipped mode the built dist/ is served by FastAPI itself
// on the same origin, so the same relative /api/generate fetch works unchanged.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    // FastAPI mounts frontend/dist in production.
    outDir: "dist",
  },
});
