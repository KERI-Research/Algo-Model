import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
	// Relative base so the bundle works when served from a nested preview path.
	base: "./",
	plugins: [react()],
	server: {
		port: 5173,
		// Local development only: the production build is served by FastAPI itself.
		proxy: { "/api": "http://127.0.0.1:5000" },
	},
	build: { outDir: "dist", emptyOutDir: true, sourcemap: false },
	test: {
		environment: "jsdom",
		globals: true,
		setupFiles: ["./src/test-setup.js"],
		include: ["src/**/*.test.{js,jsx}"],
	},
});
