import { defineConfig } from "vitest/config";
export default defineConfig({
  build: {
    outDir: "../src/rolloutdiff/_web",
    emptyOutDir: true,
    sourcemap: false,
  },
  test: {
    environment: "jsdom",
    setupFiles: "./src/test-setup.ts",
    restoreMocks: true, exclude: ["e2e/**", "node_modules/**"],
  },
});
