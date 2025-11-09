import { defineConfig } from 'vitest/config';
import path from 'path';
import { fileURLToPath } from 'url';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';
import { playwright } from '@vitest/browser-playwright';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
  test: {
    globals: true,
    include: ['test/performance/*webgpu*.test.ts', 'src/test/webgpu/**/*.test.ts'],
    exclude: ['node_modules', 'dist', '.git', '.cache'],
    setupFiles: ['./src/test/setup-browser.ts'],
    testTimeout: 15000,
    browser: {
      enabled: true,
      headless: true,
      provider: playwright({
        launchOptions: {
          channel: 'chrome',
          args: [
            '--enable-unsafe-webgpu',
            '--enable-features=Vulkan,WebGPUDeveloperFeatures',
          ],
        },
      }),
      instances: [
        {
          browser: 'chromium',
        },
      ],
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
