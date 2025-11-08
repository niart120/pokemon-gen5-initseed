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
    include: [
      'src/test/generation/early-termination.test.ts',
      'src/test/generation/generation-worker.test.ts',
      'src/test/generation/generation-manager.test.ts',
    ],
    exclude: ['node_modules', 'dist', '.git', '.cache'],
    setupFiles: ['./src/test/setup-browser.ts'],
    testTimeout: 15000,
    browser: {
      enabled: true,
      headless: true,
      provider: playwright({
        launchOptions: {
          channel: 'chrome',
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
