import { defineConfig } from 'vitest/config'
import path from 'path'
import { fileURLToPath } from 'url'
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
  test: {
    globals: true,
    // WebAssemblyロジックは node 環境でも動作するが、Reactコンポーネント（Testing Library）にはDOMが必要。
    // happy-dom を統一利用。純Rustテストに影響する場合は、将来ファイルパターンで分離可能。
    environment: 'happy-dom',
    include: [
      'src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}',
      'test/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}',
    ],
    exclude: [
      'node_modules',
      'dist',
      '.git',
      '.cache',
      // WebGPU / worker tests are run in browser-specific configs
      'src/test/webgpu/**',
      'src/test/mt-seed-search/**',
    ],
    setupFiles: ['./src/test/setup.ts'],
    testTimeout: 10000,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
