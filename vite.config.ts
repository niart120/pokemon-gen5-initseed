import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react-swc";
import { defineConfig } from "vite";

import fs from 'node:fs'
import { resolve } from 'path'

const projectRoot = process.env.PROJECT_ROOT || import.meta.dirname
const certPath = resolve(projectRoot, 'certs', 'localhost.pem')
const keyPath = resolve(projectRoot, 'certs', 'localhost-key.pem')
const httpsConfig = fs.existsSync(certPath) && fs.existsSync(keyPath)
  ? {
      cert: fs.readFileSync(certPath),
      key: fs.readFileSync(keyPath),
    }
  : undefined

// GitHub Pages用のベースパス設定（本番ビルド時のみ適用）
const isProduction = process.env.NODE_ENV === 'production'
const base = isProduction ? '/pokemon-gen5-initseed/' : '/'

// https://vite.dev/config/
export default defineConfig({
  base,
  plugins: [
    react(),
    tailwindcss(),
  ],
  resolve: {
    alias: {
      '@': resolve(projectRoot, 'src')
    }
  },
  worker: {
    format: 'es',
  },
  server: {
    host: true,
    https: httpsConfig,
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor': ['react', 'react-dom'],
          'ui': ['@radix-ui/react-dialog', '@radix-ui/react-select'],
          'utils': ['date-fns', 'zustand']
        }
      }
    }
  }
});
