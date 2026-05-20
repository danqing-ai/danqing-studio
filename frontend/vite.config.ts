import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import { resolve } from 'path';

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
    dedupe: ['vue'],
  },
  optimizeDeps: {
    include: ['reka-ui', '@danqing/dq-ui', '@danqing/dq-shell'],
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:7860',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: resolve(__dirname, '../out/frontend/dist'),
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      onwarn(warning, warn) {
        // @vueuse/core (via dq-ui): harmless PURE annotation placement noise
        if (warning.message?.includes('__PURE__')) return;
        warn(warning);
      },
    },
  },
});