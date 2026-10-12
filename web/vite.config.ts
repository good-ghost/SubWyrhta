import { fileURLToPath, URL } from 'node:url'

import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vitest/config'

// vue-smartview 는 vendor 에 둔 배포본 dist/ 를 번들에 넣는다.
const SMARTVIEW_DIST = fileURLToPath(new URL('./vendor/vue-smartview/dist/', import.meta.url))
// 개발 서버가 /api 를 넘길 serve 주소 (설정 web.port 기본값)
const SERVER_URL = process.env.SUBWYRHTA_WEB_URL ?? 'http://127.0.0.1:8950'

export default defineConfig({
  // 서버가 / 아래에 정적 파일로 낸다. 상대 경로면 리버스 프록시의 하위 경로에서도 열린다
  base: './',
  plugins: [
    vue({ template: { compilerOptions: { isCustomElement: (tag) => tag.startsWith('smartview-') } } }),
  ],
  resolve: {
    alias: [
      { find: /^vue-smartview$/, replacement: `${SMARTVIEW_DIST}vue-smartview.js` },
      // 요소별 진입점: import 하면 그 요소만 등록한다
      { find: /^vue-smartview\/elements\/(.+)$/, replacement: `${SMARTVIEW_DIST}elements/$1.js` },
    ],
  },
  server: {
    proxy: { '/api': { target: SERVER_URL, changeOrigin: false } },
  },
  build: {
    // 번들에 Vue·Vuetify 가 들어 있어 기본 경고 기준(500kB)을 넘는다 (vue-smartview CONSUMING 2.2)
    chunkSizeWarningLimit: 2048,
  },
  test: {
    environment: 'happy-dom',
    include: ['src/**/*.test.ts'],
  },
})
