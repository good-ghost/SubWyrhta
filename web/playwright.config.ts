import { defineConfig } from '@playwright/test'

// 브라우저는 테스트가 직접 띄운다 (확장 로드에 persistent context 가 필요하다, e2e/lib/env.ts).
// 대상은 빌드한 화면(web/dist)을 내는 로컬 serve 서버다. `npm run test:e2e` 가 먼저 빌드한다.
export default defineConfig({
  testDir: './e2e',
  testMatch: '**/*.spec.ts',
  // 캡처 정리·요약은 실행 전후 한 번만 (실패하면 워커가 다시 떠 파일 수준 beforeAll 이 다시 돈다)
  globalSetup: './e2e/lib/setup.ts',
  globalTeardown: './e2e/lib/teardown.ts',
  timeout: 20 * 60 * 1000,
  expect: { timeout: 15_000 },
  workers: 1,
  retries: 0,
  reporter: [['list']],
  outputDir: './test-results/playwright',
  webServer: {
    command: 'bash e2e/server/run.sh',
    url: `http://127.0.0.1:${process.env.E2E_PORT ?? 18950}/api/session`,
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
  use: {
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
  },
})
