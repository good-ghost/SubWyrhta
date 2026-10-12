// E2E 전체 실행 전에 한 번: 이전 캡처·결과를 지운다 (워커가 다시 떠도 다시 지우지 않는다)
import fs from 'node:fs'
import path from 'node:path'

import { COVERAGE_DIR, MOVIE_DIR, RESULTS_FILE, SCREENSHOT_DIR, VIDEO_TMP } from './env'

export default function globalSetup(): void {
  for (const dir of [SCREENSHOT_DIR, MOVIE_DIR, COVERAGE_DIR, VIDEO_TMP]) {
    fs.mkdirSync(dir, { recursive: true })
    for (const file of fs.readdirSync(dir)) fs.rmSync(path.join(dir, file), { recursive: true })
  }
  fs.rmSync(RESULTS_FILE, { force: true })
}
