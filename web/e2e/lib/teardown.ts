// E2E 전체 실행 뒤에 한 번: 장면 결과로 스크린샷 색인·요약·조작 커버리지를 쓴다
import fs from 'node:fs'
import path from 'node:path'

import { COVERAGE_DIR, REPO_DIR, RESULTS_FILE, SCREENSHOT_DIR, type SceneResult } from './env'

/** 행마다 반복되는 요소(행 액션 아이콘)는 한 종류로 본다: 한 행에서 눌렀으면 그 기능은 확인했다 */
function coverageKey(key: string): string {
  return key.replace(/ \| row=[^|]*/, '')
}

function comboCoverage(results: SceneResult[], combo: string): { total: number; touched: number; untouched: string[] } {
  const scoped = results.filter((r) => r.combo === combo)
  const seen = new Set(scoped.flatMap((r) => r.seen.map(coverageKey)))
  const touched = new Set(scoped.flatMap((r) => r.touched.map(coverageKey)))
  const untouched = [...seen].filter((key) => !touched.has(key)).sort()
  return { total: seen.size, touched: seen.size - untouched.length, untouched }
}

export default function globalTeardown(): void {
  if (!fs.existsSync(RESULTS_FILE)) return
  const results = fs
    .readFileSync(RESULTS_FILE, 'utf8')
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line) as SceneResult)
  const cell = (v: string) => String(v || '').replace(/\|/g, '\\|').replace(/\n/g, ' ')
  const shotLines = [
    '# 웹 화면 E2E 스크린샷',
    '',
    '`web/e2e/walkthrough.spec.ts`가 만든다. 번호: 조합(1000 ko·라이트, 2000 en·다크) + 장면(100 단위).',
    '',
  ]
  for (const result of results) {
    shotLines.push(`## ${result.label}`, '', '| 번호 | 파일 | 장면 | 설명 |', '|---|---|---|---|')
    for (const shot of result.shots) {
      shotLines.push(`| ${shot.file.slice(0, 4)} | [${shot.file}](${shot.file}) | ${cell(shot.scene)} | ${cell(shot.caption || shot.name)} |`)
    }
    shotLines.push('')
  }
  fs.writeFileSync(path.join(SCREENSHOT_DIR, 'README.md'), shotLines.join('\n'))
  const summary = [
    '# 웹 화면 E2E 결과',
    '',
    '`cd web && npm run test:e2e` (개발 컨테이너, 로컬 serve 서버, Playwright `channel: chromium`, Sandbox Bar Toggle 확장 로드).',
    'STT 는 실제로 처리한다 (faster-whisper tiny, CPU int8). 정지 구간 검사: `bash web/e2e/check-freeze.sh`.',
    '',
    '| 장면 | 결과 | 길이(초) | 스크린샷 | 동영상 |',
    '|---|---|---|---|---|',
    ...results.map((r) => {
      const movie = `${r.label}${r.failed ? '-FAILED' : ''}`
      return `| ${r.label} | ${r.failed ? '실패' : '통과'} | ${r.seconds} | ${r.shots.length} | [webm](movie/${movie}.webm) · [vtt](movie/${movie}.vtt) |`
    }),
    '',
    '## 조작 커버리지',
    '',
    '조합의 모든 장면에서 보인 상호작용 요소(버튼·링크·입력칸·표 머리글, Shadow DOM 포함) 중 누르거나 입력하지 않은 것이다. 행마다 반복되는 아이콘은 한 종류로 센다 (`coverage/<조합>.json`).',
    '',
    ...[...new Set(results.map((r) => r.combo))].flatMap((combo) => {
      const coverage = comboCoverage(results, combo)
      fs.mkdirSync(COVERAGE_DIR, { recursive: true })
      fs.writeFileSync(path.join(COVERAGE_DIR, `${combo}.json`), JSON.stringify(coverage, null, 2) + '\n')
      return [
        `### ${combo}: ${coverage.touched} / ${coverage.total}`,
        '',
        ...(coverage.untouched.length ? coverage.untouched.map((key) => `- \`${key}\``) : ['- 없음']),
        '',
      ]
    }),
  ]
  fs.writeFileSync(path.join(REPO_DIR, 'docs', 'e2e', 'README.md'), summary.join('\n'))
}
