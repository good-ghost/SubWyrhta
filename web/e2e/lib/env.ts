// E2E 실행 환경 값과 브라우저 실행 헬퍼
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { chromium } from '@playwright/test'
import type { BrowserContext } from '@playwright/test'

/** web/ 폴더 */
export const WEB_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
export const REPO_DIR = path.dirname(WEB_DIR)

// 테스트·녹화에 반드시 로드해야 하는 브라우저 확장 (Salesforce Sandbox Bar Toggle, MV3)의 폴더.
// 경로는 실행하는 사람마다 달라 저장소에 적지 않는다: 환경 변수, 없으면 web/e2e/.env.local 의 같은 이름
export const EXTENSION_PATH_ENV = 'E2E_EXTENSION_PATH'
const LOCAL_ENV_FILE = path.join(WEB_DIR, 'e2e', '.env.local')

export const E2E_PORT = Number(process.env.E2E_PORT ?? 18950)
export const BASE_URL = `http://127.0.0.1:${E2E_PORT}/`
/** e2e/server/run.sh 가 데이터·설정·비밀번호를 두는 곳 (실행마다 새로 만든다) */
export const SERVER_ROOT = path.join(WEB_DIR, 'test-results', 'e2e-server')
export const FIXTURES = path.join(WEB_DIR, 'e2e', 'fixtures')
export const SCREENSHOT_DIR = path.join(REPO_DIR, 'docs', 'e2e', 'screenshots')
export const MOVIE_DIR = path.join(REPO_DIR, 'docs', 'e2e', 'movie')
// 녹화 해상도 (공통 규칙: 기본 1920×1080)
export const VIDEO_VIEWPORT = { width: 1920, height: 1080 }
export const ADMIN = 'admin'
export const COVERAGE_DIR = path.join(REPO_DIR, 'docs', 'e2e', 'coverage')
export const VIDEO_TMP = path.join(WEB_DIR, 'test-results', 'video')
/** 장면 결과 (한 줄에 JSON 하나). globalSetup 이 지우고 globalTeardown 이 요약한다 */
export const RESULTS_FILE = path.join(WEB_DIR, 'test-results', 'e2e-results.jsonl')

export interface SceneResult {
  combo: string
  label: string
  failed: boolean
  shots: { file: string; name: string; scene: string; caption: string }[]
  seen: string[]
  touched: string[]
  seconds: number
}

/** 관리 계정 비밀번호 (run.sh 가 실행마다 새로 만든다) */
export function adminPassword(): string {
  return fs.readFileSync(path.join(SERVER_ROOT, 'token'), 'utf8').trim()
}

/** web/e2e/.env.local 의 `이름=값` 줄에서 값을 읽는다 (파일이나 이름이 없으면 ''). */
function localEnvValue(name: string): string {
  if (!fs.existsSync(LOCAL_ENV_FILE)) return ''
  for (const line of fs.readFileSync(LOCAL_ENV_FILE, 'utf8').split('\n')) {
    const [key, ...rest] = line.split('=')
    if (key.trim() === name) return rest.join('=').trim()
  }
  return ''
}

/** 확장 경로. 정하지 않았거나 그 폴더에 확장이 없으면 조용히 건너뛰지 않고 실패시킨다. */
export function requireExtensionPath(): string {
  const extensionPath = process.env[EXTENSION_PATH_ENV]?.trim() || localEnvValue(EXTENSION_PATH_ENV)
  if (!extensionPath) {
    throw new Error(`브라우저 확장 경로가 없습니다: 환경 변수 ${EXTENSION_PATH_ENV} 또는 ${LOCAL_ENV_FILE} 에 넣으세요`)
  }
  if (!fs.existsSync(path.join(extensionPath, 'manifest.json'))) {
    throw new Error(`브라우저 확장을 찾지 못했습니다: ${extensionPath}`)
  }
  return extensionPath
}

/** 확장을 로드한 persistent context (확장은 persistent context 에서만 로드된다), 녹화 포함. */
export async function launchRecorded(videoDir: string, colorScheme: 'light' | 'dark'): Promise<BrowserContext> {
  const extensionPath = requireExtensionPath()
  const userDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'subwyrhta-e2e-'))
  return chromium.launchPersistentContext(userDataDir, {
    channel: 'chromium',
    viewport: VIDEO_VIEWPORT,
    colorScheme,
    acceptDownloads: true,
    args: [`--disable-extensions-except=${extensionPath}`, `--load-extension=${extensionPath}`],
    recordVideo: { dir: videoDir, size: VIDEO_VIEWPORT },
    permissions: ['clipboard-read', 'clipboard-write'],
  })
}

/** 로그인 화면을 거치지 않고 세션 쿠키를 받는다 (로그인 장면이 아닌 녹화에서 로그인 과정을 빼기 위해) */
export async function sessionCookie(username = ADMIN, password = adminPassword()): Promise<{ name: string; value: string }> {
  const response = await fetch(`${BASE_URL}api/session`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!response.ok) throw new Error(`로그인 실패: ${response.status}`)
  const cookie = response.headers.getSetCookie()[0] ?? ''
  const [pair] = cookie.split(';')
  const index = pair.indexOf('=')
  return { name: pair.slice(0, index), value: pair.slice(index + 1) }
}
