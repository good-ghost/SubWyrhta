// 웹 화면 전체 E2E·녹화 (로컬 serve 서버, 실제 STT: faster-whisper tiny CPU)
// - 조합: ko·라이트, en·다크. 장면마다 동영상 하나(docs/e2e/movie/<조합>-<장면>.webm + .vtt)
// - 스크린샷: docs/e2e/screenshots/NNNN-<조합>-<단계>.png (조합 1000·2000번대, 장면 100단위)
// - 모든 버튼을 누르고 모든 입력칸에 값을 넣는다. 장면이 끝나면 누르지 않은 요소를 커버리지 파일로 남긴다
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

import { expect, test, type BrowserContext, type Locator, type Page } from '@playwright/test'

import { MESSAGES, type MessageKey } from '../src/i18n'
import { FIELD_TABS, TAB_FIELDS, type SettingField } from '../src/settings/fields'
import { Director, escapeRe, PACE } from './lib/director'
import {
  ADMIN,
  adminPassword,
  BASE_URL,
  FIXTURES,
  REPO_DIR,
  SERVER_ROOT,
  launchRecorded,
  MOVIE_DIR,
  RESULTS_FILE,
  SCREENSHOT_DIR,
  sessionCookie,
  VIDEO_TMP,
  type SceneResult,
} from './lib/env'

type Locale = 'ko' | 'en'
type Theme = 'light' | 'dark'

interface Combo {
  locale: Locale
  theme: Theme
  base: number
  // 첫 조합은 처음 설정(관리 계정 만들기)부터 시작한다
  firstRun: boolean
  // 마지막 조합은 계정을 실제로 바꾼다 (이후 장면이 없다)
  changeAccount: boolean
}

const COMBOS: Combo[] = [
  { locale: 'ko', theme: 'light', base: 1000, firstRun: true, changeAccount: false },
  { locale: 'en', theme: 'dark', base: 2000, firstRun: false, changeAccount: true },
]
// 실제 STT 처리를 기다리는 상한 (CPU, tiny 모델, 첫 실행은 모델을 내려받는다)
const JOB_TIMEOUT_MS = 6 * 60_000
// 처리를 기다리는 동안 커서를 움직이는 간격 (녹화 정지 구간 5초 이하)
const MOTION_MS = 600
// 알림 닫기를 기다리는 상한. 그 사이에 알림이 스스로 닫히면 넘어간다
const TOAST_CLOSE_TIMEOUT_MS = 3_000
const NEW_PASSWORD = 'changed-password-1'


interface ApiJob {
  id: number
  name: string
  status: string
  stage: string | null
  outputs: string[]
}

/** 장면 결과를 파일에 쌓는다 (실패하면 워커가 다시 떠 메모리가 사라진다. 요약은 e2e/lib/teardown.ts) */
function saveResult(result: SceneResult): void {
  fs.mkdirSync(path.dirname(RESULTS_FILE), { recursive: true })
  fs.appendFileSync(RESULTS_FILE, JSON.stringify(result) + '\n')
}

function fixture(name: string, as?: string): { name: string; mimeType: string; buffer: Buffer } {
  const mimeType = name.endsWith('.wav') ? 'audio/wav' : 'video/mp4'
  return { name: as ?? name, mimeType, buffer: fs.readFileSync(path.join(FIXTURES, name)) }
}

/**
 * 약 9분 음성 (e2e/server/run.sh 가 speech.wav 를 이어 붙여 만든다). 짧은 샘플은 tiny 모델이 3초 안에 끝내
 * 처리 중 화면·실행 중 취소를 보일 수 없다
 */
function longSpeech(as: string): { name: string; mimeType: string; buffer: Buffer } {
  return { name: as, mimeType: 'audio/wav', buffer: fs.readFileSync(path.join(SERVER_ROOT, 'speech-long.wav')) }
}

const TERMINAL = new Set(['done', 'failed', 'cancelled'])

/** 한 장면: 녹화 context·연출·조합별 문구 */
class Scene {
  constructor(
    readonly context: BrowserContext,
    readonly page: Page,
    readonly d: Director,
    readonly combo: Combo,
    readonly name: string,
  ) {}

  t(key: MessageKey, params: Record<string, string | number> = {}): string {
    return MESSAGES[this.combo.locale][key].replace(/\{(\w+)\}/g, (w, k: string) => String(params[k] ?? w))
  }

  /** 자막: 조합 언어로 (한국어 문장, 영어 문장) */
  async say(ko: string, en: string, holdMs?: number): Promise<void> {
    await this.d.say(this.combo.locale === 'ko' ? ko : en, holdMs)
  }

  byTest(id: string): Locator {
    return this.page.locator(`[data-testid="${id}"]`)
  }

  dialog(): Locator {
    return this.page.getByRole('dialog')
  }

  /**
   * 누르고 알림을 확인한다. 알림은 몇 초 뒤 사라지므로 누르기 전에 기다리기 시작한다.
   * count 를 모르면 숫자 자리를 아무 숫자로 본다.
   */
  async clickToast(target: Locator, key: MessageKey, params: Record<string, string | number> = {}, shot?: string): Promise<void> {
    const text = escapeRe(MESSAGES[this.combo.locale][key]).replace(/\\\{(\w+)\\\}/g, (_w, k: string) =>
      k in params ? escapeRe(String(params[k])) : '.+',
    )
    const toast = this.page.locator('[data-part="snackbar"]').filter({ hasText: new RegExp(text) })
    const seen = expect(toast.first()).toBeVisible({ timeout: 60_000 })
    await this.d.click(target, null, { pace: 'input' })
    await seen
    if (shot) await this.d.shot(shot)
    await this.page.waitForTimeout(PACE.transition)
  }

  async closeToast(): Promise<void> {
    const close = this.page.locator('[data-part="snackbar"] [data-part="close"]').first()
    if (!(await close.isVisible())) return
    try {
      await this.d.click(close, null, { pace: 'input', timeout: TOAST_CLOSE_TIMEOUT_MS })
    } catch (error) {
      // 알림은 시간이 지나면 스스로 닫힌다. 커서가 닫기 버튼으로 가는 사이에 닫혔으면 닫을 것이 없다
      if (await close.isVisible()) throw error
    }
  }

  /** 콤보박스·필터 메뉴에서 글자가 같은 항목을 고른다 */
  async choose(field: Locator, label: string, shot?: string): Promise<void> {
    await this.d.select(field, label, shot ?? null)
  }

  toggleOf(host: Locator): Locator {
    return host.locator('[data-part="toggle"] .v-selection-control__input')
  }

  async toggle(host: Locator, shot?: string): Promise<void> {
    await this.d.click(this.toggleOf(host), shot ?? null, { pace: 'input' })
  }

  /** 정렬할 수 있는 머리글을 모두 한 번씩 누른다 (마지막에 첫 열로 돌아온다) */
  async sortAll(table: Locator): Promise<void> {
    const headers = table.locator('th[tabindex]')
    const count = await headers.count()
    for (let index = 0; index < count; index += 1) await this.d.click(headers.nth(index), null, { pace: 'input' })
    await this.d.click(headers.first(), null, { pace: 'input' })
  }

  /** 표 아래 페이지 단추(다음·이전·마지막·처음)를 누를 수 있으면 누른다 */
  async pageAll(table: Locator): Promise<void> {
    const buttons = table.locator('.v-data-table-footer__pagination button')
    for (const index of [2, 1, 3, 0]) {
      const button = buttons.nth(index)
      if ((await button.count()) && (await button.isEnabled())) await this.d.click(button, null, { pace: 'input' })
    }
  }

  async navigate(key: string): Promise<void> {
    await this.d.click(this.page.locator(`smartview-app-shell [data-part="item"][data-key="${key}"]`), null)
  }

  async jobs(): Promise<ApiJob[]> {
    const response = await this.page.request.get(`${BASE_URL}api/jobs`)
    expect(response.ok()).toBe(true)
    return (await response.json()) as ApiJob[]
  }

  async job(name: string): Promise<ApiJob> {
    const found = (await this.jobs()).find((job) => job.name === name)
    if (!found) throw new Error(`작업 없음: ${name}`)
    return found
  }

  /**
   * 작업이 조건을 만족할 때까지 기다린다. 기다리는 동안 커서를 작업 행에 올렸다 내리기를 번갈아
   * 행 강조가 켜지고 꺼지게 해 녹화가 멈춰 보이지 않게 한다. 커서만 움직이는 변화는 정지 검사
   * (check-freeze.sh) 문턱에 못 미치고, 진행 막대는 구간 하나가 끝날 때(small 모델은 약 5초)만 바뀐다.
   */
  async waitJob(name: string, predicate: (job: ApiJob) => boolean, timeoutMs = JOB_TIMEOUT_MS): Promise<ApiJob> {
    const deadline = Date.now() + timeoutMs
    let step = 0
    while (Date.now() < deadline) {
      const job = (await this.jobs()).find((candidate) => candidate.name === name)
      if (job && predicate(job)) return job
      // 기다리던 상태가 아닌 종료 상태면 시간 초과까지 기다리지 않는다
      if (job && TERMINAL.has(job.status)) throw new Error(`작업이 다른 상태로 끝남: ${name} ${job.status}`)
      step += 1
      const row = this.page.locator('tbody tr').filter({ has: this.page.locator('[data-part="name"]', { hasText: name }) }).first()
      // isVisible 는 기다리지 않는다. 행이 없으면 표 아래 빈 곳만 오간다
      const box = step % 2 === 0 && (await row.isVisible()) ? await row.boundingBox() : null
      const x = box ? box.x + box.width * 0.3 : 900 + Math.round(Math.cos(step / 2) * 120)
      const y = box ? box.y + box.height / 2 : 620 + Math.round(Math.sin(step / 2) * 60)
      await this.page.mouse.move(x, y, { steps: 12 })
      await this.page.waitForTimeout(MOTION_MS)
    }
    throw new Error(`작업 대기 시간 초과: ${name}`)
  }

  row(table: Locator, name: string): Locator {
    return table.locator('tbody tr').filter({ has: this.page.locator('[data-part="name"]', { hasText: name }) })
  }

  action(table: Locator, name: string, action: string): Locator {
    return this.row(table, name).locator(`[data-part="icon-action"][data-action="${action}"]`)
  }

  /** 상세 창 출력 파일 표의 행 동작 (파일 이름 열은 일반 글자 열이다) */
  outputAction(name: string, action: string): Locator {
    return this.page
      .locator('[data-testid="detail-outputs"] tbody tr')
      .filter({ hasText: name })
      .locator(`[data-part="icon-action"][data-action="${action}"]`)
  }
}

async function openScene(combo: Combo, name: string, sceneNo: number, route: string, loggedIn = true): Promise<Scene> {
  const context = await launchRecorded(VIDEO_TMP, combo.theme)
  await Director.install(context)
  // 처음 한 번만 언어·테마를 정한다 (장면 안에서 메뉴로 바꾼 값은 새로고침해도 유지)
  await context.addInitScript(
    ([locale, theme]) => {
      if (sessionStorage.getItem('e2e-prefs')) return
      localStorage.setItem('subwyrhta.locale', locale)
      localStorage.setItem('subwyrhta.theme', theme)
      sessionStorage.setItem('e2e-prefs', '1')
    },
    [combo.locale, combo.theme],
  )
  if (loggedIn) {
    const cookie = await sessionCookie()
    await context.addCookies([{ ...cookie, url: BASE_URL }])
  }
  const page = context.pages()[0] ?? (await context.newPage())
  const d = new Director(page, {
    screenshotDir: SCREENSHOT_DIR,
    prefix: `${combo.locale}-${combo.theme}`,
    firstShot: combo.base + sceneNo * 100 + 1,
  })
  d.markVideoStart()
  await page.goto(`${BASE_URL}#/${route}`)
  const ready = loggedIn
    ? page.locator('smartview-app-shell [data-part="main"]')
    : page.locator('[data-testid="setup-form"], smartview-login [data-part="form"]').first()
  await ready.waitFor()
  await Promise.all([page.waitForTimeout(800), d.checkpoint()])
  return new Scene(context, page, d, combo, name)
}

async function closeScene(scene: Scene, failed: boolean): Promise<void> {
  const label = `${scene.combo.locale}-${scene.combo.theme}-${scene.name}`
  await scene.d.coverageSummary().catch(() => undefined)
  saveResult({
    combo: `${scene.combo.locale}-${scene.combo.theme}`,
    label,
    failed,
    shots: scene.d.shots,
    seen: [...scene.d.seenAll],
    touched: [...scene.d.touchedAll],
    seconds: Math.round((Date.now() - scene.d.t0) / 1000),
  })
  await scene.d.clearCaption().catch(() => undefined)
  scene.d.writeVtt(path.join(MOVIE_DIR, `${label}${failed ? '-FAILED' : ''}.vtt`))
  const video = scene.page.video()
  await scene.context.close()
  if (video) {
    // persistent context 는 닫으면 브라우저도 닫혀 saveAs 를 쓸 수 없다. 끝난 파일을 옮긴다
    const recorded = await video.path()
    fs.copyFileSync(recorded, path.join(MOVIE_DIR, `${label}${failed ? '-FAILED' : ''}.webm`))
    fs.rmSync(recorded)
  }
}

/** 장면 하나를 녹화한다. 실패하면 영상 이름에 -FAILED 를 붙이고 다시 던진다 */
async function record(
  combo: Combo,
  name: string,
  sceneNo: number,
  route: string,
  body: (scene: Scene) => Promise<void>,
  loggedIn = true,
): Promise<void> {
  const scene = await openScene(combo, name, sceneNo, route, loggedIn)
  try {
    await body(scene)
  } catch (error) {
    await closeScene(scene, true).catch((closeError: unknown) => console.error('closeScene failed', closeError))
    throw error
  }
  await closeScene(scene, false)
}

// ──────────────────────────────────────────────
// 장면 본문
// ──────────────────────────────────────────────

async function setupAccount(s: Scene): Promise<void> {
  const { d } = s
  d.setScene(s.t('setup.heading'))
  await s.say('처음 접속하면 관리 계정을 만드는 화면이 나옵니다', 'On first access the server asks you to create the admin account')
  await d.shot('setup')
  await d.type(s.byTest('setup-username').locator('input'), ADMIN)
  await s.say('8자보다 짧은 비밀번호는 바로 안내합니다', 'Passwords shorter than 8 characters are flagged immediately')
  await d.type(s.byTest('setup-password').locator('input'), 'short', { shot: 'setup-short' })
  await d.type(s.byTest('setup-password').locator('input'), adminPassword(), { delay: 10 })
  await s.say('확인 칸이 다르면 만들 수 없습니다', 'The confirmation must match')
  await d.type(s.byTest('setup-confirm').locator('input'), 'mismatch-password', { shot: 'setup-mismatch' })
  await d.type(s.byTest('setup-confirm').locator('input'), adminPassword(), { delay: 10 })
  await s.say('눈 아이콘으로 입력한 비밀번호를 확인할 수 있습니다', 'The eye icon shows the typed password')
  for (const id of ['setup-password', 'setup-confirm']) {
    const eye = s.byTest(id).locator('[data-part="toggle"]')
    if (await eye.count()) {
      await d.click(eye, null, { pace: 'input' })
      await d.click(eye, null, { pace: 'input' })
    }
  }
  await d.click(s.byTest('setup-submit').locator('button'), null)
  await s.page.locator('smartview-app-shell [data-part="main"]').waitFor()
  await d.shot('setup-done-dashboard')
}

async function loginScreen(s: Scene, password: string): Promise<void> {
  const { d, page } = s
  const login = page.locator('smartview-login')
  await login.locator('[data-part="form"]').waitFor()
  await d.shot('login')
  await s.say('빈 칸으로 로그인하면 필수 항목 안내가 보입니다', 'Submitting empty fields shows the required hints')
  await d.click(login.locator('[data-part="submit"]'), 'login-empty', { pace: 'input' })
  await s.say('틀린 비밀번호는 1초 늦게 거절합니다 (대입 공격 완화)', 'A wrong password is rejected after a 1 s delay (slows brute force)')
  await d.type(login.locator('[data-part="username"] input'), ADMIN)
  await d.type(login.locator('[data-part="password"] input'), 'wrong-password-demo')
  await d.click(login.locator('[data-part="toggle"]'), 'login-visible', { pace: 'input' })
  await d.click(login.locator('[data-part="submit"]'), null)
  await expect(login.locator('[data-part="error"]')).toContainText(s.t('login.invalid'))
  await d.shot('login-invalid')
  const closeError = login.locator('[data-part="error"] button').first()
  if (await closeError.count()) await d.click(closeError, null, { pace: 'input' })
  await d.click(login.locator('[data-part="toggle"]'), null, { pace: 'input' })
  await s.say('맞는 비밀번호로 로그인합니다', 'Sign in with the correct password')
  await d.type(login.locator('[data-part="password"] input'), password, { delay: 10 })
  await d.click(login.locator('[data-part="submit"]'), null)
  await page.locator('smartview-app-shell [data-part="main"]').waitFor()
  await d.shot('logged-in')
}

async function sceneShell(s: Scene): Promise<void> {
  const { d, page } = s
  await d.titleCard({
    badge: `${s.combo.locale.toUpperCase()} · ${s.combo.theme}`,
    title: 'SubWyrhta',
    subtitle: s.combo.locale === 'ko' ? '영상·음성 자막 생성 웹 화면 — 계정과 화면 구성' : 'Subtitle generation web UI — account and layout',
  })
  if (s.combo.firstRun) await setupAccount(s)
  else await loginScreen(s, adminPassword())

  d.setScene(s.t('nav.dashboard'))
  await s.say('왼쪽 메뉴: 대시보드, 작업 대기열, 완료 목록, 로그, 설정', 'Left menu: dashboard, job queue, completed, logs, settings')
  for (const key of ['queue', 'completed', 'logs', 'settings', 'dashboard']) {
    await s.navigate(key)
    await d.shot(`nav-${key}`)
  }
  await s.say('상단 바의 메뉴 버튼으로 메뉴를 접고 펼칩니다', 'The menu button collapses and expands the navigation')
  await d.click(page.locator('smartview-app-shell [data-part="menu"]'), 'rail-collapsed', { pace: 'input' })
  await d.click(page.locator('smartview-app-shell [data-part="menu"]'), null, { pace: 'input' })
  await d.click(page.locator('smartview-app-shell [data-part="footer-item"][data-key="rail"]'), null, { pace: 'input' })
  await d.click(page.locator('smartview-app-shell [data-part="footer-item"][data-key="rail"]'), 'rail-expanded', { pace: 'input' })
  await s.say('브레드크럼의 첫 항목은 대시보드로 돌아갑니다', 'The first breadcrumb returns to the dashboard')
  await s.navigate('logs')
  await d.click(page.locator('smartview-app-shell [data-part="crumb"] a').first(), 'breadcrumb-home')
  await s.say('언어를 바꾸면 메뉴와 화면 문구가 함께 바뀝니다', 'Switching the language changes the menu and screen text')
  await d.click(page.locator('smartview-app-shell [data-part="footer-item"][data-key="locale"]'), 'locale-switched')
  await d.click(page.locator('smartview-app-shell [data-part="footer-item"][data-key="locale"]'), 'locale-restored')
  await s.say('테마도 바꿀 수 있고, 새로고침 뒤에도 유지됩니다', 'The theme can be switched and survives a reload')
  await d.click(page.locator('smartview-app-shell [data-part="footer-item"][data-key="theme"]'), 'theme-switched')
  await d.checkpoint()
  await page.reload()
  await page.locator('smartview-app-shell [data-part="main"]').waitFor()
  await page.waitForTimeout(PACE.transition)
  await d.shot('theme-after-reload')
  await d.click(page.locator('smartview-app-shell [data-part="footer-item"][data-key="theme"]'), 'theme-restored')
  await s.say('로그아웃하면 로그인 화면으로 돌아갑니다', 'Signing out returns to the login screen')
  await d.click(page.locator('smartview-app-shell [data-part="footer-item"][data-key="logout"]'), 'logged-out')
  await loginScreen(s, adminPassword())
}

/** 새 작업 모달에서 파일을 고르고 (설정 함수로 옵션을 바꾼 뒤) 올려서 등록한다 */
async function submitJob(
  s: Scene,
  file: { name: string; mimeType: string; buffer: Buffer },
  configure?: (modal: Locator) => Promise<void>,
  shot?: string,
): Promise<void> {
  const { d, page } = s
  await d.click(page.locator('smartview-page-header [data-part="action"]'), null)
  const modal = s.dialog()
  // 모달 본문은 슬롯으로 보이는 호스트 요소라 dialog 의 자손이 아니다: 본문은 페이지에서 찾는다
  await d.chooseFile(page.locator('[data-testid="new-job-file"] [data-part="button"]'), file, shot ? `${shot}-file` : null)
  if (configure) await configure(page.locator('[data-testid="new-job-form"]'))
  await s.clickToast(modal.locator('[data-part="confirm"]'), 'newJob.registered', { name: file.name }, shot)
}

async function configureAllOptions(s: Scene, modal: Locator): Promise<void> {
  const { d } = s
  const by = (id: string): Locator => modal.locator(`[data-testid="${id}"]`)
  await s.say('음성 인식: 엔진, 모델 크기, 언어, 연산 정밀도를 고릅니다', 'Recognition: choose the engine, model size, language, and compute type')
  await s.choose(by('new-job-engine').locator('[data-part="field"]'), 'faster-whisper')
  await s.choose(by('new-job-size').locator('[data-part="field"]'), 'tiny')
  await s.choose(by('new-job-language').locator('[data-part="field"]'), 'en')
  await s.choose(by('new-job-compute').locator('[data-part="field"]'), 'int8', 'options-stt')
  await s.say('오디오 트랙을 비우면 음성 언어에 맞는 트랙을 자동으로 고릅니다', 'Leave the audio track empty to pick the track matching the spoken language')
  await d.type(by('new-job-audio-track').locator('input'), '-1', { shot: 'options-audio-track-invalid' })
  await d.type(by('new-job-audio-track').locator('input'), '')
  await s.say('Pre-VAD 를 끄면 파일 전체를 한 번에 인식합니다. 다시 켭니다', 'Turning Pre-VAD off transcribes the whole file at once. Turn it back on')
  await s.toggle(by('new-job-prevad'))
  await s.toggle(by('new-job-prevad'))
  await s.say('자막: 포맷을 여러 개 고르고 세그멘테이션 모드와 최대 CPS 를 정합니다', 'Subtitles: pick several formats, the segmentation mode, and max CPS')
  for (const format of ['vtt', 'ass', 'json']) {
    await d.click(by('new-job-formats').locator(`[data-part="chip"][data-value="${format}"]`), null, { pace: 'input' })
  }
  await s.choose(by('new-job-mode').locator('[data-part="field"]'), s.t('mode.advanced'))
  await d.type(by('new-job-cps').locator('input'), 'abc', { shot: 'options-cps-invalid' })
  await d.type(by('new-job-cps').locator('input'), '17')
  for (const id of ['new-job-punctuation', 'new-job-report', 'new-job-words']) await s.toggle(by(id))
  await s.say(
    'LLM 오인식 교정·부호 넣기는 자막을 나누기 전에, LLM 마지막 검토는 경계·줄바꿈·검수 리포트를 맡습니다',
    'LLM recognition fixes and punctuation run before splitting; LLM final review handles boundaries, line breaks, and a review report',
  )
  // E2E 서버에는 LLM 프로바이더가 없다: 켰다가 다시 끈다
  for (const id of ['new-job-llm-correction', 'new-job-llm-punctuation', 'new-job-llm-review']) {
    await s.toggle(by(id))
    await s.toggle(by(id))
  }
  await d.shot('options-output')
  await s.say('오디오: 정규화·노이즈 억제·배경음 분리를 켜고 끌 수 있습니다', 'Audio: normalization, noise reduction, and vocal separation can be toggled')
  await s.toggle(by('new-job-normalize'))
  await s.toggle(by('new-job-normalize'))
  for (const id of ['new-job-denoise', 'new-job-vocals']) {
    await s.toggle(by(id))
    await s.toggle(by(id))
  }
  await s.say('화자 분리를 켜면 화자 수 칸이 나옵니다 (pyannote·HF_TOKEN 필요). 이번에는 끕니다', 'Diarization shows a speaker count field (needs pyannote and HF_TOKEN). Off for this run')
  await s.toggle(by('new-job-diarize'))
  await d.type(by('new-job-speakers').locator('input'), '0', { shot: 'options-speakers-invalid' })
  await d.type(by('new-job-speakers').locator('input'), '2', { shot: 'options-diarize' })
  await s.toggle(by('new-job-diarize'))
  await d.shot('options-all')
}

async function sceneJobs(s: Scene): Promise<void> {
  const { d, page } = s
  const table = s.byTest('queue-table')
  const bar = s.byTest('queue-filter')
  d.setScene(s.t('nav.queue'))
  await s.say('작업 대기열은 등록한 순서대로 한 번에 하나씩 처리합니다', 'The queue processes one job at a time in the order added')
  await d.shot('queue-empty')

  d.setScene(s.t('newJob.heading'))
  await s.say('새 작업 버튼으로 업로드 창을 엽니다. 취소하면 그냥 닫힙니다', 'New job opens the upload dialog. Cancel just closes it')
  await d.click(page.locator('smartview-page-header [data-part="action"]'), 'new-job-modal')
  await d.click(s.dialog().locator('[data-part="cancel"]'), null)
  await s.say('영상 파일을 고르고 모든 처리 옵션을 정한 뒤 올립니다', 'Choose a video, set every processing option, and upload')
  await submitJob(s, fixture('demo-talk.mp4'), (modal) => configureAllOptions(s, modal), 'job1-registered')
  await s.closeToast()

  d.setScene(s.t('status.done'))
  await s.say('업로드가 끝나면 바로 처리합니다. 짧은 영상은 몇 초 만에 끝납니다', 'Processing starts right away. A short clip finishes in seconds')
  await s.waitJob('demo-talk.mp4', (job) => job.status === 'done')
  await page.waitForTimeout(PACE.transition)
  await d.shot('job1-done')

  // 긴 작업은 small 모델로 처리한다: tiny 는 9분 음성도 20초 안에 끝내 실행 중 취소·진행 화면을 보일 틈이 없다
  const smallModel = async (form: Locator): Promise<void> => {
    await s.choose(form.locator('[data-testid="new-job-size"] [data-part="field"]'), 'small')
  }

  d.setScene(s.t('queue.cancel'))
  await s.say('9분 길이 음성을 small 모델로 올립니다', 'Upload a 9-minute recording with the small model')
  await submitJob(s, longSpeech('speech-cancel.wav'), smallModel)
  await s.waitJob('speech-cancel.wav', (job) => job.status === 'running' && job.stage === 'transcribe')
  // 표는 3초마다 갱신된다: 표에 음성 인식 단계가 보인 뒤에 찍는다
  await expect(s.row(table, 'speech-cancel.wav')).toContainText(s.t('stage.transcribe'))
  await d.shot('running-progress')
  await s.say('실행 중인 작업을 취소하면 확인 창이 뜹니다. 한 번은 닫고, 다시 눌러 취소합니다', 'Cancelling a running job asks first. Dismiss once, then confirm')
  await d.click(s.action(table, 'speech-cancel.wav', 'cancel'), 'cancel-prompt')
  await d.click(s.dialog().locator('[data-part="cancel"]'), null, { pace: 'input' })
  await d.click(s.action(table, 'speech-cancel.wav', 'cancel'), null, { pace: 'input' })
  await s.clickToast(s.dialog().locator('[data-part="confirm"]'), 'queue.cancelled', {}, 'cancelled')
  await s.say('취소한 작업은 다시 시도할 수 있습니다 (끝낸 구간은 체크포인트에서 이어서)', 'A cancelled job can be retried (finished chunks resume from the checkpoint)')
  await s.clickToast(s.action(table, 'speech-cancel.wav', 'retry'), 'queue.retriedOne', {}, 'retried')
  await s.waitJob('speech-cancel.wav', (job) => job.status === 'done')

  d.setScene(s.t('status.queued'))
  await s.say('두 작업을 잇달아 올리면 두 번째는 대기합니다. 대기 작업은 확인 없이 바로 취소됩니다', 'Two uploads in a row: the second waits. A queued job cancels immediately')
  await submitJob(s, longSpeech('speech-a.wav'), smallModel)
  await submitJob(s, fixture('speech.wav', 'speech-b.wav'))
  await d.shot('two-jobs')
  await s.clickToast(s.action(table, 'speech-b.wav', 'cancel'), 'queue.cancelled', {}, 'queued-cancelled')
  await s.say('삭제도 확인 창을 거칩니다. 입력과 자막 파일이 함께 지워집니다', 'Delete also asks first. Input and subtitle files are removed together')
  await d.click(s.action(table, 'speech-b.wav', 'delete'), 'delete-prompt')
  await d.click(s.dialog().locator('[data-part="cancel"]'), null)
  await d.click(s.action(table, 'speech-b.wav', 'delete'), null)
  await s.clickToast(s.dialog().locator('[data-part="confirm"]'), 'queue.deleted', {}, 'deleted')

  d.setScene(s.t('status.running'))
  await s.say('진행 막대는 단계와 처리한 구간 수를 보입니다. 행을 누르면 상세 창이 열립니다', 'The bar shows the stage and processed chunks. Click a row for details')
  await s.waitJob('speech-a.wav', (job) => job.status === 'running' && job.stage === 'transcribe')
  await d.click(s.row(table, 'speech-a.wav').locator('td').first(), 'running-detail')
  await s.say('새로 고침으로 지금 상태를 다시 읽습니다', 'Refresh reloads the current state')
  await d.click(s.dialog().locator('[data-part="confirm"]'), 'running-detail-refreshed', { pace: 'input' })
  await d.click(s.dialog().locator('[data-part="cancel"]'), null)
  await s.say('처리 중에는 대시보드의 현재 작업 카드에서도 진행을 봅니다', 'The dashboard current-job card also shows progress')
  await s.navigate('dashboard')
  await d.shot('dashboard-running')
  await s.navigate('queue')
  await s.waitJob('speech-a.wav', (job) => job.status === 'done')

  d.setScene(s.t('status.failed'))
  await s.say('미디어가 아닌 파일은 오디오 추출에서 실패합니다', 'A file that is not media fails at audio extraction')
  await submitJob(s, fixture('broken.mp4'))
  await s.waitJob('broken.mp4', (job) => job.status === 'failed')
  await page.waitForTimeout(PACE.transition)
  await d.shot('failed')
  await d.click(s.action(table, 'broken.mp4', 'detail'), 'failed-detail')
  await d.click(s.dialog().locator('[data-part="cancel"]'), null)
  await s.say('실패한 작업을 모두 다시 시도합니다. 같은 이유로 다시 실패합니다', 'Retry all failed jobs. It fails again for the same reason')
  await s.clickToast(s.byTest('retry-failed').locator('button'), 'queue.retried', { count: 1 }, 'retry-failed')
  await s.waitJob('broken.mp4', (job) => job.status === 'failed')
  await d.click(s.action(table, 'broken.mp4', 'delete'), null)
  await s.clickToast(s.dialog().locator('[data-part="confirm"]'), 'queue.deleted', {}, 'failed-deleted')

  d.setScene(s.t('queue.filter'))
  await s.say('상태 필터·검색·정렬로 작업을 찾습니다', 'Filter, search, and sort to find jobs')
  for (const key of ['queued', 'running', 'failed', 'cancelled', 'all', 'open'] as const) {
    await s.choose(bar.locator('[data-part="filter"]'), s.t(`queue.filter.${key}`), `filter-${key}`)
  }
  await s.choose(bar.locator('[data-part="filter"]'), s.t('queue.filter.all'))
  await d.type(bar.locator('[data-part="search"] input'), 'speech', { shot: 'search' })
  await d.type(bar.locator('[data-part="search"] input'), '')
  await d.click(bar.locator('[data-part="refresh"]'), null, { pace: 'input' })
  await s.sortAll(table)
  await s.choose(table.locator('.v-data-table-footer__items-per-page .v-select'), '25', 'per-page')
  await s.pageAll(table)
}

async function sceneDashboard(s: Scene): Promise<void> {
  const { d, page } = s
  d.setScene(s.t('nav.dashboard'))
  await s.say('대시보드는 워커 상태, 작업 수, 시스템 환경, 기본 처리 설정을 보입니다', 'The dashboard shows the worker, job counts, system, and default settings')
  await d.shot('dashboard')
  // info-list 는 이름이 길면 값 칸만 줄인다. 값이 잘리거나 0폭이 되면 캡처로는 놓치기 쉬워 잰다
  const values = await page.locator('.cards [data-part="value"]').all()
  expect(values.length, '대시보드 카드 값 개수').toBeGreaterThan(0)
  for (const value of values) {
    const size = await value.evaluate((el) => ({ text: el.textContent?.trim() ?? '', width: el.clientWidth, full: el.scrollWidth }))
    expect(size.width, `값 칸 폭: ${size.text}`).toBeGreaterThan(0)
    expect(size.full, `값 잘림: ${size.text}`).toBeLessThanOrEqual(size.width)
  }
  await s.say('시스템 카드: 디바이스, PyTorch, ffmpeg, HF_TOKEN, 여유 공간', 'System card: devices, PyTorch, ffmpeg, HF_TOKEN, free space')
  await d.hover(s.byTest('system-card'), 'system-card')
  await s.say('패키지 카드는 엔진·기능별 설치 여부를 보입니다', 'The packages card shows which engines and features are installed')
  await d.hover(s.byTest('packages-card'), 'packages-card')
  await d.click(s.byTest('refresh').locator('button'), 'dashboard-refreshed', { pace: 'input' })
  await s.say('카드를 누르면 해당 목록으로 이동합니다', 'Clicking a card opens the matching list')
  await d.click(s.byTest('stat-queued').locator('[data-part="root"]'), 'card-queued')
  await expect(page).toHaveURL(/#\/queue$/)
  await s.navigate('dashboard')
  await d.click(s.byTest('stat-failed').locator('[data-part="root"]'), 'card-failed')
  await s.navigate('dashboard')
  await d.click(s.byTest('stat-done').locator('[data-part="root"]'), 'card-done')
  await expect(page).toHaveURL(/#\/completed$/)
}

const REVIEW_TYPES = ['misrecognition', 'punctuation', 'other', 'suggestion', 'corrected'] as const
// 검수 장면에서 고쳐 넣는 자막 (STT 결과와 달라야 저장 버튼이 켜진다)
const REVIEWED_TEXT = 'Welcome to the subtitle demo (reviewed).'

/** E2E 서버는 LLM 없이 돌아 검수 리포트가 없다: 작업의 SRT 에 맞춘 리포트를 만들어 붙인다 */
function seedReview(jobId: number): void {
  execFileSync(
    path.join(REPO_DIR, '.venv', 'bin', 'python'),
    [path.join(REPO_DIR, 'web', 'e2e', 'server', 'seed-review.py'), path.join(SERVER_ROOT, 'data'), String(jobId)],
    { cwd: REPO_DIR, env: { ...process.env, PYTHONPATH: REPO_DIR }, stdio: 'pipe' },
  )
}

/** 자막 검수: 자막 전체와 메모를 보고 종류별로 걸러 보며, 자막을 고쳐 저장하거나 그대로 저장해 검수 완료로 표시한다 */
async function sceneReview(s: Scene, completed: Locator, jobName: string): Promise<void> {
  const { d, page } = s
  const job = await s.job(jobName)
  const srt = job.outputs.find((name) => name.endsWith('.srt'))
  if (!srt) throw new Error(`SRT 출력이 없습니다: ${jobName}`)
  seedReview(job.id)

  d.setScene(s.t('review.heading'))
  await s.say('자막 검수: SRT 자막 전체에 LLM 검수 메모를 붙여 보이고, 앞뒤 자막을 보며 고칩니다', 'Subtitle review: every SRT subtitle with the LLM review notes, edited in context')
  await d.click(s.action(completed, jobName, 'review'), null)
  const table = s.byTest('review-table')
  const bar = s.byTest('review-filter')
  await expect(table.locator('[data-part="row"]').first()).toBeVisible()
  await d.shot('review')

  await s.say('검수 종류로 걸러 봅니다', 'Filter by review type')
  for (const type of REVIEW_TYPES) {
    await s.choose(bar.locator('[data-part="filter"]'), s.t(`review.type.${type}`), `review-${type}`)
    await expect(table.locator('[data-part="row"]')).toHaveCount(1)
  }
  await s.choose(bar.locator('[data-part="filter"]'), s.t('review.type.all'))
  await d.type(bar.locator('[data-part="search"] input'), 'misrecognition', { shot: 'review-search' })
  await d.type(bar.locator('[data-part="search"] input'), '')
  await s.say('"검수 항목만" 을 켜면 메모가 있는 자막만 보입니다', 'Turn on "With notes only" to see only subtitles with notes')
  const all = await table.locator('[data-part="row"]').count()
  await d.click(bar.locator('[data-part="switch"] .v-selection-control__input'), 'review-flagged', { pace: 'input' })
  expect(await table.locator('[data-part="row"]').count()).toBeLessThanOrEqual(all)
  await d.click(bar.locator('[data-part="switch"] .v-selection-control__input'), null, { pace: 'input' })
  await d.click(bar.locator('[data-part="refresh"]'), null, { pace: 'input' })
  await s.sortAll(table)
  await s.choose(table.locator('.v-data-table-footer__items-per-page .v-select'), '25')
  await s.pageAll(table)

  await s.say('자막을 누르면 메모와 함께 수정 창이 열립니다', 'Click a subtitle to open the editor with its notes')
  await d.click(table.locator('[data-part="row"]').first().locator('td').nth(2), 'review-editor')
  await expect(s.byTest('review-note').first()).toBeVisible()
  await d.click(s.dialog().locator('[data-part="cancel"]'), null)
  await s.say('메모를 보고 자막을 고쳐 저장하면 검수 완료로 표시합니다. 처음 만든 자막은 따로 남습니다', 'Fix the subtitle and save to mark it reviewed. The original file is kept')
  await d.click(table.locator('[data-part="icon-action"][data-action="edit"]').first(), null)
  // 자동 높이 조절(auto-grow)이 높이를 재는 보조 textarea 를 하나 더 둔다: 앞의 것이 입력칸이다
  await d.type(s.byTest('review-text').locator('textarea').first(), REVIEWED_TEXT, { shot: 'review-typed' })
  await s.clickToast(s.dialog().locator('[data-part="confirm"]'), 'review.saved', {}, 'review-saved')
  await expect(s.byTest('review-edited')).toBeVisible()
  // 상태 열 (검수 종류 다음): 저장한 자막은 검수 완료
  const status = (row: number): Locator => table.locator('[data-part="row"]').nth(row).locator('td').nth(4)
  await expect(status(0)).toContainText(s.t('review.status.reviewed'))
  await expect(status(1)).toContainText(s.t('review.status.pending'))
  await s.say('고칠 것이 없으면 그대로 저장해 검수 완료로 표시합니다', 'If nothing needs fixing, save as is to mark it reviewed')
  await d.click(table.locator('[data-part="icon-action"][data-action="edit"]').nth(1), null)
  await s.clickToast(s.dialog().locator('[data-part="confirm"]'), 'review.saved', {}, 'review-confirmed')
  await expect(status(1)).toContainText(s.t('review.status.reviewed'))

  // 저장한 글자가 SRT 에 들어갔고 원본이 출력 목록에 남았는지 (화면이 아니라 서버 값으로)
  const saved = await page.request.get(`${BASE_URL}api/jobs/${job.id}/files/${encodeURIComponent(srt)}`)
  expect(await saved.text()).toContain(REVIEWED_TEXT)
  expect((await s.job(jobName)).outputs).toContain(srt.replace(/\.srt$/, '.orig.srt'))
  await d.click(s.byTest('review-back').locator('button'), 'review-back')
  await expect(completed.locator('[data-part="row"]').first()).toBeVisible()
}

async function sceneCompleted(s: Scene): Promise<void> {
  const { d, page } = s
  const table = s.byTest('completed-table')
  const bar = s.byTest('completed-filter')
  d.setScene(s.t('nav.completed'))
  await s.say('완료 목록은 언어·길이·자막 수·품질 등급·출력 포맷을 보입니다', 'Completed jobs show language, length, subtitle count, quality grade, and formats')
  await d.shot('completed')
  await d.type(bar.locator('[data-part="search"] input'), 'demo', { shot: 'completed-search' })
  await d.type(bar.locator('[data-part="search"] input'), '')
  await d.click(bar.locator('[data-part="refresh"]'), null, { pace: 'input' })
  await s.sortAll(table)
  await s.choose(table.locator('.v-data-table-footer__items-per-page .v-select'), '25')
  await s.pageAll(table)

  d.setScene(s.t('detail.outputs'))
  await s.say('상세 창: 상태·처리 옵션·결과 요약과 출력 파일', 'Details: status, options, result summary, and output files')
  await d.click(s.action(table, 'demo-talk.mp4', 'detail'), 'completed-detail')
  const job = await s.job('demo-talk.mp4')
  await s.say('출력 파일마다 미리 보기로 내용을 확인합니다', 'Preview each output file')
  for (const name of job.outputs) {
    await d.click(s.outputAction(name, 'preview'), `preview-${name.split('.').pop() ?? name}`, { pace: 'input' })
  }
  await s.say('파일 이름 머리글을 눌러 출력 파일을 정렬합니다', 'Click the file header to sort the outputs')
  await d.click(page.locator('[data-testid="detail-outputs"] th[tabindex]').first(), null, { pace: 'input' })
  await s.say('내려받기 아이콘으로 파일을 저장합니다', 'The download icon saves the file')
  for (const name of job.outputs) {
    const download = page.waitForEvent('download')
    await d.click(s.outputAction(name, 'download'), null, { pace: 'input' })
    expect((await download).suggestedFilename()).toBe(name)
  }
  await d.shot('downloaded')
  await d.click(s.dialog().locator('[data-part="confirm"]'), null, { pace: 'input' })
  await d.click(s.dialog().locator('[data-part="cancel"]'), null)
  await s.say('행을 눌러도 상세 창이 열립니다', 'Clicking a row also opens details')
  await d.click(s.row(table, 'speech-a.wav').locator('td').first(), null)
  await d.click(s.dialog().locator('[data-part="cancel"]'), null)

  await sceneReview(s, table, 'demo-talk.mp4')

  d.setScene(s.t('queue.delete'))
  await s.say('작업 하나를 지웁니다 (확인 창을 한 번 닫았다가 다시)', 'Delete one job (dismiss the prompt once, then confirm)')
  await d.click(s.action(table, 'speech-a.wav', 'delete'), 'completed-delete-prompt')
  await d.click(s.dialog().locator('[data-part="cancel"]'), null)
  await d.click(s.action(table, 'speech-a.wav', 'delete'), null)
  await s.clickToast(s.dialog().locator('[data-part="confirm"]'), 'queue.deleted', {}, 'completed-deleted')
  await s.say('끝난 작업 모두 지우기는 완료·실패·취소한 작업을 한 번에 지웁니다', 'Clear finished removes all done, failed, and cancelled jobs')
  await d.click(s.byTest('clear-finished').locator('button'), 'clear-prompt')
  await d.click(s.dialog().locator('[data-part="cancel"]'), null)
  await d.click(s.byTest('clear-finished').locator('button'), null)
  await s.clickToast(s.dialog().locator('[data-part="confirm"]'), 'completed.cleared', {}, 'cleared')
}

async function sceneLogs(s: Scene): Promise<void> {
  const { d } = s
  const bar = s.byTest('logs-filter')
  const table = s.byTest('logs-table')
  d.setScene(s.t('nav.logs'))
  await s.say('로그는 서버와 작업 프로세스의 기록을 시간 순서로 보입니다', 'Logs show records from the server and job processes')
  await d.shot('logs')
  for (const level of ['DEBUG', 'WARNING', 'ERROR', 'INFO'] as const) {
    await s.choose(bar.locator('[data-part="filter"]'), s.t(`logs.level.${level}`), `level-${level.toLowerCase()}`)
  }
  await d.type(bar.locator('[data-part="search"] input'), '작업', { shot: 'logs-search' })
  await d.type(bar.locator('[data-part="search"] input'), '')
  await s.say('자동 갱신을 끄거나 일시 정지 버튼으로 멈춥니다', 'Turn auto refresh off or pause it')
  await d.click(bar.locator('[data-part="switch"] .v-selection-control__input'), 'auto-off', { pace: 'input' })
  await d.click(bar.locator('[data-part="switch"] .v-selection-control__input'), null, { pace: 'input' })
  await d.click(s.byTest('logs-pause').locator('button'), 'paused', { pace: 'input' })
  await d.click(s.byTest('logs-pause').locator('button'), null, { pace: 'input' })
  await d.click(bar.locator('[data-part="refresh"]'), null, { pace: 'input' })
  await s.sortAll(table)
  await s.choose(table.locator('.v-data-table-footer__items-per-page .v-select'), '25')
  await s.pageAll(table)
  await s.say('항목을 누르면 전체 메시지를 보고 복사합니다', 'Click an entry to see and copy the full message')
  await d.click(table.locator('[data-part="row"]').first(), 'log-entry')
  await s.clickToast(s.dialog().locator('[data-part="confirm"]'), 'logs.copied', {}, 'copied')
  await d.click(s.dialog().locator('[data-part="cancel"]'), null)
}

// 설정 칸에 넣을 값. select 는 고를 항목의 글자, 나머지는 입력할 글자. 없으면 지금 값을 다시 넣는다
function selectLabel(s: Scene, field: SettingField): string {
  const labels: Record<string, string> = {
    'stt.model': 'faster-whisper',
    'stt.model_size': 'tiny',
    'stt.device': 'cpu',
    'stt.compute_type': 'int8',
    'stt.source_lang': s.t('settings.option.auto'),
    'audio.denoise.model': s.t('settings.option.denoise.noisereduce'),
    'segmentation.mode': s.t('mode.advanced'),
    'output.format': 'srt',
    'logging.level': 'INFO',
    'llm.provider': s.t('settings.llm.providerOption.local'),
    'llm.reasoning_effort': s.t('settings.llm.reasoningEffortOption.low'),
  }
  const label = labels[field.path]
  if (!label) throw new Error(`select 값 없음: ${field.path}`)
  return label
}

const TEXT_VALUES: Record<string, string> = {
  'segmentation.max_chars_per_line': '40',
  'web.max_upload_mb': '300',
  'web.session_hours': '48',
  'llm.model': 'qwen3',
  'llm.base_url': 'http://127.0.0.1:11434/v1',
  'llm.rpm': '20',
}

async function fillField(s: Scene, field: SettingField): Promise<void> {
  const host = s.byTest(`field-${field.path}`)
  // 다른 칸 값에 따라 숨는 칸 (엔드포인트는 로컬 프로바이더만)
  if ((await host.count()) === 0) return
  if (field.kind === 'toggle') {
    await s.toggle(host)
    await s.toggle(host)
  } else if (field.kind === 'select') {
    if (field.path === 'llm.provider') {
      // 구독(CLI 설치 여부)과 클라우드 API 키(.env 설정 여부)의 준비 상태 배너를 보인 뒤, 로컬 서버로 저장한다
      await s.choose(host.locator('[data-part="field"]'), s.t('settings.llm.providerOption.claude-code'), 'settings-llm-subscription')
      await expect(s.byTest('llm-readiness')).toBeVisible()
      await s.choose(host.locator('[data-part="field"]'), s.t('settings.llm.providerOption.gemini'), 'settings-llm-api-key')
      // E2E 서버는 빈 env 파일로 띄워 키가 없다
      await expect(s.byTest('llm-readiness')).toContainText('GEMINI_API_KEY')
    }
    await s.choose(host.locator('[data-part="field"]'), selectLabel(s, field))
  } else if (field.kind === 'tags') {
    // 태그마다 숨은 input 이 있다: 입력칸은 combobox 역할의 글자 input 이다
    const input = host.locator('[data-part="field"] input[role="combobox"]')
    const tag = field.path.endsWith('languages') ? 'zh' : '\\[Music\\]'
    await s.d.type(input, tag)
    await input.press('Enter')
    await s.page.waitForTimeout(PACE.input)
    // 더한 태그를 지워 원래 목록으로 돌린다 (태그 닫기 버튼)
    await s.d.click(host.locator('.v-chip').filter({ hasText: tag }).locator('[data-testid="close-chip"], .v-chip__close').first(), null, {
      pace: 'input',
    })
  } else {
    const input = host.locator('input').first()
    const value = TEXT_VALUES[field.path] ?? (await input.inputValue())
    await s.d.type(input, value, { delay: 25 })
  }
}

async function sceneSettings(s: Scene): Promise<void> {
  const { d, page } = s
  const tabs = s.byTest('settings-tabs')
  d.setScene(s.t('nav.settings'))
  await s.say('설정은 config.yaml 의 값을 고칩니다. 저장하면 다음 작업부터 적용됩니다', 'Settings edit config.yaml. Saved values apply from the next job')
  await d.shot('settings')
  for (const tab of FIELD_TABS) {
    d.setScene(`${s.t('nav.settings')} · ${s.t(`settings.tab.${tab}`)}`)
    await d.click(tabs.locator(`[data-part="tabs"] [data-tab="${tab}"]`), null, { pace: 'input' })
    await s.say(`${s.t(`settings.tab.${tab}`)} 탭의 모든 칸에 값을 넣습니다`, `Fill every field on the ${s.t(`settings.tab.${tab}`)} tab`, 900)
    for (const field of TAB_FIELDS[tab]) await fillField(s, field)
    await d.shot(`settings-${tab}`)
  }
  await s.clickToast(s.byTest('settings-save').locator('button'), 'settings.saved', {}, 'settings-saved')

  d.setScene(s.t('error.invalid_settings'))
  await s.say('잘못된 값은 저장하지 않고 칸과 탭에 오류를 표시합니다', 'Invalid values are not saved; the field and tab show the error')
  await d.click(tabs.locator('[data-part="tabs"] [data-tab="stt"]'), null, { pace: 'input' })
  await d.type(s.byTest('field-stt.batch_size').locator('input'), '0')
  await d.click(tabs.locator('[data-part="tabs"] [data-tab="segmentation"]'), null, { pace: 'input' })
  await d.type(s.byTest('field-segmentation.force_split_after_s').locator('input'), '9')
  await d.click(tabs.locator('[data-part="tabs"] [data-tab="llm"]'), null, { pace: 'input' })
  await d.type(s.byTest('field-llm.model').locator('input'), '')
  await s.clickToast(s.byTest('settings-save').locator('button'), 'error.invalid_settings', {}, 'settings-invalid')
  await expect(page.locator('smartview-hint-banner').filter({ hasText: s.t('validation.code.llm_model_required') })).toHaveCount(1)
  await s.say('강제 분할 시간은 최대 표시 시간보다 작아야 하고, API 방식 LLM 은 모델이 필요합니다 (섹션 오류)', 'Force split must be less than the maximum display time, and API LLM providers need a model (section errors)')
  await d.click(tabs.locator('[data-part="tabs"] [data-tab="llm"]'), null, { pace: 'input' })
  await d.type(s.byTest('field-llm.model').locator('input'), TEXT_VALUES['llm.model'], { shot: 'settings-llm-invalid' })
  // 저장에 실패하면 화면이 첫 오류 탭으로 옮겨 간다: 세그멘테이션 탭으로 돌아가 고친다
  await d.click(tabs.locator('[data-part="tabs"] [data-tab="segmentation"]'), null, { pace: 'input' })
  await d.type(s.byTest('field-segmentation.force_split_after_s').locator('input'), '5')
  await d.click(tabs.locator('[data-part="tabs"] [data-tab="stt"]'), null, { pace: 'input' })
  await d.type(s.byTest('field-stt.batch_size').locator('input'), '8')
  await s.clickToast(s.byTest('settings-save').locator('button'), 'settings.saved', {}, 'settings-fixed')
  await d.click(s.byTest('settings-reload').locator('button'), 'settings-reloaded', { pace: 'input' })

  d.setScene(s.t('settings.tab.account'))
  await d.click(tabs.locator('[data-part="tabs"] [data-tab="account"]'), 'account-tab', { pace: 'input' })
  const panel = s.byTest('account-panel')
  await d.type(panel.locator('[data-testid="account-username"] input'), ADMIN)
  await d.type(panel.locator('[data-testid="account-new"] input'), NEW_PASSWORD)
  await d.type(panel.locator('[data-testid="account-confirm"] input'), 'not-the-same', { shot: 'account-mismatch' })
  await d.type(panel.locator('[data-testid="account-confirm"] input'), NEW_PASSWORD)
  for (const id of ['account-current', 'account-new', 'account-confirm']) {
    const eye = panel.locator(`[data-testid="${id}"] [data-part="toggle"]`)
    if (await eye.count()) {
      await d.click(eye, null, { pace: 'input' })
      await d.click(eye, null, { pace: 'input' })
    }
  }
  if (!s.combo.changeAccount) {
    await s.say('지금 비밀번호가 틀리면 바꾸지 않습니다', 'A wrong current password leaves the account unchanged')
    await d.type(panel.locator('[data-testid="account-current"] input'), 'wrong-password-demo')
    await d.click(panel.locator('[data-testid="account-save"] button'), null, { pace: 'input' })
    await expect(panel.locator('[data-testid="account-current"]')).toContainText(s.t('account.error.wrong_password'))
    await d.shot('account-wrong-password')
    return
  }
  await s.say('지금 비밀번호를 넣고 저장하면 모든 로그인이 끝나 다시 로그인합니다', 'Saving with the current password signs everyone out')
  await d.type(panel.locator('[data-testid="account-current"] input'), adminPassword(), { delay: 10 })
  await d.click(panel.locator('[data-testid="account-save"] button'), null)
  const login = page.locator('smartview-login')
  await expect(login.locator('[data-part="form"]')).toBeVisible()
  await d.shot('account-changed-login')
  await s.say('새 비밀번호로 다시 로그인합니다', 'Sign in again with the new password')
  await d.type(login.locator('[data-part="username"] input'), ADMIN)
  await d.type(login.locator('[data-part="password"] input'), NEW_PASSWORD, { delay: 10 })
  await d.click(login.locator('[data-part="submit"]'), null)
  await page.locator('smartview-app-shell [data-part="main"]').waitFor()
  await d.shot('relogged-in')
}

for (const combo of COMBOS) {
  test.describe(`${combo.locale}-${combo.theme}`, () => {
    test.describe.configure({ mode: 'serial' })
    test('01 계정·셸', async () => {
      await record(combo, 'shell', 1, 'dashboard', sceneShell, false)
    })
    test('02 작업 처리', async () => {
      await record(combo, 'jobs', 2, 'queue', sceneJobs)
    })
    test('03 대시보드', async () => {
      await record(combo, 'dashboard', 3, 'dashboard', sceneDashboard)
    })
    test('04 완료 목록', async () => {
      await record(combo, 'completed', 4, 'completed', sceneCompleted)
    })
    test('05 로그', async () => {
      await record(combo, 'logs', 5, 'logs', sceneLogs)
    })
    test('06 설정', async () => {
      await record(combo, 'settings', 6, 'settings', sceneSettings)
    })
  })
}
