// 표시용 변환 (시각·크기·길이·오류 코드)
import { ApiError } from './api/client'
import type { FieldError } from './api/types'
import { t, type MessageKey, MESSAGES } from './i18n'

const pad = (n: number): string => String(n).padStart(2, '0')

/** 서버의 epoch 초 → ISO 문자열 (vue-smartview date 열이 로컬 시각으로 보인다). 0·없음은 ''. */
export function isoFromEpoch(seconds: number | null | undefined): string {
  return seconds ? new Date(seconds * 1000).toISOString() : ''
}

/** epoch 초 → 보는 사람의 로컬 시각 `YYYY-MM-DD HH:mm:ss`. 0·없음은 '-'. */
export function localTime(seconds: number | null | undefined): string {
  if (!seconds) return t('common.none')
  const date = new Date(seconds * 1000)
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
  )
}

/** 표 칸용 짧은 시각: 오늘이면 `HH:mm:ss`, 아니면 날짜까지 (ISO·`YYYY-MM-DD HH:mm:ss` 문자열도 받는다). */
export function compactTime(value: number | string | null | undefined, now: Date = new Date()): string {
  let text: string
  if (typeof value === 'string' && value.includes('T')) {
    const date = new Date(value)
    text = Number.isNaN(date.getTime()) ? value : localTime(date.getTime() / 1000)
  } else {
    text = typeof value === 'string' ? value : localTime(value)
  }
  const today = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} `
  return text.startsWith(today) ? text.slice(today.length) : text
}

const SIZE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'] as const
const KIB = 1024

/** 바이트 → `12.3 MB`. */
export function fileSize(bytes: number): string {
  let value = bytes
  let unit = 0
  while (value >= KIB && unit < SIZE_UNITS.length - 1) {
    value /= KIB
    unit += 1
  }
  return `${unit === 0 ? value : value.toFixed(1)} ${SIZE_UNITS[unit]}`
}

/** 초 → `H:MM:SS` (1시간 미만이면 `M:SS`). 없음은 '-'. */
export function durationText(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return t('common.none')
  const total = Math.max(0, Math.round(seconds))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`
}

const MS_PER_SECOND = 1000

/** 초 → 자막 시각 `HH:MM:SS.mmm` (자릿수를 맞춰 글자 순서가 시간 순서와 같다). */
export function timecode(seconds: number): string {
  const total = Math.max(0, Math.round(seconds * MS_PER_SECOND))
  const ms = total % MS_PER_SECOND
  const s = Math.floor(total / MS_PER_SECOND)
  return `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}.${String(ms).padStart(3, '0')}`
}

/** API 오류를 화면 문구로. 서버 detail 코드에 문구가 있으면 그것, 없으면 일반 문구. */
export function errorText(error: unknown): string {
  if (error instanceof ApiError) {
    const key = `error.${error.code}`
    if (key in MESSAGES.ko) return t(key as MessageKey)
    return t('common.unknownError', { code: error.code })
  }
  if (error instanceof TypeError) return t('common.unreachable') // fetch·XHR 네트워크 오류
  return t('common.error', { message: error instanceof Error ? error.message : String(error) })
}

/** 작업 오류: 서버가 정한 코드(interrupted 등)는 문구로, 파이프라인 오류 문구는 그대로. */
export function jobErrorText(error: string | null): string {
  if (!error) return ''
  const key = `jobError.${error}`
  return key in MESSAGES.ko ? t(key as MessageKey) : error
}

/** 숫자 칸 값: 숫자로 읽히면 숫자, 아니면 그대로 (서버가 검증해 오류를 돌려준다). */
export function numberOrText(value: string | number): string | number {
  if (typeof value === 'number') return value
  const trimmed = value.trim()
  return trimmed !== '' && Number.isFinite(Number(trimmed)) ? Number(trimmed) : value
}

// pydantic 이 검증기 ValueError 를 감싸는 머리말 (서버는 그 뒤에 코드를 둔다: src/web/settings_io.py)
const VALUE_ERROR_PREFIX = 'Value error, '

/** 설정 검증 오류 문구. 아는 pydantic 오류 종류·서버 검증 코드는 번역하고, 모르면 서버 원문. */
export function fieldErrorText(error: FieldError): string {
  if (error.type === 'value_error' && error.message.startsWith(VALUE_ERROR_PREFIX)) {
    const body = error.message.slice(VALUE_ERROR_PREFIX.length)
    const [code, ...rest] = body.split(': ')
    const codeKey = `validation.code.${code}`
    if (codeKey in MESSAGES.ko) return t(codeKey as MessageKey, { detail: rest.join(': ') })
    return body
  }
  const key = `validation.${error.type ?? ''}`
  if (key in MESSAGES.ko) return t(key as MessageKey, error.ctx ?? {})
  return error.message
}
