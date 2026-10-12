import { afterEach, describe, expect, it } from 'vitest'

import { ApiError } from './api/client'
import {
  compactTime,
  durationText,
  errorText,
  fieldErrorText,
  fileSize,
  isoFromEpoch,
  jobErrorText,
  localTime,
  numberOrText,
  timecode,
} from './format'
import { setAppLocale } from './i18n'

describe('format', () => {
  afterEach(() => setAppLocale('ko'))

  it('epoch 초를 ISO·로컬 시각으로', () => {
    expect(isoFromEpoch(0)).toBe('')
    expect(isoFromEpoch(1_700_000_000)).toBe('2023-11-14T22:13:20.000Z')
    expect(localTime(null)).toBe('-')
    expect(localTime(1_700_000_000)).toMatch(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/)
  })

  it('오늘 시각은 시간만, ISO 로그 시각도 받는다', () => {
    const now = new Date(2026, 9, 8, 15, 0, 0)
    const today = new Date(2026, 9, 8, 9, 5, 7).getTime() / 1000
    expect(compactTime(today, now)).toBe('09:05:07')
    expect(compactTime(today - 86_400, now)).toBe('2026-10-07 09:05:07')
    const iso = new Date(2026, 9, 8, 10, 0, 1).toISOString()
    expect(compactTime(iso, now)).toBe('10:00:01')
  })

  it('파일 크기와 길이', () => {
    expect(fileSize(512)).toBe('512 B')
    expect(fileSize(1536)).toBe('1.5 KB')
    expect(fileSize(5 * 1024 ** 3)).toBe('5.0 GB')
    expect(durationText(65)).toBe('1:05')
    expect(durationText(3_725)).toBe('1:02:05')
    expect(durationText(null)).toBe('-')
  })

  it('자막 시각은 자릿수를 맞춘 HH:MM:SS.mmm', () => {
    expect(timecode(0.885)).toBe('00:00:00.885')
    expect(timecode(3_723.004)).toBe('01:02:03.004')
    expect(timecode(59.9996)).toBe('00:01:00.000')
    expect(timecode(-1)).toBe('00:00:00.000')
  })

  it('오류 코드를 문구로', () => {
    expect(errorText(new ApiError(409, 'job_running'))).toBe('실행 중인 작업은 지울 수 없습니다. 먼저 취소하세요')
    expect(errorText(new ApiError(500, 'http_500'))).toBe('알 수 없는 오류 (http_500)')
    expect(errorText(new TypeError('Failed to fetch'))).toBe('서버에 연결할 수 없습니다')
    setAppLocale('en')
    expect(errorText(new ApiError(413, 'file_too_large'))).toBe('The file exceeds the upload limit')
  })

  it('작업 오류: 서버 코드는 번역, 파이프라인 문구는 그대로', () => {
    expect(jobErrorText('interrupted')).toContain('다시 시도')
    expect(jobErrorText('ffmpeg를 찾을 수 없습니다')).toBe('ffmpeg를 찾을 수 없습니다')
    expect(jobErrorText(null)).toBe('')
  })

  it('숫자 칸과 검증 오류', () => {
    expect(numberOrText(' 12.5 ')).toBe(12.5)
    expect(numberOrText('abc')).toBe('abc')
    expect(numberOrText('')).toBe('')
    expect(fieldErrorText({ loc: 'stt.batch_size', message: 'x', type: 'greater_than_equal', ctx: { ge: '1' } })).toBe('1 이상이어야 합니다')
    expect(fieldErrorText({ loc: 'stt.model', message: 'Value error, invalid_choice: a', type: 'value_error' })).toBe('허용 값: a')
  })
})

describe('검증 코드', () => {
  afterEach(() => setAppLocale('ko'))

  it('서버 검증 코드를 화면 언어로 바꾼다', () => {
    const error = { loc: 'segmentation', message: 'Value error, force_split_lt_max_duration', type: 'value_error' }
    expect(fieldErrorText(error)).toBe('강제 분할 시작 시간은 최대 표시 시간보다 작아야 합니다')
    setAppLocale('en')
    expect(fieldErrorText(error)).toBe('Force split must be less than the maximum display time')
    expect(fieldErrorText({ loc: 'noise_filter.custom_patterns', message: 'Value error, invalid_regex: (', type: 'value_error' })).toBe(
      'Invalid regular expression: (',
    )
    expect(fieldErrorText({ loc: 'x', message: 'Value error, unknown thing', type: 'value_error' })).toBe('unknown thing')
  })
})
