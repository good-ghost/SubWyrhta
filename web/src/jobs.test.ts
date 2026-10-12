import { afterEach, describe, expect, it } from 'vitest'

import { setAppLocale } from './i18n'
import { canCancel, progressPercent, progressText, statusMap } from './jobs'

describe('jobs', () => {
  afterEach(() => setAppLocale('ko'))

  it('진행 문구: 실행 중이면 단계와 개수', () => {
    expect(progressText({ status: 'running', stage: 'transcribe', progress: { done: 3, total: 10 } })).toBe('음성 인식 3/10')
    expect(progressText({ status: 'running', stage: 'extract', progress: { done: 0, total: 0 } })).toBe('오디오 추출')
    expect(progressText({ status: 'running', stage: null, progress: null })).toBe('처리 중')
    expect(progressText({ status: 'failed', stage: 'vad', progress: null })).toBe('실패')
  })

  it('진행률: 단계 구간 안에서 청크 비율로 채운다', () => {
    expect(progressPercent({ status: 'queued', stage: null, progress: null })).toBeNull()
    expect(progressPercent({ status: 'done', stage: null, progress: null })).toBe(100)
    expect(progressPercent({ status: 'running', stage: 'transcribe', progress: { done: 0, total: 10 } })).toBe(15)
    expect(progressPercent({ status: 'running', stage: 'transcribe', progress: { done: 5, total: 10 } })).toBe(50)
    expect(progressPercent({ status: 'running', stage: 'write', progress: { done: 0, total: 0 } })).toBe(98)
  })

  it('취소 가능 상태와 상태 표', () => {
    expect(canCancel({ status: 'queued' })).toBe(true)
    expect(canCancel({ status: 'running' })).toBe(true)
    expect(canCancel({ status: 'done' })).toBe(false)
    setAppLocale('en')
    expect(statusMap().cancelled.label).toBe('Cancelled')
  })
})
