import { describe, expect, it } from 'vitest'

import type { Choices } from '../api/types'
import { MESSAGES, type MessageKey } from '../i18n'
import { allFields, getPath, isFieldVisible, LLM_PROVIDERS, resolveOptions, setPath, TAB_FIELDS } from './fields'

const CHOICES: Choices = {
  engines: ['faster-whisper', 'whisperx'],
  model_sizes: ['small'],
  compute_types: ['auto', 'int8'],
  devices: ['auto', 'cpu'],
  languages: ['auto', 'en'],
  formats: ['srt', 'vtt'],
  modes: ['advanced', 'basic'],
  stages: ['extract'],
}

describe('settings fields', () => {
  it('모든 칸의 문구 키가 있다', () => {
    const keys = allFields().flatMap((field) => [field.label, field.hint, field.suffix].filter(Boolean) as MessageKey[])
    expect(keys.filter((key) => !(key in MESSAGES.ko))).toEqual([])
    const optionKeys = allFields().flatMap((field) => (field.options ?? []).filter((o) => !o.literal).map((o) => o.label))
    expect(optionKeys.filter((key) => !(key in MESSAGES.ko))).toEqual([])
  })

  it('엔드포인트 칸은 로컬 프로바이더(ollama, local)일 때만 보인다', () => {
    const baseUrl = TAB_FIELDS.llm.find((f) => f.path === 'llm.base_url')!
    const visible = LLM_PROVIDERS.filter((provider) => isFieldVisible(baseUrl, { llm: { provider } }))
    expect(visible).toEqual(['ollama', 'local'])
    expect(isFieldVisible(TAB_FIELDS.llm[0], {})).toBe(true)
  })

  it('경로가 겹치지 않는다', () => {
    const paths = allFields().map((field) => field.path)
    expect(new Set(paths).size).toBe(paths.length)
  })

  it('서버 선택지를 채운다 (모드는 번역 키, 언어 auto 는 자동)', () => {
    const mode = resolveOptions(TAB_FIELDS.segmentation[0], CHOICES)
    expect(mode.options?.map((o) => o.label)).toEqual(['mode.advanced', 'mode.basic'])
    expect(mode.options?.every((o) => `${o.label}` in MESSAGES.ko)).toBe(true)
    const language = resolveOptions(TAB_FIELDS.stt.find((f) => f.path === 'stt.source_lang')!, CHOICES)
    expect(language.options?.[0]).toEqual({ value: 'auto', label: 'settings.option.auto' })
    expect(language.options?.[1]).toEqual({ value: 'en', label: 'en', literal: true })
    const engine = resolveOptions(TAB_FIELDS.stt[0], CHOICES)
    expect(engine.options?.map((o) => o.value)).toEqual(CHOICES.engines)
  })

  it('점 경로로 읽고 쓴다', () => {
    const values = { audio: { vad: { threshold: 0.5 } } }
    expect(getPath(values, 'audio.vad.threshold')).toBe(0.5)
    setPath(values, 'audio.vad.threshold', 0.3)
    expect(values.audio.vad.threshold).toBe(0.3)
    expect(getPath(values, 'audio.missing.key')).toBeUndefined()
  })
})
