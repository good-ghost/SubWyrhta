// 설정 화면의 필드 정의. path 는 values 안의 점 경로이고 서버 검증 오류 loc 과 같다 (src/web/settings_io.py).
import type { Choices } from '../api/types'
import type { MessageKey } from '../i18n'

export type FieldKind = 'text' | 'number' | 'toggle' | 'select' | 'tags'

export interface FieldOption {
  value: string
  // literal 이 아니면 번역 키 (MessageKey)
  label: string
  // label 이 번역 키가 아니라 그대로 보일 글자일 때 (엔진 이름·언어 코드 등)
  literal?: boolean
}

// 화면이 서버 응답(choices)으로 채우는 선택지
export type DynamicOptions = keyof Pick<Choices, 'engines' | 'model_sizes' | 'devices' | 'compute_types' | 'languages' | 'formats' | 'modes'>

export interface SettingField {
  path: string
  kind: FieldKind
  label: MessageKey
  hint?: MessageKey
  suffix?: MessageKey
  options?: FieldOption[]
  dynamicOptions?: DynamicOptions
  // 글자를 쳐서 선택지를 거른다 (긴 목록)
  searchable?: boolean
  // 숫자 칸을 비우면 null (없음)로 보낸다
  nullable?: boolean
  // 다른 칸(path)이 values 중 하나일 때만 보인다 (예: 엔드포인트는 로컬 프로바이더만)
  visibleWhen?: { path: string; values: readonly string[] }
}

export const TABS = ['stt', 'audio', 'segmentation', 'speaker', 'quality', 'llm', 'web', 'account'] as const
export type SettingsTab = (typeof TABS)[number]
// 계정 탭은 config.yaml 칸이 아니라 자기 저장 버튼이 있는 패널이다
export type FieldTab = Exclude<SettingsTab, 'account'>
export const FIELD_TABS: FieldTab[] = ['stt', 'audio', 'segmentation', 'speaker', 'quality', 'llm', 'web']

// 서버 src/segmentation/llm_client.py LLM_PROVIDERS 와 같은 순서 (구독, 클라우드 API 키, 로컬)
export const LLM_PROVIDERS = ['codex', 'claude-code', 'gemini', 'openrouter', 'nvidia-nim', 'ollama', 'local'] as const
// 엔드포인트를 설정에서 정하는 프로바이더 (클라우드는 고정 주소)
export const LLM_ENDPOINT_PROVIDERS = ['ollama', 'local'] as const
// OpenAI 호환 API 로 부르는 프로바이더 (구독 CLI 제외). 추론 끄기는 이 요청에만 넣는다
export const LLM_OPENAI_COMPATIBLE_PROVIDERS = ['gemini', 'openrouter', 'nvidia-nim', 'ollama', 'local'] as const

const literal = (values: string[]): FieldOption[] => values.map((value) => ({ value, label: value, literal: true }))

export const TAB_FIELDS: Record<FieldTab, SettingField[]> = {
  stt: [
    { path: 'stt.model', kind: 'select', label: 'settings.stt.model', dynamicOptions: 'engines' },
    { path: 'stt.model_size', kind: 'select', label: 'settings.stt.model_size', dynamicOptions: 'model_sizes' },
    { path: 'stt.device', kind: 'select', label: 'settings.stt.device', dynamicOptions: 'devices' },
    {
      path: 'stt.compute_type', kind: 'select', label: 'settings.stt.compute_type', hint: 'settings.stt.compute_typeHint',
      dynamicOptions: 'compute_types',
    },
    { path: 'stt.source_lang', kind: 'select', label: 'settings.stt.source_lang', dynamicOptions: 'languages', searchable: true },
    { path: 'stt.batch_size', kind: 'number', label: 'settings.stt.batch_size' },
    { path: 'stt.beam_size', kind: 'number', label: 'settings.stt.beam_size' },
    { path: 'stt.temperature', kind: 'number', label: 'settings.stt.temperature' },
    { path: 'stt.model_dir', kind: 'text', label: 'settings.stt.model_dir', hint: 'settings.stt.model_dirHint' },
  ],
  audio: [
    { path: 'audio.normalize', kind: 'toggle', label: 'settings.audio.normalize' },
    { path: 'audio.separate_vocals', kind: 'toggle', label: 'settings.audio.separate_vocals' },
    { path: 'audio.denoise.enabled', kind: 'toggle', label: 'settings.audio.denoise.enabled' },
    {
      path: 'audio.denoise.model', kind: 'select', label: 'settings.audio.denoise.model',
      options: [{ value: 'noisereduce', label: 'settings.option.denoise.noisereduce' }],
    },
    { path: 'audio.vad.enabled', kind: 'toggle', label: 'settings.audio.vad.enabled' },
    { path: 'audio.vad.pre_vad', kind: 'toggle', label: 'settings.audio.vad.pre_vad' },
    { path: 'audio.vad.threshold', kind: 'number', label: 'settings.audio.vad.threshold' },
    { path: 'audio.vad.min_speech_ms', kind: 'number', label: 'settings.audio.vad.min_speech_ms', suffix: 'settings.unit.ms' },
    { path: 'audio.vad.min_silence_ms', kind: 'number', label: 'settings.audio.vad.min_silence_ms', suffix: 'settings.unit.ms' },
    { path: 'audio.vad.merge_gap_s', kind: 'number', label: 'settings.audio.vad.merge_gap_s', suffix: 'settings.unit.seconds' },
    { path: 'audio.vad.min_chunk_s', kind: 'number', label: 'settings.audio.vad.min_chunk_s', suffix: 'settings.unit.seconds' },
    { path: 'audio.vad.max_chunk_s', kind: 'number', label: 'settings.audio.vad.max_chunk_s', suffix: 'settings.unit.seconds' },
  ],
  segmentation: [
    { path: 'segmentation.mode', kind: 'select', label: 'settings.segmentation.mode', dynamicOptions: 'modes' },
    {
      path: 'segmentation.max_chars_per_line', kind: 'number', label: 'settings.segmentation.max_chars_per_line',
      hint: 'settings.segmentation.max_chars_per_lineHint', nullable: true,
    },
    { path: 'segmentation.max_lines', kind: 'number', label: 'settings.segmentation.max_lines' },
    {
      path: 'segmentation.max_cps', kind: 'number', label: 'settings.segmentation.max_cps',
      hint: 'settings.segmentation.max_cpsHint', nullable: true,
    },
    { path: 'segmentation.min_duration_s', kind: 'number', label: 'settings.segmentation.min_duration_s', suffix: 'settings.unit.seconds' },
    { path: 'segmentation.max_duration_s', kind: 'number', label: 'settings.segmentation.max_duration_s', suffix: 'settings.unit.seconds' },
    {
      path: 'segmentation.force_split_after_s', kind: 'number', label: 'settings.segmentation.force_split_after_s',
      hint: 'settings.segmentation.force_split_after_sHint', suffix: 'settings.unit.seconds',
    },
    { path: 'segmentation.min_gap_s', kind: 'number', label: 'settings.segmentation.min_gap_s', suffix: 'settings.unit.seconds' },
    { path: 'segmentation.dynamic_threshold', kind: 'toggle', label: 'settings.segmentation.dynamic_threshold' },
    {
      path: 'segmentation.dynamic_threshold_start_s', kind: 'number', label: 'settings.segmentation.dynamic_threshold_start_s',
      suffix: 'settings.unit.seconds',
    },
    { path: 'segmentation.dynamic_threshold_min_ratio', kind: 'number', label: 'settings.segmentation.dynamic_threshold_min_ratio' },
    { path: 'segmentation.min_chars', kind: 'number', label: 'settings.segmentation.min_chars' },
  ],
  speaker: [
    { path: 'diarization.enabled', kind: 'toggle', label: 'settings.diarization.enabled' },
    { path: 'diarization.num_speakers', kind: 'text', label: 'settings.diarization.num_speakers' },
    { path: 'segmentation.speaker_split', kind: 'toggle', label: 'settings.segmentation.speaker_split' },
    { path: 'segmentation.overlap_text_separator', kind: 'text', label: 'settings.segmentation.overlap_text_separator' },
    { path: 'noise_filter.enabled', kind: 'toggle', label: 'settings.noise_filter.enabled' },
    { path: 'noise_filter.confidence_threshold', kind: 'number', label: 'settings.noise_filter.confidence_threshold' },
    {
      path: 'noise_filter.remove_low_confidence', kind: 'toggle', label: 'settings.noise_filter.remove_low_confidence',
      hint: 'settings.noise_filter.remove_low_confidenceHint',
    },
    {
      path: 'noise_filter.remove_interjections', kind: 'toggle', label: 'settings.noise_filter.remove_interjections',
      hint: 'settings.noise_filter.remove_interjectionsHint',
    },
    { path: 'noise_filter.languages', kind: 'tags', label: 'settings.noise_filter.languages' },
    { path: 'noise_filter.custom_patterns', kind: 'tags', label: 'settings.noise_filter.custom_patterns' },
    {
      path: 'postprocess.restore_punctuation', kind: 'toggle', label: 'settings.postprocess.restore_punctuation',
      hint: 'settings.postprocess.restore_punctuationHint',
    },
    {
      path: 'postprocess.punctuation_model', kind: 'toggle', label: 'settings.postprocess.punctuation_model',
      hint: 'settings.postprocess.punctuation_modelHint',
    },
    {
      path: 'postprocess.llm_correction', kind: 'toggle', label: 'settings.postprocess.llm_correction',
      hint: 'settings.postprocess.llm_correctionHint',
    },
    {
      path: 'postprocess.llm_punctuation', kind: 'toggle', label: 'settings.postprocess.llm_punctuation',
      hint: 'settings.postprocess.llm_punctuationHint',
    },
    {
      path: 'postprocess.llm_review', kind: 'toggle', label: 'settings.postprocess.llm_review',
      hint: 'settings.postprocess.llm_reviewHint',
    },
  ],
  quality: [
    { path: 'quality.enabled', kind: 'toggle', label: 'settings.quality.enabled' },
    { path: 'quality.auto_fix', kind: 'toggle', label: 'settings.quality.auto_fix' },
    { path: 'quality.export_report', kind: 'toggle', label: 'settings.quality.export_report' },
    { path: 'scene.enabled', kind: 'toggle', label: 'settings.scene.enabled', hint: 'settings.scene.enabledHint' },
    { path: 'scene.gap_s', kind: 'number', label: 'settings.scene.gap_s', suffix: 'settings.unit.seconds' },
    { path: 'scene.min_segments', kind: 'number', label: 'settings.scene.min_segments' },
    { path: 'scene.boundary_gap_s', kind: 'number', label: 'settings.scene.boundary_gap_s', suffix: 'settings.unit.seconds' },
    { path: 'scene.use_video', kind: 'toggle', label: 'settings.scene.use_video', hint: 'settings.scene.use_videoHint' },
    { path: 'output.format', kind: 'select', label: 'settings.output.format', dynamicOptions: 'formats' },
    { path: 'output.export_words', kind: 'toggle', label: 'settings.output.export_words' },
    {
      path: 'logging.level', kind: 'select', label: 'settings.logging.level',
      options: literal(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
    },
  ],
  llm: [
    {
      path: 'llm.provider', kind: 'select', label: 'settings.llm.provider', hint: 'settings.llm.hint',
      options: LLM_PROVIDERS.map((value) => ({ value, label: `settings.llm.providerOption.${value}` })),
    },
    { path: 'llm.model', kind: 'text', label: 'settings.llm.model', hint: 'settings.llm.modelHint' },
    {
      path: 'llm.base_url', kind: 'text', label: 'settings.llm.base_url', hint: 'settings.llm.base_urlHint',
      visibleWhen: { path: 'llm.provider', values: LLM_ENDPOINT_PROVIDERS },
    },
    { path: 'llm.timeout_s', kind: 'number', label: 'settings.llm.timeout_s', suffix: 'settings.unit.seconds', hint: 'settings.llm.timeout_sHint' },
    { path: 'llm.max_retries', kind: 'number', label: 'settings.llm.max_retries' },
    { path: 'llm.rpm', kind: 'number', label: 'settings.llm.rpm', hint: 'settings.llm.rpmHint' },
    {
      path: 'llm.disable_thinking', kind: 'toggle', label: 'settings.llm.disable_thinking',
      hint: 'settings.llm.disable_thinkingHint',
      visibleWhen: { path: 'llm.provider', values: LLM_OPENAI_COMPATIBLE_PROVIDERS },
    },
    {
      path: 'llm.reasoning_effort', kind: 'select', label: 'settings.llm.reasoning_effort',
      hint: 'settings.llm.reasoning_effortHint',
      options: ['', 'low', 'medium', 'high'].map((value) => ({
        value,
        label: `settings.llm.reasoningEffortOption.${value || 'none'}`,
      })),
      visibleWhen: { path: 'llm.provider', values: LLM_OPENAI_COMPATIBLE_PROVIDERS },
    },
  ],
  web: [
    {
      path: 'web.session_hours', kind: 'number', label: 'settings.web.session_hours', hint: 'settings.web.session_hoursHint',
      suffix: 'settings.unit.hours',
    },
    { path: 'web.max_upload_mb', kind: 'number', label: 'settings.web.max_upload_mb', suffix: 'settings.unit.mb' },
    {
      path: 'web.delete_input_after_done', kind: 'toggle', label: 'settings.web.delete_input_after_done',
      hint: 'settings.web.delete_input_after_doneHint',
    },
  ],
}

export function allFields(): SettingField[] {
  return Object.values(TAB_FIELDS).flat()
}

/** 서버 선택지로 dynamicOptions 를 채운다. 모드는 번역 키(mode.*)를, 언어 auto 는 '자동' 키를 쓴다. */
export function resolveOptions(field: SettingField, choices: Choices | undefined): SettingField {
  if (!field.dynamicOptions || !choices) return field
  const values = choices[field.dynamicOptions]
  if (field.dynamicOptions === 'modes') {
    return { ...field, options: values.map((value) => ({ value, label: `mode.${value}` })) }
  }
  if (field.dynamicOptions === 'languages') {
    return {
      ...field,
      options: values.map((value) =>
        value === 'auto' ? { value, label: 'settings.option.auto' } : { value, label: value, literal: true },
      ),
    }
  }
  return { ...field, options: literal(values) }
}

/** visibleWhen 조건을 지금 값으로 판정한다 (조건이 없으면 늘 보인다) */
export function isFieldVisible(field: SettingField, values: unknown): boolean {
  if (!field.visibleWhen) return true
  return field.visibleWhen.values.includes(String(getPath(values, field.visibleWhen.path)))
}

export function getPath(values: unknown, path: string): unknown {
  return path.split('.').reduce<unknown>((node, key) => (node as Record<string, unknown> | null)?.[key], values)
}

export function setPath(values: object, path: string, value: unknown): void {
  const keys = path.split('.')
  const last = keys.pop() as string
  const parent = keys.reduce<Record<string, unknown>>(
    (node, key) => node[key] as Record<string, unknown>,
    values as Record<string, unknown>,
  )
  parent[last] = value
}
