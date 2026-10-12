// 서버 API 응답 모양 (src/web/api.py·server.py 의 pydantic 모델과 같다)

export interface SessionState {
  authenticated: boolean
  username: string | null
  // 관리 계정이 아직 없다: 처음 설정 화면
  setup_required: boolean
}

export type JobStatus = 'queued' | 'running' | 'done' | 'failed' | 'cancelled'

// 파이프라인 단계 (src/pipeline.py STAGES)
export type Stage =
  | 'extract'
  | 'enhance'
  | 'vad'
  | 'load_model'
  | 'transcribe'
  | 'diarize'
  | 'segment'
  | 'quality'
  | 'write'

export interface Progress {
  done: number
  total: number
}

export interface JobOptions {
  engine: string
  model_size: string
  source_lang: string
  compute_type: string
  formats: string[]
  segmentation_mode: string
  max_cps: number | null
  audio_track: number | null
  pre_vad: boolean
  normalize: boolean
  denoise: boolean
  separate_vocals: boolean
  diarize: boolean
  num_speakers: string
  restore_punctuation: boolean
  llm_correction: boolean
  llm_punctuation: boolean
  llm_review: boolean
  quality_report: boolean
  export_words: boolean
  // 검수 리포트 메모를 쓸 언어: 작업을 등록하는 화면의 언어 (null 이면 설정 파일 값)
  review_language: 'ko' | 'en' | null
}

export interface QualityResult {
  grade: string
  avg_confidence: number
  low_confidence: number
  errors: number
  warnings: number
  auto_fixed: number
}

export interface JobResult {
  language: string
  duration_s: number
  model: string
  segments: number
  num_speakers: number | null
  quality: QualityResult | null
}

export interface Job {
  id: number
  name: string
  size: number
  status: JobStatus
  stage: Stage | null
  progress: Progress | null
  // 오류 문구 또는 코드 (interrupted, process_crashed, invalid_options)
  error: string | null
  outputs: string[]
  result: JobResult | null
  options: JobOptions
  attempts: number
  input_deleted: boolean
  retryable: boolean
  created_at: number
  started_at: number | null
  finished_at: number | null
  updated_at: number
}

export interface CurrentJob {
  id: number
  name: string
  stage: Stage | null
  progress: Progress | null
  started_at: number
}

export interface SystemDevice {
  available_devices: string[]
  torch_version: string | null
  cuda_device_name?: string
  cuda_vram_gb?: number
}

export interface SystemData {
  device?: SystemDevice
  packages?: Record<string, boolean>
  ffmpeg?: boolean
}

export interface ServerStatus {
  version: string
  started_at: number
  worker_alive: boolean
  current: CurrentJob | null
  queue: Record<JobStatus, number>
  system_state: 'checking' | 'ready' | 'failed'
  system: SystemData
  hf_token: boolean
  disk_free: number
  disk_total: number
  defaults: {
    engine: string
    model_size: string
    device: string
    compute_type: string
    source_lang: string
    format: string
    mode: string
  }
}

export interface Choices {
  engines: string[]
  model_sizes: string[]
  compute_types: string[]
  devices: string[]
  languages: string[]
  formats: string[]
  modes: string[]
  stages: Stage[]
}

export interface JobFormOptions {
  defaults: JobOptions
  choices: Choices
  max_upload_mb: number
  extensions: string[]
}

export interface Upload {
  id: string
  name: string
  size: number
}

export interface Preview {
  name: string
  text: string
  truncated: boolean
}

export type LogLevel = 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'

export interface LogEntry {
  time: string
  level: string
  logger: string
  job_id: number | null
  message: string
}

export interface LogChunk {
  entries: LogEntry[]
  cursor: string
  reset: boolean
}

export interface SettingsValues {
  stt: {
    model: string
    model_size: string
    model_dir: string
    device: string
    compute_type: string
    batch_size: number
    beam_size: number
    temperature: number
    source_lang: string
  }
  audio: {
    normalize: boolean
    separate_vocals: boolean
    denoise: { enabled: boolean; model: string }
    vad: {
      enabled: boolean
      threshold: number
      min_speech_ms: number
      min_silence_ms: number
      pre_vad: boolean
      merge_gap_s: number
      min_chunk_s: number
      max_chunk_s: number
    }
  }
  segmentation: Record<string, string | number | boolean | null>
  llm: { provider: string; model: string; base_url: string; timeout_s: number; max_retries: number; rpm: number; disable_thinking: boolean; reasoning_effort: string }
  diarization: { enabled: boolean; num_speakers: string }
  noise_filter: {
    enabled: boolean
    languages: string[]
    confidence_threshold: number
    remove_low_confidence: boolean
    custom_patterns: string[]
  }
  postprocess: { restore_punctuation: boolean; punctuation_model: boolean; llm_correction: boolean; llm_punctuation: boolean; llm_review: boolean }
  scene: { enabled: boolean; gap_s: number; min_segments: number; boundary_gap_s: number; use_video: boolean }
  quality: { enabled: boolean; auto_fix: boolean; export_report: boolean }
  output: { format: string; export_words: boolean }
  logging: { level: string }
  web: { session_hours: number; max_upload_mb: number; delete_input_after_done: boolean }
}

export interface SettingsState {
  path: string
  writable: boolean
  values: SettingsValues
  // HF_TOKEN 이 설정돼 있는지 (값은 내려오지 않는다)
  hf_token: boolean
  // LLM 프로바이더 → 준비 상태 (구독 CLI 설치, API 키 설정 여부. 키 값은 내려오지 않는다)
  llm: Record<string, LlmReadiness>
  choices: Choices
}

export interface LlmReadiness {
  kind: 'subscription' | 'api_key' | 'local'
  // CLI 실행 파일 또는 키 환경 변수 이름 (로컬은 빈 글자)
  requirement: string
  ready: boolean
}

export interface FieldError {
  loc: string
  message: string
  // pydantic 오류 종류와 기준값 (화면이 문구를 번역한다)
  type?: string
  ctx?: Record<string, string>
}

// 검수 종류: 검수 리포트 issues 의 종류 + 반영하지 않은 교정 제안·반영한 교정 (src/web/review.py)
export type ReviewType = 'misrecognition' | 'punctuation' | 'other' | 'suggestion' | 'corrected'

export interface ReviewNote {
  type: ReviewType
  note: string
}

// 검수 화면의 자막 한 줄 (start·end 는 초)
export interface ReviewRow {
  index: number
  start: number
  end: number
  text: string
  notes: ReviewNote[]
  // 사용자가 저장해 검수를 마친 자막
  reviewed: boolean
}

export interface Review {
  // 지금 보는 SRT 출력 이름
  name: string
  // 고를 수 있는 SRT 출력 (남겨 둔 원본 *.orig.srt 는 빠진다)
  subtitles: string[]
  // 검수 리포트 출력 이름. 없으면 null (LLM 마지막 검토를 켜지 않은 작업)
  report: string | null
  // LLM 호출이 실패해 검토를 건너뛴 원인 (빈 값이면 정상)
  llm_error: string
  // 고친 적이 있다 (원본이 *.orig.srt 로 남아 있다)
  edited: boolean
  rows: ReviewRow[]
}
