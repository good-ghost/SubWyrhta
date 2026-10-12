// 작업 표시 규칙 (상태 색·단계 문구·진행률)
import type { Job, JobStatus, Stage } from './api/types'
import { t, type MessageKey } from './i18n'

export const STATUS_LOOKS: Record<JobStatus, { color: string; icon: string }> = {
  queued: { color: 'info', icon: 'mdi-clock-outline' },
  running: { color: 'primary', icon: 'mdi-loading mdi-spin' },
  done: { color: 'success', icon: 'mdi-check-circle' },
  failed: { color: 'error', icon: 'mdi-alert-circle' },
  cancelled: { color: 'grey', icon: 'mdi-cancel' },
}

/** 표의 status 열 statusMap (화면 언어로) */
export function statusMap(): Record<string, { color: string; icon: string; label: string }> {
  return Object.fromEntries(
    (Object.keys(STATUS_LOOKS) as JobStatus[]).map((s) => [s, { ...STATUS_LOOKS[s], label: t(`status.${s}`) }]),
  )
}

export function stageLabel(stage: Stage | null): string {
  return stage ? t(`stage.${stage}` as MessageKey) : t('common.none')
}

/**
 * 진행 문구: 실행 중이면 `음성 인식 12/40`처럼 단계와 개수, 개수를 모르면 단계 이름만.
 * 실행 중이 아니면 상태 이름.
 */
export function progressText(job: Pick<Job, 'status' | 'stage' | 'progress'>): string {
  if (job.status !== 'running') return t(`status.${job.status}`)
  if (!job.stage) return t('status.running')
  const progress = job.progress
  if (progress && progress.total > 0) {
    return t('stage.progress', { stage: stageLabel(job.stage), done: progress.done, total: progress.total })
  }
  return stageLabel(job.stage)
}

/**
 * 전체 진행률(0~100). 단계 순서로 대략의 위치를 정하고, 음성 인식 단계는 청크 수로 채운다.
 * 음성 인식이 처리 시간의 대부분이라 그 단계에 막대의 큰 몫을 준다.
 */
const STAGE_SPAN: Record<Stage, [number, number]> = {
  extract: [0, 4],
  enhance: [4, 8],
  vad: [8, 10],
  load_model: [10, 15],
  transcribe: [15, 85],
  diarize: [85, 93],
  segment: [93, 96],
  quality: [96, 98],
  write: [98, 100],
}

export function progressPercent(job: Pick<Job, 'status' | 'stage' | 'progress'>): number | null {
  if (job.status === 'done') return 100
  if (job.status !== 'running' || !job.stage) return null
  const [start, end] = STAGE_SPAN[job.stage]
  const progress = job.progress
  const ratio = progress && progress.total > 0 ? progress.done / progress.total : 0
  return Math.round(start + (end - start) * Math.min(1, Math.max(0, ratio)))
}

/** 취소할 수 있는 작업 (대기·실행 중) */
export function canCancel(job: Pick<Job, 'status'>): boolean {
  return job.status === 'queued' || job.status === 'running'
}
