// 자막 검수 표시 규칙 (검수 종류의 모양, 표에 넣을 줄, 걸러 보기)
import type { ReviewRow, ReviewType } from './api/types'

// 화면에 보이는 순서 (필터 목록·칩)
export const REVIEW_TYPES: readonly ReviewType[] = ['misrecognition', 'punctuation', 'other', 'suggestion', 'corrected']

// 메모 배너 색: 사람이 봐야 하는 의심은 warning, 이미 반영한 교정은 success
export const REVIEW_TYPE_LOOKS: Record<ReviewType, 'info' | 'warning' | 'success'> = {
  misrecognition: 'warning',
  punctuation: 'info',
  other: 'info',
  suggestion: 'warning',
  corrected: 'success',
}

const SRT_SUFFIX = '.srt'
// 처음 고칠 때 서버가 남기는 원본 (검수 대상이 아니다, src/web/review.py BACKUP_SUFFIX)
const BACKUP_SUFFIX = '.orig.srt'
// 자막 한 개의 최대 글자 수 (서버 src/web/review.py MAX_TEXT_LENGTH 와 같다)
export const MAX_SUBTITLE_LENGTH = 500
// 표의 한 칸에서 줄바꿈을 보이는 표시 (칸은 한 줄로 그린다)
export const LINE_BREAK_MARK = ' ↵ '

/** 검수할 SRT 출력이 있는 작업인지. */
export function hasSubtitle(outputs: readonly string[]): boolean {
  return outputs.some((name) => name.endsWith(SRT_SUFFIX) && !name.endsWith(BACKUP_SUFFIX))
}

/** 줄에 붙은 검수 종류 (겹치지 않게, 화면 순서로). */
export function rowTypes(row: Pick<ReviewRow, 'notes'>): ReviewType[] {
  const present = new Set(row.notes.map((note) => note.type))
  return REVIEW_TYPES.filter((type) => present.has(type))
}

// 검수 상태: 저장해 검수를 마쳤다(reviewed), 검수 항목이 있는데 아직 저장하지 않았다(pending), 아직 보지 않았다(open).
// 표의 status 열은 빈 값에도 빈 칩을 그리므로 모든 줄에 상태를 준다
export type ReviewStatus = 'reviewed' | 'pending' | 'open'

export const REVIEW_STATUS_LOOKS: Record<ReviewStatus, { color: string; icon: string }> = {
  reviewed: { color: 'success', icon: 'mdi-check-circle' },
  pending: { color: 'warning', icon: 'mdi-alert-circle-outline' },
  open: { color: 'grey', icon: 'mdi-circle-outline' },
}

/** 줄의 검수 상태. 검수를 마친 것이 먼저다 (항목이 있어도 저장했으면 완료). */
export function rowStatus(row: Pick<ReviewRow, 'notes' | 'reviewed'>): ReviewStatus {
  if (row.reviewed) return 'reviewed'
  return row.notes.length > 0 ? 'pending' : 'open'
}

export interface ReviewFilter {
  // 검수 항목이 있는 자막만
  flaggedOnly: boolean
  // 이 종류의 항목이 있는 자막만 ('' 이면 모두)
  type: ReviewType | ''
}

/** 걸러 볼 자막. 종류를 고르면 그 종류의 항목이 있는 자막만 남는다. */
export function filterRows(rows: readonly ReviewRow[], filter: ReviewFilter): ReviewRow[] {
  return rows.filter((row) => {
    if (filter.type) return row.notes.some((note) => note.type === filter.type)
    return !filter.flaggedOnly || row.notes.length > 0
  })
}

/** 표의 자막 칸: 줄바꿈을 표시로 바꾼 한 줄. */
export function oneLine(text: string): string {
  return text.split('\n').join(LINE_BREAK_MARK)
}
