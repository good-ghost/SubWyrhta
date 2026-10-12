import { describe, expect, it } from 'vitest'

import type { ReviewRow } from './api/types'
import { filterRows, hasSubtitle, oneLine, rowStatus, rowTypes } from './review'

const ROWS: ReviewRow[] = [
  { index: 1, start: 0, end: 1, text: 'a', notes: [], reviewed: false },
  {
    index: 2, start: 1, end: 2, text: 'b',
    notes: [
      { type: 'other', note: 'x' },
      { type: 'misrecognition', note: 'y' },
      { type: 'other', note: 'z' },
    ],
    reviewed: false,
  },
  { index: 3, start: 2, end: 3, text: 'c', notes: [{ type: 'corrected', note: 'w' }], reviewed: true },
]

describe('review', () => {
  it('남겨 둔 원본은 검수할 SRT 로 세지 않는다', () => {
    expect(hasSubtitle(['a.ja.srt', 'a.review.json'])).toBe(true)
    expect(hasSubtitle(['a.ja.orig.srt', 'a.ja.vtt'])).toBe(false)
    expect(hasSubtitle([])).toBe(false)
  })

  it('줄의 검수 종류는 겹치지 않게 화면 순서로', () => {
    expect(rowTypes(ROWS[1])).toEqual(['misrecognition', 'other'])
    expect(rowTypes(ROWS[0])).toEqual([])
  })

  it('검수 항목만, 종류별로 걸러 본다', () => {
    expect(filterRows(ROWS, { flaggedOnly: false, type: '' }).map((row) => row.index)).toEqual([1, 2, 3])
    expect(filterRows(ROWS, { flaggedOnly: true, type: '' }).map((row) => row.index)).toEqual([2, 3])
    expect(filterRows(ROWS, { flaggedOnly: false, type: 'corrected' }).map((row) => row.index)).toEqual([3])
    expect(filterRows(ROWS, { flaggedOnly: true, type: 'punctuation' })).toEqual([])
  })

  it('검수 상태: 저장한 자막은 완료, 항목이 있는데 저장하지 않았으면 확인 필요, 나머지는 미검수', () => {
    expect(ROWS.map(rowStatus)).toEqual(['open', 'pending', 'reviewed'])
    expect(rowStatus({ notes: [], reviewed: true })).toBe('reviewed')
  })

  it('표 칸에서는 줄바꿈을 표시로 바꾼다', () => {
    expect(oneLine('一行目\n二行目')).toBe('一行目 ↵ 二行目')
    expect(oneLine('한 줄')).toBe('한 줄')
  })
})
