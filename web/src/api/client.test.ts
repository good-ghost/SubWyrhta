import { afterEach, describe, expect, it, vi } from 'vitest'

import { api, ApiError, downloadHref, errorFromBody, onUnauthorized, queryString } from './client'

describe('api client', () => {
  afterEach(() => {
    vi.restoreAllMocks()
    onUnauthorized(null)
  })

  it('쿼리 문자열: 배열은 같은 키로 반복하고 undefined 는 뺀다', () => {
    expect(queryString({ status: ['queued', 'running'], after: undefined, level: 'INFO' })).toBe(
      '?status=queued&status=running&level=INFO',
    )
    expect(queryString(undefined)).toBe('')
  })

  it('서버 detail 을 오류 코드로', () => {
    expect(errorFromBody(409, { detail: 'job_running' }).code).toBe('job_running')
    const invalid = errorFromBody(400, { detail: { code: 'invalid_settings', errors: [{ loc: 'stt.beam_size', message: 'x' }] } })
    expect(invalid.code).toBe('invalid_settings')
    expect(invalid.errors).toHaveLength(1)
    expect(errorFromBody(502, null).code).toBe('http_502')
  })

  it('내려받기 주소는 파일 이름을 인코딩한다', () => {
    expect(downloadHref(3, 'a b#.srt')).toBe('api/jobs/3/files/a%20b%23.srt')
  })

  it('쓰기 요청은 JSON 으로 보내고 401 이면 로그인 화면으로 돌린다', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'not_authenticated' }), { status: 401 }))
    vi.stubGlobal('fetch', fetchMock)
    const handler = vi.fn()
    onUnauthorized(handler)
    await expect(api.cancel(7)).rejects.toBeInstanceOf(ApiError)
    expect(handler).toHaveBeenCalledOnce()
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe('api/jobs/7/cancel')
    expect(init.method).toBe('POST')
    expect((init.headers as Record<string, string>)['Content-Type']).toBe('application/json')
    vi.unstubAllGlobals()
  })
})
