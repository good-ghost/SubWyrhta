// 서버 API 호출. 세션은 HttpOnly 쿠키라 스크립트가 토큰을 다루지 않는다.
import type {
  FieldError,
  Job,
  JobFormOptions,
  JobOptions,
  JobStatus,
  LogChunk,
  LogLevel,
  Preview,
  Review,
  ReviewRow,
  ServerStatus,
  SessionState,
  SettingsState,
  SettingsValues,
  Upload,
} from './types'

/** API 오류. code 는 서버 detail 코드(화면이 번역한다), errors 는 설정 검증 오류. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    readonly errors: FieldError[] = [],
  ) {
    super(`${status} ${code}`)
    this.name = 'ApiError'
  }
}

let unauthorizedHandler: (() => void) | null = null

/** 로그인이 풀렸을 때(401) 부를 함수. 앱이 로그인 화면으로 돌아간다. */
export function onUnauthorized(handler: (() => void) | null): void {
  unauthorizedHandler = handler
}

type Query = Record<string, string | number | boolean | string[] | undefined>

export function queryString(query: Query | undefined): string {
  if (!query) return ''
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined) continue
    for (const item of Array.isArray(value) ? value : [value]) params.append(key, String(item))
  }
  const text = params.toString()
  return text ? `?${text}` : ''
}

/** 서버 detail 을 오류로. 문자열 코드 또는 {code, errors}. */
export function errorFromBody(status: number, body: unknown): ApiError {
  const detail = (body as { detail?: unknown } | null)?.detail
  if (typeof detail === 'string') return new ApiError(status, detail)
  if (detail && typeof detail === 'object' && 'code' in detail) {
    const { code, errors } = detail as { code: string; errors?: FieldError[] }
    return new ApiError(status, code, errors ?? [])
  }
  return new ApiError(status, `http_${status}`)
}

async function parseError(response: Response): Promise<ApiError> {
  let body: unknown = null
  try {
    body = await response.json()
  } catch {
    // 본문이 JSON 이 아니다 (프록시 오류 페이지 등): 상태 코드로만 알린다
  }
  return errorFromBody(response.status, body)
}

async function request<T>(method: string, path: string, body?: unknown, query?: Query): Promise<T> {
  // 쓰기 요청은 JSON 만 받는다 (서버가 다른 형식을 415 로 거절한다)
  const headers: Record<string, string> = method === 'GET' ? {} : { 'Content-Type': 'application/json' }
  // 상대 경로: 리버스 프록시의 하위 경로에서도 같은 위치의 API 를 부른다
  const response = await fetch(`api/${path}${queryString(query)}`, {
    method,
    headers,
    credentials: 'same-origin',
    body: method === 'GET' ? undefined : JSON.stringify(body ?? {}),
  })
  if (!response.ok) {
    const error = await parseError(response)
    if (response.status === 401 && path !== 'session') unauthorizedHandler?.()
    throw error
  }
  return (await response.json()) as T
}

/**
 * 파일 업로드. 진행률을 받으려고 fetch 대신 XMLHttpRequest 로 본문을 그대로 보낸다
 * (application/octet-stream: 다른 사이트의 폼으로는 보낼 수 없는 형식).
 */
export function uploadFile(file: File, onProgress: (loaded: number, total: number) => void): {
  promise: Promise<Upload>
  abort: () => void
} {
  const xhr = new XMLHttpRequest()
  const promise = new Promise<Upload>((resolve, reject) => {
    xhr.open('POST', `api/uploads${queryString({ name: file.name })}`)
    xhr.setRequestHeader('Content-Type', 'application/octet-stream')
    xhr.withCredentials = true
    xhr.upload.onprogress = (event) => onProgress(event.loaded, event.lengthComputable ? event.total : file.size)
    xhr.onload = () => {
      let body: unknown = null
      try {
        body = JSON.parse(xhr.responseText) as unknown
      } catch {
        // 본문이 JSON 이 아니다: 상태 코드로만 알린다
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as Upload)
        return
      }
      if (xhr.status === 401) unauthorizedHandler?.()
      reject(errorFromBody(xhr.status, body))
    }
    xhr.onerror = () => reject(new TypeError('network'))
    xhr.onabort = () => reject(new ApiError(0, 'upload_aborted'))
    xhr.send(file)
  })
  return { promise, abort: () => xhr.abort() }
}

/** 출력 파일 내려받기 주소 (세션 쿠키로 받는다) */
export function downloadHref(jobId: number, name: string): string {
  return `api/jobs/${jobId}/files/${encodeURIComponent(name)}`
}

export const api = {
  session: () => request<SessionState>('GET', 'session'),
  setup: (username: string, password: string) => request<SessionState>('POST', 'session/setup', { username, password }),
  login: (username: string, password: string) => request<SessionState>('POST', 'session', { username, password }),
  logout: () => request<SessionState>('DELETE', 'session'),
  status: () => request<ServerStatus>('GET', 'status'),
  options: () => request<JobFormOptions>('GET', 'options'),
  jobs: (status?: JobStatus[]) => request<Job[]>('GET', 'jobs', undefined, { status }),
  job: (id: number) => request<Job>('GET', `jobs/${id}`),
  createJob: (uploadId: string, options: JobOptions) => request<Job>('POST', 'jobs', { upload_id: uploadId, options }),
  cancel: (id: number) => request<Job>('POST', `jobs/${id}/cancel`),
  retry: (id: number) => request<Job>('POST', `jobs/${id}/retry`),
  retryFailed: () => request<{ count: number }>('POST', 'jobs/retry-failed'),
  clearFinished: () => request<{ count: number }>('POST', 'jobs/clear-finished'),
  remove: (id: number) => request<{ count: number }>('DELETE', `jobs/${id}`),
  preview: (id: number, name: string) => request<Preview>('GET', `jobs/${id}/preview/${encodeURIComponent(name)}`),
  review: (id: number, name?: string) => request<Review>('GET', `jobs/${id}/review`, undefined, { name }),
  saveReview: (id: number, name: string, index: number, text: string) =>
    request<ReviewRow>('PUT', `jobs/${id}/review`, { name, index, text }),
  logs: (level: LogLevel, after: string | undefined, limit?: number) =>
    request<LogChunk>('GET', 'logs', undefined, { level, after, limit }),
  settings: () => request<SettingsState>('GET', 'settings'),
  saveSettings: (values: SettingsValues) => request<SettingsState>('PUT', 'settings', { values }),
  account: () => request<{ username: string | null }>('GET', 'account'),
  saveAccount: (currentPassword: string, username: string, newPassword: string | null) =>
    request<{ username: string | null }>('PUT', 'account', {
      current_password: currentPassword,
      username,
      new_password: newPassword,
    }),
}
