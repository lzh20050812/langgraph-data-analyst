export class ApiError extends Error {
  status: number
  detail: string
  constructor(status: number, detail: string) {
    super(detail)
    this.status = status
    this.detail = detail
  }
}

const base = import.meta.env.DEV ? '/api' : ''

export async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers || {})
  if (options.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  const response = await fetch(`${base}${path}`, {
    ...options,
    headers,
    credentials: 'include',
  })
  if (!response.ok) {
    let detail = `请求失败 (${response.status})`
    try {
      const payload = await response.json()
      detail = payload.detail || detail
    } catch (_) { /* response is not JSON */ }
    if (response.status === 401 && path !== '/auth/login') {
      window.dispatchEvent(new CustomEvent('portal:unauthorized'))
    }
    throw new ApiError(response.status, detail)
  }
  return response.json() as Promise<T>
}

export function apiUrl(path: string): string {
  return `${base}${path}`
}

export function errorMessage(error: unknown, fallback = '请求失败'): string {
  if (error instanceof ApiError) return error.detail
  if (error instanceof Error && error.message) return error.message
  return fallback
}
