// Extract dataset name from URL path (e.g., /investment -> investment)
export function getDataset(): string {
  const path = window.location.pathname
  const match = path.match(/^\/([^/]+)/)
  return match ? match[1] : 'default'
}

// Dynamic API base that includes dataset name
function getApiBase(): string {
  return `/api/${getDataset()}`
}

// Logging helper
const log = (category: string, message: string, data?: Record<string, unknown>) => {
  const ts = new Date().toISOString().slice(11, 23)
  const dataStr = data ? ` | ${Object.entries(data).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(' ')}` : ''
  console.log(`[${ts}] [CLIENT] [${category}] ${message}${dataStr}`)
}

// Custom error class for API failures - crashes the app with clear messaging
export class ApiError extends Error {
  status: number
  endpoint: string

  constructor(message: string, status: number, endpoint: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.endpoint = endpoint
  }

  // Returns a user-friendly crash message
  get crashMessage(): string {
    if (this.status === 404) {
      return `DATA MISSING: ${this.endpoint}\n\nPre-computed data not found. Run compute_geometry_analysis.py to generate required data.`
    }
    if (this.status === 500) {
      return `SERVER ERROR: ${this.endpoint}\n\nThe backend crashed. Check server logs for details.`
    }
    return `API ERROR (${this.status}): ${this.endpoint}\n\n${this.message}`
  }
}

class ApiClient {
  private requestId: number = 0

  private get baseUrl(): string {
    return getApiBase()
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    const reqId = ++this.requestId
    const startTime = performance.now()

    log('API', `[#${reqId}] ${options.method || 'GET'} ${endpoint}`)

    const config: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    }

    const response = await fetch(url, config)
    const fetchTime = performance.now() - startTime

    if (!response.ok) {
      log('API', `[#${reqId}] ERROR ${response.status}`, { elapsed_ms: fetchTime.toFixed(1) })

      let errorMessage = `HTTP error ${response.status}`
      try {
        const errorData = await response.json()
        errorMessage = errorData.detail || errorData.message || errorMessage
      } catch {
        // Use default error message
      }

      // Throw ApiError with full context - this will crash the app
      const error = new ApiError(errorMessage, response.status, endpoint)
      console.error(`[API CRASH] ${error.crashMessage}`)
      throw error
    }

    const parseStart = performance.now()
    const data = await response.json()
    const parseTime = performance.now() - parseStart
    const totalTime = performance.now() - startTime

    log('API', `[#${reqId}] OK`, {
      fetch_ms: fetchTime.toFixed(1),
      parse_ms: parseTime.toFixed(1),
      total_ms: totalTime.toFixed(1),
      size_kb: (JSON.stringify(data).length / 1024).toFixed(1)
    })

    return data
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' })
  }

  async post<T>(endpoint: string, data?: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    })
  }

  async put<T>(endpoint: string, data?: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    })
  }

  async delete<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'DELETE' })
  }
}

export const api = new ApiClient()

// API endpoint functions
export const geoApi = {
  // Health check
  health: () => api.get<{ status: string }>('/health'),

  // Data endpoints
  getData: () => api.get<unknown>('/data'),
  getDataById: (id: string) => api.get<unknown>(`/data/${id}`),

  // Analysis endpoints
  runAnalysis: (params: unknown) => api.post<unknown>('/analysis', params),
  getAnalysisResults: (id: string) => api.get<unknown>(`/analysis/${id}`),

  // Visualization endpoints
  getVisualization: (id: string) => api.get<unknown>(`/visualization/${id}`),
}

export default api
