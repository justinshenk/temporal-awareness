const API_BASE = '/api'

interface ApiError {
  message: string
  status: number
}

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`

    const config: RequestInit = {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    }

    const response = await fetch(url, config)

    if (!response.ok) {
      const error: ApiError = {
        message: `HTTP error ${response.status}`,
        status: response.status,
      }

      try {
        const errorData = await response.json()
        error.message = errorData.detail || errorData.message || error.message
      } catch {
        // Use default error message
      }

      throw error
    }

    return response.json()
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
