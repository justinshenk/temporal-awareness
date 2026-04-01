import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

interface ViewState {
  zoom: number
  rotation: [number, number, number]
  position: [number, number, number]
}

interface DataPoint {
  id: string
  coordinates: [number, number, number]
  label?: string
  cluster?: number
  metadata?: Record<string, unknown>
}

interface AppState {
  // Loading state
  isLoading: boolean
  setLoading: (loading: boolean) => void

  // Error state
  error: string | null
  setError: (error: string | null) => void
  clearError: () => void

  // Data state
  dataPoints: DataPoint[]
  setDataPoints: (points: DataPoint[]) => void
  selectedPointId: string | null
  selectPoint: (id: string | null) => void

  // View state
  viewState: ViewState
  setViewState: (state: Partial<ViewState>) => void
  resetView: () => void

  // UI state
  sidebarOpen: boolean
  toggleSidebar: () => void
  activePanel: string
  setActivePanel: (panel: string) => void
}

const initialViewState: ViewState = {
  zoom: 1,
  rotation: [0, 0, 0],
  position: [0, 0, 5],
}

export const useAppStore = create<AppState>()(
  devtools(
    (set) => ({
      // Loading state
      isLoading: false,
      setLoading: (loading) => set({ isLoading: loading }),

      // Error state
      error: null,
      setError: (error) => set({ error }),
      clearError: () => set({ error: null }),

      // Data state
      dataPoints: [],
      setDataPoints: (points) => set({ dataPoints: points }),
      selectedPointId: null,
      selectPoint: (id) => set({ selectedPointId: id }),

      // View state
      viewState: initialViewState,
      setViewState: (state) =>
        set((prev) => ({
          viewState: { ...prev.viewState, ...state },
        })),
      resetView: () => set({ viewState: initialViewState }),

      // UI state
      sidebarOpen: true,
      toggleSidebar: () =>
        set((state) => ({ sidebarOpen: !state.sidebarOpen })),
      activePanel: 'overview',
      setActivePanel: (panel) => set({ activePanel: panel }),
    }),
    { name: 'geoapp-store' }
  )
)

// Selector hooks for optimized re-renders
export const useIsLoading = () => useAppStore((state) => state.isLoading)
export const useError = () => useAppStore((state) => state.error)
export const useDataPoints = () => useAppStore((state) => state.dataPoints)
export const useSelectedPoint = () => {
  const points = useAppStore((state) => state.dataPoints)
  const selectedId = useAppStore((state) => state.selectedPointId)
  return points.find((p) => p.id === selectedId) ?? null
}
export const useViewState = () => useAppStore((state) => state.viewState)
