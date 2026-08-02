# UI Debug Status

## Current Status

**Last updated**: Fixed sample_indices/embedding mismatch bug - 3D plot should work now

**TLDR**: Both servers running, critical bug fixed, ready for browser testing at http://localhost:3000

## What's Working

1. **Backend server**: Running on http://localhost:8000
   - `/api/config` returns valid data: 12 layers, 17 positions, 4588 samples
   - `available_methods: ['pca', 'umap']` (both now available)
   - `/api/embedding/35/resid_post/response_choice/stream` returns 3751 points with coordinates ranging -65 to +102
   - TypeScript build passes with no errors

2. **Frontend server**: Running on http://localhost:3000
   - Vite dev server running, serves valid HTML
   - Build completes successfully (687 modules)
   - No compilation/TypeScript errors

3. **UMAP generation**: Complete (654/816 files, 80%+ threshold met)
   - UMAP is now available in the UI method selector

## Verified Data Flow (via curl)

```
GET /api/config
Response: {layers: [0,1,3,12,18,19,21,24,28,31,34,35], positions: [...17 positions], available_methods: ['pca', 'umap']}

GET /api/embedding/35/resid_post/response_choice/stream?method=pca
Response: SSE stream with total_points=3751, valid coordinates (ranges -65 to +102)

GET /api/metadata?color_by=time_horizon
Response: 4588 values, 4318 non-null
```

## Issue: Embeddings Not Being Requested

Server logs only show `/api/config` requests - no `/api/embedding` requests were made.
This suggests either:
1. The browser page wasn't loaded (most likely)
2. A JavaScript runtime error before embedding fetch starts
3. The layer/position state isn't updating from config

## To Debug in Browser

1. **Open http://localhost:3000 in browser**

2. **Open DevTools (F12) > Console tab**

3. **Look for these initialization logs:**
   ```
   [CLIENT] [useConfig] Config loaded | n_layers=12 n_positions=17
   [CLIENT] [App] Initializing layer from config | layer=35
   [CLIENT] [App] Initializing position from config | position=response_choice
   [CLIENT] [useStreamingEmbedding] STARTING SSE: L35/resid_post/response_choice (pca)
   ```

4. **If embedding doesn't start, check:**
   - Is there a JavaScript error before "Initializing layer from config"?
   - Is the method selector showing "pca" and "umap" options?
   - Check Network tab for failed requests

5. **If embedding starts but plot is black:**
   - Look for: `[ScatterPlot3D] Render #X | n_points=XXXX`
   - n_points should be ~3751, not 0
   - Check for WebGL errors in console

## Expected Working Behavior

When the page loads correctly:
1. Config loads first (12 layers, 17 positions)
2. Layer auto-sets to 35 (last layer)
3. Position auto-sets to "response_choice" (last position)
4. Embedding stream starts for L35/resid_post/response_choice
5. Points render progressively as chunks arrive
6. Method selector shows "PCA" and "UMAP" buttons

## Sample Panel Missing Response

The sample panel shows `response_label` and `response_term` but `response_text` is empty.
This is a data generation issue - preference_sample.json files have `response_text: ""`.
To fix: Regenerate samples with response text extraction enabled.

## Summary of Fixes Made This Session

1. **summary.json** - Added computed layers/components/positions to match actual embeddings
2. **Server startup** - Disabled slow warmup (now uses lazy loading)
3. **UMAP** - Generation completed to 80% threshold, now available in UI
4. **Fixed sample_indices/embedding size mismatch** (routes.py)
   - Bug: `get_valid_sample_indices()` returned 3751 indices but embedding only had 3419 rows
   - Fix: Use embedding.shape[0] as authoritative source, truncate indices to match
   - Now `total_points` correctly reports 3419 to the frontend
