"""API route definitions for Geometry visualization backend."""

import asyncio
import math
import time
from typing import Literal

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
import json

from .data_loader import GeometryDataLoader

# Request counter for logging
_request_counter = 0

def _log(endpoint: str, message: str, **kwargs):
    """Log with timestamp and request info."""
    global _request_counter
    ts = time.strftime("%H:%M:%S")
    extras = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    print(f"[{ts}] [SERVER] [{endpoint}] {message}" + (f" | {extras}" if extras else ""))
from .models import (
    ColorValues,
    ConfigResponse,
    EmbeddingResponse,
    HeatmapCell,
    HeatmapResponse,
    MetricsResponse,
    PCAMetrics,
    Point3D,
    ProbeMetrics,
    SampleResponse,
    TokenInfo,
    TokensResponse,
    TrajectoryPoint,
    TrajectoryResponse,
    WarmupResponse,
    WarmupStatus,
)


def _sanitize_float(value: float) -> float | None:
    """Convert NaN/Infinity to None for JSON serialization.

    JSON spec doesn't support NaN or Infinity, so we convert them to null.
    """
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return float(value)


def create_router(data_loader: GeometryDataLoader) -> APIRouter:
    """Create API router with endpoints bound to the given data loader.

    Args:
        data_loader: GeometryDataLoader instance for accessing embedding data.

    Returns:
        FastAPI APIRouter with all endpoints configured.
    """
    router = APIRouter(prefix="/api", tags=["geometry"])

    # Warmup state (shared across requests)
    warmup_state = {
        "is_running": False,
        "progress": 0,
        "total": 0,
        "current_task": None,
    }

    def _run_warmup_background(
        methods: list[str],
        layers: list[int],
        components: list[str],
        positions: list[str],
    ):
        """Run warmup in background thread."""
        warmup_state["is_running"] = True
        warmup_state["progress"] = 0
        warmup_state["total"] = len(methods) * len(layers) * len(components) * len(positions)

        def progress_callback(current: int, total: int, desc: str):
            warmup_state["progress"] = current
            warmup_state["total"] = total
            warmup_state["current_task"] = desc

        try:
            data_loader.warmup(
                methods=methods,
                layers=layers,
                components=components,
                positions=positions,
                progress_callback=progress_callback,
            )
        finally:
            warmup_state["is_running"] = False
            warmup_state["current_task"] = None

    @router.get("/config", response_model=ConfigResponse)
    async def get_config() -> ConfigResponse:
        """Get available configuration options for the visualization.

        Returns layers, components, positions, color options, and position labels.
        Only returns positions/layers that have precomputed embeddings.
        """
        _log("/config", "GET request received")
        # Use ONLY precomputed positions/layers to avoid 500 errors
        layers = data_loader.get_precomputed_layers()
        positions = data_loader.get_precomputed_positions()
        available_methods = data_loader.get_available_methods()
        _log("/config", f"Returning config", n_layers=len(layers), n_positions=len(positions), n_samples=data_loader.n_samples, methods=available_methods)
        return ConfigResponse(
            layers=layers,
            components=data_loader.get_components(),
            positions=positions,
            color_options=data_loader.get_color_options(),
            n_samples=data_loader.n_samples,
            model_name=data_loader.get_model_name(),
            position_labels=data_loader.get_enriched_position_labels(),
            prompt_template=data_loader.get_prompt_template_structure(),
            semantic_to_positions=data_loader.get_semantic_to_positions_mapping(),
            markers=data_loader.get_markers(),
            rel_pos_counts=data_loader.get_rel_pos_counts(),
            available_methods=available_methods,
        )

    @router.get("/embedding/{layer}/{component}/{position}", response_model=EmbeddingResponse)
    async def get_embedding(
        layer: int,
        component: str,
        position: str,
        method: Literal["pca", "umap", "tsne"] = Query(default="pca", description="Dimensionality reduction method"),
        response: Response = None,
    ) -> EmbeddingResponse:
        """Get 3D embedding coordinates for a specific layer/component/position.

        Args:
            layer: Transformer layer number.
            component: Activation component (resid_pre, attn_out, mlp_out, resid_post).
            position: Token position identifier.
            method: Dimensionality reduction method (pca, umap, or tsne).

        Returns:
            3D coordinates for all samples.
        """
        _log("/embedding", f"GET request", layer=layer, component=component, position=position, method=method)
        start_time = time.time()

        # Validate layer
        if layer not in data_loader.get_layers():
            raise HTTPException(status_code=404, detail=f"Layer {layer} not found")

        # Validate component
        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

        # Validate position - must have precomputed embeddings
        precomputed_positions = data_loader.get_precomputed_positions()
        if position not in precomputed_positions:
            if position in data_loader.get_positions():
                # Position exists but no precomputed embedding
                raise HTTPException(
                    status_code=404,
                    detail=f"Position '{position}' exists but has no precomputed embeddings. "
                    f"Available positions: {precomputed_positions}"
                )
            raise HTTPException(status_code=404, detail=f"Position '{position}' not found")

        # Load embedding based on method
        try:
            if method == "pca":
                embedding = data_loader.load_pca(layer, component, position, n_components=3)
            elif method == "umap":
                embedding = data_loader.load_umap(layer, component, position, n_components=3)
            elif method == "tsne":
                embedding = data_loader.load_tsne(layer, component, position, n_components=3)
            else:
                raise HTTPException(status_code=400, detail=f"Invalid method: {method}")
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Get the sample indices that correspond to the embedding rows
        sample_indices = data_loader.get_valid_sample_indices(layer, component, position)

        # Optimized conversion: sanitize NaN/Infinity in bulk using numpy
        # Replace NaN/Inf with 0.0 in-place
        clean_embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

        # Convert to flat list of floats for maximum performance
        # Format: [x0, y0, z0, x1, y1, z1, ...] - frontend will reshape
        coordinates_flat = clean_embedding.flatten().tolist()

        # Set cache headers for browser caching (30 minutes)
        if response:
            response.headers["Cache-Control"] = "max-age=1800, stale-while-revalidate=3600"

        elapsed = time.time() - start_time
        _log("/embedding", f"Returning embedding", n_samples=len(sample_indices), n_coords=len(coordinates_flat), elapsed_ms=f"{elapsed*1000:.1f}")

        return EmbeddingResponse(
            layer=layer,
            component=component,
            position=position,
            method=method,
            n_samples=len(sample_indices),
            coordinates_flat=coordinates_flat,
            sample_indices=sample_indices,
        )

    @router.get("/embedding/{layer}/{component}/{position}/stream")
    async def get_embedding_stream(
        layer: int,
        component: str,
        position: str,
        method: Literal["pca", "umap", "tsne"] = Query(default="pca"),
        chunk_size: int = Query(default=500, description="Points per chunk"),
    ):
        """Stream embedding coordinates via Server-Sent Events.

        Sends data in chunks so UI can render progressively.
        Each chunk contains coordinates for chunk_size points.
        """
        _log("/embedding/stream", f"🚀 SSE request received", layer=layer, component=component, position=position, method=method, chunk_size=chunk_size)

        # Validate inputs
        if layer not in data_loader.get_layers():
            raise HTTPException(status_code=404, detail=f"Layer {layer} not found")
        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")
        precomputed_positions = data_loader.get_precomputed_positions()
        if position not in precomputed_positions:
            if position in data_loader.get_positions():
                raise HTTPException(
                    status_code=404,
                    detail=f"Position '{position}' exists but has no precomputed embeddings"
                )
            raise HTTPException(status_code=404, detail=f"Position '{position}' not found")

        async def generate_chunks():
            """Generator that yields SSE chunks."""
            import asyncio

            stream_start = time.time()
            _log("/embedding/stream", f"Loading embedding...", method=method)

            # Load embedding (this is the slow part) - STRICT: raises ValueError if missing
            if method == "pca":
                embedding = data_loader.load_pca(layer, component, position, n_components=3)
            elif method == "umap":
                embedding = data_loader.load_umap(layer, component, position, n_components=3)
            elif method == "tsne":
                embedding = data_loader.load_tsne(layer, component, position, n_components=3)
            else:
                _log("/embedding/stream", f"ERROR: Invalid method {method}")
                yield f"data: {json.dumps({'error': f'Invalid method: {method}'})}\n\n"
                return

            load_elapsed = time.time() - stream_start
            _log("/embedding/stream", f"Embedding loaded", elapsed_ms=f"{load_elapsed*1000:.1f}", shape=embedding.shape)

            sample_indices = data_loader.get_valid_sample_indices(layer, component, position)
            clean_embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)

            total_points = len(sample_indices)
            _log("/embedding/stream", f"Sending metadata", total_points=total_points)

            # Send metadata first
            metadata = {
                "type": "metadata",
                "layer": layer,
                "component": component,
                "position": position,
                "method": method,
                "total_points": total_points,
                "chunk_size": chunk_size,
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            await asyncio.sleep(0)  # Yield to event loop

            # Stream chunks of points
            n_chunks = (total_points + chunk_size - 1) // chunk_size
            for chunk_num, i in enumerate(range(0, total_points, chunk_size)):
                end_idx = min(i + chunk_size, total_points)
                chunk_coords = clean_embedding[i:end_idx].flatten().tolist()
                chunk_indices = sample_indices[i:end_idx]

                chunk_data = {
                    "type": "chunk",
                    "start_idx": i,
                    "end_idx": end_idx,
                    "coordinates": chunk_coords,
                    "sample_indices": chunk_indices,
                }
                _log("/embedding/stream", f"Sending chunk {chunk_num+1}/{n_chunks}", start=i, end=end_idx)
                yield f"data: {json.dumps(chunk_data)}\n\n"
                await asyncio.sleep(0)  # Yield to event loop between chunks

            total_elapsed = time.time() - stream_start
            _log("/embedding/stream", f"Stream complete", total_ms=f"{total_elapsed*1000:.1f}", total_points=total_points)
            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        return StreamingResponse(
            generate_chunks(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )

    @router.get("/metadata", response_model=ColorValues)
    async def get_metadata(
        color_by: str = Query(default="log_time_horizon", description="Metadata field to use for coloring"),
        response: Response = None,
    ) -> ColorValues:
        """Get color values for all samples based on a metadata field.

        Args:
            color_by: Metadata field name (e.g., log_time_horizon, time_scale, choice_type).

        Returns:
            Color values for all samples with data type information.
        """
        _log("/metadata", f"GET request", color_by=color_by)
        valid_options = data_loader.get_color_options()
        if color_by not in valid_options:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid color_by option: {color_by}. Valid options: {valid_options}",
            )

        values = data_loader.get_sample_metadata(color_by)

        # Determine data type - handle numpy types
        if len(values) == 0:
            dtype = "numeric"
            values_list = []
        elif isinstance(values[0], (bool, np.bool_)):
            dtype = "boolean"
            values_list = [bool(v) for v in values]
        elif isinstance(values[0], (int, float, np.integer, np.floating)):
            dtype = "numeric"
            # Convert sentinel values (-1) to null for time_horizon fields
            # This ensures consistency with the /sample endpoint
            # Also sanitize NaN/Infinity values for valid JSON
            if color_by in ("time_horizon", "log_time_horizon"):
                values_list = [None if v < 0 else _sanitize_float(v) for v in values]
            else:
                values_list = [_sanitize_float(v) for v in values]
        else:
            dtype = "categorical"
            values_list = [str(v) for v in values]

        # Set cache headers (1 hour - metadata doesn't change)
        if response:
            response.headers["Cache-Control"] = "max-age=3600, stale-while-revalidate=7200"

        _log("/metadata", f"Returning values", dtype=dtype, n_values=len(values_list))
        return ColorValues(
            color_by=color_by,
            values=values_list,
            dtype=dtype,
        )

    @router.get("/sample/{idx}", response_model=SampleResponse)
    async def get_sample(idx: int) -> SampleResponse:
        """Get detailed information for a specific sample.

        Args:
            idx: Sample index (0-based).

        Returns:
            Full sample information including text and metadata.
        """
        _log("/sample", f"GET request", idx=idx)
        if idx < 0 or idx >= data_loader.n_samples:
            raise HTTPException(
                status_code=404,
                detail=f"Sample index {idx} out of range (0-{data_loader.n_samples - 1})",
            )

        sample_info = data_loader.get_sample_info(idx)

        # Extract time horizon from prompt structure (same source as coloring)
        # This is the prompt's explicit time horizon, NOT the chosen option's delay
        prompt = sample_info.get("prompt", {})
        time_horizon_data = prompt.get("time_horizon") if isinstance(prompt, dict) else None

        if time_horizon_data and isinstance(time_horizon_data, dict):
            th_value = time_horizon_data.get("value")
            th_unit = time_horizon_data.get("unit", "months")
            if th_value is not None:
                # Convert to months for consistency
                time_horizon_months = data_loader._convert_to_months(th_value, th_unit)
            else:
                time_horizon_months = None
        else:
            time_horizon_months = None

        # Extract known fields
        known_fields = {"text", "time_horizon_months", "time_scale", "choice_type", "short_term_first"}
        metadata = {k: v for k, v in sample_info.items() if k not in known_fields}

        return SampleResponse(
            idx=idx,
            text=sample_info.get("text", ""),
            time_horizon_months=time_horizon_months,
            time_scale=sample_info.get("time_scale"),
            choice_type=sample_info.get("choice_type"),
            short_term_first=sample_info.get("short_term_first"),
            metadata=metadata,
        )

    @router.get("/metrics/{layer}/{component}/{position}", response_model=MetricsResponse)
    async def get_metrics(
        layer: int,
        component: str,
        position: str,
    ) -> MetricsResponse:
        """Get probe metrics for a specific layer/component/position.

        Args:
            layer: Transformer layer number.
            component: Activation component.
            position: Token position identifier.

        Returns:
            Linear probe and PCA metrics if available.
        """
        # Validate inputs
        if layer not in data_loader.get_layers():
            raise HTTPException(status_code=404, detail=f"Layer {layer} not found")

        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

        if position not in data_loader.get_positions():
            raise HTTPException(status_code=404, detail=f"Position {position} not found")

        # Load linear probe metrics
        linear_probe_data = data_loader.load_linear_probe_metrics(layer, component, position)
        linear_probe = None
        if linear_probe_data:
            linear_probe = ProbeMetrics(
                train_accuracy=linear_probe_data.get("train_accuracy"),
                test_accuracy=linear_probe_data.get("test_accuracy"),
                train_r2=linear_probe_data.get("train_r2"),
                test_r2=linear_probe_data.get("test_r2"),
            )

        # Load PCA metrics
        pca_data = data_loader.load_pca_metrics(layer, component, position)
        pca = None
        if pca_data:
            pca = PCAMetrics(
                explained_variance_ratio=pca_data.get("explained_variance_ratio"),
                cumulative_variance=pca_data.get("cumulative_variance"),
                n_components=pca_data.get("n_components"),
            )

        return MetricsResponse(
            layer=layer,
            component=component,
            position=position,
            linear_probe=linear_probe,
            pca=pca,
        )

    @router.get("/heatmap/{component}", response_model=HeatmapResponse)
    async def get_heatmap(
        component: str,
        metric: Literal["r2", "accuracy", "variance"] = Query(default="r2", description="Metric to display"),
    ) -> HeatmapResponse:
        """Get heatmap data for linear probe R² across layers and positions.

        Args:
            component: Activation component (resid_pre, attn_out, mlp_out, resid_post).
            metric: Metric to display (r2, accuracy, or variance).

        Returns:
            Heatmap data with values for each layer/position combination.
        """
        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

        layers = data_loader.get_layers()
        positions = data_loader.get_positions()

        cells = []
        values = []

        for layer in layers:
            for position in positions:
                value = None

                if metric == "r2":
                    probe_data = data_loader.load_linear_probe_metrics(layer, component, position)
                    if probe_data:
                        # Try r2_mean first (newer format), then test_r2 (older format)
                        value = probe_data.get("r2_mean") or probe_data.get("test_r2")
                elif metric == "accuracy":
                    probe_data = data_loader.load_linear_probe_metrics(layer, component, position)
                    if probe_data:
                        value = probe_data.get("accuracy") or probe_data.get("test_accuracy")
                elif metric == "variance":
                    pca_data = data_loader.load_pca_metrics(layer, component, position)
                    if pca_data and pca_data.get("explained_variance_ratio"):
                        # Sum of first 3 components
                        ratios = pca_data["explained_variance_ratio"][:3]
                        value = sum(ratios)

                # Sanitize value for JSON serialization
                sanitized_value = _sanitize_float(value) if value is not None else None
                cells.append(HeatmapCell(layer=layer, position=position, value=sanitized_value))
                if sanitized_value is not None:
                    values.append(sanitized_value)

        min_val = min(values) if values else None
        max_val = max(values) if values else None

        return HeatmapResponse(
            metric=metric,
            component=component,
            layers=layers,
            positions=positions,
            cells=cells,
            min_value=min_val,
            max_value=max_val,
        )

    @router.get("/warmup/status", response_model=WarmupResponse)
    async def get_warmup_status() -> WarmupResponse:
        """Get current warmup/precomputation status.

        Returns current cache status and any ongoing warmup progress.
        """
        return WarmupResponse(
            message="Warmup in progress" if warmup_state["is_running"] else "Idle",
            status=WarmupStatus(
                is_running=warmup_state["is_running"],
                progress=warmup_state["progress"],
                total=warmup_state["total"],
                current_task=warmup_state["current_task"],
                cached_pca=len(data_loader._pca_cache),
                cached_umap=len(data_loader._umap_cache),
                cached_tsne=len(data_loader._tsne_cache),
            ),
        )

    @router.post("/warmup", response_model=WarmupResponse)
    async def start_warmup(
        background_tasks: BackgroundTasks,
        methods: str = Query(
            default="pca,umap",
            description="Comma-separated methods to precompute (pca,umap,tsne)",
        ),
        components: str = Query(
            default="resid_pre",
            description="Comma-separated components to precompute",
        ),
        all_positions: bool = Query(
            default=True,
            description="Precompute all positions (not just named ones)",
        ),
    ) -> WarmupResponse:
        """Start background precomputation of embeddings.

        This allows seamless navigation between layers, positions, and methods
        by precomputing all combinations in the background.

        Args:
            methods: Comma-separated list of methods (pca, umap, tsne).
            components: Comma-separated list of components.
            all_positions: If True, include all positions; if False, only named positions.

        Returns:
            Warmup status with cache counts.
        """
        if warmup_state["is_running"]:
            return WarmupResponse(
                message="Warmup already in progress",
                status=WarmupStatus(
                    is_running=True,
                    progress=warmup_state["progress"],
                    total=warmup_state["total"],
                    current_task=warmup_state["current_task"],
                    cached_pca=len(data_loader._pca_cache),
                    cached_umap=len(data_loader._umap_cache),
                    cached_tsne=len(data_loader._tsne_cache),
                ),
            )

        # Parse parameters
        method_list = [m.strip() for m in methods.split(",") if m.strip() in ("pca", "umap", "tsne")]
        if not method_list:
            method_list = ["pca"]

        component_list = [c.strip() for c in components.split(",") if c.strip() in data_loader.get_components()]
        if not component_list:
            component_list = ["resid_pre"]

        position_list = data_loader.get_positions() if all_positions else data_loader.get_positions()[:1]
        layer_list = data_loader.get_layers()

        # Start background warmup
        background_tasks.add_task(
            _run_warmup_background,
            method_list,
            layer_list,
            component_list,
            position_list,
        )

        return WarmupResponse(
            message=f"Started warmup for {len(method_list)} methods, {len(layer_list)} layers, {len(component_list)} components, {len(position_list)} positions",
            status=WarmupStatus(
                is_running=True,
                progress=0,
                total=len(method_list) * len(layer_list) * len(component_list) * len(position_list),
                current_task="Starting...",
                cached_pca=len(data_loader._pca_cache),
                cached_umap=len(data_loader._umap_cache),
                cached_tsne=len(data_loader._tsne_cache),
            ),
        )

    @router.post("/warmup/prefetch", response_model=WarmupResponse)
    async def prefetch_adjacent(
        background_tasks: BackgroundTasks,
        layer: int = Query(description="Current layer"),
        component: str = Query(description="Current component"),
        position: str = Query(description="Current position"),
        method: str = Query(default="pca", description="Current method"),
    ) -> WarmupResponse:
        """Prefetch embeddings for adjacent layers/positions.

        Call this after navigating to a new view to preload nearby data
        for smoother navigation.

        Runs in background to avoid blocking the main request.
        """
        # Skip prefetch if warmup is already running to avoid overloading
        if warmup_state["is_running"]:
            return WarmupResponse(
                message="Skipped prefetch - warmup already running",
                status=WarmupStatus(
                    is_running=True,
                    progress=warmup_state.get("progress", 0),
                    total=warmup_state.get("total", 0),
                    current_task=warmup_state.get("current_task"),
                    cached_pca=len(data_loader._pca_cache),
                    cached_umap=len(data_loader._umap_cache),
                    cached_tsne=len(data_loader._tsne_cache),
                ),
            )

        def do_prefetch() -> None:
            """Background prefetch task."""
            # Use ONLY precomputed positions/layers to avoid crashes
            layers = data_loader.get_precomputed_layers()
            positions = data_loader.get_precomputed_positions()

            # Find current layer index and prefetch +/- 2 layers
            layer_idx = layers.index(layer) if layer in layers else 0
            adjacent_layers = [
                layers[i] for i in range(max(0, layer_idx - 2), min(len(layers), layer_idx + 3))
            ]

            # Find current position index and prefetch +/- 2 positions
            pos_idx = positions.index(position) if position in positions else 0
            adjacent_positions = [
                positions[i] for i in range(max(0, pos_idx - 2), min(len(positions), pos_idx + 3))
            ]

            # Quick prefetch for PCA only (UMAP/t-SNE are too slow)
            for l in adjacent_layers:
                for p in adjacent_positions:
                    try:
                        if method == "pca":
                            data_loader.load_pca(l, component, p)
                        # Skip UMAP/t-SNE prefetch - too slow
                    except ValueError:
                        # Skip missing embeddings silently during prefetch
                        pass

        # Run prefetch in background - return immediately
        background_tasks.add_task(do_prefetch)

        return WarmupResponse(
            message="Prefetch started in background",
            status=WarmupStatus(
                is_running=False,
                progress=0,
                total=0,
                current_task=None,
                cached_pca=len(data_loader._pca_cache),
                cached_umap=len(data_loader._umap_cache),
                cached_tsne=len(data_loader._tsne_cache),
            ),
        )

    @router.get("/trajectory/layers/{component}/{position}", response_model=TrajectoryResponse)
    async def get_layer_trajectory(
        component: str,
        position: str,
    ) -> TrajectoryResponse:
        """Get PC1 trajectory across all layers for a given component/position.

        Returns PC1 values for each sample at each layer, enabling visualization
        of how representations evolve through the model.
        """
        _log("/trajectory/layers", f"GET request", component=component, position=position)
        start_time = time.time()
        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

        if position not in data_loader.get_positions():
            raise HTTPException(status_code=404, detail=f"Position {position} not found")

        # Try to use pre-cached trajectory data first, fall back to individual PCA
        try:
            layers, pc1_matrix, shared_sample_indices = data_loader.load_layer_trajectory(component, position)
            data = []
            x_values = []
            for i, layer in enumerate(layers):
                pc1_values = pc1_matrix[i].astype(float).tolist()
                x_values.append(str(layer))
                data.append(TrajectoryPoint(
                    x_value=str(layer),
                    values=[_sanitize_float(v) or 0.0 for v in pc1_values],
                    sample_indices=shared_sample_indices,
                ))
            n_samples = len(shared_sample_indices)
            elapsed = time.time() - start_time
            _log("/trajectory/layers", f"Returning cached trajectory", n_layers=len(x_values), n_samples=n_samples, elapsed_ms=f"{elapsed*1000:.1f}")
            return TrajectoryResponse(
                component=component,
                position=position,
                method="pca",
                x_axis="layer",
                x_values=x_values,
                n_samples=n_samples,
                sample_indices=shared_sample_indices,
                data=data,
            )
        except ValueError:
            # Trajectory cache not available, fall back to individual PCA embeddings
            pass

        # Fall back to computing from individual PCA embeddings
        layers = data_loader.get_layers()
        data = []
        x_values = []
        n_samples = 0
        shared_sample_indices: list[int] = []

        for layer in layers:
            # STRICT: load_pca raises ValueError if embedding is missing
            embedding = data_loader.load_pca(layer, component, position, n_components=3)
            # Get sample indices for this layer/component/position
            sample_indices = data_loader.get_valid_sample_indices(layer, component, position)
            # Extract PC1 (first column) and normalize
            pc1_values = embedding[:, 0].astype(float).tolist()
            n_samples = len(pc1_values)
            x_values.append(str(layer))
            data.append(TrajectoryPoint(
                x_value=str(layer),
                values=[_sanitize_float(v) or 0.0 for v in pc1_values],
                sample_indices=sample_indices,
            ))
            # For layer trajectory, all layers share the same sample indices (same position)
            if not shared_sample_indices:
                shared_sample_indices = sample_indices

        elapsed = time.time() - start_time
        _log("/trajectory/layers", f"Returning trajectory", n_layers=len(x_values), n_samples=n_samples, elapsed_ms=f"{elapsed*1000:.1f}")

        return TrajectoryResponse(
            component=component,
            position=position,
            method="pca",
            x_axis="layer",
            x_values=x_values,
            n_samples=n_samples,
            sample_indices=shared_sample_indices,
            data=data,
        )

    @router.get("/trajectory/positions/{layer}/{component}", response_model=TrajectoryResponse)
    async def get_position_trajectory(
        layer: int,
        component: str,
        positions_filter: str = Query(default="", description="Comma-separated positions to include (empty = all named)"),
    ) -> TrajectoryResponse:
        """Get PC1 trajectory across semantic positions for a given layer/component.

        Returns PC1 values for each sample at each position, enabling visualization
        of how representations vary across different parts of the prompt.
        """
        _log("/trajectory/positions", f"GET request", layer=layer, component=component, positions_filter=positions_filter or "default")
        start_time = time.time()
        if layer not in data_loader.get_layers():
            raise HTTPException(status_code=404, detail=f"Layer {layer} not found")

        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

        # Use filtered positions or default to all positions
        if positions_filter:
            positions = [p.strip() for p in positions_filter.split(",") if p.strip()]
        else:
            positions = data_loader.get_positions()[:10]

        # Try to use pre-cached trajectory data first (only if no filter applied)
        if not positions_filter:
            try:
                cached_positions, pc1_values_list, sample_indices_list = data_loader.load_position_trajectory(layer, component)
                # Filter to first 10 positions (default behavior)
                data = []
                x_values = []
                n_samples = 0
                for i, pos in enumerate(cached_positions[:10]):
                    pc1_arr = pc1_values_list[i]
                    pc1_values = pc1_arr.astype(float).tolist() if hasattr(pc1_arr, 'astype') else list(pc1_arr)
                    n_samples = max(n_samples, len(pc1_values))
                    x_values.append(pos)
                    data.append(TrajectoryPoint(
                        x_value=pos,
                        values=[_sanitize_float(v) or 0.0 for v in pc1_values],
                        sample_indices=sample_indices_list[i] if i < len(sample_indices_list) else [],
                    ))
                elapsed = time.time() - start_time
                _log("/trajectory/positions", f"Returning cached trajectory", n_positions=len(x_values), n_samples=n_samples, elapsed_ms=f"{elapsed*1000:.1f}")
                return TrajectoryResponse(
                    layer=layer,
                    component=component,
                    method="pca",
                    x_axis="position",
                    x_values=x_values,
                    n_samples=n_samples,
                    sample_indices=[],  # Position trajectory has per-point indices since each position may differ
                    data=data,
                )
            except ValueError:
                # Trajectory cache not available, fall back to individual PCA embeddings
                pass

        # Fall back to computing from individual PCA embeddings
        data = []
        x_values = []
        n_samples = 0

        for pos in positions:
            # STRICT: load_pca raises ValueError if embedding is missing
            embedding = data_loader.load_pca(layer, component, pos, n_components=3)
            # Get sample indices for this layer/component/position
            sample_indices = data_loader.get_valid_sample_indices(layer, component, pos)
            pc1_values = embedding[:, 0].astype(float).tolist()
            n_samples = len(pc1_values)
            x_values.append(pos)
            data.append(TrajectoryPoint(
                x_value=pos,
                values=[_sanitize_float(v) or 0.0 for v in pc1_values],
                sample_indices=sample_indices,
            ))

        elapsed = time.time() - start_time
        _log("/trajectory/positions", f"Returning trajectory", n_positions=len(x_values), n_samples=n_samples, elapsed_ms=f"{elapsed*1000:.1f}")

        return TrajectoryResponse(
            layer=layer,
            component=component,
            method="pca",
            x_axis="position",
            x_values=x_values,
            n_samples=n_samples,
            sample_indices=[],  # Position trajectory has per-point indices since each position may differ
            data=data,
        )

    @router.get("/tokens/{sample_idx}", response_model=TokensResponse)
    async def get_sample_tokens(sample_idx: int) -> TokensResponse:
        """Get token-level position mapping for a sample.

        Returns all tokens with their absolute position, decoded text,
        traj_section (prompt/response), and format_pos (semantic position).
        """
        if sample_idx < 0 or sample_idx >= data_loader.n_samples:
            raise HTTPException(
                status_code=404,
                detail=f"Sample index {sample_idx} out of range",
            )

        tokens_data = data_loader.get_tokens_for_sample(sample_idx)
        if not tokens_data:
            raise HTTPException(
                status_code=404,
                detail=f"No token mapping found for sample {sample_idx}",
            )

        tokens = [
            TokenInfo(
                abs_pos=t["abs_pos"],
                decoded_token=t["decoded_token"],
                traj_section=t["traj_section"],
                format_pos=t.get("format_pos"),
                rel_pos=t.get("rel_pos", -1),
            )
            for t in tokens_data
        ]

        return TokensResponse(sample_idx=sample_idx, tokens=tokens)

    return router
