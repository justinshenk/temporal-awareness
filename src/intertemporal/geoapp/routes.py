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


def align_pc_signs_continuity(pc_values: list[np.ndarray]) -> list[np.ndarray]:
    """Align PC signs using continuity (unbiased method).

    Works backwards from last layer (most structured) to ensure smooth trajectories.
    Flips signs where correlation with next layer is negative.

    Args:
        pc_values: List of PC projections, one per layer. Shape (n_samples,) each.

    Returns:
        Sign-aligned PC values (same structure, signs may be flipped).
    """
    if len(pc_values) <= 1:
        return pc_values

    aligned = [arr.copy() for arr in pc_values]

    # Work backwards from last layer (most structured)
    for i in range(len(aligned) - 2, -1, -1):
        # Correlation with next layer
        corr = np.corrcoef(aligned[i], aligned[i + 1])[0, 1]
        if np.isfinite(corr) and corr < 0:
            aligned[i] *= -1

    return aligned

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
    ExampleSample,
    HeatmapCell,
    HeatmapResponse,
    MetricsResponse,
    PCAMetrics,
    Point3D,
    ProbeMetrics,
    SampleResponse,
    TokenInfo,
    TokensResponse,
    Trajectory2DPoint,
    Trajectory2DResponse,
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


def _validate_pc_values(values: list[float], context: str) -> list[float]:
    """Validate PC values and raise error if NaN/Infinity found.

    Args:
        values: List of PC values to validate.
        context: Description of where these values came from (for error message).

    Returns:
        The same list of values as floats if all are valid.

    Raises:
        ValueError: If any value is NaN or Infinity.
    """
    for i, v in enumerate(values):
        if v is None or math.isnan(v) or math.isinf(v):
            raise ValueError(
                f"Invalid PC value at index {i} in {context}: {v}. "
                "This indicates a computation failure in PCA/embedding generation."
            )
    return [float(v) for v in values]


def _validate_rel_pos(position: str, data_loader: GeometryDataLoader) -> None:
    """Validate rel_pos index if present in position string.

    Args:
        position: Position string, optionally with :rel_pos suffix (e.g., "response_choice:2")
        data_loader: GeometryDataLoader instance to get rel_pos counts

    Raises:
        HTTPException: If rel_pos is out of bounds for the position
    """
    if ":" not in position:
        return  # No rel_pos suffix, nothing to validate

    parts = position.split(":")
    base_pos = parts[0]
    try:
        rel_pos = int(parts[1])
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid rel_pos format in position: {position}")

    rel_pos_counts = data_loader.get_rel_pos_counts()
    max_rel_pos = rel_pos_counts.get(base_pos, 0)
    if rel_pos >= max_rel_pos:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid rel_pos {rel_pos} for position {base_pos}. Valid range: 0-{max_rel_pos - 1}."
        )


def _validate_embedding(embedding: np.ndarray, context: str) -> np.ndarray:
    """Validate embedding array and raise error if NaN/Infinity found.

    Args:
        embedding: Numpy array of embedding coordinates.
        context: Description of where this embedding came from (for error message).

    Returns:
        The same embedding array if all values are valid.

    Raises:
        ValueError: If any value is NaN or Infinity.
    """
    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
        nan_count = np.sum(np.isnan(embedding))
        inf_count = np.sum(np.isinf(embedding))
        raise ValueError(
            f"Invalid embedding values in {context}: {nan_count} NaN, {inf_count} Infinity. "
            "This indicates a computation failure in PCA/UMAP/t-SNE generation."
        )
    return embedding


def create_router(data_loader: GeometryDataLoader, dataset_name: str = "geometry") -> APIRouter:
    """Create API router with endpoints bound to the given data loader.

    Args:
        data_loader: GeometryDataLoader instance for accessing embedding data.
        dataset_name: Name of this dataset, used as URL prefix.

    Returns:
        FastAPI APIRouter with all endpoints configured.
    """
    router = APIRouter(prefix=f"/api/{dataset_name}", tags=[dataset_name])

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

        # Get example sample with time horizon for UI illustration
        example_data = data_loader.get_example_sample_with_horizon()
        example_sample = None
        if example_data:
            example_sample = ExampleSample(
                sample_idx=example_data["sample_idx"],
                format_texts=example_data["format_texts"],
            )

        _log("/config", f"Returning config", n_layers=len(layers), n_positions=len(positions), n_samples=data_loader.n_samples, methods=available_methods)
        return ConfigResponse(
            layers=layers,
            components=data_loader.get_components(),
            positions=positions,
            color_options=data_loader.get_color_options(),
            n_samples=data_loader.n_samples,
            model_name=data_loader.get_model_name(),
            position_labels=data_loader.get_position_labels(),
            prompt_template=data_loader.get_prompt_template_structure(),
            semantic_to_positions={pos: [pos] for pos in data_loader.get_positions()},
            markers=data_loader.get_markers(),
            rel_pos_counts=data_loader.get_rel_pos_counts(),
            available_methods=available_methods,
            example_sample=example_sample,
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
            position: Token position identifier (supports format_pos:rel_pos syntax).
            method: Dimensionality reduction method (pca, umap, or tsne).

        Returns:
            3D coordinates for all samples.
        """
        _log("/embedding", f"GET request", layer=layer, component=component, position=position, method=method)
        start_time = time.time()

        # Parse position to extract base format_pos and optional rel_pos
        base_position, rel_pos = data_loader._parse_position(position)

        # Validate layer
        if layer not in data_loader.get_layers():
            raise HTTPException(status_code=404, detail=f"Layer {layer} not found")

        # Validate component
        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

        # Validate method is available (UMAP/t-SNE may not be precomputed)
        available_methods = data_loader.get_available_methods()
        if method not in available_methods:
            raise HTTPException(
                status_code=404,
                detail=f"Method '{method}' not available. Available methods: {available_methods}"
            )

        # Validate base position - must have precomputed embeddings
        precomputed_positions = data_loader.get_precomputed_positions()
        if base_position not in precomputed_positions:
            if base_position in data_loader.get_positions():
                # Position exists but no precomputed embedding
                raise HTTPException(
                    status_code=404,
                    detail=f"Position '{base_position}' exists but has no precomputed embeddings. "
                    f"Available positions: {precomputed_positions}"
                )
            raise HTTPException(status_code=404, detail=f"Position '{base_position}' not found")

        # Validate rel_pos if specified
        _validate_rel_pos(position, data_loader)

        # Load embedding based on method
        # Use full position (with rel_pos) if specified - per-rel_pos embeddings are precomputed
        load_position = position if rel_pos is not None else base_position
        used_fallback = False
        try:
            if method == "pca":
                embedding = data_loader.load_pca(layer, component, load_position, n_components=3)
            elif method == "umap":
                embedding = data_loader.load_umap(layer, component, load_position, n_components=3)
            elif method == "tsne":
                embedding = data_loader.load_tsne(layer, component, load_position, n_components=3)
            else:
                raise HTTPException(status_code=400, detail=f"Invalid method: {method}")
        except ValueError as e:
            # If per-rel_pos embedding doesn't exist, fall back to combined embedding
            if rel_pos is not None:
                _log("/embedding", f"Per-rel_pos embedding not found, falling back to combined", position=position, rel_pos=rel_pos)
                load_position = base_position
                used_fallback = True
                try:
                    if method == "pca":
                        embedding = data_loader.load_pca(layer, component, load_position, n_components=3)
                    elif method == "umap":
                        embedding = data_loader.load_umap(layer, component, load_position, n_components=3)
                    elif method == "tsne":
                        embedding = data_loader.load_tsne(layer, component, load_position, n_components=3)
                except ValueError:
                    raise HTTPException(status_code=404, detail=f"No embedding data for {base_position}.")
            else:
                raise HTTPException(status_code=404, detail=f"No embedding data for {position}.")

        # Validate embedding - crash if NaN/Infinity found (indicates computation failure)
        _validate_embedding(embedding, f"L{layer}_{component}_{position}_{method}")
        n_embedding_rows = embedding.shape[0]

        # Get sample indices for the position we loaded (either per-rel_pos or combined)
        # NOTE: We use embedding row count as authoritative since indices may mismatch
        all_sample_indices = data_loader.get_valid_sample_indices(layer, component, load_position)

        # Truncate/pad indices to match embedding rows
        if len(all_sample_indices) > n_embedding_rows:
            all_sample_indices = all_sample_indices[:n_embedding_rows]
        elif len(all_sample_indices) < n_embedding_rows:
            all_sample_indices = list(all_sample_indices) + list(range(len(all_sample_indices), n_embedding_rows))

        # No additional filtering needed - per-rel_pos embeddings are pre-filtered
        sample_indices = list(all_sample_indices)
        if rel_pos is not None:
            _log("/embedding", f"Loaded per-rel_pos embedding", rel_pos=rel_pos, n_samples=len(sample_indices))

        # Use embedding size as n_samples
        n_samples = embedding.shape[0]

        # Convert to flat list of floats for maximum performance
        # Format: [x0, y0, z0, x1, y1, z1, ...] - frontend will reshape
        coordinates_flat = embedding.flatten().tolist()

        # Set cache headers for browser caching (30 minutes)
        if response:
            response.headers["Cache-Control"] = "max-age=1800, stale-while-revalidate=3600"

        elapsed = time.time() - start_time
        _log("/embedding", f"Returning embedding", n_samples=n_samples, n_coords=len(coordinates_flat), elapsed_ms=f"{elapsed*1000:.1f}")

        return EmbeddingResponse(
            layer=layer,
            component=component,
            position=position,
            method=method,
            n_samples=n_samples,
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

        # Parse position to extract base format_pos and optional rel_pos
        base_position, rel_pos = data_loader._parse_position(position)

        # Validate inputs
        if layer not in data_loader.get_layers():
            raise HTTPException(status_code=404, detail=f"Layer {layer} not found")
        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")
        precomputed_positions = data_loader.get_precomputed_positions()
        if base_position not in precomputed_positions:
            if base_position in data_loader.get_positions():
                raise HTTPException(
                    status_code=404,
                    detail=f"Position '{base_position}' exists but has no precomputed embeddings"
                )
            raise HTTPException(status_code=404, detail=f"Position '{base_position}' not found")

        # Validate rel_pos if specified
        _validate_rel_pos(position, data_loader)

        # Validate method is available (UMAP/t-SNE may not be precomputed)
        available_methods = data_loader.get_available_methods()
        if method not in available_methods:
            async def error_generator():
                yield f"data: {json.dumps({'error': f'Method {method} not available. Available methods: {available_methods}'})}\n\n"
            return StreamingResponse(
                error_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        async def generate_chunks():
            """Generator that yields SSE chunks."""
            import asyncio

            stream_start = time.time()
            _log("/embedding/stream", f"Loading embedding...", method=method)

            # Load embedding using full position (with rel_pos) if specified
            load_position = position if rel_pos is not None else base_position
            used_fallback = False
            try:
                if method == "pca":
                    embedding = data_loader.load_pca(layer, component, load_position, n_components=3)
                elif method == "umap":
                    embedding = data_loader.load_umap(layer, component, load_position, n_components=3)
                elif method == "tsne":
                    embedding = data_loader.load_tsne(layer, component, load_position, n_components=3)
                else:
                    _log("/embedding/stream", f"ERROR: Invalid method {method}")
                    yield f"data: {json.dumps({'error': f'Invalid method: {method}'})}\n\n"
                    return
            except ValueError as e:
                # If per-rel_pos embedding doesn't exist, fall back to combined embedding
                if rel_pos is not None:
                    _log("/embedding/stream", f"Per-rel_pos embedding not found, falling back to combined", position=position, rel_pos=rel_pos)
                    load_position = base_position
                    used_fallback = True
                    try:
                        if method == "pca":
                            embedding = data_loader.load_pca(layer, component, load_position, n_components=3)
                        elif method == "umap":
                            embedding = data_loader.load_umap(layer, component, load_position, n_components=3)
                        elif method == "tsne":
                            embedding = data_loader.load_tsne(layer, component, load_position, n_components=3)
                    except ValueError:
                        _log("/embedding/stream", f"Embedding not found", position=base_position)
                        yield f"data: {json.dumps({'error': f'No embedding data for {base_position}.'})}\n\n"
                        return
                else:
                    _log("/embedding/stream", f"Embedding not found", position=load_position, error=str(e))
                    yield f"data: {json.dumps({'error': f'No embedding data for {position}.'})}\n\n"
                    return

            load_elapsed = time.time() - stream_start
            _log("/embedding/stream", f"Embedding loaded", elapsed_ms=f"{load_elapsed*1000:.1f}", shape=embedding.shape)

            # Validate embedding - crash if NaN/Infinity found (indicates computation failure)
            _validate_embedding(embedding, f"L{layer}_{component}_{position}_{method}")
            n_embedding_rows = embedding.shape[0]

            # Get sample indices for the position we loaded
            all_sample_indices = data_loader.get_valid_sample_indices(layer, component, load_position)

            # Truncate/pad indices to match embedding rows
            if len(all_sample_indices) > n_embedding_rows:
                all_sample_indices = all_sample_indices[:n_embedding_rows]
            elif len(all_sample_indices) < n_embedding_rows:
                all_sample_indices = list(all_sample_indices) + list(range(len(all_sample_indices), n_embedding_rows))

            # No additional filtering needed - per-rel_pos embeddings are pre-filtered
            sample_indices = list(all_sample_indices)
            if rel_pos is not None:
                _log("/embedding/stream", f"Loaded per-rel_pos embedding", rel_pos=rel_pos, n_samples=len(sample_indices))

            # Use embedding size as total_points
            total_points = embedding.shape[0]

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
                chunk_coords = embedding[i:end_idx].flatten().tolist()
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

        time_horizon_months = sample_info.get("time_horizon_months")

        # Extract known fields
        known_fields = {
            "text", "time_horizon_months", "time_scale", "choice_type", "short_term_first",
            "response_label", "response_term", "response_text", "choice_prob"
        }
        metadata = {k: v for k, v in sample_info.items() if k not in known_fields}

        # Handle text that may be a list (multi-part prompts)
        text = sample_info.get("text", "")
        if isinstance(text, list):
            text = "\n".join(text)

        # Compute time_scale at runtime from time_horizon_months (don't use pre-computed)
        TIME_SCALE_LABELS = ["Seconds", "Minutes", "Hours", "Days", "Weeks", "Months", "Years", "Decades", "Centuries"]
        if time_horizon_months is None:
            time_scale = "No Horizon"
        else:
            scale_idx = data_loader._get_time_scale(time_horizon_months)
            time_scale = TIME_SCALE_LABELS[scale_idx] if 0 <= scale_idx < len(TIME_SCALE_LABELS) else None

        # Convert choice_type from int index to string label if needed
        CHOICE_TYPE_LABELS = ["rational", "impulsive", "neutral"]
        choice_type_raw = sample_info.get("choice_type")
        if isinstance(choice_type_raw, int) and 0 <= choice_type_raw < len(CHOICE_TYPE_LABELS):
            choice_type = CHOICE_TYPE_LABELS[choice_type_raw]
        elif isinstance(choice_type_raw, str):
            choice_type = choice_type_raw
        else:
            choice_type = None

        return SampleResponse(
            idx=idx,
            text=text,
            time_horizon_months=time_horizon_months,
            time_scale=time_scale,
            choice_type=choice_type,
            short_term_first=sample_info.get("short_term_first"),
            response_label=sample_info.get("response_label"),
            response_term=sample_info.get("response_term"),
            response_text=sample_info.get("response_text"),
            choice_confidence=sample_info.get("choice_prob"),
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
            default="pca",
            description="Comma-separated methods to precompute (pca,umap,tsne) - only available methods will be used",
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

        # Parse parameters - filter by available methods
        available_methods = data_loader.get_available_methods()
        method_list = [m.strip() for m in methods.split(",") if m.strip() in available_methods]
        if not method_list:
            method_list = ["pca"] if "pca" in available_methods else []
        if not method_list:
            return WarmupResponse(
                message="No available methods to warm up",
                status=WarmupStatus(
                    is_running=False,
                    progress=0,
                    total=0,
                    current_task="No methods available",
                    cached_pca=len(data_loader._pca_cache),
                    cached_umap=len(data_loader._umap_cache),
                    cached_tsne=len(data_loader._tsne_cache),
                ),
            )

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

    @router.get("/trajectory/layers/{component}/{position:path}", response_model=TrajectoryResponse)
    async def get_layer_trajectory(
        component: str,
        position: str,
        mode: str = Query(default="aligned", description="PCA mode: 'aligned' (per-target PCA + sign align) or 'shared' (single PCA subspace)"),
    ) -> TrajectoryResponse:
        """Get PC1 trajectory across all layers for a given component/position.

        Returns PC1 values for each sample at each layer, enabling visualization
        of how representations evolve through the model.

        Position can be:
        - "response_choice" (combined/all tokens)
        - "response_choice:0" (specific rel_pos token)

        Mode can be:
        - "aligned": Per-target PCA with sign alignment (original method)
        - "shared": Single PCA subspace across all layers (new method)
        """
        _log("/trajectory/layers", f"GET request", component=component, position=position, mode=mode)
        start_time = time.time()
        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

        # Validate mode
        if mode not in ("aligned", "shared"):
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}. Must be 'aligned' or 'shared'.")

        # Parse position to extract base_pos for validation
        base_pos = position.split(":")[0] if ":" in position else position

        if base_pos not in data_loader.get_positions():
            raise HTTPException(status_code=404, detail=f"Position {base_pos} not found")

        # Validate rel_pos if specified
        _validate_rel_pos(position, data_loader)

        # Try to use pre-cached trajectory data first (handles both combined and per-relpos with fallback)
        try:
            layers, pc1_matrix, shared_sample_indices = data_loader.load_layer_trajectory(component, position, mode=mode)
            data = []
            x_values = []
            for i, layer in enumerate(layers):
                pc1_values = pc1_matrix[i].astype(float).tolist()
                x_values.append(str(layer))
                data.append(TrajectoryPoint(
                    x_value=str(layer),
                    values=_validate_pc_values(pc1_values, f"layer_trajectory L{layer}_{component}_{position}"),
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
        except ValueError as e:
            # NO FALLBACK - trajectory cache MUST exist
            raise HTTPException(
                status_code=404,
                detail=f"Trajectory cache not found for {component}/{position}. Run compute_geometry_analysis.py first. Error: {e}"
            )

    @router.get("/trajectory/positions/{layer}/{component}", response_model=TrajectoryResponse)
    async def get_position_trajectory(
        layer: int,
        component: str,
        positions_filter: str = Query(default="", description="Comma-separated positions to include (empty = all named)"),
        mode: str = Query(default="aligned", description="PCA mode: 'aligned' or 'shared'"),
    ) -> TrajectoryResponse:
        """Get PC1 trajectory across semantic positions for a given layer/component.

        Returns PC1 values for each sample at each position, enabling visualization
        of how representations vary across different parts of the prompt.
        """
        _log("/trajectory/positions", f"GET request", layer=layer, component=component, positions_filter=positions_filter or "default", mode=mode)
        start_time = time.time()
        if layer not in data_loader.get_layers():
            raise HTTPException(status_code=404, detail=f"Layer {layer} not found")

        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

        # Validate mode
        if mode not in ("aligned", "shared"):
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}. Must be 'aligned' or 'shared'.")

        # Use filtered positions or default to all positions (in actual prompt order)
        if positions_filter:
            positions = [p.strip() for p in positions_filter.split(",") if p.strip()]
        else:
            positions = data_loader.get_positions_in_prompt_order()

        # Try to use pre-cached trajectory data first (only if no filter applied)
        if not positions_filter:
            try:
                cached_positions, pc1_values_list, sample_indices_list = data_loader.load_position_trajectory(layer, component, mode=mode)
                # Build lookup from cached data
                cached_data = {
                    pos: (pc1_values_list[i], sample_indices_list[i] if i < len(sample_indices_list) else [])
                    for i, pos in enumerate(cached_positions)
                }
                # Reorder by actual prompt order
                prompt_order = data_loader.get_positions_in_prompt_order()
                ordered_positions = [p for p in prompt_order if p in cached_data]

                data = []
                x_values = []
                n_samples = 0
                for pos in ordered_positions:
                    pc1_arr, indices = cached_data[pos]
                    pc1_values = pc1_arr.astype(float).tolist() if hasattr(pc1_arr, 'astype') else list(pc1_arr)
                    n_samples = max(n_samples, len(pc1_values))
                    x_values.append(pos)
                    data.append(TrajectoryPoint(
                        x_value=pos,
                        values=_validate_pc_values(pc1_values, f"position_trajectory {pos}_{component}"),
                        sample_indices=indices,
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
            except ValueError as e:
                # NO FALLBACK - trajectory cache MUST exist
                raise HTTPException(
                    status_code=404,
                    detail=f"Position trajectory cache not found for L{layer}/{component}. Run compute_geometry_analysis.py first. Error: {e}"
                )

    # ==================== 2D TRAJECTORY ENDPOINTS (PC1 + PC2) ====================

    @router.get("/trajectory2d/layers/{component}/{position:path}", response_model=Trajectory2DResponse)
    async def get_layer_trajectory_2d(
        component: str,
        position: str,
        mode: str = Query(default="aligned", description="PCA mode: 'aligned' or 'shared'"),
    ) -> Trajectory2DResponse:
        """Get PC1+PC2 trajectory across all layers for a given component/position.

        Returns both PC1 and PC2 values for each sample at each layer, enabling
        3D visualization of how representations evolve through the model.
        X-axis: layer, Y-axis: PC1, Z-axis: PC2
        """
        _log("/trajectory2d/layers", f"GET request", component=component, position=position, mode=mode)
        start_time = time.time()

        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

        # Validate mode
        if mode not in ("aligned", "shared"):
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}. Must be 'aligned' or 'shared'.")

        # Parse position to extract base_pos and optional rel_pos
        base_pos = position.split(":")[0] if ":" in position else position
        if base_pos not in data_loader.get_positions():
            raise HTTPException(status_code=404, detail=f"Position {base_pos} not found")

        # Validate rel_pos if specified
        _validate_rel_pos(position, data_loader)

        # Try to use pre-cached trajectory data (handles both combined and per-relpos with fallback)
        try:
            result = data_loader.load_layer_trajectory(component, position, include_pc2=True, mode=mode)
            layers, pc1_matrix, pc2_matrix, shared_sample_indices = result
            data = []
            x_values = []
            for i, layer in enumerate(layers):
                pc1_values = pc1_matrix[i].astype(float).tolist()
                pc2_values = pc2_matrix[i].astype(float).tolist()
                x_values.append(str(layer))
                data.append(Trajectory2DPoint(
                    x_value=str(layer),
                    pc1_values=_validate_pc_values(pc1_values, f"trajectory2d_layers L{layer}_{component}_{position} PC1"),
                    pc2_values=_validate_pc_values(pc2_values, f"trajectory2d_layers L{layer}_{component}_{position} PC2"),
                    sample_indices=shared_sample_indices,
                ))
            n_samples = len(shared_sample_indices)
            elapsed = time.time() - start_time
            _log("/trajectory2d/layers", f"Returning cached 2D trajectory", n_layers=len(x_values), n_samples=n_samples, elapsed_ms=f"{elapsed*1000:.1f}")
            return Trajectory2DResponse(
                component=component,
                position=position,
                method="pca",
                x_axis="layer",
                x_values=x_values,
                n_samples=n_samples,
                sample_indices=shared_sample_indices,
                data=data,
            )
        except ValueError as e:
            # NO FALLBACK - trajectory cache MUST exist
            raise HTTPException(
                status_code=404,
                detail=f"2D layer trajectory cache not found for {component}/{position}. Run compute_geometry_analysis.py first. Error: {e}"
            )

    @router.get("/trajectory2d/positions/{layer}/{component}", response_model=Trajectory2DResponse)
    async def get_position_trajectory_2d(
        layer: int,
        component: str,
        positions_filter: str = Query(default="", description="Comma-separated positions to include"),
        mode: str = Query(default="aligned", description="PCA mode: 'aligned' or 'shared'"),
    ) -> Trajectory2DResponse:
        """Get PC1+PC2 trajectory across semantic positions for a given layer/component.

        Returns both PC1 and PC2 values for each sample at each position, enabling
        3D visualization of how representations vary across positions.
        X-axis: position, Y-axis: PC1, Z-axis: PC2
        """
        _log("/trajectory2d/positions", f"GET request", layer=layer, component=component, mode=mode)
        start_time = time.time()

        if layer not in data_loader.get_layers():
            raise HTTPException(status_code=404, detail=f"Layer {layer} not found")
        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

        # Validate mode
        if mode not in ("aligned", "shared"):
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}. Must be 'aligned' or 'shared'.")

        # Use named positions if no filter
        if positions_filter:
            positions = [p.strip() for p in positions_filter.split(",") if p.strip()]
        else:
            positions = data_loader.get_positions()

        # Try cached trajectory (only if no filter)
        if not positions_filter:
            try:
                result = data_loader.load_position_trajectory(layer, component, include_pc2=True, mode=mode)
                cached_positions, pc1_values_list, pc2_values_list, sample_indices_list = result
                data = []
                x_values = []
                n_samples = 0
                for i, pos in enumerate(cached_positions):
                    if pos not in positions:
                        continue
                    pc1_values = pc1_values_list[i].astype(float).tolist()
                    pc2_values = pc2_values_list[i].astype(float).tolist()
                    sample_indices = sample_indices_list[i]
                    n_samples = max(n_samples, len(pc1_values))
                    x_values.append(pos)
                    data.append(Trajectory2DPoint(
                        x_value=pos,
                        pc1_values=_validate_pc_values(pc1_values, f"trajectory2d_positions L{layer}_{component}_{pos} PC1"),
                        pc2_values=_validate_pc_values(pc2_values, f"trajectory2d_positions L{layer}_{component}_{pos} PC2"),
                        sample_indices=sample_indices,
                    ))
                elapsed = time.time() - start_time
                _log("/trajectory2d/positions", f"Returning cached 2D trajectory", n_positions=len(x_values), n_samples=n_samples, elapsed_ms=f"{elapsed*1000:.1f}")
                return Trajectory2DResponse(
                    layer=layer,
                    component=component,
                    method="pca",
                    x_axis="position",
                    x_values=x_values,
                    n_samples=n_samples,
                    sample_indices=[],
                    data=data,
                )
            except ValueError as e:
                # NO FALLBACK - trajectory cache MUST exist
                raise HTTPException(
                    status_code=404,
                    detail=f"2D position trajectory cache not found for L{layer}/{component}. Run compute_geometry_analysis.py first. Error: {e}"
                )

        # If positions_filter was provided, still need to load data
        # This is NOT a fallback - it's loading cached data with filtering
        result = data_loader.load_position_trajectory(layer, component, include_pc2=True, mode=mode)
        cached_positions, pc1_values_list, pc2_values_list, sample_indices_list = result
        data = []
        x_values = []
        n_samples = 0
        for i, pos in enumerate(cached_positions):
            if pos not in positions:
                continue
            pc1_values = pc1_values_list[i].astype(float).tolist()
            pc2_values = pc2_values_list[i].astype(float).tolist()
            sample_indices = sample_indices_list[i]
            n_samples = max(n_samples, len(pc1_values))
            x_values.append(pos)
            data.append(Trajectory2DPoint(
                x_value=pos,
                pc1_values=_validate_pc_values(pc1_values, f"trajectory2d_positions L{layer}_{component}_{pos} PC1"),
                pc2_values=_validate_pc_values(pc2_values, f"trajectory2d_positions L{layer}_{component}_{pos} PC2"),
                sample_indices=sample_indices,
            ))

        elapsed = time.time() - start_time
        _log("/trajectory2d/positions", f"Returning 2D trajectory (filtered)", n_positions=len(x_values), n_samples=n_samples, elapsed_ms=f"{elapsed*1000:.1f}")

        return Trajectory2DResponse(
            layer=layer,
            component=component,
            method="pca",
            x_axis="position",
            x_values=x_values,
            n_samples=n_samples,
            sample_indices=[],
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

    @router.get("/scree/{position:path}")
    async def get_scree_data(position: str, n_components: int = 10) -> dict:
        """Get Scree plot data (cumulative variance explained) for a position.

        Position can include rel_pos suffix (e.g., "response_choice:0") - will use
        combined position metrics since per-rel_pos metrics are not computed.

        Returns cumulative variance explained across all layers and components.
        """
        _log("/scree", f"GET request", position=position)
        # Validate base position (strip rel_pos suffix for validation)
        base_position = position.rsplit(":", 1)[0] if ":" in position else position
        if base_position not in data_loader.get_positions():
            raise HTTPException(status_code=400, detail=f"Invalid position: {position}")

        scree_data = data_loader.get_scree_data(position, n_components)
        return scree_data

    @router.get("/alignment/{position:path}")
    async def get_direction_alignment(position: str, pc_index: int = 0) -> dict:
        """Get direction alignment (cosine similarity) matrix for a position.

        Position can include rel_pos suffix (e.g., "response_choice:0") - will use
        combined position components since per-rel_pos components are not computed.

        Returns cosine similarity between PC directions across all targets.
        """
        _log("/alignment", f"GET request", position=position, pc_index=pc_index)
        # Validate base position (strip rel_pos suffix for validation)
        base_position = position.rsplit(":", 1)[0] if ":" in position else position
        if base_position not in data_loader.get_positions():
            raise HTTPException(status_code=400, detail=f"Invalid position: {position}")

        alignment_data = data_loader.get_direction_alignment(position, pc_index)
        return alignment_data

    return router
