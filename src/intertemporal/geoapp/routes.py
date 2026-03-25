"""API route definitions for GeoViz backend."""

import asyncio
import math
from typing import Literal

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from .data_loader import GeoVizDataLoader
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


def create_router(data_loader: GeoVizDataLoader) -> APIRouter:
    """Create API router with endpoints bound to the given data loader.

    Args:
        data_loader: GeoVizDataLoader instance for accessing embedding data.

    Returns:
        FastAPI APIRouter with all endpoints configured.
    """
    router = APIRouter(prefix="/api", tags=["geoviz"])

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
        Position labels are enriched with semantic region info for absolute positions.
        """
        return ConfigResponse(
            layers=data_loader.get_layers(),
            components=data_loader.get_components(),
            positions=data_loader.get_positions(),
            color_options=data_loader.get_color_options(),
            n_samples=data_loader.n_samples,
            model_name=data_loader.get_model_name(),
            position_labels=data_loader.get_enriched_position_labels(),
            prompt_template=data_loader.get_prompt_template_structure(),
            semantic_to_positions=data_loader.get_semantic_to_positions_mapping(),
            markers=data_loader.get_markers(),
        )

    @router.get("/embedding/{layer}/{component}/{position}", response_model=EmbeddingResponse)
    async def get_embedding(
        layer: int,
        component: str,
        position: str,
        method: Literal["pca", "umap", "tsne"] = Query(default="pca", description="Dimensionality reduction method"),
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
        # Validate layer
        if layer not in data_loader.get_layers():
            raise HTTPException(status_code=404, detail=f"Layer {layer} not found")

        # Validate component
        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

        # Validate position
        if position not in data_loader.get_positions():
            raise HTTPException(status_code=404, detail=f"Position {position} not found")

        # Load embedding based on method
        if method == "pca":
            embedding = data_loader.load_pca(layer, component, position, n_components=3)
        elif method == "umap":
            embedding = data_loader.load_umap(layer, component, position, n_components=3)
        elif method == "tsne":
            embedding = data_loader.load_tsne(layer, component, position, n_components=3)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid method: {method}")

        if embedding is None:
            raise HTTPException(
                status_code=404,
                detail=f"No embedding found for L{layer}_{component}_{position}",
            )

        # Convert numpy array to list of Point3D, sanitizing NaN/Infinity
        coordinates = [
            Point3D(
                x=_sanitize_float(row[0]) or 0.0,
                y=_sanitize_float(row[1]) or 0.0,
                z=_sanitize_float(row[2]) or 0.0,
            )
            for row in embedding
        ]

        return EmbeddingResponse(
            layer=layer,
            component=component,
            position=position,
            method=method,
            n_samples=len(coordinates),
            coordinates=coordinates,
        )

    @router.get("/metadata", response_model=ColorValues)
    async def get_metadata(
        color_by: str = Query(default="log_time_horizon", description="Metadata field to use for coloring"),
    ) -> ColorValues:
        """Get color values for all samples based on a metadata field.

        Args:
            color_by: Metadata field name (e.g., log_time_horizon, time_scale, choice_type).

        Returns:
            Color values for all samples with data type information.
        """
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
        layer: int = Query(description="Current layer"),
        component: str = Query(description="Current component"),
        position: str = Query(description="Current position"),
        method: str = Query(default="pca", description="Current method"),
    ) -> WarmupResponse:
        """Prefetch embeddings for adjacent layers/positions.

        Call this after navigating to a new view to preload nearby data
        for smoother navigation.

        This runs synchronously but is fast for PCA (uses cache).
        """
        layers = data_loader.get_layers()
        positions = data_loader.get_positions()

        # Find current index and prefetch +/- 2
        layer_idx = layers.index(layer) if layer in layers else 0
        adjacent_layers = [
            layers[i] for i in range(max(0, layer_idx - 2), min(len(layers), layer_idx + 3))
        ]

        # Prefetch adjacent positions (named positions only for speed)
        pos_idx = positions.index(position) if position in positions else 0
        adjacent_positions = [
            positions[i] for i in range(max(0, pos_idx - 1), min(len(positions), pos_idx + 2))
        ]

        # Quick prefetch for PCA only (UMAP/t-SNE are slow)
        prefetched = 0
        for l in adjacent_layers:
            for p in adjacent_positions:
                if method == "pca":
                    result = data_loader.load_pca(l, component, p)
                elif method == "umap":
                    result = data_loader.load_umap(l, component, p)
                elif method == "tsne":
                    result = data_loader.load_tsne(l, component, p)
                else:
                    result = None
                if result is not None:
                    prefetched += 1

        return WarmupResponse(
            message=f"Prefetched {prefetched} embeddings for adjacent layers/positions",
            status=WarmupStatus(
                is_running=False,
                progress=prefetched,
                total=prefetched,
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
        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

        if position not in data_loader.get_positions():
            raise HTTPException(status_code=404, detail=f"Position {position} not found")

        layers = data_loader.get_layers()
        data = []
        x_values = []
        n_samples = 0

        for layer in layers:
            embedding = data_loader.load_pca(layer, component, position, n_components=3)
            if embedding is not None:
                # Extract PC1 (first column) and normalize
                pc1_values = embedding[:, 0].astype(float).tolist()
                n_samples = len(pc1_values)
                x_values.append(str(layer))
                data.append(TrajectoryPoint(
                    x_value=str(layer),
                    values=[_sanitize_float(v) or 0.0 for v in pc1_values],
                ))

        return TrajectoryResponse(
            component=component,
            position=position,
            method="pca",
            x_axis="layer",
            x_values=x_values,
            n_samples=n_samples,
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
        if layer not in data_loader.get_layers():
            raise HTTPException(status_code=404, detail=f"Layer {layer} not found")

        if component not in data_loader.get_components():
            raise HTTPException(status_code=400, detail=f"Invalid component: {component}")

        # Use filtered positions or default to all positions
        if positions_filter:
            positions = [p.strip() for p in positions_filter.split(",") if p.strip()]
        else:
            positions = data_loader.get_positions()[:10]

        data = []
        x_values = []
        n_samples = 0

        for pos in positions:
            embedding = data_loader.load_pca(layer, component, pos, n_components=3)
            if embedding is not None:
                pc1_values = embedding[:, 0].astype(float).tolist()
                n_samples = len(pc1_values)
                x_values.append(pos)
                data.append(TrajectoryPoint(
                    x_value=pos,
                    values=[_sanitize_float(v) or 0.0 for v in pc1_values],
                ))

        return TrajectoryResponse(
            layer=layer,
            component=component,
            method="pca",
            x_axis="position",
            x_values=x_values,
            n_samples=n_samples,
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
