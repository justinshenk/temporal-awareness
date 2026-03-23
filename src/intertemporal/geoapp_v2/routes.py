"""API route definitions for GeoViz backend."""

from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ..geoapp.data_loader import GeoVizDataLoader

from .models import (
    ColorValues,
    ConfigResponse,
    EmbeddingResponse,
    MetricsResponse,
    PCAMetrics,
    Point3D,
    ProbeMetrics,
    SampleResponse,
)


def create_router(data_loader: GeoVizDataLoader) -> APIRouter:
    """Create API router with endpoints bound to the given data loader.

    Args:
        data_loader: GeoVizDataLoader instance for accessing embedding data.

    Returns:
        FastAPI APIRouter with all endpoints configured.
    """
    router = APIRouter(prefix="/api", tags=["geoviz"])

    @router.get("/config", response_model=ConfigResponse)
    async def get_config() -> ConfigResponse:
        """Get available configuration options for the visualization.

        Returns layers, components, positions, and color options available in the dataset.
        """
        return ConfigResponse(
            layers=data_loader.get_layers(),
            components=data_loader.get_components(),
            positions=data_loader.get_positions(),
            color_options=data_loader.get_color_options(),
            n_samples=data_loader.n_samples,
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
                detail=f"No embedding found for L{layer}_{component}_P{position}",
            )

        # Convert numpy array to list of Point3D
        coordinates = [
            Point3D(x=float(row[0]), y=float(row[1]), z=float(row[2]))
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
            values_list = [float(v) for v in values]
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

        # Extract known fields
        known_fields = {"text", "time_horizon_months", "time_scale", "choice_type", "short_term_first"}
        metadata = {k: v for k, v in sample_info.items() if k not in known_fields}

        return SampleResponse(
            idx=idx,
            text=sample_info.get("text", ""),
            time_horizon_months=sample_info.get("time_horizon_months"),
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

    return router
