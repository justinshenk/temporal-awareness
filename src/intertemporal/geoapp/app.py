"""Interactive Dash app for geometric visualization - Modern Light Pastel Theme.

IMPORTANT: Set numba threading layer before any imports that might trigger numba.
This must be at the very top of the file to prevent threading issues.
"""

import os

# Disable numba parallelism to avoid threading issues with Dash
# This is the safest approach on macOS ARM where TBB isn't available
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, clientside_callback, dcc, html

from .data_loader import GeoVizDataLoader

# Default data directory
DEFAULT_DATA_DIR = Path("out/geo_viz")

# Pastel color palette (from reference webapp)
COLORS = {
    "pink": "#FF6B9D",
    "purple": "#C678DD",
    "cyan": "#56B6C2",
    "gold": "#E5C07B",
    "mint": "#98C379",
    "blue": "#61AFEF",
    "coral": "#E06C75",
    "orange": "#D19A66",
    "text": "#4a3f5c",
    "text_dim": "#7a6b8a",
    "text_muted": "#9a8baa",
}

# Trace color sequence for plotly
PASTEL_COLORS = [
    COLORS["pink"],
    COLORS["purple"],
    COLORS["cyan"],
    COLORS["gold"],
    COLORS["mint"],
    COLORS["blue"],
    COLORS["coral"],
    COLORS["orange"],
]

# Component display names
COMPONENT_NAMES = {
    "resid_pre": "Residual Pre",
    "attn_out": "Attention Out",
    "mlp_out": "MLP Out",
    "resid_post": "Residual Post",
}

# Color option display labels with better descriptions
COLOR_OPTION_LABELS = {
    "log_time_horizon": "Log10 Time Horizon (months)",
    "time_horizon_months": "Time Horizon (months)",
    "time_scale": "Time Scale Category",
    "choice_type": "Choice Type",
    "short_term_first": "Short-Term Option First",
    "has_horizon": "Has Time Horizon",
    "sample_idx": "Sample Index",
}

# Tooltip descriptions for controls
TOOLTIP_TEXTS = {
    "method": {
        "pca": "Principal Component Analysis - Fast linear projection that preserves global variance. Best for initial exploration.",
        "umap": "Uniform Manifold Approximation - Non-linear method that preserves local structure. Good for finding clusters.",
        "tsne": "t-SNE - Non-linear method that emphasizes local neighborhoods. Best for visualization of cluster separations.",
    },
    "color_options": {
        "log_time_horizon": "Logarithm (base 10) of the time horizon in months. Useful for visualizing across orders of magnitude.",
        "time_horizon_months": "Raw time horizon value in months. Shows linear scale differences.",
        "time_scale": "Categorical grouping: immediate, short-term, medium-term, long-term.",
        "choice_type": "Type of intertemporal choice presented to the model.",
        "short_term_first": "Whether the short-term option appears first in the prompt.",
        "has_horizon": "Whether the sample has a defined time horizon (some samples are timeless).",
        "sample_idx": "Sequential index of the sample in the dataset.",
    },
    "no_horizon": "Samples without a defined time horizon (timeless scenarios). Displayed as gold diamonds when enabled.",
    "layer": "Transformer layer number. Earlier layers capture surface features, later layers capture more abstract representations.",
    "component": {
        "resid_pre": "Residual stream before attention and MLP at this layer.",
        "attn_out": "Output of the attention mechanism at this layer.",
        "mlp_out": "Output of the MLP/feed-forward network at this layer.",
        "resid_post": "Residual stream after attention and MLP (layer output).",
    },
    "position": "Token position in the sequence. 'response' is typically where the model makes its decision.",
}

# CSS for light pastel theme
CUSTOM_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }

:root {
    --pink: #FF6B9D;
    --purple: #C678DD;
    --cyan: #56B6C2;
    --gold: #E5C07B;
    --mint: #98C379;
    --blue: #61AFEF;
    --text: #4a3f5c;
    --text-dim: #7a6b8a;
    --text-muted: #9a8baa;
}

html, body {
    width: 100%;
    min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

body {
    background: linear-gradient(135deg, #fef6f9 0%, #f8f4ff 30%, #f0f7ff 60%, #fff5f0 100%);
    color: var(--text);
}

/* Modern card design */
.glass-card {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(180, 160, 200, 0.2) !important;
    border-radius: 24px !important;
    box-shadow: 0 8px 32px rgba(100, 80, 120, 0.08);
    transition: all 0.25s ease;
}

.glass-card:hover {
    box-shadow: 0 12px 40px rgba(100, 80, 120, 0.12);
}

.card-header {
    background: transparent !important;
    border-bottom: 1px solid rgba(180, 160, 200, 0.15) !important;
    color: var(--text) !important;
    font-weight: 600;
    padding: 16px 20px !important;
}

.card-body {
    padding: 20px !important;
}

/* Info panels */
.info-panel {
    background: linear-gradient(135deg, rgba(86, 182, 194, 0.1), rgba(198, 120, 221, 0.1));
    border-radius: 12px;
    padding: 14px 18px;
    font-size: 13px;
    color: var(--text);
}

/* Sample text display */
.sample-text {
    background: rgba(248, 244, 255, 0.8);
    border: 1px solid rgba(180, 160, 200, 0.2);
    border-radius: 12px;
    padding: 18px;
    font-family: 'Roboto Mono', 'SF Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    white-space: pre-wrap;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text);
}

/* Position badge */
.position-badge {
    background: linear-gradient(135deg, #FF6B9D, #C678DD);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    color: white;
    display: inline-block;
    margin-left: 8px;
}

/* Navbar styling */
.navbar {
    background: linear-gradient(135deg, rgba(255, 107, 157, 0.15) 0%, rgba(198, 120, 221, 0.18) 50%, rgba(97, 175, 239, 0.12) 100%) !important;
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(198, 120, 221, 0.25) !important;
    box-shadow: 0 4px 24px rgba(198, 120, 221, 0.12), 0 1px 3px rgba(100, 80, 120, 0.08);
    padding: 12px 0 !important;
}

.navbar-brand {
    color: var(--text) !important;
    font-weight: 800 !important;
    font-size: 1.75rem !important;
    letter-spacing: -0.02em;
    text-shadow: 0 1px 2px rgba(198, 120, 221, 0.15), 0 0 20px rgba(198, 120, 221, 0.08);
}

.navbar .badge {
    font-weight: 600;
    padding: 8px 14px;
    border-radius: 16px;
    font-size: 12px;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Badge styling */
.badge {
    font-weight: 600;
    padding: 6px 12px;
    border-radius: 12px;
}

.badge.bg-info {
    background: linear-gradient(135deg, #56B6C2, #61AFEF) !important;
}

.badge.bg-success {
    background: linear-gradient(135deg, #98C379, #56B6C2) !important;
}

.badge.bg-warning {
    background: linear-gradient(135deg, #E5C07B, #D19A66) !important;
    color: white !important;
}

/* Tab styling - Modern pastel glass aesthetic */
.nav-tabs {
    border-bottom: none !important;
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(12px);
    padding: 8px 12px 0 12px;
    border-radius: 20px 20px 0 0;
    gap: 6px;
    display: flex;
    border: 1px solid rgba(180, 160, 200, 0.15);
    border-bottom: none;
    box-shadow: 0 -4px 20px rgba(100, 80, 120, 0.04);
}

.nav-tabs .nav-item {
    margin-bottom: 0;
}

.nav-tabs .nav-link {
    color: var(--text-dim) !important;
    border: 1px solid transparent !important;
    padding: 14px 28px !important;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.01em;
    border-radius: 16px 16px 0 0 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    background: transparent;
    position: relative;
    overflow: hidden;
    margin-bottom: -1px;
}

/* Subtle shine effect on tabs */
.nav-tabs .nav-link::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.4),
        transparent
    );
    transition: left 0.5s ease;
}

.nav-tabs .nav-link:hover::before {
    left: 100%;
}

.nav-tabs .nav-link:hover {
    color: var(--purple) !important;
    background: linear-gradient(
        135deg,
        rgba(198, 120, 221, 0.08) 0%,
        rgba(255, 107, 157, 0.06) 50%,
        rgba(97, 175, 239, 0.05) 100%
    ) !important;
    border-color: rgba(198, 120, 221, 0.2) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(198, 120, 221, 0.1);
}

.nav-tabs .nav-link:focus {
    box-shadow: 0 0 0 3px rgba(198, 120, 221, 0.15) !important;
    outline: none;
}

.nav-tabs .nav-link.active {
    color: white !important;
    background: linear-gradient(
        135deg,
        #FF6B9D 0%,
        #C678DD 50%,
        #9b7cb8 100%
    ) !important;
    border-color: transparent !important;
    box-shadow:
        0 4px 16px rgba(255, 107, 157, 0.35),
        0 2px 8px rgba(198, 120, 221, 0.25),
        inset 0 1px 0 rgba(255, 255, 255, 0.25);
    transform: translateY(-2px);
}

.nav-tabs .nav-link.active::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #FF6B9D, #C678DD);
    border-radius: 3px 3px 0 0;
}

/* Tab content area connection */
.tab-content {
    background: rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(180, 160, 200, 0.15);
    border-top: none;
    border-radius: 0 0 20px 20px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(100, 80, 120, 0.06);
}

/* Active tab glow animation */
@keyframes tab-glow {
    0%, 100% {
        box-shadow:
            0 4px 16px rgba(255, 107, 157, 0.35),
            0 2px 8px rgba(198, 120, 221, 0.25);
    }
    50% {
        box-shadow:
            0 6px 20px rgba(255, 107, 157, 0.45),
            0 3px 10px rgba(198, 120, 221, 0.35);
    }
}

.nav-tabs .nav-link.active:hover {
    animation: tab-glow 2s ease-in-out infinite;
}

/* Dropdown styling - light theme with proper contrast */
.Select-control, .Select-menu-outer {
    background: #ffffff !important;
    border-color: rgba(180, 160, 200, 0.4) !important;
    border-radius: 12px !important;
}

.Select-control {
    min-height: 40px !important;
    box-shadow: 0 2px 8px rgba(100, 80, 120, 0.06) !important;
}

.Select-control:hover {
    border-color: rgba(198, 120, 221, 0.5) !important;
}

.Select-value-label, .Select-placeholder {
    color: #4a3f5c !important;
}

.Select-placeholder {
    color: #9a8baa !important;
}

.Select-arrow {
    border-color: #7a6b8a transparent transparent !important;
}

.Select-input input {
    color: #4a3f5c !important;
}

.Select-menu {
    background: #ffffff !important;
    border-radius: 12px !important;
}

.Select-menu-outer {
    margin-top: 4px !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 24px rgba(100, 80, 120, 0.15) !important;
    overflow: hidden !important;
}

.Select-option {
    background: #ffffff !important;
    color: #4a3f5c !important;
    padding: 10px 14px !important;
}

.Select-option:hover, .Select-option.is-focused {
    background: rgba(198, 120, 221, 0.12) !important;
    color: #4a3f5c !important;
}

.Select-option.is-selected {
    background: linear-gradient(135deg, #C678DD, #9b7cb8) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

.is-focused .Select-control {
    border-color: #C678DD !important;
    box-shadow: 0 0 0 3px rgba(198, 120, 221, 0.2) !important;
}

/* Dash dropdown overrides */
.dash-dropdown .Select-control {
    background: #ffffff !important;
    border: 1px solid rgba(180, 160, 200, 0.4) !important;
    border-radius: 12px !important;
    min-height: 40px;
    box-shadow: 0 2px 8px rgba(100, 80, 120, 0.06);
}

.dash-dropdown .Select-control:hover {
    border-color: rgba(198, 120, 221, 0.5) !important;
}

.dash-dropdown .is-focused .Select-control {
    border-color: #C678DD !important;
    box-shadow: 0 0 0 3px rgba(198, 120, 221, 0.2) !important;
}

.dash-dropdown .Select-value-label {
    color: #4a3f5c !important;
}

.dash-dropdown .Select-placeholder {
    color: #9a8baa !important;
}

.dash-dropdown .Select-input input {
    color: #4a3f5c !important;
}

.dash-dropdown .Select-menu-outer {
    background: #ffffff !important;
    border: 1px solid rgba(180, 160, 200, 0.3) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 24px rgba(100, 80, 120, 0.15);
    margin-top: 4px;
    overflow: hidden;
}

.dash-dropdown .Select-menu {
    background: #ffffff !important;
    max-height: 280px !important;
}

.dash-dropdown .VirtualizedSelectOption {
    background: #ffffff !important;
    color: #4a3f5c !important;
    padding: 10px 14px !important;
}

.dash-dropdown .VirtualizedSelectOption:hover,
.dash-dropdown .VirtualizedSelectFocusedOption {
    background: rgba(198, 120, 221, 0.12) !important;
    color: #4a3f5c !important;
}

.dash-dropdown .VirtualizedSelectSelectedOption {
    background: linear-gradient(135deg, #C678DD, #9b7cb8) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Multi-select tag styling */
.Select-value {
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.15), rgba(255, 107, 157, 0.15)) !important;
    border: 1px solid rgba(198, 120, 221, 0.3) !important;
    border-radius: 8px !important;
    color: #4a3f5c !important;
}

.Select-value-icon {
    border-right: 1px solid rgba(198, 120, 221, 0.3) !important;
}

.Select-value-icon:hover {
    background: rgba(224, 108, 117, 0.2) !important;
    color: #c05060 !important;
}

/* Clear indicator */
.Select-clear {
    color: #9a8baa !important;
}

.Select-clear:hover {
    color: #c05060 !important;
}

/* ========================================
   Radio Buttons & Checkboxes - Modern Pastel Style
   ======================================== */

/* Base form-check container */
.form-check {
    position: relative;
    padding-left: 0;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Inline radio/checkbox layout */
.form-check-inline {
    display: inline-flex;
    align-items: center;
    margin-right: 16px;
}

/* Hide default input */
.form-check-input {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
    margin: 0;
}

/* Custom radio button styling */
.form-check-input[type="radio"] + .form-check-label {
    position: relative;
    padding-left: 32px;
    cursor: pointer;
    color: var(--text) !important;
    font-weight: 500;
    font-size: 14px;
    transition: color 0.2s ease;
    user-select: none;
    display: flex;
    align-items: center;
    min-height: 24px;
}

/* Radio button outer circle */
.form-check-input[type="radio"] + .form-check-label::before {
    content: '';
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 22px;
    height: 22px;
    border: 2px solid rgba(180, 160, 200, 0.5);
    border-radius: 50%;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 244, 255, 0.9));
    box-shadow: 0 2px 6px rgba(100, 80, 120, 0.08), inset 0 1px 2px rgba(255, 255, 255, 0.8);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Radio button inner dot */
.form-check-input[type="radio"] + .form-check-label::after {
    content: '';
    position: absolute;
    left: 6px;
    top: 50%;
    transform: translateY(-50%) scale(0);
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--purple), var(--pink));
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    opacity: 0;
}

/* Radio hover state */
.form-check-input[type="radio"]:not(:checked):not(:disabled) + .form-check-label:hover::before {
    border-color: rgba(198, 120, 221, 0.6);
    background: linear-gradient(135deg, rgba(255, 255, 255, 1), rgba(248, 244, 255, 1));
    box-shadow: 0 3px 10px rgba(198, 120, 221, 0.15), inset 0 1px 2px rgba(255, 255, 255, 0.9);
}

/* Radio checked state */
.form-check-input[type="radio"]:checked + .form-check-label::before {
    border-color: var(--purple);
    background: linear-gradient(135deg, rgba(255, 255, 255, 1), rgba(248, 244, 255, 1));
    box-shadow: 0 3px 12px rgba(198, 120, 221, 0.25), inset 0 1px 2px rgba(255, 255, 255, 0.9);
}

.form-check-input[type="radio"]:checked + .form-check-label::after {
    transform: translateY(-50%) scale(1);
    opacity: 1;
}

.form-check-input[type="radio"]:checked + .form-check-label {
    color: var(--purple) !important;
}

/* Radio focus state */
.form-check-input[type="radio"]:focus + .form-check-label::before {
    outline: none;
    border-color: var(--purple);
    box-shadow: 0 0 0 4px rgba(198, 120, 221, 0.2), 0 3px 10px rgba(198, 120, 221, 0.15);
}

/* Radio disabled state */
.form-check-input[type="radio"]:disabled + .form-check-label {
    opacity: 0.5;
    cursor: not-allowed;
}

/* ========================================
   Switch/Toggle Styling (pill-shaped)
   ======================================== */

/* Switch container */
.form-switch {
    padding-left: 0 !important;
}

.form-switch .form-check-input {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
}

/* Switch label with track */
.form-switch .form-check-label {
    position: relative;
    padding-left: 56px;
    cursor: pointer;
    color: var(--text) !important;
    font-weight: 500;
    font-size: 14px;
    transition: color 0.2s ease;
    user-select: none;
    display: flex;
    align-items: center;
    min-height: 28px;
}

/* Switch track (pill shape) */
.form-switch .form-check-label::before {
    content: '';
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 48px;
    height: 26px;
    border-radius: 13px;
    background: linear-gradient(135deg, rgba(180, 160, 200, 0.3), rgba(200, 180, 220, 0.4));
    border: 1px solid rgba(180, 160, 200, 0.4);
    box-shadow: inset 0 2px 4px rgba(100, 80, 120, 0.1), 0 1px 2px rgba(255, 255, 255, 0.8);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Switch knob (circular) */
.form-switch .form-check-label::after {
    content: '';
    position: absolute;
    left: 3px;
    top: 50%;
    transform: translateY(-50%);
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: linear-gradient(135deg, #ffffff, #f8f4ff);
    box-shadow: 0 2px 6px rgba(100, 80, 120, 0.2), 0 1px 2px rgba(0, 0, 0, 0.1);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Switch hover state (unchecked) */
.form-switch .form-check-input:not(:checked):not(:disabled) + .form-check-label:hover::before {
    background: linear-gradient(135deg, rgba(180, 160, 200, 0.4), rgba(200, 180, 220, 0.5));
    border-color: rgba(198, 120, 221, 0.5);
}

.form-switch .form-check-input:not(:checked):not(:disabled) + .form-check-label:hover::after {
    box-shadow: 0 3px 8px rgba(100, 80, 120, 0.25), 0 1px 3px rgba(0, 0, 0, 0.15);
}

/* Switch checked state */
.form-switch .form-check-input:checked + .form-check-label::before {
    background: linear-gradient(135deg, var(--purple), var(--pink)) !important;
    border-color: transparent;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1), 0 2px 8px rgba(198, 120, 221, 0.3);
}

.form-switch .form-check-input:checked + .form-check-label::after {
    left: 25px;
    background: linear-gradient(135deg, #ffffff, #fff5f8);
    box-shadow: 0 2px 8px rgba(198, 120, 221, 0.3), 0 1px 3px rgba(0, 0, 0, 0.15);
}

.form-switch .form-check-input:checked + .form-check-label {
    color: var(--purple) !important;
}

/* Switch focus state */
.form-switch .form-check-input:focus + .form-check-label::before {
    outline: none;
    box-shadow: 0 0 0 4px rgba(198, 120, 221, 0.2), inset 0 2px 4px rgba(100, 80, 120, 0.1);
}

.form-switch .form-check-input:checked:focus + .form-check-label::before {
    box-shadow: 0 0 0 4px rgba(198, 120, 221, 0.2), inset 0 2px 4px rgba(0, 0, 0, 0.1), 0 2px 8px rgba(198, 120, 221, 0.3);
}

/* Switch disabled state */
.form-switch .form-check-input:disabled + .form-check-label {
    opacity: 0.5;
    cursor: not-allowed;
}

/* ========================================
   Standard Checkbox Styling
   ======================================== */

.form-check-input[type="checkbox"]:not(.form-switch .form-check-input) + .form-check-label {
    position: relative;
    padding-left: 32px;
    cursor: pointer;
    color: var(--text) !important;
    font-weight: 500;
    font-size: 14px;
    transition: color 0.2s ease;
    user-select: none;
    display: flex;
    align-items: center;
    min-height: 24px;
}

/* Checkbox box */
.form-check-input[type="checkbox"]:not(.form-switch .form-check-input) + .form-check-label::before {
    content: '';
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 22px;
    height: 22px;
    border: 2px solid rgba(180, 160, 200, 0.5);
    border-radius: 6px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 244, 255, 0.9));
    box-shadow: 0 2px 6px rgba(100, 80, 120, 0.08), inset 0 1px 2px rgba(255, 255, 255, 0.8);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Checkbox checkmark */
.form-check-input[type="checkbox"]:not(.form-switch .form-check-input) + .form-check-label::after {
    content: '';
    position: absolute;
    left: 5px;
    top: 50%;
    width: 6px;
    height: 11px;
    border: solid transparent;
    border-width: 0 2.5px 2.5px 0;
    transform: translateY(-60%) rotate(45deg) scale(0);
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Checkbox hover state */
.form-check-input[type="checkbox"]:not(:checked):not(:disabled):not(.form-switch .form-check-input) + .form-check-label:hover::before {
    border-color: rgba(198, 120, 221, 0.6);
    background: linear-gradient(135deg, rgba(255, 255, 255, 1), rgba(248, 244, 255, 1));
    box-shadow: 0 3px 10px rgba(198, 120, 221, 0.15), inset 0 1px 2px rgba(255, 255, 255, 0.9);
}

/* Checkbox checked state */
.form-check-input[type="checkbox"]:checked:not(.form-switch .form-check-input) + .form-check-label::before {
    background: linear-gradient(135deg, var(--purple), var(--pink)) !important;
    border-color: transparent;
    box-shadow: 0 3px 12px rgba(198, 120, 221, 0.3);
}

.form-check-input[type="checkbox"]:checked:not(.form-switch .form-check-input) + .form-check-label::after {
    border-color: white;
    transform: translateY(-60%) rotate(45deg) scale(1);
}

.form-check-input[type="checkbox"]:checked:not(.form-switch .form-check-input) + .form-check-label {
    color: var(--purple) !important;
}

/* Checkbox focus state */
.form-check-input[type="checkbox"]:focus:not(.form-switch .form-check-input) + .form-check-label::before {
    outline: none;
    border-color: var(--purple);
    box-shadow: 0 0 0 4px rgba(198, 120, 221, 0.2), 0 3px 10px rgba(198, 120, 221, 0.15);
}

/* Checkbox disabled state */
.form-check-input[type="checkbox"]:disabled:not(.form-switch .form-check-input) + .form-check-label {
    opacity: 0.5;
    cursor: not-allowed;
}

/* ========================================
   Bootstrap dbc.RadioItems / dbc.Checklist overrides
   ======================================== */

/* For dbc components which use slightly different structure */
.form-check.form-check-inline {
    margin-right: 20px;
}

/* Ensure proper alignment in card headers */
.card-body .form-check {
    margin-bottom: 12px;
}

.card-body .form-check:last-child {
    margin-bottom: 0;
}

/* Animation keyframes for check appearance */
@keyframes checkmark-appear {
    0% {
        transform: translateY(-60%) rotate(45deg) scale(0);
        opacity: 0;
    }
    50% {
        transform: translateY(-60%) rotate(45deg) scale(1.2);
    }
    100% {
        transform: translateY(-60%) rotate(45deg) scale(1);
        opacity: 1;
    }
}

@keyframes radio-dot-appear {
    0% {
        transform: translateY(-50%) scale(0);
        opacity: 0;
    }
    50% {
        transform: translateY(-50%) scale(1.3);
    }
    100% {
        transform: translateY(-50%) scale(1);
        opacity: 1;
    }
}

/* Apply animations */
.form-check-input[type="radio"]:checked + .form-check-label::after {
    animation: radio-dot-appear 0.3s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}

.form-check-input[type="checkbox"]:checked:not(.form-switch .form-check-input) + .form-check-label::after {
    animation: checkmark-appear 0.3s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}

/* Slider styling */
.rc-slider {
    padding: 8px 0 !important;
    margin: 12px 0 !important;
}

.rc-slider-rail {
    background: linear-gradient(90deg, rgba(248, 244, 255, 0.8), rgba(240, 247, 255, 0.8)) !important;
    height: 8px !important;
    border-radius: 4px !important;
    border: 1px solid rgba(180, 160, 200, 0.25) !important;
}

.rc-slider-track {
    background: linear-gradient(90deg, #C678DD, #FF6B9D) !important;
    height: 8px !important;
    border-radius: 4px !important;
    box-shadow: 0 2px 8px rgba(198, 120, 221, 0.25) !important;
}

.rc-slider-handle {
    width: 20px !important;
    height: 20px !important;
    margin-top: -6px !important;
    background: linear-gradient(135deg, #ffffff 0%, #f8f4ff 100%) !important;
    border: 3px solid var(--purple) !important;
    border-radius: 50% !important;
    box-shadow: 0 3px 10px rgba(198, 120, 221, 0.35), 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    cursor: grab !important;
    transition: all 0.2s ease !important;
}

.rc-slider-handle:hover {
    background: linear-gradient(135deg, #ffffff 0%, #fff5f8 100%) !important;
    border-color: var(--pink) !important;
    box-shadow: 0 4px 14px rgba(255, 107, 157, 0.4), 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    transform: scale(1.1) !important;
}

.rc-slider-handle:active,
.rc-slider-handle-dragging {
    background: linear-gradient(135deg, #fff5f8 0%, #fef0f5 100%) !important;
    border-color: var(--pink) !important;
    box-shadow: 0 2px 8px rgba(255, 107, 157, 0.5), 0 1px 2px rgba(0, 0, 0, 0.15) !important;
    cursor: grabbing !important;
    transform: scale(1.05) !important;
}

.rc-slider-handle:focus {
    outline: none !important;
    box-shadow: 0 0 0 4px rgba(198, 120, 221, 0.2), 0 3px 10px rgba(198, 120, 221, 0.35) !important;
}

/* Tick marks / dots */
.rc-slider-step {
    height: 8px !important;
}

.rc-slider-dot {
    width: 10px !important;
    height: 10px !important;
    bottom: -1px !important;
    background: rgba(255, 255, 255, 0.95) !important;
    border: 2px solid rgba(180, 160, 200, 0.4) !important;
    border-radius: 50% !important;
    transition: all 0.2s ease !important;
}

.rc-slider-dot-active {
    background: linear-gradient(135deg, #C678DD, #FF6B9D) !important;
    border-color: var(--purple) !important;
    box-shadow: 0 1px 4px rgba(198, 120, 221, 0.3) !important;
}

/* Mark text (labels below slider) */
.rc-slider-mark {
    top: 22px !important;
}

.rc-slider-mark-text {
    color: var(--text-muted) !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    transition: color 0.2s ease !important;
}

.rc-slider-mark-text-active {
    color: var(--purple) !important;
    font-weight: 600 !important;
}

/* Tooltip styling */
.rc-slider-tooltip {
    padding: 0 !important;
}

.rc-slider-tooltip-inner {
    background: linear-gradient(135deg, #C678DD, #9b7cb8) !important;
    color: white !important;
    padding: 6px 12px !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(198, 120, 221, 0.35), 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    min-width: 32px !important;
    text-align: center !important;
}

.rc-slider-tooltip-arrow {
    border-top-color: #C678DD !important;
}

.rc-slider-tooltip-placement-top .rc-slider-tooltip-arrow {
    border-top-color: #9b7cb8 !important;
}

/* Disabled state */
.rc-slider-disabled {
    opacity: 0.5 !important;
}

.rc-slider-disabled .rc-slider-handle {
    cursor: not-allowed !important;
}

/* Loading overlay and spinner */
._dash-loading {
    position: relative;
}

._dash-loading-callback {
    position: absolute !important;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.88) !important;
    backdrop-filter: blur(12px);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    border-radius: 20px;
}

.dash-spinner {
    color: var(--purple) !important;
}

/* Custom spinner animations */
@keyframes spin-gradient {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@keyframes pulse-glow {
    0%, 100% {
        box-shadow: 0 0 20px rgba(198, 120, 221, 0.4),
                    0 0 40px rgba(86, 182, 194, 0.2);
    }
    50% {
        box-shadow: 0 0 30px rgba(198, 120, 221, 0.6),
                    0 0 60px rgba(86, 182, 194, 0.4);
    }
}

@keyframes fade-pulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}

/* Dash default loading override - circle spinner with gradient */
.dash-spinner.dash-spinner--circle {
    width: 50px !important;
    height: 50px !important;
    border-width: 3px !important;
    border-style: solid !important;
    border-color: rgba(198, 120, 221, 0.15) !important;
    border-top-color: var(--purple) !important;
    border-right-color: var(--cyan) !important;
    border-radius: 50% !important;
    animation: spin-gradient 0.9s linear infinite !important;
    box-shadow: 0 0 20px rgba(198, 120, 221, 0.2);
}

/* Loading container with overlay effect */
.dash-loading-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
}

/* Loading text styling */
.loading-text-container {
    text-align: center;
    margin-top: 12px;
}

.loading-text {
    color: var(--text);
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.02em;
    animation: fade-pulse 2s ease-in-out infinite;
}

.loading-subtext {
    color: var(--text-muted);
    font-size: 12px;
    margin-top: 4px;
}

/* Progress bar for long computations (UMAP/t-SNE) */
.loading-progress {
    width: 120px;
    height: 4px;
    background: rgba(180, 160, 200, 0.2);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 8px;
}

.loading-progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--cyan), var(--purple), var(--pink), var(--cyan));
    background-size: 300% 100%;
    animation: progress-shimmer 2s ease-in-out infinite;
    border-radius: 2px;
}

@keyframes progress-shimmer {
    0% { background-position: 100% 0; }
    100% { background-position: -100% 0; }
}

/* Input styling */
.form-control {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(180, 160, 200, 0.3) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

.form-control:focus {
    border-color: var(--purple) !important;
    box-shadow: 0 0 0 3px rgba(198, 120, 221, 0.2) !important;
}

/* Labels */
label, .form-label {
    color: var(--text) !important;
    font-weight: 600;
    font-size: 13px;
}

/* Error message styling */
.error-message {
    background: rgba(224, 108, 117, 0.15);
    border: 1px solid rgba(224, 108, 117, 0.3);
    border-radius: 12px;
    padding: 16px 20px;
    color: #c05060;
    font-size: 14px;
    text-align: center;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(248, 244, 255, 0.5);
}

::-webkit-scrollbar-thumb {
    background: rgba(198, 120, 221, 0.4);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(198, 120, 221, 0.6);
}

/* Metrics panel */
.metrics-item {
    padding: 12px 14px;
    border-bottom: 1px solid rgba(180, 160, 200, 0.15);
    display: flex;
    align-items: center;
    gap: 12px;
    transition: background 0.2s ease;
    border-radius: 8px;
    margin-bottom: 4px;
}

.metrics-item:hover {
    background: rgba(248, 244, 255, 0.5);
}

.metrics-item:last-child {
    border-bottom: none;
    margin-bottom: 0;
}

.metrics-icon {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}

.metrics-icon.probe {
    background: linear-gradient(135deg, rgba(86, 182, 194, 0.2), rgba(97, 175, 239, 0.2));
    color: var(--cyan);
}

.metrics-icon.pca {
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.2), rgba(255, 107, 157, 0.2));
    color: var(--purple);
}

.metrics-content {
    flex: 1;
}

.metrics-label {
    color: var(--text-dim);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 2px;
}

.metrics-value {
    color: var(--text);
    font-size: 20px;
    font-weight: 700;
    font-family: 'Roboto Mono', monospace;
}

.metrics-subtext {
    color: var(--text-muted);
    font-size: 11px;
    margin-top: 2px;
}

/* Info icon for tooltips */
.info-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.15), rgba(86, 182, 194, 0.15));
    color: var(--purple);
    font-size: 10px;
    font-weight: 700;
    cursor: help;
    margin-left: 6px;
    transition: all 0.2s ease;
    vertical-align: middle;
}

.info-icon:hover {
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.25), rgba(86, 182, 194, 0.25));
    transform: scale(1.1);
}

/* Help card styling */
.help-card {
    background: linear-gradient(135deg, rgba(86, 182, 194, 0.08), rgba(198, 120, 221, 0.08), rgba(255, 107, 157, 0.05));
    border: 1px solid rgba(180, 160, 200, 0.2);
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.help-card-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.help-card-content {
    font-size: 12px;
    color: var(--text-dim);
    line-height: 1.6;
}

.help-card-content ul {
    margin: 8px 0 0 0;
    padding-left: 20px;
}

.help-card-content li {
    margin-bottom: 4px;
}

/* Bootstrap tooltip styling override for pastel theme */
.tooltip-inner {
    background: linear-gradient(135deg, #4a3f5c, #3a2f4c) !important;
    color: #ffffff !important;
    font-size: 12px !important;
    padding: 10px 14px !important;
    border-radius: 10px !important;
    max-width: 280px !important;
    text-align: left !important;
    box-shadow: 0 8px 24px rgba(74, 63, 92, 0.3) !important;
}

.tooltip.bs-tooltip-top .tooltip-arrow::before,
.tooltip.bs-tooltip-auto[data-popper-placement^="top"] .tooltip-arrow::before {
    border-top-color: #4a3f5c !important;
}

.tooltip.bs-tooltip-bottom .tooltip-arrow::before,
.tooltip.bs-tooltip-auto[data-popper-placement^="bottom"] .tooltip-arrow::before {
    border-bottom-color: #4a3f5c !important;
}

.tooltip.bs-tooltip-start .tooltip-arrow::before,
.tooltip.bs-tooltip-auto[data-popper-placement^="left"] .tooltip-arrow::before {
    border-left-color: #4a3f5c !important;
}

.tooltip.bs-tooltip-end .tooltip-arrow::before,
.tooltip.bs-tooltip-auto[data-popper-placement^="right"] .tooltip-arrow::before {
    border-right-color: #4a3f5c !important;
}

/* Keyboard shortcuts panel */
.keyboard-shortcuts-toggle {
    position: fixed;
    bottom: 24px;
    right: 24px;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--purple), var(--pink));
    border: none;
    color: white;
    font-size: 20px;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 4px 16px rgba(198, 120, 221, 0.4);
    transition: all 0.3s ease;
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
}

.keyboard-shortcuts-toggle:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 24px rgba(198, 120, 221, 0.5);
}

.keyboard-shortcuts-panel {
    position: fixed;
    bottom: 84px;
    right: 24px;
    width: 320px;
    max-height: 70vh;
    overflow-y: auto;
    background: rgba(255, 255, 255, 0.98);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(180, 160, 200, 0.3);
    border-radius: 20px;
    box-shadow: 0 12px 40px rgba(100, 80, 120, 0.2);
    z-index: 999;
    padding: 0;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.keyboard-shortcuts-panel.hidden {
    opacity: 0;
    transform: translateY(20px) scale(0.95);
    pointer-events: none;
}

.keyboard-shortcuts-panel.visible {
    opacity: 1;
    transform: translateY(0) scale(1);
}

.shortcuts-header {
    padding: 16px 20px;
    border-bottom: 1px solid rgba(180, 160, 200, 0.2);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.shortcuts-header h4 {
    margin: 0;
    font-size: 15px;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 8px;
}

.shortcuts-close {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 20px;
    padding: 4px;
    line-height: 1;
    transition: color 0.2s ease;
}

.shortcuts-close:hover {
    color: var(--purple);
}

.shortcuts-body {
    padding: 16px 20px;
}

.shortcuts-section {
    margin-bottom: 16px;
}

.shortcuts-section:last-child {
    margin-bottom: 0;
}

.shortcuts-section-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 8px;
    font-weight: 600;
}

.shortcut-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(180, 160, 200, 0.1);
}

.shortcut-item:last-child {
    border-bottom: none;
}

.shortcut-keys {
    display: flex;
    gap: 4px;
}

.shortcut-key {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 28px;
    height: 28px;
    padding: 0 8px;
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.1), rgba(86, 182, 194, 0.1));
    border: 1px solid rgba(180, 160, 200, 0.3);
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'Roboto Mono', monospace;
    color: var(--text);
}

.shortcut-description {
    font-size: 13px;
    color: var(--text-dim);
}

/* ========================================
   Sample Viewer Enhanced Styles
   ======================================== */

/* Sample info panel container */
.sample-info-panel {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

/* Sample metadata badges container */
.sample-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 12px;
}

/* Individual metadata badges */
.sample-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 16px;
    font-size: 12px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.sample-badge:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.sample-badge.choice-type {
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.15), rgba(255, 107, 157, 0.15));
    border: 1px solid rgba(198, 120, 221, 0.3);
    color: var(--purple);
}

.sample-badge.time-scale {
    background: linear-gradient(135deg, rgba(86, 182, 194, 0.15), rgba(97, 175, 239, 0.15));
    border: 1px solid rgba(86, 182, 194, 0.3);
    color: var(--cyan);
}

.sample-badge.time-horizon {
    background: linear-gradient(135deg, rgba(152, 195, 121, 0.15), rgba(86, 182, 194, 0.15));
    border: 1px solid rgba(152, 195, 121, 0.3);
    color: var(--mint);
}

.sample-badge.short-term-first {
    background: linear-gradient(135deg, rgba(229, 192, 123, 0.15), rgba(209, 154, 102, 0.15));
    border: 1px solid rgba(229, 192, 123, 0.3);
    color: var(--gold);
}

.sample-badge.no-horizon {
    background: linear-gradient(135deg, rgba(229, 192, 123, 0.2), rgba(209, 154, 102, 0.2));
    border: 1px solid rgba(229, 192, 123, 0.4);
    color: #a08050;
}

.sample-badge.sample-index {
    background: linear-gradient(135deg, rgba(74, 63, 92, 0.1), rgba(122, 107, 138, 0.1));
    border: 1px solid rgba(74, 63, 92, 0.2);
    color: var(--text);
}

/* Badge label styling */
.sample-badge-label {
    font-weight: 500;
    opacity: 0.7;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.sample-badge-value {
    font-weight: 700;
}

/* Mini summary card */
.sample-summary-card {
    background: linear-gradient(135deg, rgba(248, 244, 255, 0.9), rgba(240, 247, 255, 0.9));
    border: 1px solid rgba(180, 160, 200, 0.2);
    border-radius: 16px;
    padding: 16px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 12px;
}

.summary-stat {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 8px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.6);
}

.summary-stat-value {
    font-size: 16px;
    font-weight: 700;
    color: var(--text);
    font-family: 'Roboto Mono', monospace;
}

.summary-stat-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
    margin-top: 2px;
}

/* Prompt text with syntax highlighting */
.sample-prompt-container {
    position: relative;
}

.sample-prompt {
    background: rgba(248, 244, 255, 0.8);
    border: 1px solid rgba(180, 160, 200, 0.2);
    border-radius: 12px;
    padding: 18px;
    padding-top: 40px;
    font-family: 'Roboto Mono', 'SF Mono', monospace;
    font-size: 13px;
    line-height: 1.6;
    white-space: pre-wrap;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text);
}

/* Syntax highlighting for prompt */
.prompt-keyword {
    color: var(--purple);
    font-weight: 600;
}

.prompt-number {
    color: var(--cyan);
    font-weight: 500;
}

.prompt-option {
    color: var(--pink);
}

.prompt-time {
    color: var(--mint);
    font-weight: 500;
}

/* Copy button */
.copy-button {
    position: absolute;
    top: 8px;
    right: 8px;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(248, 244, 255, 0.9));
    border: 1px solid rgba(180, 160, 200, 0.3);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-dim);
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    gap: 6px;
    z-index: 10;
}

.copy-button:hover {
    background: linear-gradient(135deg, var(--purple), var(--pink));
    color: white;
    border-color: transparent;
    box-shadow: 0 4px 12px rgba(198, 120, 221, 0.3);
}

.copy-button.copied {
    background: linear-gradient(135deg, var(--mint), var(--cyan));
    color: white;
    border-color: transparent;
}

/* Click to select instruction */
.click-to-select-hint {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: rgba(86, 182, 194, 0.1);
    border: 1px solid rgba(86, 182, 194, 0.2);
    border-radius: 8px;
    font-size: 11px;
    color: var(--text-dim);
    margin-bottom: 12px;
}

.click-to-select-hint .hint-icon {
    color: var(--cyan);
    font-weight: 700;
}

/* Selected sample highlight style */
.selected-from-plot {
    background: linear-gradient(135deg, rgba(255, 107, 157, 0.08), rgba(198, 120, 221, 0.08));
    border-left: 3px solid var(--pink);
    padding-left: 12px;
}
"""


def create_loading_wrapper(
    children,
    loading_text: str = "Computing embeddings...",
    show_progress: bool = False,
):
    """Create a styled loading wrapper with custom text and optional progress bar.

    Args:
        children: The content to wrap (typically a dcc.Graph)
        loading_text: Text to display while loading
        show_progress: Whether to show an indeterminate progress bar (useful for UMAP/t-SNE)

    Returns:
        A dcc.Loading component with custom styling
    """
    return dcc.Loading(
        children,
        type="circle",
        color=COLORS["purple"],
        className="loading-wrapper",
        custom_spinner=html.Div(
            [
                html.Div(className="dash-spinner dash-spinner--circle"),
                html.Div(
                    [
                        html.Div(loading_text, className="loading-text"),
                        html.Div(
                            html.Div(className="loading-progress-bar"),
                            className="loading-progress",
                        )
                        if show_progress
                        else None,
                    ],
                    className="loading-text-container",
                ),
            ],
            className="dash-loading-container",
        ),
    )


def create_3d_scatter(
    embedding: np.ndarray,
    colors: np.ndarray,
    color_by: str,
    title: str,
    is_categorical: bool = False,
    no_horizon_mask: np.ndarray | None = None,
    show_no_horizon: bool = True,
    method: str = "pca",
) -> go.Figure:
    """Create a 3D scatter plot with pastel styling and PNG export."""
    axis_prefix = method.upper()

    # Track original indices for hover info
    sample_indices = np.arange(len(embedding))

    if no_horizon_mask is not None and not show_no_horizon:
        mask = ~no_horizon_mask
        embedding = embedding[mask]
        colors = colors[mask]
        sample_indices = sample_indices[mask]
        no_horizon_mask = None

    # Get human-readable label for the color scale
    colorbar_title = COLOR_OPTION_LABELS.get(color_by, color_by.replace("_", " ").title())

    # Common colorbar styling for continuous data
    colorbar_style = dict(
        title=dict(
            text=colorbar_title,
            font=dict(color=COLORS["text"], size=12),
        ),
        tickfont=dict(color=COLORS["text_dim"], size=10),
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="rgba(180, 160, 200, 0.3)",
        borderwidth=1,
        thickness=16,
        len=0.6,
        outlinewidth=0,
        tickcolor="rgba(180, 160, 200, 0.5)",
    )

    if is_categorical:
        fig = px.scatter_3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2] if embedding.shape[1] > 2 else np.zeros(len(embedding)),
            color=colors.astype(str),
            color_discrete_sequence=PASTEL_COLORS,
            labels={
                "x": f"{axis_prefix}0",
                "y": f"{axis_prefix}1",
                "z": f"{axis_prefix}2",
                "color": colorbar_title,
            },
            title=title,
        )
    elif no_horizon_mask is not None and show_no_horizon:
        horizon_mask = ~no_horizon_mask

        fig = go.Figure()

        if horizon_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=embedding[horizon_mask, 0],
                    y=embedding[horizon_mask, 1],
                    z=embedding[horizon_mask, 2]
                    if embedding.shape[1] > 2
                    else np.zeros(horizon_mask.sum()),
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=colors[horizon_mask],
                        colorscale=[
                            [0, COLORS["cyan"]],
                            [0.5, COLORS["purple"]],
                            [1, COLORS["pink"]],
                        ],
                        opacity=0.8,
                        colorbar=colorbar_style,
                    ),
                    name="With Horizon",
                    hovertemplate=f"{axis_prefix}0: %{{x:.3f}}<br>{axis_prefix}1: %{{y:.3f}}<br>{axis_prefix}2: %{{z:.3f}}<extra>With Horizon</extra>",
                )
            )

        if no_horizon_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=embedding[no_horizon_mask, 0],
                    y=embedding[no_horizon_mask, 1],
                    z=embedding[no_horizon_mask, 2]
                    if embedding.shape[1] > 2
                    else np.zeros(no_horizon_mask.sum()),
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=COLORS["gold"],
                        opacity=0.9,
                        symbol="diamond",
                    ),
                    name="No Horizon",
                    hovertemplate=f"{axis_prefix}0: %{{x:.3f}}<br>{axis_prefix}1: %{{y:.3f}}<br>{axis_prefix}2: %{{z:.3f}}<extra>No Horizon</extra>",
                )
            )

        fig.update_layout(
            title=dict(text=title, font=dict(color=COLORS["text"], size=16)),
            scene=dict(
                xaxis_title=f"{axis_prefix}0",
                yaxis_title=f"{axis_prefix}1",
                zaxis_title=f"{axis_prefix}2",
            ),
        )
    else:
        fig = px.scatter_3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2] if embedding.shape[1] > 2 else np.zeros(len(embedding)),
            color=colors,
            color_continuous_scale=[
                [0, COLORS["cyan"]],
                [0.5, COLORS["purple"]],
                [1, COLORS["pink"]],
            ],
            labels={
                "x": f"{axis_prefix}0",
                "y": f"{axis_prefix}1",
                "z": f"{axis_prefix}2",
                "color": color_by.replace("_", " ").title(),
            },
            title=title,
        )
        # Apply colorbar styling to px-generated figure
        fig.update_coloraxes(colorbar=colorbar_style)

    fig.update_traces(marker=dict(size=4, opacity=0.8), selector=dict(type="scatter3d"))

    # Common axis styling with pastel accents
    axis_common = dict(
        showbackground=True,
        showgrid=True,
        gridwidth=1,
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="rgba(180, 160, 200, 0.4)",
        tickfont=dict(color=COLORS["text_muted"], size=10),
        title=dict(font=dict(color=COLORS["text"], size=12)),
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(255, 255, 255, 0)",
        plot_bgcolor="rgba(255, 255, 255, 0)",
        font=dict(family="system-ui, sans-serif", color=COLORS["text"]),
        title=dict(font=dict(size=16, color=COLORS["text"])),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="rgba(180, 160, 200, 0.3)",
            borderwidth=1,
            font=dict(color=COLORS["text"], size=11),
            itemsizing="constant",
            tracegroupgap=4,
        ),
        scene=dict(
            bgcolor="rgba(252, 250, 255, 0.6)",
            xaxis=dict(
                **axis_common,
                backgroundcolor="rgba(255, 255, 255, 0.95)",
                gridcolor="rgba(198, 120, 221, 0.12)",
            ),
            yaxis=dict(
                **axis_common,
                backgroundcolor="rgba(255, 255, 255, 0.95)",
                gridcolor="rgba(86, 182, 194, 0.12)",
            ),
            zaxis=dict(
                **axis_common,
                backgroundcolor="rgba(255, 255, 255, 0.95)",
                gridcolor="rgba(255, 107, 157, 0.12)",
            ),
        ),
    )

    return fig


def create_empty_figure(message: str = "No data available") -> go.Figure:
    """Create an empty figure with an error/info message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=COLORS["text_muted"]),
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def precompute_embeddings(loader: GeoVizDataLoader, verbose: bool = True) -> int:
    """Precompute common embeddings during startup to avoid callback timeouts.

    This enhanced version:
    1. Pre-loads all activations for fast PCA/UMAP/t-SNE computation
    2. Precomputes PCA for all layers at the default position
    3. Precomputes UMAP for key layers (first, middle, last)
    """
    layers = loader.get_layers()
    positions = loader.get_positions()

    if not layers or not positions:
        return 0

    total_cached = 0
    default_position = "response" if "response" in positions else positions[0]

    def progress(current: int, total: int, desc: str) -> None:
        if verbose:
            pct = int(100 * current / total) if total > 0 else 0
            print(f"  [{pct:3d}%] {desc}")

    # Step 1: Pre-load all activations for faster embedding computation
    if verbose:
        print("Step 1/3: Pre-loading activation arrays...")
    loaded = loader.preload_activations(
        layers=layers,
        components=["resid_post"],  # Focus on resid_post as it's most commonly used
        positions=[default_position],
        progress_callback=progress if verbose else None,
    )
    if verbose:
        print(f"  Loaded {loaded} activation arrays")

    # Step 2: Precompute PCA for all layers at default position (fast)
    if verbose:
        print("Step 2/3: Computing PCA embeddings for all layers...")
    for i, layer in enumerate(layers):
        for component in ["resid_pre", "attn_out", "mlp_out", "resid_post"]:
            emb = loader.load_pca(layer, component, default_position, n_components=3)
            if emb is not None:
                total_cached += 1
        if verbose:
            pct = int(100 * (i + 1) / len(layers))
            print(f"  [{pct:3d}%] Layer {layer} complete")

    # Step 3: Precompute UMAP for key layers (slower, so only do a few)
    if verbose:
        print("Step 3/3: Computing UMAP for key layers (this may take a moment)...")
    key_layers = [layers[0], layers[len(layers) // 2], layers[-1]]
    for i, layer in enumerate(key_layers):
        emb = loader.load_umap(layer, "resid_post", default_position, n_components=3)
        if emb is not None:
            total_cached += 1
        if verbose:
            print(f"  [{int(100 * (i + 1) / len(key_layers)):3d}%] UMAP L{layer} @ {default_position}")

    if verbose:
        print(f"Warmup complete: {total_cached} embeddings cached")
    return total_cached


def create_app(data_dir: str | Path | None = None) -> Dash:
    """Create the Dash application with modern light pastel UI."""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    loader = GeoVizDataLoader(data_dir)

    layers = loader.get_layers()
    components = loader.get_components()
    positions = loader.get_positions()
    color_options = loader.get_color_options()

    precompute_embeddings(loader)

    app = Dash(
        __name__,
        title="GeoViz Explorer",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    # Generate timestamp for default filename
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plotly_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"geoviz_{timestamp}",
            "height": 1200,
            "width": 1600,
            "scale": 2,
        },
        "modeBarButtonsToAdd": ["toImage"],
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        # Style the modebar
        "modeBarStyle": {
            "backgroundColor": "rgba(255, 255, 255, 0.9)",
            "borderRadius": "8px",
        },
    }

    header = dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand(
                    "GeoViz Explorer",
                    className="ms-2",
                    style={
                        "fontSize": "1.75rem",
                        "fontWeight": "800",
                        "color": COLORS["text"],
                        "letterSpacing": "-0.02em",
                        "textShadow": f"0 1px 2px rgba(198, 120, 221, 0.15), 0 0 20px rgba(198, 120, 221, 0.08)",
                    },
                ),
                dbc.Nav(
                    [
                        dbc.NavItem(
                            dbc.Badge(
                                f"{loader.n_samples} samples",
                                className="me-2",
                                style={
                                    "background": f"linear-gradient(135deg, {COLORS['pink']}, {COLORS['purple']})",
                                    "padding": "8px 14px",
                                    "borderRadius": "16px",
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                    "boxShadow": "0 2px 8px rgba(255, 107, 157, 0.25)",
                                    "border": "1px solid rgba(255, 255, 255, 0.3)",
                                },
                            )
                        ),
                        dbc.NavItem(
                            dbc.Badge(
                                f"{len(layers)} layers",
                                className="me-2",
                                style={
                                    "background": f"linear-gradient(135deg, {COLORS['cyan']}, {COLORS['blue']})",
                                    "padding": "8px 14px",
                                    "borderRadius": "16px",
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                    "boxShadow": "0 2px 8px rgba(86, 182, 194, 0.25)",
                                    "border": "1px solid rgba(255, 255, 255, 0.3)",
                                },
                            )
                        ),
                        dbc.NavItem(
                            dbc.Badge(
                                f"{len(positions)} positions",
                                style={
                                    "background": f"linear-gradient(135deg, {COLORS['mint']}, {COLORS['cyan']})",
                                    "padding": "8px 14px",
                                    "borderRadius": "16px",
                                    "fontSize": "12px",
                                    "fontWeight": "600",
                                    "boxShadow": "0 2px 8px rgba(152, 195, 121, 0.25)",
                                    "border": "1px solid rgba(255, 255, 255, 0.3)",
                                },
                            )
                        ),
                    ],
                    className="ms-auto d-flex align-items-center",
                ),
            ],
            fluid=True,
            className="py-2",
        ),
        className="mb-4",
        style={
            "background": f"linear-gradient(135deg, rgba(255, 107, 157, 0.15) 0%, rgba(198, 120, 221, 0.18) 50%, rgba(97, 175, 239, 0.12) 100%)",
            "backdropFilter": "blur(20px)",
            "borderBottom": "1px solid rgba(198, 120, 221, 0.25)",
            "boxShadow": "0 4px 24px rgba(198, 120, 221, 0.12), 0 1px 3px rgba(100, 80, 120, 0.08)",
            "padding": "12px 0",
        },
    )

    # Help card for the visualization
    help_card = html.Div(
        [
            html.Div(
                [
                    html.Span("?", style={"fontSize": "14px", "fontWeight": "700", "color": COLORS["purple"]}),
                    html.Span(" About This Visualization", style={"fontWeight": "600"}),
                ],
                className="help-card-title",
            ),
            html.Div(
                [
                    "Explore neural network activations for intertemporal choice tasks. ",
                    "Each point represents a sample's representation at a specific layer and position.",
                    html.Ul(
                        [
                            html.Li("Use the tabs to switch between different exploration modes"),
                            html.Li("Color by time horizon to see temporal structure in representations"),
                            html.Li("Compare different components (attention, MLP, residual) to understand information flow"),
                            html.Li("Drag to rotate 3D plots, scroll to zoom, click camera icon to export"),
                        ]
                    ),
                ],
                className="help-card-content",
            ),
        ],
        className="help-card",
    )

    global_controls = dbc.Card(
        [
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dbc.Label("Reduction Method", className="fw-bold", style={"color": COLORS["text"]}),
                                            html.Span("?", id="method-info-icon", className="info-icon"),
                                        ],
                                        style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
                                    ),
                                    dbc.Tooltip(
                                        "Choose how to reduce high-dimensional activations to 3D for visualization. "
                                        "PCA is fast and preserves global structure. UMAP and t-SNE are slower but reveal clusters.",
                                        target="method-info-icon",
                                        placement="top",
                                    ),
                                    dbc.RadioItems(
                                        id="method-radio",
                                        options=[
                                            {"label": "PCA", "value": "pca"},
                                            {"label": "UMAP", "value": "umap"},
                                            {"label": "t-SNE", "value": "tsne"},
                                        ],
                                        value="pca",
                                        inline=True,
                                    ),
                                    # Individual method tooltips
                                    dbc.Tooltip(
                                        TOOLTIP_TEXTS["method"]["pca"],
                                        target={"type": "method-radio", "index": 0},
                                        placement="bottom",
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dbc.Label("Color by", className="fw-bold", style={"color": COLORS["text"]}),
                                            html.Span("?", id="color-info-icon", className="info-icon"),
                                        ],
                                        style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
                                    ),
                                    dbc.Tooltip(
                                        "Select which sample property to use for coloring points. "
                                        "Time horizon options show how temporal information is encoded.",
                                        target="color-info-icon",
                                        placement="top",
                                    ),
                                    dcc.Dropdown(
                                        id="color-dropdown",
                                        options=[
                                            {"label": COLOR_OPTION_LABELS.get(c, c.replace("_", " ").title()), "value": c}
                                            for c in color_options
                                        ],
                                        value="log_time_horizon",
                                        clearable=False,
                                        placeholder="Select color variable...",
                                        style={"borderRadius": "10px"},
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dbc.Label("No-Horizon Samples", className="fw-bold", style={"color": COLORS["text"]}),
                                            html.Span("?", id="horizon-info-icon", className="info-icon"),
                                        ],
                                        style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
                                    ),
                                    dbc.Tooltip(
                                        TOOLTIP_TEXTS["no_horizon"],
                                        target="horizon-info-icon",
                                        placement="top",
                                    ),
                                    dbc.Checklist(
                                        id="show-no-horizon",
                                        options=[{"label": "Show (gold diamonds)", "value": "show"}],
                                        value=["show"],
                                        switch=True,
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Div(id="global-info", className="info-panel"),
                                ],
                                md=3,
                            ),
                        ]
                    ),
                ]
            ),
        ],
        className="mb-3 glass-card",
    )

    component_explorer_tab = dbc.Tab(
        label="Component Explorer",
        tab_id="tab-component",
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            "Select Target",
                                            html.Span("?", id="ce-target-info", className="info-icon", style={"marginLeft": "8px"}),
                                        ]
                                    ),
                                    dbc.Tooltip(
                                        "Choose which part of the model to visualize. Layer controls depth, "
                                        "component selects the activation type, and position chooses the token.",
                                        target="ce-target-info",
                                        placement="right",
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label("Layer", style={"color": COLORS["text"]}),
                                                    html.Span("?", id="ce-layer-info", className="info-icon"),
                                                ],
                                                style={"display": "flex", "alignItems": "center"},
                                            ),
                                            dbc.Tooltip(
                                                TOOLTIP_TEXTS["layer"],
                                                target="ce-layer-info",
                                                placement="right",
                                            ),
                                            dcc.Slider(
                                                id="ce-layer-slider",
                                                min=min(layers) if layers else 0,
                                                max=max(layers) if layers else 0,
                                                value=layers[len(layers) // 2] if layers else 0,
                                                marks={l: str(l) for l in layers[:: max(1, len(layers) // 10)]},
                                                step=None,
                                                tooltip={"placement": "bottom", "always_visible": True},
                                            ),
                                            html.Hr(style={"borderColor": "rgba(180,160,200,0.2)"}),
                                            html.Div(
                                                [
                                                    dbc.Label("Component", style={"color": COLORS["text"]}),
                                                    html.Span("?", id="ce-component-info", className="info-icon"),
                                                ],
                                                style={"display": "flex", "alignItems": "center"},
                                            ),
                                            dbc.Tooltip(
                                                "Select which activation stream to visualize. Residual streams "
                                                "show cumulative information, while attention and MLP show "
                                                "contributions from those specific sub-layers.",
                                                target="ce-component-info",
                                                placement="right",
                                            ),
                                            dbc.RadioItems(
                                                id="ce-component-radio",
                                                options=[{"label": COMPONENT_NAMES[c], "value": c} for c in components],
                                                value="resid_post",
                                            ),
                                            html.Hr(style={"borderColor": "rgba(180,160,200,0.2)"}),
                                            html.Div(
                                                [
                                                    dbc.Label("Position", style={"color": COLORS["text"]}),
                                                    html.Span("?", id="ce-position-info", className="info-icon"),
                                                ],
                                                style={"display": "flex", "alignItems": "center"},
                                            ),
                                            dbc.Tooltip(
                                                TOOLTIP_TEXTS["position"],
                                                target="ce-position-info",
                                                placement="right",
                                            ),
                                            dcc.Dropdown(
                                                id="ce-position-dropdown",
                                                options=[{"label": p, "value": p} for p in positions],
                                                value="response" if "response" in positions else (positions[0] if positions else None),
                                                clearable=False,
                                                placeholder="Select token position...",
                                            ),
                                        ]
                                    ),
                                ],
                                className="glass-card mb-3",
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            "Metrics",
                                            html.Span("?", id="ce-metrics-info", className="info-icon", style={"marginLeft": "8px"}),
                                        ]
                                    ),
                                    dbc.Tooltip(
                                        "Quality metrics for the current selection. Linear Probe R2 shows how well "
                                        "time horizon can be predicted from activations. Variance Explained shows "
                                        "how much information is captured in 3 principal components.",
                                        target="ce-metrics-info",
                                        placement="right",
                                    ),
                                    dbc.CardBody(id="ce-metrics-panel"),
                                ],
                                className="glass-card",
                            ),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            create_loading_wrapper(
                                                dcc.Graph(
                                                    id="ce-main-plot",
                                                    style={"height": "65vh"},
                                                    config=plotly_config,
                                                ),
                                                loading_text="Computing embeddings...",
                                                show_progress=True,
                                            ),
                                        ]
                                    ),
                                ],
                                className="glass-card",
                            ),
                        ],
                        md=9,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            "Sample Viewer",
                                            html.Span("?", id="ce-sample-info", className="info-icon", style={"marginLeft": "8px"}),
                                            dbc.Badge(id="ce-sample-badge", className="ms-2", style={"background": COLORS["purple"]}),
                                        ]
                                    ),
                                    dbc.Tooltip(
                                        "View the raw text of individual samples. Use the slider or type a sample "
                                        "index to navigate. The badge shows the sample's time horizon.",
                                        target="ce-sample-info",
                                        placement="top",
                                    ),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dcc.Slider(
                                                                id="ce-sample-slider",
                                                                min=0,
                                                                max=loader.n_samples - 1 if loader.n_samples > 0 else 0,
                                                                value=0,
                                                                marks={i: str(i) for i in range(0, loader.n_samples, max(1, loader.n_samples // 10))},
                                                                tooltip={"placement": "bottom", "always_visible": True},
                                                            ),
                                                        ],
                                                        md=10,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Input(
                                                                id="ce-sample-input",
                                                                type="number",
                                                                min=0,
                                                                max=loader.n_samples - 1,
                                                                value=0,
                                                                size="sm",
                                                                placeholder="Index",
                                                            ),
                                                        ],
                                                        md=2,
                                                    ),
                                                ]
                                            ),
                                            html.Div(id="ce-sample-text", className="sample-text mt-3"),
                                        ]
                                    ),
                                ],
                                className="glass-card mt-3",
                            ),
                        ]
                    ),
                ]
            ),
        ],
    )

    layer_explorer_tab = dbc.Tab(
        label="Layer Explorer",
        tab_id="tab-layer",
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "Compare all four component types at a single layer. ",
                                                        style={"color": COLORS["text_dim"], "fontSize": "13px"},
                                                    ),
                                                    html.Span("?", id="le-intro-info", className="info-icon"),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            dbc.Tooltip(
                                                "The Layer Explorer shows side-by-side visualizations of all component "
                                                "types (resid_pre, attn_out, mlp_out, resid_post) for the selected layer. "
                                                "Useful for understanding how each sub-component transforms representations.",
                                                target="le-intro-info",
                                                placement="top",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dbc.Label("Layer", className="fw-bold", style={"color": COLORS["text"]}),
                                                                    html.Span("?", id="le-layer-info", className="info-icon"),
                                                                ],
                                                                style={"display": "flex", "alignItems": "center"},
                                                            ),
                                                            dbc.Tooltip(
                                                                TOOLTIP_TEXTS["layer"],
                                                                target="le-layer-info",
                                                                placement="top",
                                                            ),
                                                            dcc.Slider(
                                                                id="le-layer-slider",
                                                                min=min(layers) if layers else 0,
                                                                max=max(layers) if layers else 0,
                                                                value=layers[len(layers) // 2] if layers else 0,
                                                                marks={l: str(l) for l in layers},
                                                                step=None,
                                                                tooltip={"placement": "bottom", "always_visible": True},
                                                            ),
                                                        ],
                                                        md=8,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Position", className="fw-bold", style={"color": COLORS["text"]}),
                                                            dcc.Dropdown(
                                                                id="le-position-dropdown",
                                                                options=[{"label": p, "value": p} for p in positions],
                                                                value="response" if "response" in positions else (positions[0] if positions else None),
                                                                clearable=False,
                                                                placeholder="Select position...",
                                                            ),
                                                        ],
                                                        md=4,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                className="glass-card mb-3",
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            "Residual Pre",
                                            html.Span("?", id="le-resid-pre-info", className="info-icon", style={"marginLeft": "8px"}),
                                        ]
                                    ),
                                    dbc.Tooltip(
                                        TOOLTIP_TEXTS["component"]["resid_pre"],
                                        target="le-resid-pre-info",
                                        placement="top",
                                    ),
                                    dbc.CardBody(
                                        [
                                            create_loading_wrapper(
                                                dcc.Graph(id="le-resid-pre", style={"height": "40vh"}, config=plotly_config),
                                                loading_text="Computing embeddings...",
                                                show_progress=True,
                                            ),
                                        ]
                                    ),
                                ],
                                className="glass-card",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            "Attention Out",
                                            html.Span("?", id="le-attn-out-info", className="info-icon", style={"marginLeft": "8px"}),
                                        ]
                                    ),
                                    dbc.Tooltip(
                                        TOOLTIP_TEXTS["component"]["attn_out"],
                                        target="le-attn-out-info",
                                        placement="top",
                                    ),
                                    dbc.CardBody(
                                        [
                                            create_loading_wrapper(
                                                dcc.Graph(id="le-attn-out", style={"height": "40vh"}, config=plotly_config),
                                                loading_text="Computing embeddings...",
                                                show_progress=True,
                                            ),
                                        ]
                                    ),
                                ],
                                className="glass-card",
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            "MLP Out",
                                            html.Span("?", id="le-mlp-out-info", className="info-icon", style={"marginLeft": "8px"}),
                                        ]
                                    ),
                                    dbc.Tooltip(
                                        TOOLTIP_TEXTS["component"]["mlp_out"],
                                        target="le-mlp-out-info",
                                        placement="top",
                                    ),
                                    dbc.CardBody(
                                        [
                                            create_loading_wrapper(
                                                dcc.Graph(id="le-mlp-out", style={"height": "40vh"}, config=plotly_config),
                                                loading_text="Computing embeddings...",
                                                show_progress=True,
                                            ),
                                        ]
                                    ),
                                ],
                                className="glass-card",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            "Residual Post",
                                            html.Span("?", id="le-resid-post-info", className="info-icon", style={"marginLeft": "8px"}),
                                        ]
                                    ),
                                    dbc.Tooltip(
                                        TOOLTIP_TEXTS["component"]["resid_post"],
                                        target="le-resid-post-info",
                                        placement="top",
                                    ),
                                    dbc.CardBody(
                                        [
                                            create_loading_wrapper(
                                                dcc.Graph(id="le-resid-post", style={"height": "40vh"}, config=plotly_config),
                                                loading_text="Computing embeddings...",
                                                show_progress=True,
                                            ),
                                        ]
                                    ),
                                ],
                                className="glass-card",
                            ),
                        ],
                        md=6,
                    ),
                ]
            ),
        ],
    )

    trajectory_tab = dbc.Tab(
        label="Trajectory View",
        tab_id="tab-trajectory",
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Component", className="fw-bold", style={"color": COLORS["text"]}),
                                                            dbc.RadioItems(
                                                                id="traj-component-radio",
                                                                options=[{"label": COMPONENT_NAMES[c], "value": c} for c in components],
                                                                value="resid_post",
                                                                inline=True,
                                                            ),
                                                        ],
                                                        md=5,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Position", className="fw-bold", style={"color": COLORS["text"]}),
                                                            dcc.Dropdown(
                                                                id="traj-position-dropdown",
                                                                options=[{"label": p, "value": p} for p in positions],
                                                                value="response" if "response" in positions else (positions[0] if positions else None),
                                                                clearable=False,
                                                                placeholder="Select position...",
                                                            ),
                                                        ],
                                                        md=3,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dbc.Label("Sample Index", className="fw-bold", style={"color": COLORS["text"]}),
                                                                    html.Span("?", id="traj-sample-info", className="info-icon"),
                                                                ],
                                                                style={"display": "flex", "alignItems": "center"},
                                                            ),
                                                            dbc.Tooltip(
                                                                "Select a specific sample to trace its path through layers.",
                                                                target="traj-sample-info",
                                                                placement="top",
                                                            ),
                                                            dbc.Input(
                                                                id="traj-sample-input",
                                                                type="number",
                                                                min=0,
                                                                max=loader.n_samples - 1,
                                                                value=0,
                                                                size="sm",
                                                                placeholder="0",
                                                            ),
                                                        ],
                                                        md=2,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dbc.Label("Show All", className="fw-bold", style={"color": COLORS["text"]}),
                                                                    html.Span("?", id="traj-showall-info", className="info-icon"),
                                                                ],
                                                                style={"display": "flex", "alignItems": "center"},
                                                            ),
                                                            dbc.Tooltip(
                                                                "Toggle to show trajectories for up to 100 samples simultaneously. "
                                                                "Useful for seeing if samples follow similar paths through layers.",
                                                                target="traj-showall-info",
                                                                placement="top",
                                                            ),
                                                            dbc.Checklist(
                                                                id="traj-show-all",
                                                                options=[{"label": "100 samples", "value": "all"}],
                                                                value=[],
                                                                switch=True,
                                                            ),
                                                        ],
                                                        md=2,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                className="glass-card mb-3",
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            "Layer Trajectory (3D)",
                                            html.Span("?", id="traj-3d-info", className="info-icon", style={"marginLeft": "8px"}),
                                        ]
                                    ),
                                    dbc.Tooltip(
                                        "Visualize how a sample's representation moves through the model's layers. "
                                        "Each point is a layer, connected to show the trajectory.",
                                        target="traj-3d-info",
                                        placement="top",
                                    ),
                                    dbc.CardBody(
                                        [
                                            create_loading_wrapper(
                                                dcc.Graph(id="traj-plot", style={"height": "70vh"}, config=plotly_config),
                                                loading_text="Computing trajectories...",
                                                show_progress=True,
                                            ),
                                        ]
                                    ),
                                ],
                                className="glass-card",
                            ),
                        ],
                        md=8,
                    ),
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Layer Progression"),
                                    dbc.CardBody(
                                        [
                                            create_loading_wrapper(
                                                dcc.Graph(id="traj-line-plot", style={"height": "70vh"}, config=plotly_config),
                                                loading_text="Rendering plot...",
                                            ),
                                        ]
                                    ),
                                ],
                                className="glass-card",
                            ),
                        ],
                        md=4,
                    ),
                ]
            ),
        ],
    )

    position_slider_tab = dbc.Tab(
        label="Position Slider",
        tab_id="tab-position",
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    html.Span(
                                                        "Animate through token positions at a fixed layer. ",
                                                        style={"color": COLORS["text_dim"], "fontSize": "13px"},
                                                    ),
                                                    html.Span("?", id="ps-intro-info", className="info-icon"),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            dbc.Tooltip(
                                                "The Position Slider lets you see how representations change across "
                                                "different token positions in the input sequence. Useful for understanding "
                                                "where temporal information emerges.",
                                                target="ps-intro-info",
                                                placement="top",
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Layer", className="fw-bold", style={"color": COLORS["text"]}),
                                                            dcc.Slider(
                                                                id="ps-layer-slider",
                                                                min=min(layers) if layers else 0,
                                                                max=max(layers) if layers else 0,
                                                                value=layers[len(layers) // 2] if layers else 0,
                                                                marks={l: str(l) for l in layers[:: max(1, len(layers) // 5)]},
                                                                step=None,
                                                                tooltip={"placement": "bottom", "always_visible": True},
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Component", className="fw-bold", style={"color": COLORS["text"]}),
                                                            dbc.RadioItems(
                                                                id="ps-component-radio",
                                                                options=[{"label": COMPONENT_NAMES[c], "value": c} for c in components],
                                                                value="resid_post",
                                                                inline=True,
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                className="glass-card mb-3",
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label("Position", className="fw-bold", style={"color": COLORS["text"]}),
                                                    html.Span("?", id="ps-position-info-icon", className="info-icon"),
                                                ],
                                                style={"display": "flex", "alignItems": "center"},
                                            ),
                                            dbc.Tooltip(
                                                "Slide through different token positions to see how representations evolve "
                                                "across the input sequence. The 'response' position is typically where "
                                                "the model makes its final decision.",
                                                target="ps-position-info-icon",
                                                placement="top",
                                            ),
                                            dcc.Slider(
                                                id="ps-position-slider",
                                                min=0,
                                                max=len(positions) - 1 if positions else 0,
                                                value=0,
                                                marks={i: pos for i, pos in enumerate(positions)},
                                                step=1,
                                            ),
                                            html.Div(id="ps-position-info", className="info-panel mt-3"),
                                        ]
                                    ),
                                ],
                                className="glass-card mb-3",
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            create_loading_wrapper(
                                                dcc.Graph(id="ps-main-plot", style={"height": "60vh"}, config=plotly_config),
                                                loading_text="Computing embeddings...",
                                                show_progress=True,
                                            ),
                                        ]
                                    ),
                                ],
                                className="glass-card",
                            ),
                        ]
                    ),
                ]
            ),
        ],
    )

    app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
        {CUSTOM_CSS}
        </style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

    # Keyboard shortcuts help panel
    keyboard_shortcuts_panel = html.Div(
        [
            # Toggle button
            html.Button(
                "?",
                id="shortcuts-toggle",
                className="keyboard-shortcuts-toggle",
                title="Keyboard Shortcuts",
            ),
            # Panel
            html.Div(
                [
                    html.Div(
                        [
                            html.H4(["Keyboard Shortcuts"]),
                            html.Button(
                                "x",
                                id="shortcuts-close",
                                className="shortcuts-close",
                            ),
                        ],
                        className="shortcuts-header",
                    ),
                    html.Div(
                        [
                            # Navigation section
                            html.Div(
                                [
                                    html.Div("Navigation", className="shortcuts-section-title"),
                                    html.Div(
                                        [
                                            html.Div(
                                                [html.Span("<", className="shortcut-key"), html.Span(">", className="shortcut-key")],
                                                className="shortcut-keys",
                                            ),
                                            html.Span("Navigate layers", className="shortcut-description"),
                                        ],
                                        className="shortcut-item",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [html.Span("1", className="shortcut-key"), html.Span("-", className="shortcut-key"), html.Span("4", className="shortcut-key")],
                                                className="shortcut-keys",
                                            ),
                                            html.Span("Switch tabs", className="shortcut-description"),
                                        ],
                                        className="shortcut-item",
                                    ),
                                ],
                                className="shortcuts-section",
                            ),
                            # Methods section
                            html.Div(
                                [
                                    html.Div("Reduction Methods", className="shortcuts-section-title"),
                                    html.Div(
                                        [
                                            html.Div([html.Span("P", className="shortcut-key")], className="shortcut-keys"),
                                            html.Span("Switch to PCA", className="shortcut-description"),
                                        ],
                                        className="shortcut-item",
                                    ),
                                    html.Div(
                                        [
                                            html.Div([html.Span("U", className="shortcut-key")], className="shortcut-keys"),
                                            html.Span("Switch to UMAP", className="shortcut-description"),
                                        ],
                                        className="shortcut-item",
                                    ),
                                    html.Div(
                                        [
                                            html.Div([html.Span("T", className="shortcut-key")], className="shortcut-keys"),
                                            html.Span("Switch to t-SNE", className="shortcut-description"),
                                        ],
                                        className="shortcut-item",
                                    ),
                                ],
                                className="shortcuts-section",
                            ),
                            # Actions section
                            html.Div(
                                [
                                    html.Div("Actions", className="shortcuts-section-title"),
                                    html.Div(
                                        [
                                            html.Div([html.Span("S", className="shortcut-key")], className="shortcut-keys"),
                                            html.Span("Save plot as PNG", className="shortcut-description"),
                                        ],
                                        className="shortcut-item",
                                    ),
                                    html.Div(
                                        [
                                            html.Div([html.Span("?", className="shortcut-key")], className="shortcut-keys"),
                                            html.Span("Toggle this panel", className="shortcut-description"),
                                        ],
                                        className="shortcut-item",
                                    ),
                                ],
                                className="shortcuts-section",
                            ),
                        ],
                        className="shortcuts-body",
                    ),
                ],
                id="shortcuts-panel",
                className="keyboard-shortcuts-panel hidden",
            ),
        ]
    )

    app.layout = dbc.Container(
        [
            # Stores for keyboard state
            dcc.Store(id="keyboard-event-store", data=None),
            dcc.Store(id="shortcuts-panel-visible", data=False),
            # Store for tracking visited tabs (lazy loading)
            dcc.Store(id="visited-tabs-store", data=["tab-component"]),
            # Hidden div for keyboard listener
            html.Div(id="keyboard-listener", style={"display": "none"}),
            header,
            help_card,
            global_controls,
            dbc.Tabs(
                id="main-tabs",
                active_tab="tab-component",
                children=[
                    component_explorer_tab,
                    layer_explorer_tab,
                    trajectory_tab,
                    position_slider_tab,
                ],
            ),
            keyboard_shortcuts_panel,
        ],
        fluid=True,
        style={"paddingBottom": "40px"},
    )

    # Callback to track visited tabs for lazy loading
    @callback(
        Output("visited-tabs-store", "data"),
        Input("main-tabs", "active_tab"),
        State("visited-tabs-store", "data"),
    )
    def track_visited_tabs(active_tab, visited_tabs):
        """Track which tabs have been visited for lazy loading."""
        if active_tab and active_tab not in visited_tabs:
            return visited_tabs + [active_tab]
        return visited_tabs

    @callback(
        [Output("ce-main-plot", "figure"), Output("ce-metrics-panel", "children")],
        [
            Input("ce-layer-slider", "value"),
            Input("ce-component-radio", "value"),
            Input("ce-position-dropdown", "value"),
            Input("method-radio", "value"),
            Input("color-dropdown", "value"),
            Input("show-no-horizon", "value"),
        ],
    )
    def update_ce_plot(layer, component, position, method, color_by, show_no_horizon):
        try:
            if layer is None or component is None or position is None:
                return create_empty_figure("Select a target"), html.Div(
                    "Select a target to view metrics",
                    className="error-message",
                )

            if method == "pca":
                embedding = loader.load_pca(layer, component, position, n_components=3)
            elif method == "umap":
                embedding = loader.load_umap(layer, component, position, n_components=3)
            else:
                embedding = loader.load_tsne(layer, component, position, n_components=3)

            if embedding is None:
                return create_empty_figure(f"No data for L{layer}_{component}_P{position}"), html.Div(
                    f"No data available for L{layer} {component} @ {position}",
                    className="error-message",
                )

            colors = loader.get_sample_metadata(color_by)
            no_horizon_mask = loader.get_no_horizon_mask()

            if len(colors) > embedding.shape[0]:
                colors = colors[: embedding.shape[0]]
                no_horizon_mask = no_horizon_mask[: embedding.shape[0]]

            is_categorical = color_by in ["time_scale", "choice_type", "short_term_first", "has_horizon"]

            fig = create_3d_scatter(
                embedding=embedding,
                colors=colors,
                color_by=color_by,
                title=f"L{layer} {COMPONENT_NAMES[component]} @ {position} ({method.upper()})",
                is_categorical=is_categorical,
                no_horizon_mask=no_horizon_mask,
                show_no_horizon="show" in show_no_horizon,
                method=method,
            )

            metrics_parts = []
            linear_metrics = loader.load_linear_probe_metrics(layer, component, position)
            if linear_metrics:
                r2 = linear_metrics.get("r2_mean", 0)
                # Determine quality indicator
                r2_quality = "Good" if r2 > 0.5 else "Moderate" if r2 > 0.2 else "Low"
                metrics_parts.append(
                    html.Div(
                        [
                            html.Div(
                                html.Span("R", style={"fontSize": "14px", "fontWeight": "700"}),
                                className="metrics-icon probe",
                            ),
                            html.Div(
                                [
                                    html.Div("Linear Probe R2", className="metrics-label"),
                                    html.Div(f"{r2:.3f}", className="metrics-value", style={"color": COLORS["cyan"]}),
                                    html.Div(f"Predictability: {r2_quality}", className="metrics-subtext"),
                                ],
                                className="metrics-content",
                            ),
                        ],
                        className="metrics-item",
                    )
                )

            pca_metrics = loader.load_pca_metrics(layer, component, position)
            if pca_metrics:
                var_explained = pca_metrics.get("explained_variance_ratio", [])
                if var_explained:
                    total_var = sum(var_explained[:3])
                    # Show individual PC contributions
                    pc_details = ", ".join([f"PC{i}: {v:.1%}" for i, v in enumerate(var_explained[:3])])
                    metrics_parts.append(
                        html.Div(
                            [
                                html.Div(
                                    html.Span("V", style={"fontSize": "14px", "fontWeight": "700"}),
                                    className="metrics-icon pca",
                                ),
                                html.Div(
                                    [
                                        html.Div("Variance Explained (3 PCs)", className="metrics-label"),
                                        html.Div(f"{total_var:.1%}", className="metrics-value", style={"color": COLORS["purple"]}),
                                        html.Div(pc_details, className="metrics-subtext"),
                                    ],
                                    className="metrics-content",
                                ),
                            ],
                            className="metrics-item",
                        )
                    )

            if not metrics_parts:
                metrics_parts = [
                    html.Div(
                        [
                            html.Div(
                                html.Span("-", style={"fontSize": "14px", "fontWeight": "700"}),
                                className="metrics-icon",
                                style={"background": "rgba(180, 160, 200, 0.15)", "color": COLORS["text_muted"]},
                            ),
                            html.Div(
                                [
                                    html.Div("No metrics available", className="metrics-label"),
                                    html.Div(
                                        "Metrics will appear when data is loaded",
                                        className="metrics-subtext",
                                    ),
                                ],
                                className="metrics-content",
                            ),
                        ],
                        className="metrics-item",
                    )
                ]

            return fig, html.Div(metrics_parts)

        except Exception as e:
            return create_empty_figure(f"Error: {str(e)}"), html.Div(
                f"Error loading data: {str(e)}",
                className="error-message",
            )

    @callback(
        [Output("ce-sample-text", "children"), Output("ce-sample-badge", "children")],
        [Input("ce-sample-slider", "value"), Input("ce-sample-input", "value")],
    )
    def update_ce_sample(slider_val, input_val):
        try:
            idx = input_val if input_val is not None else slider_val
            idx = max(0, min(idx, loader.n_samples - 1))

            sample = loader.get_sample_info(idx)
            text = sample.get("text", "No text available")

            horizon = sample.get("time_horizon_months")
            badge_text = f"Sample {idx}"
            if horizon is not None:
                badge_text += f" | {horizon} months"
            else:
                badge_text += " | No Horizon"

            return text, badge_text
        except Exception as e:
            return f"Error: {str(e)}", f"Sample {slider_val or 0}"

    @callback(
        Output("ce-sample-slider", "value"),
        Input("ce-sample-input", "value"),
        prevent_initial_call=True,
    )
    def sync_sample_input_to_slider(val):
        if val is not None:
            return max(0, min(val, loader.n_samples - 1))
        return 0

    @callback(
        [
            Output("le-resid-pre", "figure"),
            Output("le-attn-out", "figure"),
            Output("le-mlp-out", "figure"),
            Output("le-resid-post", "figure"),
        ],
        [
            Input("le-layer-slider", "value"),
            Input("le-position-dropdown", "value"),
            Input("method-radio", "value"),
            Input("color-dropdown", "value"),
            Input("show-no-horizon", "value"),
            Input("main-tabs", "active_tab"),
        ],
    )
    def update_le_plots(layer, position, method, color_by, show_no_horizon, active_tab):
        try:
            # Lazy loading: only compute if this tab is active
            if active_tab != "tab-layer":
                empty = create_empty_figure("Switch to Layer Explorer tab to load")
                return empty, empty, empty, empty

            if layer is None or position is None:
                empty = create_empty_figure("Select layer and position")
                return empty, empty, empty, empty

            colors = loader.get_sample_metadata(color_by)
            no_horizon_mask = loader.get_no_horizon_mask()
            is_categorical = color_by in ["time_scale", "choice_type", "short_term_first", "has_horizon"]

            figs = []
            for comp in ["resid_pre", "attn_out", "mlp_out", "resid_post"]:
                if method == "pca":
                    emb = loader.load_pca(layer, comp, position, n_components=3)
                elif method == "umap":
                    emb = loader.load_umap(layer, comp, position, n_components=3)
                else:
                    emb = loader.load_tsne(layer, comp, position, n_components=3)

                if emb is None:
                    figs.append(create_empty_figure(f"No data for {comp}"))
                    continue

                n = emb.shape[0]
                c = colors[:n] if len(colors) > n else colors
                m = no_horizon_mask[:n] if len(no_horizon_mask) > n else no_horizon_mask

                fig = create_3d_scatter(
                    embedding=emb,
                    colors=c,
                    color_by=color_by,
                    title=f"L{layer} {COMPONENT_NAMES[comp]}",
                    is_categorical=is_categorical,
                    no_horizon_mask=m,
                    show_no_horizon="show" in show_no_horizon,
                    method=method,
                )
                figs.append(fig)

            return figs[0], figs[1], figs[2], figs[3]

        except Exception as e:
            error_fig = create_empty_figure(f"Error: {str(e)}")
            return error_fig, error_fig, error_fig, error_fig

    # Maximum samples to render in trajectory "show all" mode (reduced for performance)
    MAX_TRAJECTORY_SAMPLES = 50

    @callback(
        [Output("traj-plot", "figure"), Output("traj-line-plot", "figure")],
        [
            Input("traj-component-radio", "value"),
            Input("traj-position-dropdown", "value"),
            Input("traj-sample-input", "value"),
            Input("traj-show-all", "value"),
            Input("method-radio", "value"),
            Input("color-dropdown", "value"),
            Input("main-tabs", "active_tab"),
        ],
    )
    def update_trajectory(component, position, sample_idx, show_all, method, color_by, active_tab):
        try:
            # Lazy loading: only compute if this tab is active
            if active_tab != "tab-trajectory":
                empty = create_empty_figure("Switch to Trajectory View tab to load")
                return empty, empty

            if component is None or position is None:
                return create_empty_figure("Select component and position"), create_empty_figure("Select component and position")

            embeddings = loader.load_all_layer_embeddings(component, position, method, n_components=3)

            if not embeddings:
                return create_empty_figure("No embeddings available"), create_empty_figure("No embeddings available")

            layers_sorted = sorted(embeddings.keys())
            n_layers = len(layers_sorted)

            fig3d = go.Figure()

            if "all" in show_all:
                colors = loader.get_sample_metadata(color_by)
                n_samples = embeddings[layers_sorted[0]].shape[0]

                # Reduced from 100 to MAX_TRAJECTORY_SAMPLES for better performance
                for i in range(min(n_samples, MAX_TRAJECTORY_SAMPLES)):
                    xs, ys, zs = [], [], []
                    for layer in layers_sorted:
                        if i < embeddings[layer].shape[0]:
                            xs.append(embeddings[layer][i, 0])
                            ys.append(embeddings[layer][i, 1])
                            zs.append(embeddings[layer][i, 2] if embeddings[layer].shape[1] > 2 else 0)

                    color_val = colors[i] if i < len(colors) else i
                    fig3d.add_trace(
                        go.Scatter3d(
                            x=xs,
                            y=ys,
                            z=zs,
                            mode="lines+markers",
                            marker=dict(size=3, color=color_val, colorscale=[[0, COLORS["cyan"]], [0.5, COLORS["purple"]], [1, COLORS["pink"]]]),
                            line=dict(width=1, color=f"rgba(180, 160, 200, 0.3)"),
                            showlegend=False,
                            hovertemplate=f"Sample {i}<br>%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}}<extra></extra>",
                        )
                    )
            else:
                idx = sample_idx if sample_idx is not None else 0
                xs, ys, zs, layer_labels = [], [], [], []

                for layer in layers_sorted:
                    if idx < embeddings[layer].shape[0]:
                        xs.append(embeddings[layer][idx, 0])
                        ys.append(embeddings[layer][idx, 1])
                        zs.append(embeddings[layer][idx, 2] if embeddings[layer].shape[1] > 2 else 0)
                        layer_labels.append(f"L{layer}")

                fig3d.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        line=dict(width=4, color=COLORS["cyan"]),
                        showlegend=False,
                    )
                )

                fig3d.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="markers+text",
                        marker=dict(
                            size=10,
                            color=list(range(n_layers)),
                            colorscale=[[0, COLORS["cyan"]], [0.5, COLORS["purple"]], [1, COLORS["pink"]]],
                            colorbar=dict(title="Layer", titlefont=dict(color=COLORS["text"]), tickfont=dict(color=COLORS["text"])),
                        ),
                        text=layer_labels,
                        textposition="top center",
                        textfont=dict(color=COLORS["text"], size=10),
                        hovertemplate="Layer: %{text}<br>%{x:.3f}, %{y:.3f}, %{z:.3f}<extra></extra>",
                        showlegend=False,
                    )
                )

            prefix = method.upper()
            fig3d.update_layout(
                title=dict(text=f"Trajectory: {COMPONENT_NAMES[component]} @ {position}", font=dict(color=COLORS["text"], size=16)),
                template="plotly_white",
                paper_bgcolor="rgba(255,255,255,0)",
                font=dict(color=COLORS["text"]),
                scene=dict(
                    xaxis_title=f"{prefix}0",
                    yaxis_title=f"{prefix}1",
                    zaxis_title=f"{prefix}2",
                    xaxis=dict(backgroundcolor="rgba(248,244,255,0.3)", gridcolor="rgba(180,160,200,0.3)"),
                    yaxis=dict(backgroundcolor="rgba(240,247,255,0.3)", gridcolor="rgba(180,160,200,0.3)"),
                    zaxis=dict(backgroundcolor="rgba(254,246,249,0.3)", gridcolor="rgba(180,160,200,0.3)"),
                ),
            )

            fig_line = go.Figure()
            idx = sample_idx if sample_idx is not None else 0

            line_colors = [COLORS["pink"], COLORS["cyan"], COLORS["purple"]]
            for dim, (color, name) in enumerate(zip(line_colors, [f"{prefix}0", f"{prefix}1", f"{prefix}2"])):
                vals = []
                for layer in layers_sorted:
                    if idx < embeddings[layer].shape[0] and embeddings[layer].shape[1] > dim:
                        vals.append(embeddings[layer][idx, dim])
                    else:
                        vals.append(None)

                fig_line.add_trace(
                    go.Scatter(
                        x=layers_sorted,
                        y=vals,
                        mode="lines+markers",
                        name=name,
                        line=dict(color=color, width=3),
                        marker=dict(size=8),
                    )
                )

            fig_line.update_layout(
                title=dict(text=f"Sample {idx} Layer Progression", font=dict(color=COLORS["text"], size=16)),
                template="plotly_white",
                paper_bgcolor="rgba(255,255,255,0)",
                font=dict(color=COLORS["text"]),
                xaxis_title="Layer",
                yaxis_title="Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(255,255,255,0.9)"),
                xaxis=dict(gridcolor="rgba(180,160,200,0.2)"),
                yaxis=dict(gridcolor="rgba(180,160,200,0.2)"),
            )

            return fig3d, fig_line

        except Exception as e:
            error_fig = create_empty_figure(f"Error: {str(e)}")
            return error_fig, error_fig

    @callback(
        [Output("ps-main-plot", "figure"), Output("ps-position-info", "children")],
        [
            Input("ps-layer-slider", "value"),
            Input("ps-component-radio", "value"),
            Input("ps-position-slider", "value"),
            Input("method-radio", "value"),
            Input("color-dropdown", "value"),
            Input("show-no-horizon", "value"),
            Input("main-tabs", "active_tab"),
        ],
    )
    def update_ps_plot(layer, component, pos_idx, method, color_by, show_no_horizon, active_tab):
        try:
            # Lazy loading: only compute if this tab is active
            if active_tab != "tab-position":
                return create_empty_figure("Switch to Position Slider tab to load"), ""

            if layer is None or component is None:
                return create_empty_figure("Select a target"), "Select a target"

            if pos_idx >= len(positions):
                return create_empty_figure("Invalid position"), "Invalid position"

            position = positions[pos_idx]

            if method == "pca":
                embedding = loader.load_pca(layer, component, position, n_components=3)
            elif method == "umap":
                embedding = loader.load_umap(layer, component, position, n_components=3)
            else:
                embedding = loader.load_tsne(layer, component, position, n_components=3)

            if embedding is None:
                return create_empty_figure(f"No data for position {position}"), f"No data for position {position}"

            colors = loader.get_sample_metadata(color_by)
            no_horizon_mask = loader.get_no_horizon_mask()

            n = embedding.shape[0]
            colors = colors[:n] if len(colors) > n else colors
            no_horizon_mask = no_horizon_mask[:n] if len(no_horizon_mask) > n else no_horizon_mask

            is_categorical = color_by in ["time_scale", "choice_type", "short_term_first", "has_horizon"]

            fig = create_3d_scatter(
                embedding=embedding,
                colors=colors,
                color_by=color_by,
                title=f"L{layer} {COMPONENT_NAMES[component]} @ {position}",
                is_categorical=is_categorical,
                no_horizon_mask=no_horizon_mask,
                show_no_horizon="show" in show_no_horizon,
                method=method,
            )

            info = html.Div(
                [
                    html.Strong(f"Position {pos_idx + 1}/{len(positions)}: ", style={"color": COLORS["text"]}),
                    html.Span(position, className="position-badge"),
                ]
            )

            return fig, info

        except Exception as e:
            return create_empty_figure(f"Error: {str(e)}"), f"Error: {str(e)}"

    @callback(
        Output("global-info", "children"),
        [Input("method-radio", "value"), Input("color-dropdown", "value")],
    )
    def update_global_info(method, color_by):
        return html.Div(
            [
                html.Strong("Method: ", style={"color": COLORS["text_dim"]}),
                html.Span(method.upper(), style={"color": COLORS["purple"], "fontWeight": "600"}),
                html.Br(),
                html.Strong("Color: ", style={"color": COLORS["text_dim"]}),
                html.Span(color_by.replace("_", " ").title(), style={"color": COLORS["cyan"], "fontWeight": "600"}),
            ]
        )

    return app


def run_app(
    data_dir: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = True,
):
    """Run the Dash application with threading disabled to avoid numba issues."""
    app = create_app(data_dir)
    app.run(host=host, port=port, debug=debug, threaded=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GeoViz Explorer")
    parser.add_argument("--data-dir", type=str, default="out/geo_viz", help="Data directory")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=8050, help="Port")
    parser.add_argument("--no-debug", action="store_true", help="Disable debug mode")
    args = parser.parse_args()

    run_app(
        data_dir=args.data_dir,
        host=args.host,
        port=args.port,
        debug=not args.no_debug,
    )
