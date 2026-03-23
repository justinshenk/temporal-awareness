"""Interactive Dash app for geometric visualization - Modern Light Pastel Theme.

IMPORTANT: Set numba threading layer before any imports that might trigger numba.
This must be at the very top of the file to prevent threading issues.
"""

import os

# Disable numba parallelism to avoid threading issues with Dash
# This is the safest approach on macOS ARM where TBB isn't available
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, clientside_callback, dcc, html, no_update
from sklearn.decomposition import PCA as SkPCA

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
/* ========================================
   CSS Reset & Variables
   ======================================== */

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
    /* Light mode backgrounds */
    --bg-gradient: linear-gradient(135deg, #fef6f9 0%, #f8f4ff 25%, #f0f7ff 50%, #fff5f0 75%, #fef6f9 100%);
    --bg-card: rgba(255, 255, 255, 0.85);
    --bg-card-border: rgba(180, 160, 200, 0.2);
    --bg-input: #ffffff;
    --bg-panel: rgba(248, 244, 255, 0.8);
    --bg-navbar: linear-gradient(135deg, rgba(255, 107, 157, 0.15) 0%, rgba(198, 120, 221, 0.18) 50%, rgba(97, 175, 239, 0.12) 100%);
    --scrollbar-track: rgba(248, 244, 255, 0.5);
    --scrollbar-thumb: rgba(198, 120, 221, 0.4);
    --loading-overlay: rgba(255, 255, 255, 0.88);
    --tooltip-bg: linear-gradient(135deg, #4a3f5c, #3a2f4c);
}

/* Dark mode variables */
body.dark-mode {
    --text: #e0e0e0;
    --text-dim: #b0a8c0;
    --text-muted: #8a7c9a;
    --bg-gradient: linear-gradient(135deg, #0a0a12 0%, #12101a 25%, #1a1525 50%, #151020 75%, #0a0a12 100%);
    --bg-card: rgba(26, 21, 37, 0.9);
    --bg-card-border: rgba(120, 100, 150, 0.3);
    --bg-input: rgba(20, 16, 28, 0.95);
    --bg-panel: rgba(26, 21, 37, 0.8);
    --bg-navbar: linear-gradient(135deg, rgba(255, 107, 157, 0.12) 0%, rgba(198, 120, 221, 0.15) 50%, rgba(97, 175, 239, 0.1) 100%);
    --scrollbar-track: rgba(26, 21, 37, 0.5);
    --scrollbar-thumb: rgba(198, 120, 221, 0.5);
    --loading-overlay: rgba(10, 10, 18, 0.92);
    --tooltip-bg: linear-gradient(135deg, #2a2538, #1a1528);
}

/* ========================================
   Base Styles & Typography
   ======================================== */

html, body {
    width: 100%;
    min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

body {
    background: var(--bg-gradient);
    background-size: 400% 400%;
    animation: gradientShift 20s ease infinite;
    color: var(--text);
    transition: background 0.4s ease, color 0.3s ease;
}

/* ========================================
   Modern Card Design
   ======================================== */

.glass-card {
    background: var(--bg-card) !important;
    -webkit-backdrop-filter: blur(20px);
    backdrop-filter: blur(20px);
    border: 1px solid var(--bg-card-border) !important;
    border-radius: 24px !important;
    box-shadow: 0 8px 32px rgba(100, 80, 120, 0.08);
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1),
                box-shadow 0.3s cubic-bezier(0.4, 0, 0.2, 1),
                background 0.3s ease,
                border-color 0.3s ease;
    animation: fadeInUp 0.5s ease-out both;
}

/* Stagger delays for multiple cards */
.glass-card:nth-child(1) { animation-delay: 0.05s; }
.glass-card:nth-child(2) { animation-delay: 0.1s; }
.glass-card:nth-child(3) { animation-delay: 0.15s; }
.glass-card:nth-child(4) { animation-delay: 0.2s; }
.glass-card:nth-child(5) { animation-delay: 0.25s; }
.glass-card:nth-child(6) { animation-delay: 0.3s; }

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 48px rgba(100, 80, 120, 0.14);
}

/* Row-based stagger for cards in rows */
.row > .col .glass-card,
.row > [class*="col-"] .glass-card {
    animation: fadeInUp 0.5s ease-out both;
}

.row:nth-child(1) > .col .glass-card,
.row:nth-child(1) > [class*="col-"] .glass-card { animation-delay: 0.1s; }
.row:nth-child(2) > .col .glass-card,
.row:nth-child(2) > [class*="col-"] .glass-card { animation-delay: 0.2s; }
.row:nth-child(3) > .col .glass-card,
.row:nth-child(3) > [class*="col-"] .glass-card { animation-delay: 0.3s; }

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
    transition: background 0.3s ease, border-color 0.3s ease;
}

.sample-text:hover {
    background: rgba(248, 244, 255, 0.95);
    border-color: rgba(198, 120, 221, 0.3);
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

/* ========================================
   Navbar Styling
   ======================================== */

.navbar {
    background: linear-gradient(135deg, rgba(255, 107, 157, 0.15) 0%, rgba(198, 120, 221, 0.18) 50%, rgba(97, 175, 239, 0.12) 100%) !important;
    -webkit-backdrop-filter: blur(20px);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(198, 120, 221, 0.25) !important;
    box-shadow: 0 4px 24px rgba(198, 120, 221, 0.12), 0 1px 3px rgba(100, 80, 120, 0.08);
    padding: 12px 0 !important;
    animation: fadeInUp 0.4s ease-out;
}

.navbar-brand {
    color: var(--text) !important;
    font-weight: 800 !important;
    font-size: 1.75rem !important;
    letter-spacing: -0.02em;
    text-shadow: 0 1px 2px rgba(198, 120, 221, 0.15), 0 0 20px rgba(198, 120, 221, 0.08);
    transition: transform 0.2s ease, text-shadow 0.2s ease;
}

.navbar-brand:hover {
    transform: scale(1.02);
    text-shadow: 0 2px 4px rgba(198, 120, 221, 0.25), 0 0 24px rgba(198, 120, 221, 0.12);
}

.navbar .badge {
    font-weight: 600;
    padding: 8px 14px;
    border-radius: 16px;
    font-size: 12px;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.navbar .badge:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
}

/* ========================================
   Badge Styling
   ======================================== */

.badge {
    font-weight: 600;
    padding: 6px 12px;
    border-radius: 12px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.badge:hover,
.position-badge:hover {
    transform: translateY(-1px) scale(1.03);
    box-shadow: 0 4px 12px rgba(100, 80, 120, 0.2);
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

/* ========================================
   Tab Styling - Modern Pastel Glass Aesthetic
   ======================================== */

.nav-tabs {
    border-bottom: none !important;
    background: rgba(255, 255, 255, 0.6);
    -webkit-backdrop-filter: blur(12px);
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
    -webkit-backdrop-filter: blur(8px);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(180, 160, 200, 0.15);
    border-top: none;
    border-radius: 0 0 20px 20px;
    padding: 20px;
    box-shadow: 0 8px 32px rgba(100, 80, 120, 0.06);
}

/* Tab content fade/slide transition */
.tab-content > .tab-pane {
    animation: none;
}

.tab-content > .tab-pane.active {
    animation: fadeIn 0.35s ease-out, slideInLeft 0.35s ease-out;
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
    background: var(--loading-overlay) !important;
    -webkit-backdrop-filter: blur(12px);
    backdrop-filter: blur(12px);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    border-radius: 20px;
    transition: background 0.3s ease;
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
    background: var(--scrollbar-track);
    transition: background 0.3s ease;
}

::-webkit-scrollbar-thumb {
    background: var(--scrollbar-thumb);
    border-radius: 4px;
    transition: background 0.3s ease;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(198, 120, 221, 0.6);
}

/* ========================================
   Metrics Panel
   ======================================== */

.metrics-item {
    padding: 12px 14px;
    border-bottom: 1px solid rgba(180, 160, 200, 0.15);
    display: flex;
    align-items: center;
    gap: 12px;
    transition: background 0.2s ease, transform 0.2s ease;
    border-radius: 8px;
    margin-bottom: 4px;
}

.metrics-item:hover {
    background: rgba(248, 244, 255, 0.5);
    transform: translateX(4px);
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
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.metrics-item:hover .metrics-icon {
    transform: scale(1.1) rotate(-5deg);
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
    transition: transform 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
    vertical-align: middle;
}

.info-icon:hover {
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.25), rgba(86, 182, 194, 0.25));
    transform: scale(1.15) rotate(5deg);
    box-shadow: 0 2px 8px rgba(198, 120, 221, 0.3);
}

/* Help card styling */
.help-card {
    background: linear-gradient(135deg, rgba(86, 182, 194, 0.08), rgba(198, 120, 221, 0.08), rgba(255, 107, 157, 0.05));
    border: 1px solid rgba(180, 160, 200, 0.2);
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 16px;
    animation: fadeInUp 0.6s ease-out 0.1s both;
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

/* ========================================
   Keyboard Shortcuts Panel
   ======================================== */

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
    animation: fadeInUp 0.5s ease-out 0.5s both;
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
    -webkit-backdrop-filter: blur(20px);
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
    transition: transform 0.15s ease, background 0.15s ease;
}

.shortcut-key:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.2), rgba(86, 182, 194, 0.2));
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


/* ========================================
   Animation Keyframes
   ======================================== */

/* Card entrance animation - fade in from below */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Shimmer effect for loading states */
@keyframes shimmer {
    0% {
        background-position: -200% 0;
    }
    100% {
        background-position: 200% 0;
    }
}

/* Floating animation for decorative elements */
@keyframes float {
    0%, 100% {
        transform: translateY(0) rotate(0deg);
    }
    25% {
        transform: translateY(-10px) rotate(1deg);
    }
    50% {
        transform: translateY(-5px) rotate(0deg);
    }
    75% {
        transform: translateY(-12px) rotate(-1deg);
    }
}

/* Subtle gradient shift for background */
@keyframes gradientShift {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

/* Subtle scale pulse for interactive elements */
@keyframes subtlePulse {
    0%, 100% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.02);
    }
}

/* Fade transition for tab content */
@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

/* Slide in from left for tab panels */
@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-10px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

/* ========================================
   Floating Decorative Shapes
   ======================================== */

.app-container {
    position: relative;
    overflow: hidden;
}

.app-container::before,
.app-container::after {
    content: '';
    position: fixed;
    border-radius: 50%;
    opacity: 0.03;
    pointer-events: none;
    z-index: -1;
}

.app-container::before {
    width: 400px;
    height: 400px;
    background: linear-gradient(135deg, var(--purple), var(--pink));
    top: -100px;
    right: -100px;
    animation: float 15s ease-in-out infinite;
}

.app-container::after {
    width: 300px;
    height: 300px;
    background: linear-gradient(135deg, var(--cyan), var(--blue));
    bottom: -50px;
    left: -50px;
    animation: float 18s ease-in-out infinite reverse;
}

/* ========================================
   Loading State Animations
   ======================================== */

/* Shimmer loading placeholder */
.loading-shimmer {
    background: linear-gradient(
        90deg,
        rgba(248, 244, 255, 0.6) 0%,
        rgba(255, 255, 255, 0.8) 50%,
        rgba(248, 244, 255, 0.6) 100%
    );
    background-size: 200% 100%;
    animation: shimmer 1.5s ease-in-out infinite;
}

/* Graph container loading state */
.js-plotly-plot.loading {
    position: relative;
}

.js-plotly-plot.loading::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(198, 120, 221, 0.1) 50%,
        transparent 100%
    );
    background-size: 200% 100%;
    animation: shimmer 2s ease-in-out infinite;
    pointer-events: none;
}

/* ========================================
   Additional Hover Animations
   ======================================== */

/* Dropdown hover glow */
.dash-dropdown .Select-control {
    transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease;
}

.dash-dropdown .Select-control:hover {
    transform: translateY(-1px);
}

/* Form control focus animation */
.form-control {
    transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease;
}

.form-control:focus {
    transform: translateY(-1px);
}

/* Input number hover */
input[type="number"] {
    transition: all 0.2s ease;
}

input[type="number"]:hover:not(:focus) {
    border-color: rgba(198, 120, 221, 0.4) !important;
}

/* Sample stat card animations */
.sample-stat-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.sample-stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(100, 80, 120, 0.12);
}

/* ========================================
   Reduced Motion Preferences
   ======================================== */

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }

    body {
        animation: none;
        background-size: 100% 100%;
    }

    .app-container::before,
    .app-container::after {
        animation: none;
    }

    .glass-card,
    .help-card,
    .navbar,
    .keyboard-shortcuts-toggle {
        animation: none;
        opacity: 1;
        transform: none;
    }

    .tab-content > .tab-pane.active {
        animation: none;
    }

    .nav-tabs .nav-link.active:hover {
        animation: none;
    }
}

/* ========================================
   Dark Mode Toggle Button
   ======================================== */

.dark-mode-toggle {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.3);
    border: 1px solid rgba(180, 160, 200, 0.3);
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin-left: 12px;
}

.dark-mode-toggle:hover {
    background: rgba(255, 255, 255, 0.5);
    border-color: rgba(198, 120, 221, 0.5);
    transform: scale(1.05);
}

.dark-mode-toggle:active {
    transform: scale(0.95);
}

.dark-mode-toggle .toggle-icon {
    font-size: 20px;
    transition: transform 0.3s ease, opacity 0.3s ease;
}

/* Dark mode specific toggle styling */
body.dark-mode .dark-mode-toggle {
    background: rgba(26, 21, 37, 0.6);
    border-color: rgba(198, 120, 221, 0.4);
}

body.dark-mode .dark-mode-toggle:hover {
    background: rgba(40, 30, 55, 0.8);
    border-color: rgba(198, 120, 221, 0.6);
}

/* Dark mode overrides for various elements */
body.dark-mode .navbar {
    background: var(--bg-navbar) !important;
    border-bottom-color: rgba(198, 120, 221, 0.2) !important;
}

body.dark-mode .navbar-brand {
    color: var(--text) !important;
}

body.dark-mode .card-header {
    border-bottom-color: rgba(120, 100, 150, 0.25) !important;
}

body.dark-mode .sample-text,
body.dark-mode .sample-prompt {
    background: var(--bg-panel);
    border-color: var(--bg-card-border);
}

body.dark-mode .info-panel {
    background: linear-gradient(135deg, rgba(86, 182, 194, 0.08), rgba(198, 120, 221, 0.08));
}

body.dark-mode .help-card {
    background: linear-gradient(135deg, rgba(86, 182, 194, 0.06), rgba(198, 120, 221, 0.06), rgba(255, 107, 157, 0.04));
    border-color: var(--bg-card-border);
}

body.dark-mode .nav-tabs {
    background: rgba(26, 21, 37, 0.6);
    border-color: var(--bg-card-border);
}

body.dark-mode .tab-content {
    background: rgba(26, 21, 37, 0.5);
    border-color: var(--bg-card-border);
}

body.dark-mode .nav-tabs .nav-link {
    color: var(--text-dim) !important;
}

body.dark-mode .nav-tabs .nav-link:hover {
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.12), rgba(255, 107, 157, 0.08), rgba(97, 175, 239, 0.06)) !important;
    border-color: rgba(198, 120, 221, 0.25) !important;
}

body.dark-mode .metrics-item:hover {
    background: rgba(40, 30, 55, 0.5);
}

body.dark-mode .keyboard-shortcuts-panel {
    background: rgba(26, 21, 37, 0.98);
    border-color: var(--bg-card-border);
}

body.dark-mode .shortcut-key {
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.15), rgba(86, 182, 194, 0.15));
    border-color: rgba(120, 100, 150, 0.4);
}

body.dark-mode .tooltip-inner {
    background: var(--tooltip-bg) !important;
}

/* Dark mode dropdown overrides */
body.dark-mode .dash-dropdown .Select-control {
    background: var(--bg-input) !important;
    border-color: var(--bg-card-border) !important;
}

body.dark-mode .dash-dropdown .Select-value-label,
body.dark-mode .dash-dropdown .Select-input input {
    color: var(--text) !important;
}

body.dark-mode .dash-dropdown .Select-placeholder {
    color: var(--text-muted) !important;
}

body.dark-mode .dash-dropdown .Select-menu-outer {
    background: var(--bg-input) !important;
    border-color: var(--bg-card-border) !important;
}

body.dark-mode .dash-dropdown .Select-menu {
    background: var(--bg-input) !important;
}

body.dark-mode .dash-dropdown .VirtualizedSelectOption {
    background: var(--bg-input) !important;
    color: var(--text) !important;
}

body.dark-mode .dash-dropdown .VirtualizedSelectOption:hover,
body.dark-mode .dash-dropdown .VirtualizedSelectFocusedOption {
    background: rgba(198, 120, 221, 0.2) !important;
}

/* Dark mode form controls */
body.dark-mode .form-control {
    background: var(--bg-input) !important;
    border-color: var(--bg-card-border) !important;
    color: var(--text) !important;
}

body.dark-mode .form-check-input[type="radio"] + .form-check-label::before,
body.dark-mode .form-check-input[type="checkbox"]:not(.form-switch .form-check-input) + .form-check-label::before {
    background: linear-gradient(135deg, rgba(26, 21, 37, 0.95), rgba(40, 30, 55, 0.9));
    border-color: rgba(120, 100, 150, 0.5);
}

body.dark-mode .form-switch .form-check-label::before {
    background: linear-gradient(135deg, rgba(60, 50, 80, 0.5), rgba(80, 60, 100, 0.6));
    border-color: rgba(120, 100, 150, 0.5);
}

body.dark-mode .form-switch .form-check-label::after {
    background: linear-gradient(135deg, #e0e0e0, #c0c0c0);
}

body.dark-mode .rc-slider-rail {
    background: linear-gradient(90deg, rgba(40, 30, 55, 0.8), rgba(50, 40, 70, 0.8)) !important;
    border-color: rgba(120, 100, 150, 0.3) !important;
}

body.dark-mode .rc-slider-dot {
    background: rgba(40, 30, 55, 0.95) !important;
    border-color: rgba(120, 100, 150, 0.5) !important;
}

/* ========================================
   Share Button & Modal
   ======================================== */

.share-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(152, 195, 121, 0.2), rgba(86, 182, 194, 0.2));
    border: 1px solid rgba(152, 195, 121, 0.4);
    color: var(--text);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin-left: 12px;
}

.share-btn:hover {
    background: linear-gradient(135deg, rgba(152, 195, 121, 0.35), rgba(86, 182, 194, 0.35));
    border-color: rgba(152, 195, 121, 0.6);
    transform: scale(1.05);
    box-shadow: 0 4px 16px rgba(152, 195, 121, 0.25);
}

.share-btn:active {
    transform: scale(0.95);
}

.share-btn .share-icon {
    font-size: 16px;
}

body.dark-mode .share-btn {
    background: linear-gradient(135deg, rgba(152, 195, 121, 0.15), rgba(86, 182, 194, 0.15));
    border-color: rgba(152, 195, 121, 0.3);
}

body.dark-mode .share-btn:hover {
    background: linear-gradient(135deg, rgba(152, 195, 121, 0.25), rgba(86, 182, 194, 0.25));
    border-color: rgba(152, 195, 121, 0.5);
}

/* Share Modal Styling */
.share-modal .modal-content {
    background: var(--bg-card) !important;
    border: 1px solid var(--bg-card-border) !important;
    border-radius: 20px !important;
    -webkit-backdrop-filter: blur(20px);
    backdrop-filter: blur(20px);
    box-shadow: 0 12px 40px rgba(100, 80, 120, 0.2);
}

.share-modal .modal-header {
    border-bottom: 1px solid rgba(180, 160, 200, 0.2) !important;
    padding: 20px 24px !important;
}

.share-modal .modal-title {
    color: var(--text) !important;
    font-weight: 700;
    font-size: 1.25rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

.share-modal .modal-body {
    padding: 24px !important;
}

.share-modal .modal-footer {
    border-top: 1px solid rgba(180, 160, 200, 0.2) !important;
    padding: 16px 24px !important;
}

.share-modal .btn-close {
    filter: none;
    opacity: 0.6;
}

.share-modal .btn-close:hover {
    opacity: 1;
}

body.dark-mode .share-modal .btn-close {
    filter: invert(1);
}

/* Share Option Cards */
.share-option-card {
    background: linear-gradient(135deg, rgba(248, 244, 255, 0.5), rgba(255, 255, 255, 0.5));
    border: 1px solid rgba(180, 160, 200, 0.2);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    transition: all 0.3s ease;
}

.share-option-card:hover {
    border-color: rgba(198, 120, 221, 0.3);
    box-shadow: 0 4px 16px rgba(100, 80, 120, 0.08);
}

body.dark-mode .share-option-card {
    background: linear-gradient(135deg, rgba(26, 21, 37, 0.5), rgba(40, 30, 55, 0.5));
    border-color: rgba(120, 100, 150, 0.3);
}

.share-option-title {
    font-size: 14px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.share-option-title .option-icon {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
}

.share-option-title .option-icon.url {
    background: linear-gradient(135deg, rgba(97, 175, 239, 0.2), rgba(86, 182, 194, 0.2));
    color: var(--blue);
}

.share-option-title .option-icon.export {
    background: linear-gradient(135deg, rgba(152, 195, 121, 0.2), rgba(86, 182, 194, 0.2));
    color: var(--mint);
}

.share-option-title .option-icon.import {
    background: linear-gradient(135deg, rgba(229, 192, 123, 0.2), rgba(209, 154, 102, 0.2));
    color: var(--gold);
}

.share-option-desc {
    font-size: 12px;
    color: var(--text-dim);
    margin-bottom: 12px;
    line-height: 1.5;
}

/* URL Display */
.url-display {
    display: flex;
    gap: 8px;
    align-items: stretch;
}

.url-input {
    flex: 1;
    background: var(--bg-input) !important;
    border: 1px solid rgba(180, 160, 200, 0.3) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-family: 'Roboto Mono', 'SF Mono', monospace;
    font-size: 12px;
    color: var(--text) !important;
}

.url-input:focus {
    border-color: var(--purple) !important;
    box-shadow: 0 0 0 3px rgba(198, 120, 221, 0.2) !important;
    outline: none;
}

body.dark-mode .url-input {
    background: rgba(20, 16, 28, 0.8) !important;
    border-color: rgba(120, 100, 150, 0.4) !important;
}

/* Action Buttons */
.share-action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    padding: 10px 18px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    border: none;
}

.share-action-btn.primary {
    background: linear-gradient(135deg, var(--purple), var(--pink));
    color: white;
}

.share-action-btn.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(198, 120, 221, 0.35);
}

.share-action-btn.secondary {
    background: rgba(180, 160, 200, 0.15);
    color: var(--text);
    border: 1px solid rgba(180, 160, 200, 0.3);
}

.share-action-btn.secondary:hover {
    background: rgba(180, 160, 200, 0.25);
    border-color: rgba(180, 160, 200, 0.5);
}

.share-action-btn.copied {
    background: linear-gradient(135deg, var(--mint), var(--cyan)) !important;
}

/* JSON Display Area */
.json-display {
    background: var(--bg-input);
    border: 1px solid rgba(180, 160, 200, 0.3);
    border-radius: 12px;
    padding: 14px;
    font-family: 'Roboto Mono', 'SF Mono', monospace;
    font-size: 11px;
    line-height: 1.5;
    color: var(--text);
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
}

body.dark-mode .json-display {
    background: rgba(20, 16, 28, 0.8);
    border-color: rgba(120, 100, 150, 0.4);
}

/* Import Textarea */
.import-textarea {
    width: 100%;
    min-height: 120px;
    background: var(--bg-input) !important;
    border: 1px solid rgba(180, 160, 200, 0.3) !important;
    border-radius: 12px !important;
    padding: 14px !important;
    font-family: 'Roboto Mono', 'SF Mono', monospace;
    font-size: 12px;
    color: var(--text) !important;
    resize: vertical;
}

.import-textarea:focus {
    border-color: var(--purple) !important;
    box-shadow: 0 0 0 3px rgba(198, 120, 221, 0.2) !important;
    outline: none;
}

body.dark-mode .import-textarea {
    background: rgba(20, 16, 28, 0.8) !important;
    border-color: rgba(120, 100, 150, 0.4) !important;
}

/* Status Messages */
.share-status {
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 500;
    margin-top: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.share-status.success {
    background: rgba(152, 195, 121, 0.15);
    border: 1px solid rgba(152, 195, 121, 0.3);
    color: var(--mint);
}

.share-status.error {
    background: rgba(224, 108, 117, 0.15);
    border: 1px solid rgba(224, 108, 117, 0.3);
    color: var(--coral);
}

.share-status.info {
    background: rgba(97, 175, 239, 0.15);
    border: 1px solid rgba(97, 175, 239, 0.3);
    color: var(--blue);
}

/* ========================================
   Sample Filter Panel Styles
   ======================================== */

.sample-filter-panel {
    margin-bottom: 16px;
}

.filter-toggle-btn {
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.1), rgba(86, 182, 194, 0.1)) !important;
    border: 1px solid rgba(180, 160, 200, 0.3) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

.filter-toggle-btn:hover {
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.2), rgba(86, 182, 194, 0.2)) !important;
    border-color: var(--purple) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(198, 120, 221, 0.15) !important;
}

.filter-toggle-btn.active {
    background: linear-gradient(135deg, var(--purple), var(--pink)) !important;
    color: white !important;
    border-color: transparent !important;
}

.filter-collapse-content {
    margin-top: 12px;
}

.quick-filter-btn {
    background: rgba(255, 255, 255, 0.8) !important;
    border: 1px solid rgba(180, 160, 200, 0.3) !important;
    color: var(--text-dim) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 12px !important;
    border-radius: 16px !important;
    transition: all 0.2s ease !important;
    margin: 2px !important;
}

.quick-filter-btn:hover {
    background: rgba(198, 120, 221, 0.15) !important;
    border-color: var(--purple) !important;
    color: var(--purple) !important;
}

.quick-filter-btn.active {
    background: linear-gradient(135deg, var(--purple), var(--pink)) !important;
    color: white !important;
    border-color: transparent !important;
}

.filter-match-count {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: linear-gradient(135deg, rgba(152, 195, 121, 0.15), rgba(86, 182, 194, 0.15));
    border: 1px solid rgba(152, 195, 121, 0.3);
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    color: var(--mint);
}

.filter-match-count.warning {
    background: linear-gradient(135deg, rgba(229, 192, 123, 0.15), rgba(209, 154, 102, 0.15));
    border-color: rgba(229, 192, 123, 0.3);
    color: var(--gold);
}

.filter-match-count.error {
    background: linear-gradient(135deg, rgba(224, 108, 117, 0.15), rgba(209, 154, 102, 0.15));
    border-color: rgba(224, 108, 117, 0.3);
    color: var(--coral);
}

.filter-section-title {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
    margin-bottom: 8px;
}

.search-input-container {
    position: relative;
}

.search-input-container .search-icon {
    position: absolute;
    left: 12px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--text-muted);
    font-size: 14px;
}

.search-input {
    padding-left: 36px !important;
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid rgba(180, 160, 200, 0.3) !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    transition: all 0.2s ease !important;
}

.search-input:focus {
    border-color: var(--purple) !important;
    box-shadow: 0 0 0 3px rgba(198, 120, 221, 0.15) !important;
}

.search-input::placeholder {
    color: var(--text-muted) !important;
}

.horizon-range-slider {
    padding: 0 8px;
}

.clear-filters-btn {
    background: rgba(224, 108, 117, 0.1) !important;
    border: 1px solid rgba(224, 108, 117, 0.3) !important;
    color: var(--coral) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.clear-filters-btn:hover {
    background: var(--coral) !important;
    color: white !important;
    border-color: transparent !important;
}

/* Highlighted points in 3D scatter */
.highlighted-point {
    opacity: 1 !important;
}

.dimmed-point {
    opacity: 0.15 !important;
}

/* ========================================
   Statistics Panel - Slide-out Drawer
   ======================================== */

/* Toggle button */
.stats-panel-toggle {
    position: fixed;
    bottom: 24px;
    left: 24px;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--cyan), var(--blue));
    border: none;
    color: white;
    font-size: 18px;
    font-weight: 700;
    cursor: pointer;
    box-shadow: 0 4px 16px rgba(86, 182, 194, 0.4);
    transition: all 0.3s ease;
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fadeInUp 0.5s ease-out 0.5s both;
}

.stats-panel-toggle:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 24px rgba(86, 182, 194, 0.5);
}

.stats-panel-toggle.active {
    background: linear-gradient(135deg, var(--purple), var(--pink));
    box-shadow: 0 4px 16px rgba(198, 120, 221, 0.4);
}

/* Panel container */
.stats-panel {
    position: fixed;
    bottom: 84px;
    left: 24px;
    width: 380px;
    max-height: 75vh;
    overflow-y: auto;
    background: rgba(255, 255, 255, 0.98);
    -webkit-backdrop-filter: blur(20px);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(180, 160, 200, 0.3);
    border-radius: 20px;
    box-shadow: 0 12px 40px rgba(100, 80, 120, 0.2);
    z-index: 999;
    padding: 0;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

body.dark-mode .stats-panel {
    background: rgba(26, 21, 37, 0.98);
    border-color: rgba(120, 100, 150, 0.4);
}

.stats-panel.hidden {
    opacity: 0;
    transform: translateY(20px) scale(0.95);
    pointer-events: none;
}

.stats-panel.visible {
    opacity: 1;
    transform: translateY(0) scale(1);
}

/* Panel header */
.stats-header {
    padding: 16px 20px;
    border-bottom: 1px solid rgba(180, 160, 200, 0.2);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, rgba(86, 182, 194, 0.08), rgba(97, 175, 239, 0.08));
    border-radius: 20px 20px 0 0;
}

.stats-header h4 {
    margin: 0;
    font-size: 15px;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 8px;
}

.stats-close {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    font-size: 20px;
    padding: 4px;
    line-height: 1;
    transition: color 0.2s ease;
}

.stats-close:hover {
    color: var(--purple);
}

/* Panel body */
.stats-body {
    padding: 16px 20px;
}

/* Collapsible sections */
.stats-section {
    margin-bottom: 16px;
    border: 1px solid rgba(180, 160, 200, 0.15);
    border-radius: 14px;
    overflow: hidden;
    transition: all 0.3s ease;
}

.stats-section:last-child {
    margin-bottom: 0;
}

.stats-section:hover {
    border-color: rgba(180, 160, 200, 0.3);
}

.stats-section-header {
    padding: 12px 16px;
    background: linear-gradient(135deg, rgba(248, 244, 255, 0.8), rgba(240, 247, 255, 0.8));
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: background 0.2s ease;
    user-select: none;
}

body.dark-mode .stats-section-header {
    background: linear-gradient(135deg, rgba(40, 35, 55, 0.8), rgba(35, 30, 50, 0.8));
}

.stats-section-header:hover {
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.1), rgba(86, 182, 194, 0.1));
}

.stats-section-title {
    font-size: 13px;
    font-weight: 700;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 8px;
}

.stats-section-icon {
    font-size: 14px;
}

.stats-section-toggle {
    font-size: 12px;
    color: var(--text-muted);
    transition: transform 0.3s ease;
}

.stats-section.collapsed .stats-section-toggle {
    transform: rotate(-90deg);
}

.stats-section-content {
    padding: 14px 16px;
    background: rgba(255, 255, 255, 0.5);
    max-height: 400px;
    overflow: hidden;
    transition: max-height 0.3s ease, padding 0.3s ease, opacity 0.3s ease;
}

body.dark-mode .stats-section-content {
    background: rgba(20, 16, 28, 0.5);
}

.stats-section.collapsed .stats-section-content {
    max-height: 0;
    padding: 0 16px;
    opacity: 0;
}

/* Summary cards grid */
.stats-summary-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}

.stats-summary-card {
    background: linear-gradient(135deg, rgba(248, 244, 255, 0.9), rgba(255, 255, 255, 0.9));
    border: 1px solid rgba(180, 160, 200, 0.2);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
    transition: all 0.2s ease;
}

body.dark-mode .stats-summary-card {
    background: linear-gradient(135deg, rgba(40, 35, 55, 0.9), rgba(30, 25, 45, 0.9));
    border-color: rgba(120, 100, 150, 0.3);
}

.stats-summary-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(100, 80, 120, 0.15);
}

.stats-summary-card.highlight {
    border-color: rgba(198, 120, 221, 0.4);
    background: linear-gradient(135deg, rgba(198, 120, 221, 0.08), rgba(255, 107, 157, 0.05));
}

.stats-card-value {
    font-size: 24px;
    font-weight: 800;
    font-family: 'Roboto Mono', monospace;
    background: linear-gradient(135deg, var(--purple), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}

.stats-card-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-top: 4px;
    font-weight: 600;
}

/* Stats list items */
.stats-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.stats-list-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(180, 160, 200, 0.1);
    font-size: 13px;
}

.stats-list-item:last-child {
    border-bottom: none;
}

.stats-list-label {
    color: var(--text-dim);
    display: flex;
    align-items: center;
    gap: 6px;
}

.stats-list-value {
    font-weight: 600;
    color: var(--text);
    font-family: 'Roboto Mono', monospace;
}

.stats-list-value.positive {
    color: var(--mint);
}

.stats-list-value.negative {
    color: var(--coral);
}

/* Mini chart container */
.stats-chart-container {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(180, 160, 200, 0.15);
    border-radius: 10px;
    padding: 8px;
    margin-top: 10px;
}

body.dark-mode .stats-chart-container {
    background: rgba(20, 16, 28, 0.6);
    border-color: rgba(120, 100, 150, 0.3);
}

/* Distribution bar styling */
.stats-distribution {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.stats-dist-row {
    display: flex;
    align-items: center;
    gap: 10px;
}

.stats-dist-label {
    width: 90px;
    font-size: 11px;
    color: var(--text-dim);
    text-align: right;
    flex-shrink: 0;
}

.stats-dist-bar-container {
    flex: 1;
    height: 18px;
    background: rgba(180, 160, 200, 0.15);
    border-radius: 9px;
    overflow: hidden;
    position: relative;
}

.stats-dist-bar {
    height: 100%;
    border-radius: 9px;
    transition: width 0.5s ease;
    position: relative;
}

.stats-dist-bar.pink { background: linear-gradient(90deg, var(--pink), rgba(255, 107, 157, 0.7)); }
.stats-dist-bar.purple { background: linear-gradient(90deg, var(--purple), rgba(198, 120, 221, 0.7)); }
.stats-dist-bar.cyan { background: linear-gradient(90deg, var(--cyan), rgba(86, 182, 194, 0.7)); }
.stats-dist-bar.gold { background: linear-gradient(90deg, var(--gold), rgba(229, 192, 123, 0.7)); }
.stats-dist-bar.mint { background: linear-gradient(90deg, var(--mint), rgba(152, 195, 121, 0.7)); }
.stats-dist-bar.blue { background: linear-gradient(90deg, var(--blue), rgba(97, 175, 239, 0.7)); }

.stats-dist-value {
    width: 45px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text);
    font-family: 'Roboto Mono', monospace;
    text-align: right;
    flex-shrink: 0;
}

/* Variance explained bars */
.variance-bar-container {
    margin-top: 8px;
}

.variance-bar-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}

.variance-bar-label {
    width: 35px;
    font-size: 11px;
    color: var(--text-muted);
    font-weight: 600;
}

.variance-bar-bg {
    flex: 1;
    height: 10px;
    background: rgba(180, 160, 200, 0.15);
    border-radius: 5px;
    overflow: hidden;
}

.variance-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.5s ease;
}

.variance-bar-fill.pc1 { background: linear-gradient(90deg, var(--pink), var(--purple)); }
.variance-bar-fill.pc2 { background: linear-gradient(90deg, var(--cyan), var(--blue)); }
.variance-bar-fill.pc3 { background: linear-gradient(90deg, var(--mint), var(--cyan)); }
.variance-bar-fill.cumulative { background: linear-gradient(90deg, var(--gold), var(--orange)); }

.variance-bar-value {
    width: 42px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text);
    font-family: 'Roboto Mono', monospace;
    text-align: right;
}

/* Loading state for stats */
.stats-loading {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
    color: var(--text-muted);
    font-size: 13px;
    gap: 10px;
}

.stats-loading-spinner {
    width: 20px;
    height: 20px;
    border: 2px solid rgba(198, 120, 221, 0.2);
    border-top-color: var(--purple);
    border-radius: 50%;
    animation: spin-gradient 0.8s linear infinite;
}

/* No data state */
.stats-no-data {
    text-align: center;
    padding: 16px;
    color: var(--text-muted);
    font-size: 12px;
    font-style: italic;
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
    highlight_mask: np.ndarray | None = None,
) -> go.Figure:
    """Create a 3D scatter plot with pastel styling, PNG export, and optional highlighting.

    Args:
        embedding: 3D coordinates for each sample
        colors: Color values for each sample
        color_by: Name of the color variable
        title: Plot title
        is_categorical: Whether colors are categorical
        no_horizon_mask: Boolean mask for samples without time horizon
        show_no_horizon: Whether to show no-horizon samples
        method: Dimension reduction method used (for axis labels)
        highlight_mask: Boolean mask for samples to highlight (matching filter)
    """
    axis_prefix = method.upper()

    # Track original indices for hover info
    sample_indices = np.arange(len(embedding))

    # Handle no_horizon visibility
    original_highlight_mask = highlight_mask
    if no_horizon_mask is not None and not show_no_horizon:
        mask = ~no_horizon_mask
        embedding = embedding[mask]
        colors = colors[mask]
        sample_indices = sample_indices[mask]
        if highlight_mask is not None:
            highlight_mask = highlight_mask[mask]
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
            custom_data=[sample_indices],
            labels={
                "x": f"{axis_prefix}0",
                "y": f"{axis_prefix}1",
                "z": f"{axis_prefix}2",
                "color": colorbar_title,
            },
            title=title,
        )
        # Update hover template to show sample index
        fig.update_traces(
            hovertemplate=f"<b>Sample %{{customdata[0]}}</b><br>{axis_prefix}0: %{{x:.3f}}<br>{axis_prefix}1: %{{y:.3f}}<br>{axis_prefix}2: %{{z:.3f}}<br>{colorbar_title}: %{{marker.color}}<extra></extra>"
        )
    elif no_horizon_mask is not None and show_no_horizon:
        horizon_mask = ~no_horizon_mask

        fig = go.Figure()

        if horizon_mask.any():
            horizon_indices = sample_indices[horizon_mask]
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
                    customdata=horizon_indices,
                    name="With Horizon",
                    hovertemplate=f"<b>Sample %{{customdata}}</b><br>{axis_prefix}0: %{{x:.3f}}<br>{axis_prefix}1: %{{y:.3f}}<br>{axis_prefix}2: %{{z:.3f}}<extra>With Horizon</extra>",
                )
            )

        if no_horizon_mask.any():
            no_horizon_indices = sample_indices[no_horizon_mask]
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
                    customdata=no_horizon_indices,
                    name="No Horizon",
                    hovertemplate=f"<b>Sample %{{customdata}}</b><br>{axis_prefix}0: %{{x:.3f}}<br>{axis_prefix}1: %{{y:.3f}}<br>{axis_prefix}2: %{{z:.3f}}<extra>No Horizon</extra>",
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
            custom_data=[sample_indices],
            labels={
                "x": f"{axis_prefix}0",
                "y": f"{axis_prefix}1",
                "z": f"{axis_prefix}2",
                "color": color_by.replace("_", " ").title(),
            },
            title=title,
        )
        # Apply colorbar styling and hover template
        fig.update_coloraxes(colorbar=colorbar_style)
        fig.update_traces(
            hovertemplate=f"<b>Sample %{{customdata[0]}}</b><br>{axis_prefix}0: %{{x:.3f}}<br>{axis_prefix}1: %{{y:.3f}}<br>{axis_prefix}2: %{{z:.3f}}<br>{colorbar_title}: %{{marker.color:.3f}}<extra></extra>"
        )

    # Apply highlight mask if provided (dim non-matching samples)
    if highlight_mask is not None and len(highlight_mask) > 0:
        # Create opacity array: highlighted samples get full opacity, others get dimmed
        opacities = np.where(highlight_mask, 0.9, 0.15)
        # Also increase size for highlighted samples
        sizes = np.where(highlight_mask, 6, 3)

        # Update each trace with per-point opacity
        for trace in fig.data:
            if hasattr(trace, 'customdata') and trace.customdata is not None:
                # Map sample indices to opacity values
                trace_indices = trace.customdata
                if isinstance(trace_indices, np.ndarray) and trace_indices.ndim == 1:
                    trace_opacities = [opacities[int(idx)] if int(idx) < len(opacities) else 0.15 for idx in trace_indices]
                    trace_sizes = [sizes[int(idx)] if int(idx) < len(sizes) else 3 for idx in trace_indices]
                elif hasattr(trace_indices, '__iter__'):
                    # Handle nested customdata format
                    try:
                        if len(trace_indices) > 0 and hasattr(trace_indices[0], '__iter__'):
                            trace_opacities = [opacities[int(cd[0])] if int(cd[0]) < len(opacities) else 0.15 for cd in trace_indices]
                            trace_sizes = [sizes[int(cd[0])] if int(cd[0]) < len(sizes) else 3 for cd in trace_indices]
                        else:
                            trace_opacities = [opacities[int(idx)] if int(idx) < len(opacities) else 0.15 for idx in trace_indices]
                            trace_sizes = [sizes[int(idx)] if int(idx) < len(sizes) else 3 for idx in trace_indices]
                    except (TypeError, IndexError):
                        trace_opacities = 0.8
                        trace_sizes = 4
                else:
                    trace_opacities = 0.8
                    trace_sizes = 4

                trace.update(marker=dict(opacity=trace_opacities, size=trace_sizes))
            else:
                # Fallback for traces without customdata
                trace.update(marker=dict(size=4, opacity=0.8))
    else:
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
                        # Share button
                        dbc.NavItem(
                            html.Button(
                                [
                                    html.Span("\u21AA", className="share-icon"),
                                    "Share",
                                ],
                                id="share-btn",
                                className="share-btn",
                                n_clicks=0,
                            )
                        ),
                        # Dark mode toggle button
                        dbc.NavItem(
                            html.Button(
                                html.Span(id="dark-mode-icon", className="toggle-icon"),
                                id="dark-mode-toggle",
                                className="dark-mode-toggle",
                                n_clicks=0,
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

    # Sample filter panel with collapsible options
    # Get unique values for filter dropdowns
    time_scales_unique = sorted(set(
        s.get("time_scale", "unknown") for s in loader.samples if s.get("time_scale")
    ))
    choice_types_unique = sorted(set(
        s.get("choice_type", "unknown") for s in loader.samples if s.get("choice_type")
    ))
    # Calculate time horizon range
    horizons = [s.get("time_horizon_months", 0) for s in loader.samples if s.get("time_horizon_months") is not None]
    min_horizon = min(horizons) if horizons else 0
    max_horizon = max(horizons) if horizons else 120

    sample_filter_panel = html.Div(
        [
            # Toggle button and match count row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Button(
                                [
                                    html.Span("Filter Samples", style={"marginRight": "8px"}),
                                    html.Span(id="filter-toggle-icon", children=""),
                                ],
                                id="filter-toggle-btn",
                                className="filter-toggle-btn",
                                n_clicks=0,
                            ),
                        ],
                        width="auto",
                    ),
                    dbc.Col(
                        [
                            html.Div(id="filter-match-count", className="filter-match-count"),
                        ],
                        width="auto",
                        className="d-flex align-items-center",
                    ),
                    dbc.Col(
                        [
                            # Quick filter buttons
                            html.Div(
                                [
                                    dbc.Button("Long-term", id="quick-filter-longterm", className="quick-filter-btn", size="sm", n_clicks=0),
                                    dbc.Button("Short-term", id="quick-filter-shortterm", className="quick-filter-btn", size="sm", n_clicks=0),
                                    dbc.Button("With Horizon", id="quick-filter-with-horizon", className="quick-filter-btn", size="sm", n_clicks=0),
                                    dbc.Button("No Horizon", id="quick-filter-no-horizon", className="quick-filter-btn", size="sm", n_clicks=0),
                                ],
                                style={"display": "flex", "gap": "4px", "flexWrap": "wrap"},
                            ),
                        ],
                        className="d-flex align-items-center justify-content-end",
                    ),
                ],
                className="mb-2",
                align="center",
            ),
            # Collapsible filter content
            dbc.Collapse(
                dbc.Card(
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    # Text search
                                    dbc.Col(
                                        [
                                            html.Div("Search Text", className="filter-section-title"),
                                            html.Div(
                                                [
                                                    html.Span("", className="search-icon"),
                                                    dbc.Input(
                                                        id="filter-search-text",
                                                        type="text",
                                                        placeholder="Search in sample text...",
                                                        className="search-input",
                                                        debounce=True,
                                                    ),
                                                ],
                                                className="search-input-container",
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    # Time scale filter
                                    dbc.Col(
                                        [
                                            html.Div("Time Scale", className="filter-section-title"),
                                            dcc.Dropdown(
                                                id="filter-time-scale",
                                                options=[{"label": "All", "value": "all"}] + [
                                                    {"label": ts.replace("-", " ").title(), "value": ts}
                                                    for ts in time_scales_unique
                                                ],
                                                value="all",
                                                clearable=False,
                                                style={"borderRadius": "10px"},
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    # Choice type filter
                                    dbc.Col(
                                        [
                                            html.Div("Choice Type", className="filter-section-title"),
                                            dcc.Dropdown(
                                                id="filter-choice-type",
                                                options=[{"label": "All", "value": "all"}] + [
                                                    {"label": ct.replace("_", " ").title(), "value": ct}
                                                    for ct in choice_types_unique
                                                ],
                                                value="all",
                                                clearable=False,
                                                style={"borderRadius": "10px"},
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    # Time horizon range
                                    dbc.Col(
                                        [
                                            html.Div("Time Horizon Range (months)", className="filter-section-title"),
                                            html.Div(
                                                dcc.RangeSlider(
                                                    id="filter-horizon-range",
                                                    min=0,
                                                    max=max_horizon,
                                                    value=[0, max_horizon],
                                                    marks={
                                                        0: "0",
                                                        12: "1y",
                                                        60: "5y",
                                                        120: "10y",
                                                        max_horizon: f"{int(max_horizon)}m" if max_horizon > 120 else None,
                                                    },
                                                    tooltip={"placement": "bottom", "always_visible": False},
                                                    allowCross=False,
                                                ),
                                                className="horizon-range-slider",
                                            ),
                                        ],
                                        md=3,
                                    ),
                                    # Clear filters button
                                    dbc.Col(
                                        [
                                            html.Div(" ", className="filter-section-title"),
                                            dbc.Button(
                                                "Clear All",
                                                id="clear-filters-btn",
                                                className="clear-filters-btn",
                                                n_clicks=0,
                                            ),
                                        ],
                                        md=1,
                                        className="d-flex flex-column justify-content-end",
                                    ),
                                ],
                                className="g-3",
                            ),
                        ]
                    ),
                    className="glass-card filter-collapse-content",
                ),
                id="filter-collapse",
                is_open=False,
            ),
            # Store for filter state
            dcc.Store(id="filter-mask-store", data=[]),
        ],
        className="sample-filter-panel mb-3",
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
                                                updatemode="mouseup",  # Debounce: only fire on release
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
                                            # Click-to-select hint
                                            html.Div(
                                                [
                                                    html.Span("Tip:", className="hint-icon"),
                                                    " Click on any point in the 3D plot above to select that sample.",
                                                ],
                                                className="click-to-select-hint",
                                            ),
                                            # Sample navigation controls
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
                                                                updatemode="mouseup",  # Debounce: only fire on release
                                                            ),
                                                        ],
                                                        md=9,
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
                                                        md=3,
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Sample info panel (badges, summary, text)
                                            html.Div(id="ce-sample-info-panel", className="sample-info-panel"),
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
                                                                updatemode="mouseup",  # Debounce: only fire on release
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

    # Create Compare tab for side-by-side layer/component comparison
    compare_tab = dbc.Tab(
        label="Compare",
        tab_id="tab-compare",
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
                                                        "Compare two different targets side by side. ",
                                                        style={"color": COLORS["text_dim"], "fontSize": "13px"},
                                                    ),
                                                    html.Span("?", id="cmp-intro-info", className="info-icon"),
                                                ],
                                                style={"marginBottom": "12px"},
                                            ),
                                            dbc.Tooltip(
                                                "Select two different layer/component/position combinations to compare. "
                                                "The plots can be synchronized for rotation and zoom. Use the difference "
                                                "view to see how embeddings change between the two targets.",
                                                target="cmp-intro-info",
                                                placement="top",
                                            ),
                                            dbc.Row(
                                                [
                                                    # Left target controls
                                                    dbc.Col(
                                                        [
                                                            html.H6(
                                                                "Target A",
                                                                style={
                                                                    "color": COLORS["pink"],
                                                                    "fontWeight": "700",
                                                                    "marginBottom": "12px",
                                                                },
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Layer", style={"color": COLORS["text"], "fontSize": "12px"}),
                                                                            dcc.Dropdown(
                                                                                id="cmp-layer-a",
                                                                                options=[{"label": f"L{l}", "value": l} for l in layers],
                                                                                value=layers[len(layers) // 3] if layers else None,
                                                                                clearable=False,
                                                                                placeholder="Layer...",
                                                                            ),
                                                                        ],
                                                                        md=4,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Component", style={"color": COLORS["text"], "fontSize": "12px"}),
                                                                            dcc.Dropdown(
                                                                                id="cmp-component-a",
                                                                                options=[{"label": COMPONENT_NAMES[c], "value": c} for c in components],
                                                                                value="resid_post",
                                                                                clearable=False,
                                                                                placeholder="Component...",
                                                                            ),
                                                                        ],
                                                                        md=4,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Position", style={"color": COLORS["text"], "fontSize": "12px"}),
                                                                            dcc.Dropdown(
                                                                                id="cmp-position-a",
                                                                                options=[{"label": p, "value": p} for p in positions],
                                                                                value="response" if "response" in positions else (positions[0] if positions else None),
                                                                                clearable=False,
                                                                                placeholder="Position...",
                                                                            ),
                                                                        ],
                                                                        md=4,
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        md=5,
                                                    ),
                                                    # Sync controls in the middle
                                                    dbc.Col(
                                                        [
                                                            html.Div(
                                                                [
                                                                    dbc.Label("Sync Options", style={"color": COLORS["text"], "fontWeight": "600", "marginBottom": "8px"}),
                                                                    dbc.Checklist(
                                                                        id="cmp-sync-options",
                                                                        options=[
                                                                            {"label": "Sync Camera", "value": "camera"},
                                                                        ],
                                                                        value=["camera"],
                                                                        switch=True,
                                                                    ),
                                                                    html.Hr(style={"margin": "8px 0", "borderColor": "rgba(180,160,200,0.2)"}),
                                                                    dbc.Label("View Mode", style={"color": COLORS["text"], "fontWeight": "600", "marginBottom": "8px"}),
                                                                    dbc.RadioItems(
                                                                        id="cmp-view-mode",
                                                                        options=[
                                                                            {"label": "Side by Side", "value": "side_by_side"},
                                                                            {"label": "Difference", "value": "difference"},
                                                                            {"label": "Overlay", "value": "overlay"},
                                                                        ],
                                                                        value="side_by_side",
                                                                    ),
                                                                ],
                                                                style={"textAlign": "center"},
                                                            ),
                                                        ],
                                                        md=2,
                                                    ),
                                                    # Right target controls
                                                    dbc.Col(
                                                        [
                                                            html.H6(
                                                                "Target B",
                                                                style={
                                                                    "color": COLORS["cyan"],
                                                                    "fontWeight": "700",
                                                                    "marginBottom": "12px",
                                                                },
                                                            ),
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Layer", style={"color": COLORS["text"], "fontSize": "12px"}),
                                                                            dcc.Dropdown(
                                                                                id="cmp-layer-b",
                                                                                options=[{"label": f"L{l}", "value": l} for l in layers],
                                                                                value=layers[2 * len(layers) // 3] if layers else None,
                                                                                clearable=False,
                                                                                placeholder="Layer...",
                                                                            ),
                                                                        ],
                                                                        md=4,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Component", style={"color": COLORS["text"], "fontSize": "12px"}),
                                                                            dcc.Dropdown(
                                                                                id="cmp-component-b",
                                                                                options=[{"label": COMPONENT_NAMES[c], "value": c} for c in components],
                                                                                value="resid_post",
                                                                                clearable=False,
                                                                                placeholder="Component...",
                                                                            ),
                                                                        ],
                                                                        md=4,
                                                                    ),
                                                                    dbc.Col(
                                                                        [
                                                                            dbc.Label("Position", style={"color": COLORS["text"], "fontSize": "12px"}),
                                                                            dcc.Dropdown(
                                                                                id="cmp-position-b",
                                                                                options=[{"label": p, "value": p} for p in positions],
                                                                                value="response" if "response" in positions else (positions[0] if positions else None),
                                                                                clearable=False,
                                                                                placeholder="Position...",
                                                                            ),
                                                                        ],
                                                                        md=4,
                                                                    ),
                                                                ]
                                                            ),
                                                        ],
                                                        md=5,
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
            # Sample group filter
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
                                                            dbc.Label("Filter Sample Groups", className="fw-bold", style={"color": COLORS["text"]}),
                                                            dbc.Checklist(
                                                                id="cmp-sample-filter",
                                                                options=[
                                                                    {"label": "With Horizon", "value": "with_horizon"},
                                                                    {"label": "No Horizon", "value": "no_horizon"},
                                                                ],
                                                                value=["with_horizon", "no_horizon"],
                                                                inline=True,
                                                            ),
                                                        ],
                                                        md=6,
                                                    ),
                                                    dbc.Col(
                                                        [
                                                            dbc.Label("Time Scale Filter", className="fw-bold", style={"color": COLORS["text"]}),
                                                            dcc.Dropdown(
                                                                id="cmp-time-scale-filter",
                                                                options=[
                                                                    {"label": "All Scales", "value": "all"},
                                                                    {"label": "Immediate", "value": "immediate"},
                                                                    {"label": "Short-term", "value": "short-term"},
                                                                    {"label": "Medium-term", "value": "medium-term"},
                                                                    {"label": "Long-term", "value": "long-term"},
                                                                ],
                                                                value="all",
                                                                clearable=False,
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
            # Plots container - will change based on view mode
            html.Div(id="cmp-plots-container"),
            # Store for camera state synchronization
            dcc.Store(id="cmp-camera-store", data=None),
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
                                                                updatemode="mouseup",  # Debounce: only fire on release
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
                                                updatemode="mouseup",  # Debounce: only fire on release
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
                                                [html.Span("\u2190", className="shortcut-key"), html.Span("\u2192", className="shortcut-key")],
                                                className="shortcut-keys",
                                            ),
                                            html.Span("Navigate layers", className="shortcut-description"),
                                        ],
                                        className="shortcut-item",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [html.Span("1", className="shortcut-key"), html.Span("-", className="shortcut-key"), html.Span("5", className="shortcut-key")],
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

    # Statistics panel component
    statistics_panel = html.Div(
        [
            # Toggle button
            html.Button(
                html.Span("\u2630", style={"fontSize": "20px"}),  # Hamburger menu icon
                id="stats-panel-toggle",
                className="stats-panel-toggle",
                title="Dataset Statistics",
            ),
            # Panel
            html.Div(
                [
                    # Header
                    html.Div(
                        [
                            html.H4([
                                html.Span("\u2630", style={"color": COLORS["cyan"]}),
                                " Statistics"
                            ]),
                            html.Button(
                                "x",
                                id="stats-panel-close",
                                className="stats-close",
                            ),
                        ],
                        className="stats-header",
                    ),
                    # Body
                    html.Div(
                        [
                            # Summary cards
                            html.Div(
                                [
                                    html.Div([
                                        html.Div(str(loader.n_samples), className="stats-card-value"),
                                        html.Div("Samples", className="stats-card-label"),
                                    ], className="stats-summary-card"),
                                    html.Div([
                                        html.Div(str(len(layers)), className="stats-card-value"),
                                        html.Div("Layers", className="stats-card-label"),
                                    ], className="stats-summary-card"),
                                    html.Div([
                                        html.Div(str(len(positions)), className="stats-card-value"),
                                        html.Div("Positions", className="stats-card-label"),
                                    ], className="stats-summary-card"),
                                    html.Div([
                                        html.Div(str(len(components)), className="stats-card-value"),
                                        html.Div("Components", className="stats-card-label"),
                                    ], className="stats-summary-card"),
                                ],
                                className="stats-summary-grid",
                            ),

                            # Dataset Statistics section
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div([
                                                html.Span("\u25B8", className="stats-section-icon"),
                                                "Dataset Overview",
                                            ], className="stats-section-title"),
                                            html.Span("\u25BC", className="stats-section-toggle"),
                                        ],
                                        className="stats-section-header",
                                        id="stats-dataset-header",
                                    ),
                                    html.Div(
                                        id="stats-dataset-content",
                                        className="stats-section-content",
                                    ),
                                ],
                                id="stats-dataset-section",
                                className="stats-section",
                            ),

                            # Current View Statistics section
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div([
                                                html.Span("\u25B8", className="stats-section-icon"),
                                                "Current View",
                                            ], className="stats-section-title"),
                                            html.Span("\u25BC", className="stats-section-toggle"),
                                        ],
                                        className="stats-section-header",
                                        id="stats-view-header",
                                    ),
                                    html.Div(
                                        id="stats-view-content",
                                        className="stats-section-content",
                                    ),
                                ],
                                id="stats-view-section",
                                className="stats-section",
                            ),

                            # Embedding Statistics section
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div([
                                                html.Span("\u25B8", className="stats-section-icon"),
                                                "Embedding Stats (PCA)",
                                            ], className="stats-section-title"),
                                            html.Span("\u25BC", className="stats-section-toggle"),
                                        ],
                                        className="stats-section-header",
                                        id="stats-embedding-header",
                                    ),
                                    html.Div(
                                        id="stats-embedding-content",
                                        className="stats-section-content",
                                    ),
                                ],
                                id="stats-embedding-section",
                                className="stats-section",
                            ),

                            # Sample Distribution section
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div([
                                                html.Span("\u25B8", className="stats-section-icon"),
                                                "Sample Distribution",
                                            ], className="stats-section-title"),
                                            html.Span("\u25BC", className="stats-section-toggle"),
                                        ],
                                        className="stats-section-header",
                                        id="stats-distribution-header",
                                    ),
                                    html.Div(
                                        id="stats-distribution-content",
                                        className="stats-section-content",
                                    ),
                                ],
                                id="stats-distribution-section",
                                className="stats-section collapsed",  # Start collapsed
                            ),
                        ],
                        className="stats-body",
                    ),
                ],
                id="stats-panel",
                className="stats-panel hidden",
            ),
        ]
    )

    # Share modal component
    share_modal = dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle(
                    [
                        html.Span("\u21AA", style={"fontSize": "20px"}),
                        "Share View",
                    ]
                ),
                close_button=True,
            ),
            dbc.ModalBody(
                [
                    # URL sharing section
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("\U0001F517", className="option-icon url"),
                                    "Shareable URL",
                                ],
                                className="share-option-title",
                            ),
                            html.Div(
                                "Copy this URL to share the current view with colleagues. The URL includes all settings.",
                                className="share-option-desc",
                            ),
                            html.Div(
                                [
                                    dcc.Input(
                                        id="share-url-input",
                                        type="text",
                                        value="",
                                        readOnly=True,
                                        className="url-input",
                                    ),
                                    html.Button(
                                        "Copy",
                                        id="copy-url-btn",
                                        className="share-action-btn primary",
                                        n_clicks=0,
                                    ),
                                ],
                                className="url-display",
                            ),
                            html.Div(id="copy-url-status"),
                        ],
                        className="share-option-card",
                    ),
                    # Export settings section
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("\u2193", className="option-icon export"),
                                    "Export Settings",
                                ],
                                className="share-option-title",
                            ),
                            html.Div(
                                "Download the current view settings as a JSON file for backup or sharing.",
                                className="share-option-desc",
                            ),
                            html.Div(id="export-json-display", className="json-display"),
                            html.Div(
                                [
                                    html.Button(
                                        "Download JSON",
                                        id="download-json-btn",
                                        className="share-action-btn secondary",
                                        n_clicks=0,
                                        style={"marginTop": "12px"},
                                    ),
                                ],
                            ),
                            dcc.Download(id="download-settings"),
                        ],
                        className="share-option-card",
                    ),
                    # Import settings section
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("\u2191", className="option-icon import"),
                                    "Import Settings",
                                ],
                                className="share-option-title",
                            ),
                            html.Div(
                                "Paste a JSON settings object to restore a saved view configuration.",
                                className="share-option-desc",
                            ),
                            dcc.Textarea(
                                id="import-json-textarea",
                                placeholder='Paste JSON settings here...\n\nExample:\n{"layer": 12, "component": "resid_post", ...}',
                                className="import-textarea",
                            ),
                            html.Button(
                                "Apply Settings",
                                id="apply-import-btn",
                                className="share-action-btn primary",
                                n_clicks=0,
                                style={"marginTop": "12px"},
                            ),
                            html.Div(id="import-status"),
                        ],
                        className="share-option-card",
                    ),
                ]
            ),
        ],
        id="share-modal",
        is_open=False,
        size="lg",
        className="share-modal",
    )

    app.layout = dbc.Container(
        [
            # URL location for shareable links
            dcc.Location(id="url", refresh=False),
            # Stores for keyboard state
            dcc.Store(id="keyboard-event-store", data=None),
            dcc.Store(id="shortcuts-panel-visible", data=False),
            # Store for tracking visited tabs (lazy loading)
            dcc.Store(id="visited-tabs-store", data=["tab-component"]),
            # Store for selected sample from plot click
            dcc.Store(id="selected-sample-store", data=None),
            # Store for sample text (for copy to clipboard)
            dcc.Store(id="sample-text-store", data=""),
            # Store for dark mode preference (persisted to localStorage)
            dcc.Store(id="dark-mode-store", data=False, storage_type="local"),
            # Store for current view settings (for sharing)
            dcc.Store(id="current-settings-store", data={}),
            # Store for import trigger
            dcc.Store(id="import-settings-store", data=None),
            # Hidden div for keyboard listener
            html.Div(id="keyboard-listener", style={"display": "none"}),
            # Dummy output for clientside callback
            html.Div(id="copy-dummy-output", style={"display": "none"}),
            # Dummy output for dark mode toggle
            html.Div(id="dark-mode-dummy-output", style={"display": "none"}),
            # Dummy output for URL copy
            html.Div(id="copy-url-dummy-output", style={"display": "none"}),
            header,
            help_card,
            global_controls,
            sample_filter_panel,
            dbc.Tabs(
                id="main-tabs",
                active_tab="tab-component",
                children=[
                    component_explorer_tab,
                    layer_explorer_tab,
                    compare_tab,
                    trajectory_tab,
                    position_slider_tab,
                ],
            ),
            keyboard_shortcuts_panel,
            statistics_panel,
            share_modal,
        ],
        fluid=True,
        style={"paddingBottom": "40px"},
    )

    # ========================================
    # Statistics Panel Callbacks
    # ========================================

    # Toggle statistics panel visibility
    clientside_callback(
        """
        function(toggleClicks, closeClicks, currentClass) {
            // Initialize if first call
            if (!toggleClicks && !closeClicks) {
                return window.dash_clientside.no_update;
            }

            const isHidden = currentClass.includes('hidden');
            if (isHidden) {
                return 'stats-panel visible';
            } else {
                return 'stats-panel hidden';
            }
        }
        """,
        Output("stats-panel", "className"),
        [Input("stats-panel-toggle", "n_clicks"), Input("stats-panel-close", "n_clicks")],
        State("stats-panel", "className"),
    )

    # Toggle section collapsed state
    clientside_callback(
        """
        function(n_clicks, currentClass) {
            if (!n_clicks) return window.dash_clientside.no_update;
            const isCollapsed = currentClass.includes('collapsed');
            if (isCollapsed) {
                return 'stats-section';
            } else {
                return 'stats-section collapsed';
            }
        }
        """,
        Output("stats-dataset-section", "className"),
        Input("stats-dataset-header", "n_clicks"),
        State("stats-dataset-section", "className"),
    )

    clientside_callback(
        """
        function(n_clicks, currentClass) {
            if (!n_clicks) return window.dash_clientside.no_update;
            const isCollapsed = currentClass.includes('collapsed');
            if (isCollapsed) {
                return 'stats-section';
            } else {
                return 'stats-section collapsed';
            }
        }
        """,
        Output("stats-view-section", "className"),
        Input("stats-view-header", "n_clicks"),
        State("stats-view-section", "className"),
    )

    clientside_callback(
        """
        function(n_clicks, currentClass) {
            if (!n_clicks) return window.dash_clientside.no_update;
            const isCollapsed = currentClass.includes('collapsed');
            if (isCollapsed) {
                return 'stats-section';
            } else {
                return 'stats-section collapsed';
            }
        }
        """,
        Output("stats-embedding-section", "className"),
        Input("stats-embedding-header", "n_clicks"),
        State("stats-embedding-section", "className"),
    )

    clientside_callback(
        """
        function(n_clicks, currentClass) {
            if (!n_clicks) return window.dash_clientside.no_update;
            const isCollapsed = currentClass.includes('collapsed');
            if (isCollapsed) {
                return 'stats-section';
            } else {
                return 'stats-section collapsed';
            }
        }
        """,
        Output("stats-distribution-section", "className"),
        Input("stats-distribution-header", "n_clicks"),
        State("stats-distribution-section", "className"),
    )

    # Compute dataset overview statistics
    @callback(
        Output("stats-dataset-content", "children"),
        Input("stats-panel-toggle", "n_clicks"),
    )
    def compute_dataset_stats(_):
        """Compute and display dataset statistics."""
        samples = loader.samples

        # Count samples with time horizons
        with_horizon = sum(1 for s in samples if s.get("time_horizon_months") is not None)
        without_horizon = len(samples) - with_horizon

        # Get time horizon range
        horizons = [s.get("time_horizon_months") for s in samples if s.get("time_horizon_months") is not None]
        min_horizon = min(horizons) if horizons else 0
        max_horizon = max(horizons) if horizons else 0
        avg_horizon = sum(horizons) / len(horizons) if horizons else 0

        # Available targets
        n_targets = len(loader._target_keys)

        return html.Div([
            html.Ul([
                html.Li([
                    html.Span("Total Samples: ", className="stats-list-label"),
                    html.Span(f"{loader.n_samples}", className="stats-list-value"),
                ], className="stats-list-item"),
                html.Li([
                    html.Span("With Time Horizon: ", className="stats-list-label"),
                    html.Span(f"{with_horizon}", className="stats-list-value positive"),
                ], className="stats-list-item"),
                html.Li([
                    html.Span("Without Horizon: ", className="stats-list-label"),
                    html.Span(f"{without_horizon}", className="stats-list-value"),
                ], className="stats-list-item"),
                html.Li([
                    html.Span("Min Horizon: ", className="stats-list-label"),
                    html.Span(f"{min_horizon:.1f} mo", className="stats-list-value"),
                ], className="stats-list-item"),
                html.Li([
                    html.Span("Max Horizon: ", className="stats-list-label"),
                    html.Span(f"{max_horizon:.1f} mo", className="stats-list-value"),
                ], className="stats-list-item"),
                html.Li([
                    html.Span("Avg Horizon: ", className="stats-list-label"),
                    html.Span(f"{avg_horizon:.1f} mo", className="stats-list-value"),
                ], className="stats-list-item"),
                html.Li([
                    html.Span("Total Targets: ", className="stats-list-label"),
                    html.Span(f"{n_targets}", className="stats-list-value"),
                ], className="stats-list-item"),
            ], className="stats-list"),
        ])

    # Compute current view statistics
    @callback(
        Output("stats-view-content", "children"),
        [
            Input("ce-layer-slider", "value"),
            Input("ce-component-radio", "value"),
            Input("ce-position-dropdown", "value"),
            Input("show-no-horizon", "value"),
            Input("main-tabs", "active_tab"),
        ],
    )
    def compute_view_stats(layer, component, position, show_no_horizon, active_tab):
        """Compute and display current view statistics."""
        if layer is None or component is None or position is None:
            return html.Div("Select a target to view statistics", className="stats-no-data")

        # Get mask for samples
        samples = loader.samples
        total = len(samples)

        # Calculate filtered counts based on show_no_horizon
        no_horizon_mask = loader.get_no_horizon_mask()
        show_nh = "show" in show_no_horizon if show_no_horizon else True

        shown = total if show_nh else total - sum(no_horizon_mask)
        filtered_out = 0 if show_nh else sum(no_horizon_mask)

        # Get activation dimensions
        activations = loader.load_activations(layer, component, position)
        if activations is not None:
            n_samples, n_dims = activations.shape
        else:
            n_samples, n_dims = 0, 0

        return html.Div([
            html.Ul([
                html.Li([
                    html.Span("Current Tab: ", className="stats-list-label"),
                    html.Span(active_tab.replace("tab-", "").title() if active_tab else "-", className="stats-list-value"),
                ], className="stats-list-item"),
                html.Li([
                    html.Span("Layer: ", className="stats-list-label"),
                    html.Span(f"{layer}", className="stats-list-value"),
                ], className="stats-list-item"),
                html.Li([
                    html.Span("Component: ", className="stats-list-label"),
                    html.Span(COMPONENT_NAMES.get(component, component), className="stats-list-value"),
                ], className="stats-list-item"),
                html.Li([
                    html.Span("Position: ", className="stats-list-label"),
                    html.Span(f"{position}", className="stats-list-value"),
                ], className="stats-list-item"),
                html.Li([
                    html.Span("Points Shown: ", className="stats-list-label"),
                    html.Span(f"{shown}", className="stats-list-value positive"),
                ], className="stats-list-item"),
                html.Li([
                    html.Span("Points Filtered: ", className="stats-list-label"),
                    html.Span(f"{filtered_out}", className="stats-list-value" + (" negative" if filtered_out > 0 else "")),
                ], className="stats-list-item"),
                html.Li([
                    html.Span("Activation Dims: ", className="stats-list-label"),
                    html.Span(f"{n_dims}", className="stats-list-value"),
                ], className="stats-list-item"),
            ], className="stats-list"),
        ])

    # Compute embedding statistics (PCA variance explained)
    @callback(
        Output("stats-embedding-content", "children"),
        [
            Input("ce-layer-slider", "value"),
            Input("ce-component-radio", "value"),
            Input("ce-position-dropdown", "value"),
            Input("method-radio", "value"),
        ],
    )
    def compute_embedding_stats(layer, component, position, method):
        """Compute and display embedding statistics including variance explained."""
        if layer is None or component is None or position is None:
            return html.Div("Select a target to view statistics", className="stats-no-data")

        try:
            # Load activations to compute stats
            activations = loader.load_activations(layer, component, position)
            if activations is None:
                return html.Div("No activation data available", className="stats-no-data")

            # Compute PCA on-the-fly to get variance explained
            n_comp = min(3, activations.shape[0] - 1, activations.shape[1])
            if n_comp < 1:
                return html.Div("Not enough data for PCA", className="stats-no-data")

            pca = SkPCA(n_components=n_comp)
            pca.fit(activations)

            var_explained = pca.explained_variance_ratio_ * 100
            cumulative = sum(var_explained)

            # Build variance bars
            variance_bars = []
            colors = ["pc1", "pc2", "pc3"]
            for i, (var, color) in enumerate(zip(var_explained, colors)):
                variance_bars.append(
                    html.Div([
                        html.Div(f"PC{i+1}", className="variance-bar-label"),
                        html.Div([
                            html.Div(className=f"variance-bar-fill {color}", style={"width": f"{var}%"}),
                        ], className="variance-bar-bg"),
                        html.Div(f"{var:.1f}%", className="variance-bar-value"),
                    ], className="variance-bar-row")
                )

            # Add cumulative bar
            variance_bars.append(
                html.Div([
                    html.Div("Sum", className="variance-bar-label"),
                    html.Div([
                        html.Div(className="variance-bar-fill cumulative", style={"width": f"{min(cumulative, 100)}%"}),
                    ], className="variance-bar-bg"),
                    html.Div(f"{cumulative:.1f}%", className="variance-bar-value"),
                ], className="variance-bar-row")
            )

            return html.Div([
                html.Div(f"Method: {method.upper()}", style={
                    "fontSize": "12px",
                    "color": COLORS["text_muted"],
                    "marginBottom": "12px",
                    "fontWeight": "600",
                }),
                html.Div("Variance Explained (PCA)", style={
                    "fontSize": "11px",
                    "color": COLORS["text_dim"],
                    "marginBottom": "8px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.05em",
                }),
                html.Div(variance_bars, className="variance-bar-container"),
            ])
        except Exception as e:
            return html.Div(f"Error computing stats: {str(e)}", className="stats-no-data")

    # Compute sample distribution statistics
    @callback(
        Output("stats-distribution-content", "children"),
        Input("stats-panel-toggle", "n_clicks"),
    )
    def compute_distribution_stats(_):
        """Compute and display sample distribution by time scale and choice type."""
        samples = loader.samples
        total = len(samples)

        if total == 0:
            return html.Div("No samples available", className="stats-no-data")

        # Count by time scale
        time_scales = {}
        for s in samples:
            ts = s.get("time_scale", "unknown")
            time_scales[ts] = time_scales.get(ts, 0) + 1

        # Count by choice type
        choice_types = {}
        for s in samples:
            ct = s.get("choice_type", "unknown")
            choice_types[ct] = choice_types.get(ct, 0) + 1

        # Create mini pie chart for time scales
        ts_sorted = sorted(time_scales.items(), key=lambda x: -x[1])
        ts_labels = [ts.replace("_", " ").title() for ts, _ in ts_sorted]
        ts_values = [count for _, count in ts_sorted]

        pie_fig = go.Figure(data=[go.Pie(
            labels=ts_labels,
            values=ts_values,
            hole=0.4,
            marker=dict(colors=PASTEL_COLORS[:len(ts_labels)]),
            textinfo="percent",
            textfont=dict(size=10, color="white"),
            hovertemplate="<b>%{label}</b><br>%{value} samples<br>%{percent}<extra></extra>",
        )])
        pie_fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=9, color=COLORS["text_dim"]),
            ),
            margin=dict(l=10, r=10, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=160,
        )

        # Build distribution bars for time scale
        ts_colors = ["pink", "purple", "cyan", "gold", "mint", "blue"]
        time_scale_bars = []
        for i, (ts, count) in enumerate(ts_sorted):
            pct = count / total * 100
            color = ts_colors[i % len(ts_colors)]
            time_scale_bars.append(
                html.Div([
                    html.Div(ts.replace("_", " ").title(), className="stats-dist-label"),
                    html.Div([
                        html.Div(className=f"stats-dist-bar {color}", style={"width": f"{pct}%"}),
                    ], className="stats-dist-bar-container"),
                    html.Div(f"{count}", className="stats-dist-value"),
                ], className="stats-dist-row")
            )

        # Build distribution bars for choice type
        ct_colors = ["cyan", "purple", "pink", "gold", "mint", "blue"]
        choice_type_bars = []
        for i, (ct, count) in enumerate(sorted(choice_types.items(), key=lambda x: -x[1])):
            pct = count / total * 100
            color = ct_colors[i % len(ct_colors)]
            # Truncate long names
            label = ct[:15] + "..." if len(ct) > 15 else ct
            choice_type_bars.append(
                html.Div([
                    html.Div(label, className="stats-dist-label", title=ct),
                    html.Div([
                        html.Div(className=f"stats-dist-bar {color}", style={"width": f"{pct}%"}),
                    ], className="stats-dist-bar-container"),
                    html.Div(f"{count}", className="stats-dist-value"),
                ], className="stats-dist-row")
            )

        return html.Div([
            html.Div("By Time Scale", style={
                "fontSize": "11px",
                "color": COLORS["text_dim"],
                "marginBottom": "8px",
                "textTransform": "uppercase",
                "letterSpacing": "0.05em",
                "fontWeight": "600",
            }),
            # Mini pie chart
            html.Div([
                dcc.Graph(
                    figure=pie_fig,
                    config={"displayModeBar": False},
                    style={"height": "160px"},
                ),
            ], className="stats-chart-container"),
            html.Div(time_scale_bars, className="stats-distribution", style={"marginTop": "12px"}),
            html.Hr(style={"borderColor": "rgba(180,160,200,0.2)", "margin": "14px 0"}),
            html.Div("By Choice Type", style={
                "fontSize": "11px",
                "color": COLORS["text_dim"],
                "marginBottom": "8px",
                "textTransform": "uppercase",
                "letterSpacing": "0.05em",
                "fontWeight": "600",
            }),
            html.Div(choice_type_bars, className="stats-distribution"),
        ])

    # ========================================
    # Sample Filter Callbacks
    # ========================================

    @callback(
        [Output("filter-collapse", "is_open"), Output("filter-toggle-btn", "className")],
        Input("filter-toggle-btn", "n_clicks"),
        State("filter-collapse", "is_open"),
    )
    def toggle_filter_collapse(n_clicks, is_open):
        """Toggle the filter collapse panel."""
        if n_clicks:
            new_state = not is_open
            btn_class = "filter-toggle-btn active" if new_state else "filter-toggle-btn"
            return new_state, btn_class
        return is_open, "filter-toggle-btn"

    @callback(
        [
            Output("filter-mask-store", "data"),
            Output("filter-match-count", "children"),
            Output("filter-match-count", "className"),
        ],
        [
            Input("filter-search-text", "value"),
            Input("filter-time-scale", "value"),
            Input("filter-choice-type", "value"),
            Input("filter-horizon-range", "value"),
        ],
    )
    def update_filter_mask(search_text, time_scale, choice_type, horizon_range):
        """Calculate the filter mask based on all filter inputs."""
        n_samples = loader.n_samples
        mask = np.ones(n_samples, dtype=bool)

        # Apply text search filter
        if search_text and search_text.strip():
            search_lower = search_text.lower().strip()
            for i, sample in enumerate(loader.samples):
                text = sample.get("text", "").lower()
                if search_lower not in text:
                    mask[i] = False

        # Apply time scale filter
        if time_scale and time_scale != "all":
            for i, sample in enumerate(loader.samples):
                if sample.get("time_scale") != time_scale:
                    mask[i] = False

        # Apply choice type filter
        if choice_type and choice_type != "all":
            for i, sample in enumerate(loader.samples):
                if sample.get("choice_type") != choice_type:
                    mask[i] = False

        # Apply horizon range filter
        if horizon_range:
            min_h, max_h = horizon_range
            for i, sample in enumerate(loader.samples):
                h = sample.get("time_horizon_months")
                if h is not None:
                    if h < min_h or h > max_h:
                        mask[i] = False

        # Calculate match count and determine CSS class
        match_count = int(mask.sum())
        total = n_samples

        if match_count == total:
            count_text = f"{match_count} samples (all)"
            count_class = "filter-match-count"
        elif match_count == 0:
            count_text = "No matches"
            count_class = "filter-match-count error"
        elif match_count < total * 0.1:
            count_text = f"{match_count} / {total} samples"
            count_class = "filter-match-count warning"
        else:
            count_text = f"{match_count} / {total} samples"
            count_class = "filter-match-count"

        return mask.tolist(), count_text, count_class

    @callback(
        [
            Output("filter-search-text", "value"),
            Output("filter-time-scale", "value"),
            Output("filter-choice-type", "value"),
            Output("filter-horizon-range", "value"),
        ],
        Input("clear-filters-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def clear_all_filters(n_clicks):
        """Clear all filter values to defaults."""
        return "", "all", "all", [0, max_horizon]

    @callback(
        Output("filter-time-scale", "value", allow_duplicate=True),
        Input("quick-filter-longterm", "n_clicks"),
        prevent_initial_call=True,
    )
    def quick_filter_longterm(n_clicks):
        """Quick filter for long-term samples."""
        return "long-term"

    @callback(
        Output("filter-time-scale", "value", allow_duplicate=True),
        Input("quick-filter-shortterm", "n_clicks"),
        prevent_initial_call=True,
    )
    def quick_filter_shortterm(n_clicks):
        """Quick filter for short-term samples."""
        return "short-term"

    @callback(
        Output("filter-horizon-range", "value", allow_duplicate=True),
        Input("quick-filter-with-horizon", "n_clicks"),
        prevent_initial_call=True,
    )
    def quick_filter_with_horizon(n_clicks):
        """Quick filter for samples with horizon (exclude 0)."""
        return [1, max_horizon]

    @callback(
        Output("filter-horizon-range", "value", allow_duplicate=True),
        Input("quick-filter-no-horizon", "n_clicks"),
        prevent_initial_call=True,
    )
    def quick_filter_no_horizon(n_clicks):
        """Quick filter for samples without horizon."""
        return [0, 0]

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
            Input("filter-mask-store", "data"),
        ],
    )
    def update_ce_plot(layer, component, position, method, color_by, show_no_horizon, filter_mask):
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

            # Convert filter mask to numpy array for highlighting
            highlight_mask = None
            if filter_mask and len(filter_mask) > 0:
                highlight_mask = np.array(filter_mask[:embedding.shape[0]], dtype=bool)
                # Only apply highlighting if not all samples match
                if highlight_mask.all():
                    highlight_mask = None

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
                highlight_mask=highlight_mask,
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
        [
            Output("ce-sample-info-panel", "children"),
            Output("ce-sample-badge", "children"),
            Output("sample-text-store", "data"),
        ],
        [
            Input("ce-sample-slider", "value"),
            Input("ce-sample-input", "value"),
            Input("selected-sample-store", "data"),
        ],
    )
    def update_ce_sample(slider_val, input_val, selected_from_plot):
        """Update the enhanced sample info panel with badges, summary, and formatted text."""
        try:
            # Priority: selected from plot > input field > slider
            if selected_from_plot is not None:
                idx = selected_from_plot
            elif input_val is not None:
                idx = input_val
            else:
                idx = slider_val if slider_val is not None else 0
            idx = max(0, min(idx, loader.n_samples - 1))

            sample = loader.get_sample_info(idx)
            text = sample.get("text", "No text available")

            # Create badges for sample metadata
            badges = []

            # Sample index badge
            badges.append(
                html.Span(
                    [
                        html.Span("IDX ", className="sample-badge-label"),
                        html.Span(str(idx), className="sample-badge-value"),
                    ],
                    className="sample-badge sample-index",
                )
            )

            # Choice type badge
            choice_type = sample.get("choice_type", "unknown")
            if choice_type and choice_type != "unknown":
                badges.append(
                    html.Span(
                        [
                            html.Span("TYPE ", className="sample-badge-label"),
                            html.Span(choice_type, className="sample-badge-value"),
                        ],
                        className="sample-badge choice-type",
                    )
                )

            # Time scale badge
            time_scale = sample.get("time_scale", "unknown")
            if time_scale and time_scale != "unknown":
                badges.append(
                    html.Span(
                        [
                            html.Span("SCALE ", className="sample-badge-label"),
                            html.Span(time_scale, className="sample-badge-value"),
                        ],
                        className="sample-badge time-scale",
                    )
                )

            # Time horizon badge
            horizon = sample.get("time_horizon_months")
            if horizon is not None:
                if horizon >= 12:
                    horizon_str = f"{horizon / 12:.1f}y"
                else:
                    horizon_str = f"{horizon}mo"
                badges.append(
                    html.Span(
                        [
                            html.Span("HORIZON ", className="sample-badge-label"),
                            html.Span(horizon_str, className="sample-badge-value"),
                        ],
                        className="sample-badge time-horizon",
                    )
                )
            else:
                badges.append(
                    html.Span(
                        [
                            html.Span("NO HORIZON", className="sample-badge-value"),
                        ],
                        className="sample-badge no-horizon",
                    )
                )

            # Short-term first badge
            short_term_first = sample.get("short_term_first")
            if short_term_first is not None:
                badges.append(
                    html.Span(
                        [
                            html.Span("ORDER ", className="sample-badge-label"),
                            html.Span(
                                "Short First" if short_term_first else "Long First",
                                className="sample-badge-value",
                            ),
                        ],
                        className="sample-badge short-term-first",
                    )
                )

            badges_div = html.Div(badges, className="sample-badges")

            # Create mini summary card
            summary_stats = []
            if horizon is not None:
                summary_stats.append(
                    html.Div(
                        [
                            html.Div(f"{horizon}", className="summary-stat-value"),
                            html.Div("Months", className="summary-stat-label"),
                        ],
                        className="summary-stat",
                    )
                )
            if choice_type and choice_type != "unknown":
                summary_stats.append(
                    html.Div(
                        [
                            html.Div(choice_type[:8], className="summary-stat-value"),
                            html.Div("Choice Type", className="summary-stat-label"),
                        ],
                        className="summary-stat",
                    )
                )
            text_len = len(text)
            summary_stats.append(
                html.Div(
                    [
                        html.Div(f"{text_len}", className="summary-stat-value"),
                        html.Div("Characters", className="summary-stat-label"),
                    ],
                    className="summary-stat",
                )
            )

            summary_card = html.Div(summary_stats, className="sample-summary-card") if summary_stats else None

            # Create prompt text container with copy button
            prompt_container = html.Div(
                [
                    html.Button(
                        "Copy",
                        id="copy-sample-btn",
                        className="copy-button",
                        n_clicks=0,
                    ),
                    html.Div(text, className="sample-prompt", id="sample-prompt-text"),
                ],
                className="sample-prompt-container",
            )

            # Build the full info panel
            panel_children = [badges_div]
            if summary_card:
                panel_children.append(summary_card)
            panel_children.append(prompt_container)

            # Create badge text for header
            badge_text = f"Sample {idx}"
            if horizon is not None:
                badge_text += f" | {horizon} months"
            else:
                badge_text += " | No Horizon"

            return panel_children, badge_text, text
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return [html.Div(error_msg, className="error-message")], f"Sample {slider_val or 0}", ""

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
        [Output("selected-sample-store", "data"), Output("ce-sample-slider", "value", allow_duplicate=True)],
        Input("ce-main-plot", "clickData"),
        prevent_initial_call=True,
    )
    def handle_plot_click(click_data):
        """Handle clicking on a point in the 3D scatter plot to select that sample."""
        if click_data is None:
            return None, 0

        try:
            # Extract sample index from customdata
            point = click_data.get("points", [{}])[0]
            custom_data = point.get("customdata")

            if custom_data is not None:
                # customdata could be a single value or a list
                if isinstance(custom_data, list):
                    sample_idx = int(custom_data[0])
                else:
                    sample_idx = int(custom_data)
                return sample_idx, sample_idx
        except (KeyError, IndexError, TypeError, ValueError):
            pass

        return None, 0

    # Clientside callback for copy to clipboard functionality
    clientside_callback(
        """
        function(n_clicks, text) {
            if (n_clicks > 0 && text) {
                navigator.clipboard.writeText(text).then(function() {
                    // Find the button and temporarily change its text
                    var btn = document.getElementById('copy-sample-btn');
                    if (btn) {
                        var originalText = btn.innerText;
                        btn.innerText = 'Copied!';
                        btn.classList.add('copied');
                        setTimeout(function() {
                            btn.innerText = originalText;
                            btn.classList.remove('copied');
                        }, 2000);
                    }
                });
            }
            return '';
        }
        """,
        Output("copy-dummy-output", "children"),
        Input("copy-sample-btn", "n_clicks"),
        State("sample-text-store", "data"),
        prevent_initial_call=True,
    )

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
            Input("filter-mask-store", "data"),
        ],
    )
    def update_le_plots(layer, position, method, color_by, show_no_horizon, active_tab, filter_mask):
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

                # Convert filter mask for highlighting
                highlight_mask = None
                if filter_mask and len(filter_mask) > 0:
                    highlight_mask = np.array(filter_mask[:n], dtype=bool)
                    if highlight_mask.all():
                        highlight_mask = None

                fig = create_3d_scatter(
                    embedding=emb,
                    colors=c,
                    color_by=color_by,
                    title=f"L{layer} {COMPONENT_NAMES[comp]}",
                    is_categorical=is_categorical,
                    no_horizon_mask=m,
                    show_no_horizon="show" in show_no_horizon,
                    method=method,
                    highlight_mask=highlight_mask,
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
            Input("filter-mask-store", "data"),
        ],
    )
    def update_ps_plot(layer, component, pos_idx, method, color_by, show_no_horizon, active_tab, filter_mask):
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

            # Convert filter mask for highlighting
            highlight_mask = None
            if filter_mask and len(filter_mask) > 0:
                highlight_mask = np.array(filter_mask[:n], dtype=bool)
                if highlight_mask.all():
                    highlight_mask = None

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
                highlight_mask=highlight_mask,
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

    # Compare tab callback - creates the visualization layout based on view mode
    @callback(
        Output("cmp-plots-container", "children"),
        [
            Input("cmp-layer-a", "value"),
            Input("cmp-component-a", "value"),
            Input("cmp-position-a", "value"),
            Input("cmp-layer-b", "value"),
            Input("cmp-component-b", "value"),
            Input("cmp-position-b", "value"),
            Input("cmp-view-mode", "value"),
            Input("cmp-sample-filter", "value"),
            Input("cmp-time-scale-filter", "value"),
            Input("method-radio", "value"),
            Input("color-dropdown", "value"),
            Input("show-no-horizon", "value"),
            Input("main-tabs", "active_tab"),
        ],
    )
    def update_compare_plots(
        layer_a, comp_a, pos_a,
        layer_b, comp_b, pos_b,
        view_mode, sample_filter, time_scale_filter,
        method, color_by, show_no_horizon, active_tab
    ):
        try:
            # Lazy loading: only compute if this tab is active
            if active_tab != "tab-compare":
                return dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(
                                    "Switch to Compare tab to load",
                                    style={"textAlign": "center", "color": COLORS["text_muted"], "padding": "40px"},
                                )
                            ])
                        ], className="glass-card")
                    ])
                ])

            # Validate inputs
            if any(v is None for v in [layer_a, comp_a, pos_a, layer_b, comp_b, pos_b]):
                return dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(
                                    "Select targets A and B to compare",
                                    style={"textAlign": "center", "color": COLORS["text_muted"], "padding": "40px"},
                                )
                            ])
                        ], className="glass-card")
                    ])
                ])

            # Load embeddings for both targets
            def load_embedding(layer, component, position):
                if method == "pca":
                    return loader.load_pca(layer, component, position, n_components=3)
                elif method == "umap":
                    return loader.load_umap(layer, component, position, n_components=3)
                else:
                    return loader.load_tsne(layer, component, position, n_components=3)

            emb_a = load_embedding(layer_a, comp_a, pos_a)
            emb_b = load_embedding(layer_b, comp_b, pos_b)

            if emb_a is None or emb_b is None:
                missing = []
                if emb_a is None:
                    missing.append(f"Target A (L{layer_a}_{comp_a}_{pos_a})")
                if emb_b is None:
                    missing.append(f"Target B (L{layer_b}_{comp_b}_{pos_b})")
                return dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(
                                    f"No data for: {', '.join(missing)}",
                                    className="error-message",
                                )
                            ])
                        ], className="glass-card")
                    ])
                ])

            # Get sample metadata
            colors_data = loader.get_sample_metadata(color_by)
            no_horizon_mask = loader.get_no_horizon_mask()
            time_scales = loader.get_sample_metadata("time_scale")
            is_categorical = color_by in ["time_scale", "choice_type", "short_term_first", "has_horizon"]

            # Build filter mask
            n = min(emb_a.shape[0], emb_b.shape[0])
            filter_mask = np.ones(n, dtype=bool)

            # Apply horizon filter
            if sample_filter:
                horizon_filter = np.zeros(n, dtype=bool)
                if "with_horizon" in sample_filter:
                    horizon_filter |= ~no_horizon_mask[:n]
                if "no_horizon" in sample_filter:
                    horizon_filter |= no_horizon_mask[:n]
                filter_mask &= horizon_filter

            # Apply time scale filter
            if time_scale_filter and time_scale_filter != "all":
                scale_filter = np.array([ts == time_scale_filter for ts in time_scales[:n]])
                filter_mask &= scale_filter

            # Apply filter to data
            emb_a_filtered = emb_a[:n][filter_mask]
            emb_b_filtered = emb_b[:n][filter_mask]
            colors_filtered = colors_data[:n][filter_mask]
            no_horizon_filtered = no_horizon_mask[:n][filter_mask]
            sample_indices = np.arange(n)[filter_mask]

            if len(emb_a_filtered) == 0:
                return dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div(
                                    "No samples match the current filter criteria",
                                    className="error-message",
                                )
                            ])
                        ], className="glass-card")
                    ])
                ])

            axis_prefix = method.upper()
            title_a = f"L{layer_a} {COMPONENT_NAMES[comp_a]} @ {pos_a}"
            title_b = f"L{layer_b} {COMPONENT_NAMES[comp_b]} @ {pos_b}"

            if view_mode == "side_by_side":
                # Create two side-by-side plots
                fig_a = create_3d_scatter(
                    embedding=emb_a_filtered,
                    colors=colors_filtered,
                    color_by=color_by,
                    title=f"Target A: {title_a}",
                    is_categorical=is_categorical,
                    no_horizon_mask=no_horizon_filtered,
                    show_no_horizon="show" in show_no_horizon,
                    method=method,
                )
                fig_b = create_3d_scatter(
                    embedding=emb_b_filtered,
                    colors=colors_filtered,
                    color_by=color_by,
                    title=f"Target B: {title_b}",
                    is_categorical=is_categorical,
                    no_horizon_mask=no_horizon_filtered,
                    show_no_horizon="show" in show_no_horizon,
                    method=method,
                )

                return dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.Span("Target A", style={"color": COLORS["pink"], "fontWeight": "700"}),
                                f" - {title_a}",
                            ]),
                            dbc.CardBody([
                                create_loading_wrapper(
                                    dcc.Graph(
                                        id="cmp-plot-a",
                                        figure=fig_a,
                                        style={"height": "55vh"},
                                        config=plotly_config,
                                    ),
                                    loading_text="Computing embeddings...",
                                    show_progress=True,
                                ),
                            ]),
                        ], className="glass-card"),
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                html.Span("Target B", style={"color": COLORS["cyan"], "fontWeight": "700"}),
                                f" - {title_b}",
                            ]),
                            dbc.CardBody([
                                create_loading_wrapper(
                                    dcc.Graph(
                                        id="cmp-plot-b",
                                        figure=fig_b,
                                        style={"height": "55vh"},
                                        config=plotly_config,
                                    ),
                                    loading_text="Computing embeddings...",
                                    show_progress=True,
                                ),
                            ]),
                        ], className="glass-card"),
                    ], md=6),
                ])

            elif view_mode == "difference":
                # Show difference vectors between A and B
                diff = emb_b_filtered - emb_a_filtered
                diff_magnitude = np.linalg.norm(diff, axis=1)

                # Create figure with arrows showing movement
                fig_diff = go.Figure()

                # Plot starting points (A) with color
                fig_diff.add_trace(
                    go.Scatter3d(
                        x=emb_a_filtered[:, 0],
                        y=emb_a_filtered[:, 1],
                        z=emb_a_filtered[:, 2] if emb_a_filtered.shape[1] > 2 else np.zeros(len(emb_a_filtered)),
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=colors_filtered if not is_categorical else None,
                            colorscale=[[0, COLORS["cyan"]], [0.5, COLORS["purple"]], [1, COLORS["pink"]]],
                            opacity=0.6,
                        ),
                        name="Target A",
                        customdata=sample_indices,
                        hovertemplate="<b>Sample %{customdata}</b><br>A: %{x:.3f}, %{y:.3f}, %{z:.3f}<extra>Target A</extra>",
                    )
                )

                # Plot ending points (B)
                fig_diff.add_trace(
                    go.Scatter3d(
                        x=emb_b_filtered[:, 0],
                        y=emb_b_filtered[:, 1],
                        z=emb_b_filtered[:, 2] if emb_b_filtered.shape[1] > 2 else np.zeros(len(emb_b_filtered)),
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=colors_filtered if not is_categorical else None,
                            colorscale=[[0, COLORS["cyan"]], [0.5, COLORS["purple"]], [1, COLORS["pink"]]],
                            opacity=0.9,
                            symbol="diamond",
                        ),
                        name="Target B",
                        customdata=sample_indices,
                        hovertemplate="<b>Sample %{customdata}</b><br>B: %{x:.3f}, %{y:.3f}, %{z:.3f}<extra>Target B</extra>",
                    )
                )

                # Add lines connecting A to B (showing movement direction)
                # Subsample if too many points
                max_lines = 200
                if len(emb_a_filtered) > max_lines:
                    line_indices = np.random.choice(len(emb_a_filtered), max_lines, replace=False)
                else:
                    line_indices = np.arange(len(emb_a_filtered))

                for i in line_indices:
                    fig_diff.add_trace(
                        go.Scatter3d(
                            x=[emb_a_filtered[i, 0], emb_b_filtered[i, 0]],
                            y=[emb_a_filtered[i, 1], emb_b_filtered[i, 1]],
                            z=[
                                emb_a_filtered[i, 2] if emb_a_filtered.shape[1] > 2 else 0,
                                emb_b_filtered[i, 2] if emb_b_filtered.shape[1] > 2 else 0,
                            ],
                            mode="lines",
                            line=dict(
                                width=2,
                                color=f"rgba(198, 120, 221, {min(0.3 + diff_magnitude[i] / diff_magnitude.max() * 0.5, 0.8)})",
                            ),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

                fig_diff.update_layout(
                    title=dict(
                        text=f"Difference: {title_a} -> {title_b}",
                        font=dict(color=COLORS["text"], size=16),
                    ),
                    template="plotly_white",
                    paper_bgcolor="rgba(255, 255, 255, 0)",
                    font=dict(color=COLORS["text"]),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255, 255, 255, 0.95)",
                    ),
                    scene=dict(
                        xaxis_title=f"{axis_prefix}0",
                        yaxis_title=f"{axis_prefix}1",
                        zaxis_title=f"{axis_prefix}2",
                        xaxis=dict(backgroundcolor="rgba(248,244,255,0.3)", gridcolor="rgba(180,160,200,0.3)"),
                        yaxis=dict(backgroundcolor="rgba(240,247,255,0.3)", gridcolor="rgba(180,160,200,0.3)"),
                        zaxis=dict(backgroundcolor="rgba(254,246,249,0.3)", gridcolor="rgba(180,160,200,0.3)"),
                    ),
                )

                # Create histogram of difference magnitudes
                fig_hist = go.Figure()
                fig_hist.add_trace(
                    go.Histogram(
                        x=diff_magnitude,
                        nbinsx=30,
                        marker=dict(
                            color=COLORS["purple"],
                            line=dict(color=COLORS["pink"], width=1),
                        ),
                        name="Difference Magnitude",
                    )
                )
                fig_hist.update_layout(
                    title=dict(
                        text="Distribution of Embedding Differences",
                        font=dict(color=COLORS["text"], size=14),
                    ),
                    xaxis_title="Euclidean Distance",
                    yaxis_title="Count",
                    template="plotly_white",
                    paper_bgcolor="rgba(255, 255, 255, 0)",
                    font=dict(color=COLORS["text"]),
                    showlegend=False,
                    margin=dict(l=40, r=20, t=40, b=40),
                )

                # Stats card
                stats = html.Div([
                    html.Div([
                        html.Div(f"{diff_magnitude.mean():.4f}", className="metrics-value", style={"color": COLORS["purple"]}),
                        html.Div("Mean Distance", className="metrics-label"),
                    ], className="summary-stat"),
                    html.Div([
                        html.Div(f"{diff_magnitude.std():.4f}", className="metrics-value", style={"color": COLORS["cyan"]}),
                        html.Div("Std Distance", className="metrics-label"),
                    ], className="summary-stat"),
                    html.Div([
                        html.Div(f"{diff_magnitude.max():.4f}", className="metrics-value", style={"color": COLORS["pink"]}),
                        html.Div("Max Distance", className="metrics-label"),
                    ], className="summary-stat"),
                    html.Div([
                        html.Div(f"{len(emb_a_filtered)}", className="metrics-value", style={"color": COLORS["mint"]}),
                        html.Div("Samples", className="metrics-label"),
                    ], className="summary-stat"),
                ], className="sample-summary-card", style={"marginBottom": "16px"})

                return dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Difference Visualization"),
                            dbc.CardBody([
                                create_loading_wrapper(
                                    dcc.Graph(
                                        id="cmp-plot-diff",
                                        figure=fig_diff,
                                        style={"height": "55vh"},
                                        config=plotly_config,
                                    ),
                                    loading_text="Computing differences...",
                                    show_progress=True,
                                ),
                            ]),
                        ], className="glass-card"),
                    ], md=8),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Difference Statistics"),
                            dbc.CardBody([
                                stats,
                                dcc.Graph(
                                    id="cmp-plot-hist",
                                    figure=fig_hist,
                                    style={"height": "35vh"},
                                    config=plotly_config,
                                ),
                            ]),
                        ], className="glass-card"),
                    ], md=4),
                ])

            elif view_mode == "overlay":
                # Overlay both targets on the same plot with different colors
                fig_overlay = go.Figure()

                # Target A points
                fig_overlay.add_trace(
                    go.Scatter3d(
                        x=emb_a_filtered[:, 0],
                        y=emb_a_filtered[:, 1],
                        z=emb_a_filtered[:, 2] if emb_a_filtered.shape[1] > 2 else np.zeros(len(emb_a_filtered)),
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=COLORS["pink"],
                            opacity=0.7,
                        ),
                        name=f"A: {title_a}",
                        customdata=sample_indices,
                        hovertemplate=f"<b>Sample %{{customdata}}</b><br>{axis_prefix}0: %{{x:.3f}}<br>{axis_prefix}1: %{{y:.3f}}<br>{axis_prefix}2: %{{z:.3f}}<extra>Target A</extra>",
                    )
                )

                # Target B points
                fig_overlay.add_trace(
                    go.Scatter3d(
                        x=emb_b_filtered[:, 0],
                        y=emb_b_filtered[:, 1],
                        z=emb_b_filtered[:, 2] if emb_b_filtered.shape[1] > 2 else np.zeros(len(emb_b_filtered)),
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=COLORS["cyan"],
                            opacity=0.7,
                        ),
                        name=f"B: {title_b}",
                        customdata=sample_indices,
                        hovertemplate=f"<b>Sample %{{customdata}}</b><br>{axis_prefix}0: %{{x:.3f}}<br>{axis_prefix}1: %{{y:.3f}}<br>{axis_prefix}2: %{{z:.3f}}<extra>Target B</extra>",
                    )
                )

                fig_overlay.update_layout(
                    title=dict(
                        text=f"Overlay: {title_a} vs {title_b}",
                        font=dict(color=COLORS["text"], size=16),
                    ),
                    template="plotly_white",
                    paper_bgcolor="rgba(255, 255, 255, 0)",
                    font=dict(color=COLORS["text"]),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(255, 255, 255, 0.95)",
                    ),
                    scene=dict(
                        xaxis_title=f"{axis_prefix}0",
                        yaxis_title=f"{axis_prefix}1",
                        zaxis_title=f"{axis_prefix}2",
                        xaxis=dict(backgroundcolor="rgba(248,244,255,0.3)", gridcolor="rgba(180,160,200,0.3)"),
                        yaxis=dict(backgroundcolor="rgba(240,247,255,0.3)", gridcolor="rgba(180,160,200,0.3)"),
                        zaxis=dict(backgroundcolor="rgba(254,246,249,0.3)", gridcolor="rgba(180,160,200,0.3)"),
                    ),
                )

                return dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader([
                                "Overlay View - ",
                                html.Span("Target A", style={"color": COLORS["pink"], "fontWeight": "700"}),
                                " vs ",
                                html.Span("Target B", style={"color": COLORS["cyan"], "fontWeight": "700"}),
                            ]),
                            dbc.CardBody([
                                create_loading_wrapper(
                                    dcc.Graph(
                                        id="cmp-plot-overlay",
                                        figure=fig_overlay,
                                        style={"height": "60vh"},
                                        config=plotly_config,
                                    ),
                                    loading_text="Computing overlay...",
                                    show_progress=True,
                                ),
                            ]),
                        ], className="glass-card"),
                    ]),
                ])

            return html.Div("Unknown view mode")

        except Exception as e:
            return dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(f"Error: {str(e)}", className="error-message")
                        ])
                    ], className="glass-card")
                ])
            ])

    # Clientside callback for camera synchronization between compare plots
    clientside_callback(
        """
        function(relayout_a, relayout_b, sync_options) {
            // Check if sync is enabled
            if (!sync_options || !sync_options.includes('camera')) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }

            const ctx = dash_clientside.callback_context;
            if (!ctx.triggered || ctx.triggered.length === 0) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }

            const triggerId = ctx.triggered[0].prop_id;
            let camera = null;

            // Extract camera from the triggered relayout event
            if (triggerId.includes('cmp-plot-a') && relayout_a && relayout_a['scene.camera']) {
                camera = relayout_a['scene.camera'];
                // Return: no update for A, update B's camera
                return [window.dash_clientside.no_update, {'scene.camera': camera}];
            } else if (triggerId.includes('cmp-plot-b') && relayout_b && relayout_b['scene.camera']) {
                camera = relayout_b['scene.camera'];
                // Return: update A's camera, no update for B
                return [{'scene.camera': camera}, window.dash_clientside.no_update];
            }

            return [window.dash_clientside.no_update, window.dash_clientside.no_update];
        }
        """,
        [Output("cmp-plot-a", "figure", allow_duplicate=True), Output("cmp-plot-b", "figure", allow_duplicate=True)],
        [Input("cmp-plot-a", "relayoutData"), Input("cmp-plot-b", "relayoutData")],
        State("cmp-sync-options", "value"),
        prevent_initial_call=True,
    )

    # Clientside callback to toggle shortcuts panel visibility
    clientside_callback(
        """
        function(n_clicks_toggle, n_clicks_close, is_visible) {
            const triggered = dash_clientside.callback_context.triggered;
            if (!triggered || triggered.length === 0) {
                return is_visible;
            }
            const triggerId = triggered[0].prop_id.split('.')[0];
            if (triggerId === 'shortcuts-toggle' || triggerId === 'shortcuts-close') {
                return !is_visible;
            }
            return is_visible;
        }
        """,
        Output("shortcuts-panel-visible", "data"),
        [Input("shortcuts-toggle", "n_clicks"), Input("shortcuts-close", "n_clicks")],
        State("shortcuts-panel-visible", "data"),
    )

    # Clientside callback to update panel class based on visibility
    clientside_callback(
        """
        function(is_visible) {
            return is_visible ? 'keyboard-shortcuts-panel visible' : 'keyboard-shortcuts-panel hidden';
        }
        """,
        Output("shortcuts-panel", "className"),
        Input("shortcuts-panel-visible", "data"),
    )

    # Build the layers list for clientside access
    layers_list = list(layers)

    # Main keyboard event handler - clientside callback
    # This sets up the keyboard listener and handles all keyboard shortcuts
    clientside_callback(
        f"""
        function(_, current_method, current_tab, current_ce_layer, current_le_layer, current_ps_layer, panel_visible) {{
            // Only set up the listener once
            if (!window._keyboardListenerSetup) {{
                window._keyboardListenerSetup = true;
                window._layers = {layers_list};

                document.addEventListener('keydown', function(e) {{
                    // Don't trigger shortcuts when typing in input fields
                    const tagName = e.target.tagName.toLowerCase();
                    const isInput = tagName === 'input' || tagName === 'textarea' || tagName === 'select';
                    const isContentEditable = e.target.isContentEditable;

                    if (isInput || isContentEditable) {{
                        return;
                    }}

                    const key = e.key.toLowerCase();

                    // Tab switching with number keys 1-5
                    if (key >= '1' && key <= '5') {{
                        e.preventDefault();
                        const tabs = ['tab-component', 'tab-layer', 'tab-compare', 'tab-trajectory', 'tab-position'];
                        const tabIndex = parseInt(key) - 1;
                        if (tabIndex < tabs.length) {{
                            // Find and click the tab
                            const tabLinks = document.querySelectorAll('.nav-tabs .nav-link');
                            if (tabLinks[tabIndex]) {{
                                tabLinks[tabIndex].click();
                            }}
                        }}
                        return;
                    }}

                    // Method switching with P, U, T
                    if (key === 'p' || key === 'u' || key === 't') {{
                        e.preventDefault();
                        const methodMap = {{'p': 'pca', 'u': 'umap', 't': 'tsne'}};
                        const method = methodMap[key];
                        // Find and click the radio button
                        const radios = document.querySelectorAll('#method-radio input[type="radio"]');
                        radios.forEach(radio => {{
                            if (radio.value === method) {{
                                radio.click();
                            }}
                        }});
                        return;
                    }}

                    // Save plot with S
                    if (key === 's') {{
                        e.preventDefault();
                        // Find the visible plot's modebar and click the camera button
                        const activeTab = document.querySelector('.tab-pane.active');
                        if (activeTab) {{
                            const cameraBtn = activeTab.querySelector('[data-title="Download plot as a png"]');
                            if (cameraBtn) {{
                                cameraBtn.click();
                            }}
                        }}
                        return;
                    }}

                    // Layer navigation with arrow keys
                    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {{
                        e.preventDefault();
                        const layers = window._layers;
                        if (!layers || layers.length === 0) return;

                        // Find the active tab and its layer slider
                        const activeTab = document.querySelector('.tab-pane.active');
                        if (!activeTab) return;

                        const slider = activeTab.querySelector('.rc-slider');
                        if (!slider) return;

                        // Find the slider handle and get current position
                        const handle = slider.querySelector('.rc-slider-handle');
                        if (!handle) return;

                        // Get the slider's ID to determine which one we're working with
                        const sliderContainer = slider.closest('[id$="-slider"]') || slider.closest('.card-body');
                        const sliderInput = sliderContainer ? sliderContainer.querySelector('input[type="hidden"]') : null;

                        // Find all slider marks to determine current position
                        const marks = slider.querySelectorAll('.rc-slider-dot');
                        const activeMark = slider.querySelector('.rc-slider-dot-active');

                        let currentIdx = 0;
                        marks.forEach((mark, idx) => {{
                            if (mark.classList.contains('rc-slider-dot-active') ||
                                (activeMark && mark === activeMark)) {{
                                currentIdx = idx;
                            }}
                        }});

                        // Calculate new index
                        let newIdx = currentIdx;
                        if (e.key === 'ArrowRight') {{
                            newIdx = Math.min(currentIdx + 1, layers.length - 1);
                        }} else {{
                            newIdx = Math.max(currentIdx - 1, 0);
                        }}

                        if (newIdx !== currentIdx && marks[newIdx]) {{
                            // Click the mark to change the slider
                            marks[newIdx].click();
                        }}
                        return;
                    }}

                    // Toggle help panel with ?
                    if (key === '?' || (e.shiftKey && key === '/')) {{
                        e.preventDefault();
                        const toggleBtn = document.getElementById('shortcuts-toggle');
                        if (toggleBtn) {{
                            toggleBtn.click();
                        }}
                        return;
                    }}
                }});
            }}

            return window.dash_clientside.no_update;
        }}
        """,
        Output("keyboard-listener", "children"),
        Input("keyboard-event-store", "data"),
        [
            State("method-radio", "value"),
            State("main-tabs", "active_tab"),
            State("ce-layer-slider", "value"),
            State("le-layer-slider", "value"),
            State("ps-layer-slider", "value"),
            State("shortcuts-panel-visible", "data"),
        ],
    )

    # Trigger the keyboard listener setup on page load
    clientside_callback(
        """
        function() {
            return Date.now();
        }
        """,
        Output("keyboard-event-store", "data"),
        Input("main-tabs", "active_tab"),
    )

    # Dark mode toggle: update store when button clicked
    clientside_callback(
        """
        function(n_clicks, current_dark_mode) {
            if (n_clicks === 0) {
                // On initial load, return current state (from localStorage)
                return current_dark_mode;
            }
            // Toggle the dark mode state
            return !current_dark_mode;
        }
        """,
        Output("dark-mode-store", "data"),
        Input("dark-mode-toggle", "n_clicks"),
        State("dark-mode-store", "data"),
    )

    # Dark mode: apply class to body and update icon
    clientside_callback(
        """
        function(is_dark_mode) {
            // Apply or remove dark-mode class on body
            if (is_dark_mode) {
                document.body.classList.add('dark-mode');
            } else {
                document.body.classList.remove('dark-mode');
            }
            // Return the appropriate icon (sun for dark mode, moon for light mode)
            return is_dark_mode ? '\u2600' : '\u263D';
        }
        """,
        Output("dark-mode-icon", "children"),
        Input("dark-mode-store", "data"),
    )

    # ========================================
    # Share functionality callbacks
    # ========================================

    # Open/close share modal
    @callback(
        Output("share-modal", "is_open"),
        [Input("share-btn", "n_clicks")],
        [State("share-modal", "is_open")],
    )
    def toggle_share_modal(n_clicks, is_open):
        """Toggle the share modal when the share button is clicked."""
        if n_clicks:
            return not is_open
        return is_open

    # Update current settings store based on current control values
    @callback(
        Output("current-settings-store", "data"),
        [
            Input("main-tabs", "active_tab"),
            Input("ce-layer-slider", "value"),
            Input("ce-component-radio", "value"),
            Input("ce-position-dropdown", "value"),
            Input("le-layer-slider", "value"),
            Input("le-position-dropdown", "value"),
            Input("method-radio", "value"),
            Input("color-dropdown", "value"),
            Input("show-no-horizon", "value"),
        ],
    )
    def update_current_settings(
        active_tab,
        ce_layer,
        ce_component,
        ce_position,
        le_layer,
        le_position,
        method,
        color_by,
        show_no_horizon,
    ):
        """Store current view settings for sharing."""
        import json

        settings = {
            "tab": active_tab,
            "ce_layer": ce_layer,
            "ce_component": ce_component,
            "ce_position": ce_position,
            "le_layer": le_layer,
            "le_position": le_position,
            "method": method,
            "color_by": color_by,
            "show_no_horizon": show_no_horizon,
        }
        return settings

    # Generate shareable URL and update display
    @callback(
        [Output("share-url-input", "value"), Output("export-json-display", "children")],
        [Input("share-modal", "is_open"), Input("current-settings-store", "data")],
        [State("url", "href")],
    )
    def update_share_url(is_open, settings, current_href):
        """Generate shareable URL with query parameters."""
        import json

        if not is_open or not settings:
            return "", ""

        # Build query parameters
        params = {}
        for key, value in settings.items():
            if value is not None:
                if isinstance(value, list):
                    params[key] = ",".join(str(v) for v in value)
                else:
                    params[key] = str(value)

        # Parse current URL to get base
        if current_href:
            parsed = urlparse(current_href)
            base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        else:
            base_url = "/"

        # Generate full URL with parameters
        if params:
            share_url = f"{base_url}?{urlencode(params)}"
        else:
            share_url = base_url

        # Format JSON for display
        json_str = json.dumps(settings, indent=2)

        return share_url, json_str

    # Copy URL to clipboard
    clientside_callback(
        """
        function(n_clicks, url) {
            if (n_clicks > 0 && url) {
                navigator.clipboard.writeText(url).then(function() {
                    var btn = document.getElementById('copy-url-btn');
                    if (btn) {
                        var originalText = btn.innerText;
                        btn.innerText = 'Copied!';
                        btn.classList.add('copied');
                        setTimeout(function() {
                            btn.innerText = originalText;
                            btn.classList.remove('copied');
                        }, 2000);
                    }
                });
            }
            return '';
        }
        """,
        Output("copy-url-dummy-output", "children"),
        Input("copy-url-btn", "n_clicks"),
        State("share-url-input", "value"),
        prevent_initial_call=True,
    )

    # Download JSON settings
    @callback(
        Output("download-settings", "data"),
        Input("download-json-btn", "n_clicks"),
        State("current-settings-store", "data"),
        prevent_initial_call=True,
    )
    def download_settings(n_clicks, settings):
        """Download current settings as JSON file."""
        import json

        if n_clicks and settings:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"geoviz_settings_{timestamp}.json"
            content = json.dumps(settings, indent=2)
            return dict(content=content, filename=filename)
        return None

    # Import settings from JSON
    @callback(
        [
            Output("import-status", "children"),
            Output("import-settings-store", "data"),
        ],
        Input("apply-import-btn", "n_clicks"),
        State("import-json-textarea", "value"),
        prevent_initial_call=True,
    )
    def import_settings(n_clicks, json_text):
        """Parse and validate imported JSON settings."""
        import json

        if not n_clicks or not json_text:
            return "", None

        try:
            settings = json.loads(json_text)

            # Validate required fields
            valid_keys = {
                "tab",
                "ce_layer",
                "ce_component",
                "ce_position",
                "le_layer",
                "le_position",
                "method",
                "color_by",
                "show_no_horizon",
            }

            if not isinstance(settings, dict):
                return (
                    html.Div(
                        "Invalid format: expected a JSON object",
                        className="share-status error",
                    ),
                    None,
                )

            # Filter to only valid keys
            filtered_settings = {k: v for k, v in settings.items() if k in valid_keys}

            if not filtered_settings:
                return (
                    html.Div(
                        "No valid settings found in JSON",
                        className="share-status error",
                    ),
                    None,
                )

            return (
                html.Div(
                    f"Settings applied successfully! ({len(filtered_settings)} settings)",
                    className="share-status success",
                ),
                filtered_settings,
            )
        except json.JSONDecodeError as e:
            return (
                html.Div(
                    f"Invalid JSON: {str(e)}",
                    className="share-status error",
                ),
                None,
            )

    # Apply imported settings to controls
    @callback(
        [
            Output("main-tabs", "active_tab", allow_duplicate=True),
            Output("ce-layer-slider", "value", allow_duplicate=True),
            Output("ce-component-radio", "value", allow_duplicate=True),
            Output("ce-position-dropdown", "value", allow_duplicate=True),
            Output("le-layer-slider", "value", allow_duplicate=True),
            Output("le-position-dropdown", "value", allow_duplicate=True),
            Output("method-radio", "value", allow_duplicate=True),
            Output("color-dropdown", "value", allow_duplicate=True),
            Output("show-no-horizon", "value", allow_duplicate=True),
        ],
        Input("import-settings-store", "data"),
        prevent_initial_call=True,
    )
    def apply_imported_settings(settings):
        """Apply imported settings to all controls."""
        if not settings:
            return [no_update] * 9

        # Parse show_no_horizon if it's a string
        show_no_horizon = settings.get("show_no_horizon")
        if isinstance(show_no_horizon, str):
            show_no_horizon = show_no_horizon.split(",") if show_no_horizon else []

        return (
            settings.get("tab", no_update),
            settings.get("ce_layer", no_update),
            settings.get("ce_component", no_update),
            settings.get("ce_position", no_update),
            settings.get("le_layer", no_update),
            settings.get("le_position", no_update),
            settings.get("method", no_update),
            settings.get("color_by", no_update),
            show_no_horizon if show_no_horizon is not None else no_update,
        )

    # Read URL parameters on page load and apply settings
    @callback(
        [
            Output("main-tabs", "active_tab", allow_duplicate=True),
            Output("ce-layer-slider", "value", allow_duplicate=True),
            Output("ce-component-radio", "value", allow_duplicate=True),
            Output("ce-position-dropdown", "value", allow_duplicate=True),
            Output("le-layer-slider", "value", allow_duplicate=True),
            Output("le-position-dropdown", "value", allow_duplicate=True),
            Output("method-radio", "value", allow_duplicate=True),
            Output("color-dropdown", "value", allow_duplicate=True),
            Output("show-no-horizon", "value", allow_duplicate=True),
        ],
        Input("url", "search"),
        prevent_initial_call=True,
    )
    def apply_url_params(search):
        """Parse URL query parameters and apply them to controls."""
        if not search:
            return [no_update] * 9

        # Parse query string (remove leading ?)
        params = parse_qs(search.lstrip("?"))

        def get_param(key, default=no_update):
            value = params.get(key)
            if value:
                return value[0]
            return default

        def get_int_param(key, default=no_update):
            value = params.get(key)
            if value:
                try:
                    return int(value[0])
                except ValueError:
                    pass
            return default

        # Parse show_no_horizon list
        show_no_horizon_param = params.get("show_no_horizon")
        if show_no_horizon_param:
            show_no_horizon = show_no_horizon_param[0].split(",")
        else:
            show_no_horizon = no_update

        return (
            get_param("tab"),
            get_int_param("ce_layer"),
            get_param("ce_component"),
            get_param("ce_position"),
            get_int_param("le_layer"),
            get_param("le_position"),
            get_param("method"),
            get_param("color_by"),
            show_no_horizon,
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
