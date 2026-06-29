"""Cartopy + Matplotlib plotting utilities for LEO visualization.

This module is intentionally resilient to Cartopy binary/import problems on
platforms where compiled wheels lag behind the Python runtime (e.g. Py3.14).
If Cartopy cannot be imported, plotting gracefully falls back to plain
longitude/latitude Matplotlib axes so the Streamlit app can still run.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Sequence, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

try:
    import cartopy.crs as ccrs

    CARTOPY_AVAILABLE = True
    CARTOPY_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    ccrs = None
    CARTOPY_AVAILABLE = False
    CARTOPY_IMPORT_ERROR = exc

from .core import (
    PropagationResult,
    geodesic_circle_latlon
)

@dataclass
class AnimationArtifacts:
    fig: plt.Figure
    anim: animation.FuncAnimation

def _geo_plot_kwargs() -> dict:
    """Return keyword args for geo plotting calls.

    With Cartopy we must pass a PlateCarree transform. Without Cartopy,
    regular Matplotlib axes already use lon/lat coordinates directly.
    """
    if CARTOPY_AVAILABLE:
        return {"transform": ccrs.PlateCarree()}
    return {}


def _setup_map_axes(title: str = "LEO Satellite Ground Track") -> Tuple[plt.Figure, plt.Axes]:
    """Create map axes with Cartopy when possible, otherwise plain lon/lat axes."""
    fig = plt.figure(figsize=(12, 6))

    if CARTOPY_AVAILABLE:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(linewidth=0.8)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax = plt.axes()
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.grid(True, linewidth=0.3, alpha=0.5)

    ax.set_title(title)
    return fig, ax


def build_animation(
    result: PropagationResult,
    footprint_angular_radii_rad: Sequence[float],
    interval_ms: int = 80,
) -> AnimationArtifacts:
    """Create a Matplotlib animation with moving satellite + footprint."""
    if len(result.lats_deg) != len(footprint_angular_radii_rad):
        raise ValueError("Track and footprint radius lengths must match.")

    fig, ax = _setup_map_axes("LEO Ground Track (Static)")
    geo = _geo_plot_kwargs()

    # Ground track with dateline-safe segments
    for seg_lats, seg_lons in split_dateline_segments(result.lats_deg, result.lons_deg):
        ax.plot(seg_lons, seg_lats, color="tab:blue", linewidth=1.8, **geo)

    # Plot start/end points
    ax.scatter(result.lons_deg[0], result.lats_deg[0], s=35, color="green", label="Start", **geo)
    ax.scatter(result.lons_deg[-1], result.lats_deg[-1], s=35, color="red", label="End", **geo)

    # Draw sparse footprint circles to indicate swath envelope
    for i in range(0, len(result.lats_deg), max(1, footprint_stride)):

            ax.plot(
                seg_lons,
                seg_lats,
                color="tab:orange",
                alpha=0.15,
                linewidth=0.7,
                **geo,
            )

    ax.legend(loc="lower left")

        raise ValueError("Track and footprint radius lengths must match.")

    fig, ax = _setup_map_axes("LEO Ground Track (Animated)")
    geo = _geo_plot_kwargs()

    # Pre-draw full trajectory in faint style for spatial context
    for seg_lats, seg_lons in split_dateline_segments(result.lats_deg, result.lons_deg):
        ax.plot(
            seg_lons,
            seg_lats,
            color="tab:blue",
            linewidth=1.0,
            alpha=0.4,
            **geo,
        )

    # Artists to update
    past_track_line, = ax.plot([], [], color="tab:blue", linewidth=2.0, **geo)
    sat_marker, = ax.plot([], [], marker="o", markersize=7, color="red", **geo)
    footprint_line, = ax.plot([], [], color="tab:orange", linewidth=1.5, **geo)
    time_text = ax.text(
        0.01,
        0.01,
        "",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
    )

    lats = np.asarray(result.lats_deg)
    lons = np.asarray(result.lons_deg)

    """Render animation to bytes.

    Returns:
        (file_bytes, mime_type, extension)
    """
    if fps <= 0:
        raise ValueError("fps must be positive")

    fmt = fmt.lower().strip()
    if fmt not in {"gif", "mp4"}:
        raise ValueError("fmt must be either 'gif' or 'mp4'")

    suffix = f".{fmt}"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        if fmt == "gif":
            writer = animation.PillowWriter(fps=fps)
            mime = "image/gif"
            ext = "gif"
        else:
            writer = animation.FFMpegWriter(fps=fps, codec="libx264")
            mime = "video/mp4"
            ext = "mp4"

        anim.save(temp_path, writer=writer)

        with open(temp_path, "rb") as f:
            data = f.read()

        return data, mime, ext
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
