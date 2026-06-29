"""Cartopy + Matplotlib plotting utilities for LEO visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .core import (
    PropagationResult,
    geodesic_circle_latlon,
    split_dateline_segments,
)


@dataclass
class AnimationArtifacts:
    fig: plt.Figure
    anim: animation.FuncAnimation


def _setup_map_axes(title: str = "LEO Satellite Ground Track") -> Tuple[plt.Figure, plt.Axes]:
    """Create a PlateCarree map with common styling."""
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title)
    return fig, ax


def plot_static_ground_track(
    result: PropagationResult,
    footprint_angular_radii_rad: Sequence[float],
    footprint_stride: int = 10,
):
    """Plot full track and sparse footprint swath in static mode."""
    if len(result.lats_deg) != len(footprint_angular_radii_rad):
        raise ValueError("Track and footprint radius lengths must match.")

    fig, ax = _setup_map_axes("LEO Ground Track (Static)")

    # Ground track with dateline-safe segments
    for seg_lats, seg_lons in split_dateline_segments(result.lats_deg, result.lons_deg):
        ax.plot(seg_lons, seg_lats, transform=ccrs.PlateCarree(), color="tab:blue", linewidth=1.8)

    # Plot start/end points
    ax.scatter(result.lons_deg[0], result.lats_deg[0], transform=ccrs.PlateCarree(), s=35, color="green", label="Start")
    ax.scatter(result.lons_deg[-1], result.lats_deg[-1], transform=ccrs.PlateCarree(), s=35, color="red", label="End")

    # Draw sparse footprint circles to indicate swath envelope
    for i in range(0, len(result.lats_deg), max(1, footprint_stride)):
        latc = float(result.lats_deg[i])
        lonc = float(result.lons_deg[i])
        arad = float(footprint_angular_radii_rad[i])
        circle_lats, circle_lons = geodesic_circle_latlon(latc, lonc, arad)
        for seg_lats, seg_lons in split_dateline_segments(circle_lats, circle_lons):
            ax.plot(
                seg_lons,
                seg_lats,
                transform=ccrs.PlateCarree(),
                color="tab:orange",
                alpha=0.15,
                linewidth=0.7,
            )

    ax.legend(loc="lower left")
    return fig


def _segments_to_nan_polyline(segments: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten multiple line segments into a single NaN-separated polyline."""
    if not segments:
        return np.array([]), np.array([])

    lat_parts = []
    lon_parts = []
    for seg_lats, seg_lons in segments:
        lat_parts.append(np.asarray(seg_lats))
        lon_parts.append(np.asarray(seg_lons))
        lat_parts.append(np.array([np.nan]))
        lon_parts.append(np.array([np.nan]))

    return np.concatenate(lat_parts), np.concatenate(lon_parts)


def build_animation(
    result: PropagationResult,
    footprint_angular_radii_rad: Sequence[float],
    interval_ms: int = 80,
) -> AnimationArtifacts:
    """Create a Matplotlib animation with moving satellite + footprint."""
    if len(result.lats_deg) != len(footprint_angular_radii_rad):
        raise ValueError("Track and footprint radius lengths must match.")

    fig, ax = _setup_map_axes("LEO Ground Track (Animated)")

    # Pre-draw full trajectory in faint style for spatial context
    for seg_lats, seg_lons in split_dateline_segments(result.lats_deg, result.lons_deg):
        ax.plot(
            seg_lons,
            seg_lats,
            transform=ccrs.PlateCarree(),
            color="tab:blue",
            linewidth=1.0,
            alpha=0.4,
        )

    # Artists to update
    past_track_line, = ax.plot([], [], transform=ccrs.PlateCarree(), color="tab:blue", linewidth=2.0)
    sat_marker, = ax.plot([], [], marker="o", markersize=7, color="red", transform=ccrs.PlateCarree())
    footprint_line, = ax.plot([], [], transform=ccrs.PlateCarree(), color="tab:orange", linewidth=1.5)
    time_text = ax.text(0.01, 0.01, "", transform=ax.transAxes, fontsize=10, bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    lats = np.asarray(result.lats_deg)
    lons = np.asarray(result.lons_deg)

    def update(frame: int):
        # Up-to-current ground track (split for dateline safety)
        up_lats = lats[: frame + 1]
        up_lons = lons[: frame + 1]
        segments = split_dateline_segments(up_lats, up_lons)

        track_lats, track_lons = _segments_to_nan_polyline(segments)
        past_track_line.set_data(track_lons, track_lats)

        sat_marker.set_data([lons[frame]], [lats[frame]])

        # Moving footprint circle
        circle_lats, circle_lons = geodesic_circle_latlon(
            float(lats[frame]),
            float(lons[frame]),
            float(footprint_angular_radii_rad[frame]),
            n_points=240,
        )
        circ_segments = split_dateline_segments(circle_lats, circle_lons)
        fp_lats, fp_lons = _segments_to_nan_polyline(circ_segments)
        footprint_line.set_data(fp_lons, fp_lats)

        time_text.set_text(result.times_utc[frame].strftime("UTC: %Y-%m-%d %H:%M:%S"))
        return past_track_line, sat_marker, footprint_line, time_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(result.times_utc),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    return AnimationArtifacts(fig=fig, anim=anim)


def save_animation_bytes(
    anim: animation.FuncAnimation,
    fmt: str,
    fps: int,
) -> Tuple[bytes, str, str]:
    """Render animation to bytes.

    Returns
