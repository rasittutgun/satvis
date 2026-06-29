"""Streamlit application for LEO satellite ground-track visualization.

Run:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import streamlit as st

from satvis.core import (
    build_time_grid,
    footprint_angular_radius_rad,
    footprint_surface_radius_km,
    parse_tle,
    propagate_nadir_track,
)
from satvis.plotting import build_animation, plot_static_ground_track, save_animation_bytes


ISS_TLE_DEFAULT = """ISS (ZARYA)
1 25544U 98067A   26169.48935185  .00016717  00000+0  30053-3 0  9999
2 25544  51.6381 141.4689 0004896  81.2451 330.5118 15.50069167515373
"""


st.set_page_config(page_title="LEO Satellite Visualizer", layout="wide")
st.title("🛰️ LEO Satellite Visualizer (2D Earth Map)")
st.caption("Skyfield + Cartopy based nadir track and antenna footprint visualization")

with st.sidebar:
    st.header("Input Parameters")
    tle_text = st.text_area("TLE (2 veya 3 satır)", value=ISS_TLE_DEFAULT, height=140)
    fov_deg = st.number_input("Antenna FoV (derece)", min_value=0.1, max_value=179.0, value=30.0, step=0.5)

    now_utc = datetime.now(timezone.utc)
    default_start = now_utc
    default_end = now_utc + timedelta(minutes=90)

    start_dt = st.datetime_input("Start time (UTC)", value=default_start)
    end_dt = st.datetime_input("Finish time (UTC)", value=default_end)
    step_seconds = st.number_input("Time step (seconds)", min_value=1, max_value=3600, value=30, step=1)

    mode = st.selectbox("Visualization mode", options=["Static", "Animated"], index=0)

    st.markdown("---")
    export_enabled = st.checkbox("Export animation output (GIF/MP4)", value=False)
    export_format = st.selectbox("Export format", options=["gif", "mp4"], index=0)
    export_fps = st.slider("FPS", min_value=5, max_value=60, value=20, step=1)

run_clicked = st.button("Calculate and Visualize", type="primary")

if run_clicked:
    try:
        satellite = parse_tle(tle_text)

        # Streamlit datetime_input may return naive datetime.
        start_utc = start_dt.replace(tzinfo=timezone.utc) if start_dt.tzinfo is None else start_dt.astimezone(timezone.utc)
        end_utc = end_dt.replace(tzinfo=timezone.utc) if end_dt.tzinfo is None else end_dt.astimezone(timezone.utc)

        times_utc = build_time_grid(start_utc, end_utc, int(step_seconds))
        prop = propagate_nadir_track(satellite, times_utc)

        footprint_angular = np.array(
            [footprint_angular_radius_rad(float(h), float(fov_deg)) for h in prop.alt_km]
        )
        footprint_km = np.array(
            [footprint_surface_radius_km(float(h), float(fov_deg)) for h in prop.alt_km]
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Number of samples", f"{len(prop.times_utc)}")
        col2.metric("Mean altitude", f"{np.mean(prop.alt_km):.1f} km")
        col3.metric("Mean Footprint Radius", f"{np.mean(footprint_km):.1f} km")

        if mode == "Static":
            fig = plot_static_ground_track(prop, footprint_angular, footprint_stride=max(1, len(prop.times_utc) // 80))
            st.pyplot(fig, clear_figure=False)
        else:
            artifacts = build_animation(prop, footprint_angular, interval_ms=80)

            # Browser-side playback as GIF preview
            gif_bytes, _, _ = save_animation_bytes(artifacts.anim, fmt="gif", fps=export_fps)
            st.image(gif_bytes, caption="Animation preview", use_container_width=True)

            if export_enabled:
                try:
                    export_bytes, mime, ext = save_animation_bytes(
                        artifacts.anim,
                        fmt=export_format,
                        fps=export_fps,
                    )
                    st.download_button(
                        label=f"Download animation ({ext.upper()})",
                        data=export_bytes,
                        file_name=f"leo_visualization.{ext}",
                        mime=mime,
                    )
                except Exception as exc:
                    st.error(f"Export error: {exc}")

        with st.expander("Calculation summary"):
            st.write(
                {
                    "satellite": satellite.name,
                    "time_start_utc": prop.times_utc[0].isoformat(),
                    "time_finish_utc": prop.times_utc[-1].isoformat(),
                    "time_step_sec": int(step_seconds),
                    "fov_deg": float(fov_deg),
                    "altitude_km_min": float(np.min(prop.alt_km)),
                    "altitude_km_max": float(np.max(prop.alt_km)),
                }
            )

    except Exception as exc:
        st.error(f"Input or calculation error: {exc}")

st.markdown("---")
st.markdown(
    """
**Notes:**
- The nadir point is the geodetic sub-satellite point directly beneath the satellite.
- The footprint radius is calculated as the surface arc distance assuming a spherical Earth.
- Trajectory lines are split at dateline crossings to prevent incorrect wrap-around lines on the map.
"""
)
