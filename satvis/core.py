"""Core orbital propagation and geometry utilities for LEO visualization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Sequence, Tuple

import numpy as np
from skyfield.api import EarthSatellite, load
from skyfield.timelib import Time

# Mean Earth radius in kilometers (spherical approximation as requested)
EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class PropagationResult:
    """Container for propagated nadir track data."""

    times_utc: List[datetime]
    lats_deg: np.ndarray
    lons_deg: np.ndarray
    alt_km: np.ndarray


def _normalize_lon_deg(lon_deg: np.ndarray | float) -> np.ndarray | float:
    """Normalize longitude to [-180, 180)."""
    return ((np.asarray(lon_deg) + 180.0) % 360.0) - 180.0


def parse_tle(tle_text: str, name: str = "SAT") -> EarthSatellite:
    """Parse a TLE text block and return a Skyfield satellite object.

    Accepts:
      - 2 lines: line1 + line2
      - 3 lines: optional name + line1 + line2
    """
    lines = [ln.strip() for ln in tle_text.splitlines() if ln.strip()]
    if len(lines) == 2:
        line1, line2 = lines
    elif len(lines) >= 3 and lines[1].startswith("1 ") and lines[2].startswith("2 "):
        name = lines[0]
        line1, line2 = lines[1], lines[2]
    else:
        raise ValueError(
            "TLE input is invalid. Please provide two lines (Line 1, Line 2) or a name + two lines."
        )

    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError("TLE lines must start with '1 ' and '2 ' respectively.")

    ts = load.timescale()
    return EarthSatellite(line1, line2, name, ts)


def build_time_grid(
    start_utc: datetime,
    end_utc: datetime,
    step_seconds: int,
) -> List[datetime]:
    """Build an inclusive UTC time grid at fixed step size."""
    if step_seconds <= 0:
        raise ValueError("Time step must be positive.")

    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    else:
        start_utc = start_utc.astimezone(timezone.utc)

    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=timezone.utc)
    else:
        end_utc = end_utc.astimezone(timezone.utc)

    if end_utc <= start_utc:
        raise ValueError("End time must be after start time.")

    total_seconds = int((end_utc - start_utc).total_seconds())
    steps = total_seconds // step_seconds
    out = [start_utc + timedelta(seconds=i * step_seconds) for i in range(steps + 1)]

    # Ensure final endpoint is included when interval is not exactly divisible
    if out[-1] < end_utc:
        out.append(end_utc)

    return out


def _to_skyfield_time(times_utc: Sequence[datetime]) -> Time:
    """Convert Python UTC datetimes to Skyfield Time."""
    ts = load.timescale()
    years = [t.year for t in times_utc]
    months = [t.month for t in times_utc]
    days = [t.day for t in times_utc]
    hours = [t.hour for t in times_utc]
    minutes = [t.minute for t in times_utc]
    seconds = [t.second + t.microsecond / 1e6 for t in times_utc]
    return ts.utc(years, months, days, hours, minutes, seconds)


def propagate_nadir_track(
    satellite: EarthSatellite,
    times_utc: Sequence[datetime],
) -> PropagationResult:
    """Propagate satellite and compute nadir geodetic points + altitude.

    - Position is evaluated in ITRS (Earth-fixed) frame to obtain subpoint coordinates.
    - Nadir point is the geodetic subpoint returned by Skyfield.
    """
    if len(times_utc) < 2:
        raise ValueError("At least two time samples are required.")

    t_sf = _to_skyfield_time(times_utc)
    geocentric = satellite.at(t_sf)
    subpoints = geocentric.subpoint()

    lats_deg = np.asarray(subpoints.latitude.degrees)
    lons_deg = np.asarray(subpoints.longitude.degrees)
    lons_deg = _normalize_lon_deg(lons_deg)
    alt_km = np.asarray(subpoints.elevation.km)

    return PropagationResult(
        times_utc=list(times_utc),
        lats_deg=lats_deg,
        lons_deg=lons_deg,
        alt_km=alt_km,
    )


def footprint_angular_radius_rad(alt_km: float, fov_deg: float) -> float:
    """Compute spherical-Earth footprint angular radius (radians).

    Physics model:
      - Satellite altitude h over spherical Earth radius Re.
      - Antenna half-angle alpha = FoV/2 from nadir axis.
      - Ground-edge central angle psi is obtained from spherical triangle geometry:
            psi = asin((rs/Re) * sin(alpha)) - alpha
        where rs = Re + h.
      - If FoV exceeds visible Earth disk (alpha > alpha_max), clamp to horizon.

    Returns central angle psi in radians. Surface radius is Re * psi.
    """
    if alt_km <= 0:
        raise ValueError("Satellite altitude must be positive.")
    if fov_deg <= 0 or fov_deg >= 179.0:
        raise ValueError("FoV value must be between 0 and 179 degrees.")

    re = EARTH_RADIUS_KM
    rs = re + alt_km
    alpha = np.deg2rad(fov_deg / 2.0)

    # Maximum off-nadir half-angle that still intersects Earth surface (horizon)
    alpha_max = np.arcsin(re / rs)

    if alpha >= alpha_max:
        # Horizon-limited footprint
        psi = np.arccos(re / rs)
    else:
        arg = (rs / re) * np.sin(alpha)
        arg = np.clip(arg, -1.0, 1.0)
        psi = np.arcsin(arg) - alpha

    return float(max(psi, 0.0))


def footprint_surface_radius_km(alt_km: float, fov_deg: float) -> float:
    """Ground footprint radius along Earth surface arc length (km)."""
    return EARTH_RADIUS_KM * footprint_angular_radius_rad(alt_km, fov_deg)


def geodesic_circle_latlon(
    center_lat_deg: float,
    center_lon_deg: float,
    angular_radius_rad: float,
    n_points: int = 180,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a spherical geodesic circle around a center point.

    The circle is returned as latitude/longitude arrays in degrees.
    """
    if angular_radius_rad < 0:
        raise ValueError("Angular radius cannot be negative.")

    lat1 = np.deg2rad(center_lat_deg)
    lon1 = np.deg2rad(center_lon_deg)
    bearings = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

    sin_lat1 = np.sin(lat1)
    cos_lat1 = np.cos(lat1)
    sin_d = np.sin(angular_radius_rad)
    cos_d = np.cos(angular_radius_rad)

    lat2 = np.arcsin(sin_lat1 * cos_d + cos_lat1 * sin_d * np.cos(bearings))
    lon2 = lon1 + np.arctan2(
        np.sin(bearings) * sin_d * cos_lat1,
        cos_d - sin_lat1 * np.sin(lat2),
    )

    lat_deg = np.rad2deg(lat2)
    lon_deg = _normalize_lon_deg(np.rad2deg(lon2))
    return lat_deg, lon_deg


def split_dateline_segments(
    lats_deg: Sequence[float],
    lons_deg: Sequence[float],
    jump_threshold_deg: float = 180.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split polyline into segments at date-line jumps.

    This avoids drawing artificial lines crossing the full map width when the
    trajectory crosses +/-180 longitude.
    """
    lats = np.asarray(lats_deg)
    lons = np.asarray(lons_deg)
    if len(lats) != len(lons):
        raise ValueError("Latitude and longitude arrays must be of the same length.")
    if len(lats) == 0:
        return []

    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for i in range(1, len(lons)):
        if abs(lons[i] - lons[i - 1]) > jump_threshold_deg:
            if i - start >= 2:
                segments.append((lats[start:i], lons[start:i]))
            start = i

    if len(lons) - start >= 2:
        segments.append((lats[start:], lons[start:]))

    return segments
