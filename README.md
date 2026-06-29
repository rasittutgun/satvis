### LEO Satellite Visualizer (Streamlit + Skyfield + Cartopy)

A production-ready Python application that visualizes the ground track and antenna footprint area of LEO satellites on a 2D world map.

#### Features
- Streamlit-based GUI
- TLE input (2 or 3 lines)
- FoV (degrees), time interval, and time step inputs
- **Static** and **Animated** visualization modes
- Satellite propagation using Skyfield
- Nadir (sub-satellite) lat/lon calculation
- Footprint radius calculation using spherical Earth geometry
- Proper segmentation during date-line (±180°) crossings
- Export animations as **GIF** or **MP4**
- Built-in up-to-date ISS TLE sample by default

#### Project Structure
```text
satellite_visualizer/
├── app.py
├── requirements.txt
├── README.md
└── satvis/
    ├── __init__.py
    ├── core.py       # physics/propagation/geometry
    └── plotting.py   # cartopy plotting and animation
```

#### Installation
```bash
cd satellite_visualizer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scriptsctivate
pip install -r requirements.txt
```

> System-wide installation of `ffmpeg` is required for MP4 export. If it is not installed, please use GIF instead.

#### Running the App
```bash
streamlit run app.py
```

#### Input Descriptions
- **TLE**: Two-line element set (optional name line at the top)
- **Antenna FoV (deg)**: Antenna full field of view (half-angle = FoV/2)
- **Start/End time (UTC)**: Simulation time interval
- **Time step (s)**: Propagation sampling resolution
- **Visualization mode**:
  - `Static`: Full ground track + footprint swath
  - `Animated`: Time-dependent satellite motion + moving footprint

#### Physics Model (Footprint)
- The Earth is assumed to be spherical (`Re = 6371 km`).
- Satellite radius: `Rs = Re + h`.
- Antenna half-angle: `alpha = FoV/2`.
- Earth central angle (footprint angular radius):
  - If not limited by the horizon:
    `psi = asin((Rs/Re) * sin(alpha)) - alpha`
  - If the FoV exceeds the horizon limit, it is clamped to the horizon.
- Surface arc radius: `r = Re * psi`.

#### Outputs
- Ground track on a world map
- Real-time satellite position
- Instantaneous/sequential footprint circle
- Average altitude and footprint radius metrics
- (Optional) GIF/MP4 download

#### Notes
- If the TLE format is invalid, the application displays a meaningful error message.
- For date-line crossing scenarios, the lines are divided into separate segments to prevent incorrect "straight-line" artifacts across the map.
- Because Streamlit's `datetime_input` can return native-naive values, the application normalizes all timestamps to UTC.
