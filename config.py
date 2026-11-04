"""Input paths and study parameters for the daily rainfall analysis."""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent


def input_path(variable, default=""):
    """Resolve relative paths against the project directory."""
    value = os.environ.get(variable) or default
    if value:
        return str(PROJECT_DIR / Path(os.path.expandvars(value)).expanduser())
    return ""


# Required inputs for a full run. The raster and lines are required for maps.
NC_DATA_DIR = input_path("RAINFALL_NC_DIR", "data/netcdf")
SHP_FILE_PATH = input_path("RAINFALL_BOUNDARY", "data/study_area.shp")
MOUNTAIN_SHP_PATH = input_path("RAINFALL_MOUNTAINS", "data/mountains.shp")
BACKGROUND_TIF_PATH = input_path("RAINFALL_BACKGROUND_TIF", "data/background.tif")
OUTPUT_DIR = input_path("RAINFALL_OUTPUT_DIR", "results")

# Supply paths to enable optional map layers.
MONSOON_SHP_PATH = input_path("RAINFALL_MONSOON")
PROVINCE_SHP_PATH = input_path("RAINFALL_PROVINCES")
CAPITAL_CITY_SHP_PATH = input_path("RAINFALL_CAPITALS")
SOUTH_CHINA_SEA_DASH_SHP_PATH = input_path("RAINFALL_INSET_LINES")
SOUTH_CHINA_SEA_BOUNDARY_SHP_PATH = input_path("RAINFALL_INSET_BOUNDARY") or PROVINCE_SHP_PATH
MONSOON_ENABLED = bool(MONSOON_SHP_PATH)
PROVINCE_BOUNDARY_ENABLED = bool(PROVINCE_SHP_PATH)
CAPITAL_CITY_ENABLED = bool(CAPITAL_CITY_SHP_PATH)
SOUTH_CHINA_SEA_INSET_ENABLED = bool(SOUTH_CHINA_SEA_DASH_SHP_PATH)

# Output filenames are shared with downstream workflows.
EXTREMES_CSV_OUT = os.path.join(OUTPUT_DIR, "extreme_events_r99p.csv")
FIG_ANNUAL_SCATTER = os.path.join(OUTPUT_DIR, "annual_extreme_scatter.png")
FIG_MONTHLY_SCATTER = os.path.join(OUTPUT_DIR, "monthly_extreme_scatter.png")
FIG_SPATIAL_MAP = os.path.join(OUTPUT_DIR, "spatial_extremes_map.png")
FIG_SPATIAL_MAP_SVG = os.path.join(OUTPUT_DIR, "spatial_extremes_map.svg")
FIG_SPATIAL_MAP_PDF = os.path.join(OUTPUT_DIR, "spatial_extremes_map.pdf")
FIG_SPATIAL_MAP_TIFF = os.path.join(OUTPUT_DIR, "spatial_extremes_map.tiff")
SPATIAL_GEO_OUT = os.path.join(OUTPUT_DIR, "spatial_extremes_attr.csv")
GRID_MAX_CSV_OUT = os.path.join(OUTPUT_DIR, "grid_max_rainfall.csv")
FIG_GRID_MAX_MAP = os.path.join(OUTPUT_DIR, "grid_max_rainfall_map.png")
GRID_MAX_TIF_OUT = os.path.join(OUTPUT_DIR, "grid_max_rainfall.tif")

# Satellite raster enhancement and opacity.
BACKGROUND_ALPHA = 0.90
BACKGROUND_PERCENTILE_LOW = 2.0
BACKGROUND_PERCENTILE_HIGH = 98.0
BACKGROUND_CONTRAST = 1.20
BACKGROUND_SATURATION = 1.1
BACKGROUND_GAMMA = 1
BACKGROUND_INTERPOLATION = 'none'

# Inset extent is west, east, south, north in degrees.
SOUTH_CHINA_SEA_EXTENT = (105.0, 125.0, 3.0, 25.0)
SOUTH_CHINA_SEA_INSET_RECT = (0.805, 0.0, 0.195, 0.32)
SOUTH_CHINA_SEA_DASH_COLOR = "#222222"
SOUTH_CHINA_SEA_DASH_WIDTH = 1.2
SOUTH_CHINA_SEA_FRAME_WIDTH = 1
SOUTH_CHINA_SEA_TITLE = ""
SOUTH_CHINA_SEA_TITLE_SIZE = 9

# Monsoon shading and outline.
MONSOON_FILL_COLOR = "#e4d8ef"
MONSOON_FILL_ALPHA = 0.12
MONSOON_EDGE_COLOR = "#1E74F4"
MONSOON_EDGE_WIDTH = 1.5
MONSOON_EDGE_ALPHA = 0.9

# Administrative overlays and label style.
PROVINCE_BOUNDARY_COLOR = "#555555"
PROVINCE_BOUNDARY_WIDTH = 0.65
PROVINCE_BOUNDARY_ALPHA = 0.8
CAPITAL_CITY_COLOR = "none"
CAPITAL_CITY_SIZE = 110
PROVINCIAL_CITY_SIZE = 60
CAPITAL_CITY_EDGE_COLOR = "#1BF4E2"
CAPITAL_CITY_EDGE_WIDTH = 1.4
CAPITAL_CITY_LABEL_ENABLED = True
CAPITAL_CITY_LABEL_FIELD = "ENG_NAME"
CAPITAL_CITY_LABEL_FONT_SIZE = 14
CAPITAL_CITY_LABEL_COLOR = "#1E150F"
CAPITAL_CITY_LABEL_OFFSET = (4, 3)
CAPITAL_CITY_LABEL_HALO_COLOR = "white"
CAPITAL_CITY_LABEL_HALO_WIDTH = 0.2

# Daily precipitation variable (mm/day) and pooled threshold settings.
VAR_NAME = "prec"
# Candidate observations satisfy precipitation > WET_DAY_THRESHOLD.
WET_DAY_THRESHOLD = 50
PERCENTILE = 99
# Chunk size controls memory use and leaves the sample definition unchanged.
PROCESS_TIME_CHUNK_DAYS = 40
# Retain grid maxima strictly above this value.
GRID_MAX_THRESHOLD = 1.0

# Figure style. Spatial thinning applies only to plotted points.
PLOT_DPI = 300
SPATIAL_THIN_ENABLED = True
SPATIAL_THIN_CELL_DEGREES = 0.4
FONT_FAMILY = ["Times New Roman", "DejaVu Serif"]
FONT_SIZE_TITLE = 18
FONT_SIZE_LABEL = 18
FONT_SIZE_TICK = 18
COLOR_CMAP = "YlOrRd"
# Fixed limits support comparison across maps. Auto uses the full data range.
SPATIAL_COLORBAR_MODE = "fixed"
SPATIAL_COLORBAR_FIXED_MIN = 150.0
SPATIAL_COLORBAR_FIXED_MAX = 460.0
SPATIAL_COLORBAR_FIXED_TICKS = [150, 200, 250, 300, 350, 400, 450]

# Mountain line styles for Rank 0/1, Rank 2 and Rank 3.
MOUNTAIN_PRIMARY_COLOR = "#2f2118"
MOUNTAIN_SECONDARY_COLOR = "#5b4333"
MOUNTAIN_TERTIARY_COLOR = "#8a6f5c"
MOUNTAIN_PRIMARY_WIDTH = 4
MOUNTAIN_SECONDARY_WIDTH = 3
MOUNTAIN_TERTIARY_WIDTH = 2.3
MOUNTAIN_LINE_ALPHA = 0.98
MOUNTAIN_HALO_COLOR = "#f7f3eb"
MOUNTAIN_HALO_EXTRA_WIDTH = 2
MOUNTAIN_HALO_ALPHA = 0.92

# Label each name along its longest visible segment.
MOUNTAIN_LABEL_ENABLED = False
MOUNTAIN_LABEL_FIELD = "Ename"
MOUNTAIN_LABEL_RANKS = [1,2]
MOUNTAIN_LABEL_FONT_SIZE = 14
MOUNTAIN_LABEL_LENGTH_FIELD = "length"
MOUNTAIN_LABEL_MIN_LENGTH = None
MOUNTAIN_LABEL_COLOR = "#1E150F"
MOUNTAIN_LABEL_OFFSET_PT = 8.0
MOUNTAIN_LABEL_HALO_COLOR = "white"
MOUNTAIN_LABEL_HALO_WIDTH = 0.1
MOUNTAIN_LABEL_HALO_ALPHA = 0.95
MOUNTAIN_LABEL_MIN_FONT_SIZE = 12
MOUNTAIN_LABEL_LETTER_SPACING = 1.1
