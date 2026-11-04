# Extreme daily rainfall analysis with mountain overlays

This project extracts extreme-rainfall grid-days from daily NetCDF files,
summarizes their annual, monthly and spatial distribution, and maps the
results over ranked mountain lines. Its grid-count CSV can serve as the
frequency input to the separate RF–SHAP workflow.

## Install and run

Use Python 3.12. From the repository root, create an environment.

```bash
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` in Bash. Then install the dependencies.

```bash
python -m pip install -r requirements.txt
python main.py --help
```

Place the required inputs in the locations below or set the corresponding
environment variables. Relative paths resolve against the project folder.
Shapefile inputs need their companion files, including `.dbf`, `.shx`
and `.prj`. GeoPackage vectors can also be supplied.

| Default path | Environment variable | Input |
| --- | --- | --- |
| `data/netcdf/` | `RAINFALL_NC_DIR` | Daily `.nc` files on a common rectilinear longitude–latitude grid. |
| `data/study_area.shp` | `RAINFALL_BOUNDARY` | Study-domain polygons with a declared CRS. This layer defines the statistical domain. |
| `data/mountains.shp` | `RAINFALL_MOUNTAINS` | Mountain lines with a CRS and `Rank` field. Values 0/1, 2 and 3 define the three line styles. |
| `data/background.tif` | `RAINFALL_BACKGROUND_TIF` | Georeferenced RGB or single-band background covering the map extent. |
| `results/` | `RAINFALL_OUTPUT_DIR` | New or empty directory for a full run. |

The default NetCDF variable is `prec`. It must contain daily precipitation
totals in millimetres, with `time`, `lat` and `lon` dimensions. Supply
one sample per day on each grid and use a calendar that pandas can parse
for the temporal plots. Set `VAR_NAME` in `config.py` to match the data.

```bash
python main.py
```

The following PowerShell commands select another output folder and add
a monsoon overlay.

```powershell
$env:RAINFALL_OUTPUT_DIR = "results/national_run"
$env:RAINFALL_MONSOON = "data/monsoon.gpkg"
python main.py
```

Use `python main.py --map-only` to redraw the map from
`extreme_events_r99p.csv` in the output directory. To also export grid maxima
during a full run, use `python main.py --grid-max`. This option uses the
clipped dataset after the wet-value filter.

## Optional map layers

Set the corresponding paths to enable each layer. Each vector needs a CRS.

| Environment variable | Layer |
| --- | --- |
| `RAINFALL_MONSOON` | Polygon overlay for the monsoon region. To run the statistics within that region, set `RAINFALL_BOUNDARY` to the regional mask as well. |
| `RAINFALL_PROVINCES` | Provincial boundary polygons. |
| `RAINFALL_CAPITALS` | Capital points with `ENG_NAME` by default. The label field is configurable. |
| `RAINFALL_INSET_LINES` | Line layer for the South China Sea inset. Also provide inset polygons. |
| `RAINFALL_INSET_BOUNDARY` | Inset polygons. Defaults to the provincial layer when that layer is supplied. |

The inset layers and background must cover the selected inset extent.
Map colours, ranks, figure styles, mountain labels and inset geometry
are configured in `config.py`. Mountain names use `Ename` by default
when mountain labels are enabled. The map uses Times New Roman where
available and DejaVu Serif as a fallback.

## Calculation and outputs

The default study settings pool values **strictly above 50 mm** inside the
study polygon over the supplied dates and calculate their 99th percentile
using NumPy's linear method. Each grid-day at or above that single
threshold is counted once. The threshold is logged at run time. Temporal
counts are grid-days, so one storm may contribute many records.

| Output | Content |
| --- | --- |
| `extreme_events_r99p.csv` | Event date, latitude, longitude and daily precipitation in mm. |
| `spatial_extremes_attr.csv` | `Lon`, `Lat`, `Max_Precipitation` and `Frequency` for grids with at least one selected event. |
| `annual_extreme_scatter.png`, `monthly_extreme_scatter.png` | Temporal intensity and grid-day count plots. |
| `spatial_extremes_map.png`, `.svg`, `.pdf`, `.tiff` | Spatial plot with mountain ranks, using the selected overlays. TIFF is exported at 600 DPI. |
| `grid_max_rainfall.csv`, `grid_max_rainfall.tif`, `grid_max_rainfall_map.png` | Optional grid maxima from the clipped, wet-filtered dataset. |

The frequency CSV contains event-bearing grid centres. Confirm study-domain
coverage before interpreting absence from the table as zero. The spatial
display keeps the intensity and count peaks per 0.4° cell by default.
The CSV uses the full event-grid aggregation.

The percentile calculation stages valid samples in a temporary disk file in
the output directory. Allow sufficient disk space and memory for the event
table. `PROCESS_TIME_CHUNK_DAYS` controls the reading batch size.

## Data and rights

Supply the study rainfall data, boundary vectors, mountain lines and
background raster separately. Exact source identifiers and input versions
for study replication are pending. Publish third-party layers only under
their applicable terms.

Copyright (c) 2026 Bingxue Li. Commercial use requires the copyright holder's
permission. Software licensing terms are pending.

