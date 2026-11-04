# Terrain conditioning of extreme daily rainfall in China

This repository provides the random forest and SHAP workflow for analysing
associations between six terrain metrics and extreme daily rainfall in mainland
China and Hainan. Run the workflow for the national domain or a supplied monsoon
polygon. Choose daily rainfall maxima (mm) or cumulative extreme-rainfall
grid-days as the response.

## Installation

Use Python 3.12. From the repository root, create and activate an environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with `.\.venv\Scripts\Activate.ps1`.

```bash
python -m pip install -r requirements.txt
python rf_shap_analysis.py --help
```

## Inputs

Supply the preprocessed analysis inputs locally. Paths below are examples relative
to the current working directory. Keep the companion files of any shapefile
together, including its `.shp`, `.shx`, `.dbf` and `.prj`.

| Argument | Required input |
| --- | --- |
| `--mountains` | Mountain lines with a projected CRS in metres and the fields listed below. For the national workflow, supply the study-domain mountain inventory. |
| `--rainfall` in intensity mode | GeoTIFF with a declared CRS. Band 1 contains historical daily rainfall maxima in mm. |
| `--rainfall` in frequency mode | CSV with one row per grid centre, longitude and latitude in EPSG:4326, and a finite nonnegative extreme-day count. |
| `--monsoon-boundary` | Polygon or multipolygon vector with a CRS, required for monsoon runs. Use the same study mask as the analysis being reproduced. |
| `--taiwan-boundary` | Taiwan exclusion polygon with a CRS, required in frequency mode. The workflow excludes intersecting mountain segments because the study frequency input covers mainland China and Hainan. |
| `--national-boundary`, `--inset-line` | Optional vector layers with a CRS for the observed-response map. The inset uses the bounds of the supplied line layer. |

| Mountain field | Metric | Unit |
| --- | --- | --- |
| `macro_len` | F1, length of the parent mountain range | m |
| `intersect_` | F2, ridge–flow angle | degrees |
| `v_perp` | F3, ridge-normal climatological wind component | m/s |
| `uplift_m` | F4, windward relative relief | m |
| `dist_to_se` | F5, climatological coast-to-mountain airflow path length | m |
| `block_cnt` | F6, upwind mountain-barrier count | count |
| `wind_dir` | Meteorological wind direction used for the upwind shift | degrees |

### Data availability

Supply the analysis-ready inputs separately. The original research datasets and
study-specific processed layers are outside this repository. The
[IPCC AR6 WGI Atlas monsoon domains](https://github.com/IPCC-WG1/Atlas/blob/main/reference-regions/README.md#monsoon-regions)
are the upstream reference for the regional mask. To reproduce the study, use
the study's processed mask and input versions. The script starts from those
processed inputs. Source identifiers and preprocessing instructions for the
remaining study inputs are pending.

## Run the analysis

Run the national daily rainfall maxima analysis.

```bash
python rf_shap_analysis.py --mode intensity --study-area national --mountains data/mountains.gpkg --rainfall data/maxrain85to24.tif --output results/national_intensity
```

Run the monsoon-domain analysis for cumulative extreme-rainfall grid-days.

```bash
python rf_shap_analysis.py --mode frequency --study-area monsoon --mountains data/mountains.gpkg --rainfall data/spatial_extremes_attr.csv --taiwan-boundary data/taiwan.gpkg --monsoon-boundary data/ChinaMonsoons.shp --frequency-x-column Lon --frequency-y-column Lat --frequency-value-column Frequency --frequency-missing-is-zero --output results/monsoon_frequency
```

To run intensity in the monsoon domain, select `--study-area monsoon` and supply
`--monsoon-boundary`. For national frequency, select `--study-area national`
and omit the monsoon boundary. Choose a new or empty output directory each time.
Quote paths that contain spaces.

Use `--frequency-missing-is-zero` for a sparse event-grid CSV only after
confirming complete coverage of the study domain. An unmatched, valid sampling
area then represents zero events. The default stops on unmatched areas. By
default, the frequency response sums counts across matched grid centres and
keeps the cumulative total (`--frequency-aggregation sum --frequency-divisor 1`).
Each grid centre contributes its count to every sampling area that contains it,
so segment totals can overlap spatially.

## Workflow

1. Shift mountain segments 40 km upwind and buffer them by 30 km. In monsoon
   mode, clip segment geometry and sampling areas to the supplied polygon,
   retaining the original F1–F6 values. Frequency mode also subsets grid centres
   to the polygon.
2. Extract maximum rainfall in intensity mode or sum extreme-day counts in
   frequency mode. Unresolved F5 paths are represented by a 6,000 km sentinel.
3. Fit the F1–F6 random forest. Evaluate an 80/20 random holdout and five-fold
   spatial cross-validation using 250 km blocks in EPSG:6933. Fit an F1–F4 and
   F6 sensitivity model on the same samples and spatial folds.
4. Export held-out predictions, permutation importance, SHAP values, pairwise
   interactions, and importance intervals from resampling segment-level SHAP
   values for the fitted model.

Study defaults use 1,000 trees, depth 15, a minimum leaf size of three,
30 permutation repeats, 1,000 bootstrap replicates, and seed 42. The
`--trees`, `--permutations` and `--shap-bootstraps` options allow shorter
software checks. The spatial evaluation requires at least five populated blocks.

## Outputs

Intensity plots retain the `TERPImax` suffix. Frequency plots retain
`TERPIpotential` for compatibility. The frequency response is in grid-days
under the study defaults.

| Output | Content |
| --- | --- |
| `cleaned_model_data.csv` | Sample identifiers, model features, and observed responses. |
| `RF_random_holdout_predictions.csv` and `RF_spatial_CV_predictions.csv` | Observed and predicted values for the random test set and spatial folds, respectively. |
| `RF_random_holdout_summary.csv`, `RF_spatial_CV_summary.csv`, `RF_validation_hierarchy_summary.csv` | Validation metrics and comparison of the evaluation schemes. |
| `RF_SHAP_values_primary.csv` and `RF_SHAP_importance_primary.csv` | Sample-level SHAP and feature values, and mean absolute SHAP importance with bootstrap intervals. |
| `RF_SHAP_interaction_matrix_primary.csv` | Mean absolute SHAP interaction values. |
| `shap_dependence_plot_distance_*_binned_source.csv` and `*_sentinel_source.csv` | F5 curve medians and quartiles for resolved paths, plus a separate summary of unresolved paths. |
| `study_area_mask.gpkg`, `study_area_mountain_segments.gpkg`, `study_area_audit.json` | Monsoon geometry and clipping audit. Monsoon frequency also exports `spatial_extremes_attr_Monsoon.csv`. |
| `RF_SHAP_run_record.json` | Input paths, settings, package versions, sample counts, and key results. |

SHAP explains model associations. The F5 sentinel encodes unresolved paths,
and the dependence curve uses resolved path lengths. The workflow supports
susceptibility screening based on historical spatial associations.

## 中文使用说明

用 Python 3.12 安装 `requirements.txt` 后，按上面的命令替换本机数据路径。
`--mode intensity` 对应日降雨极值，`--mode frequency` 对应极端降雨累计网格日数。
选择 `--study-area monsoon` 时提供季风区边界。频次模式需要台湾排除边界，只有确认
事件网格表覆盖完整时才开启 `--frequency-missing-is-zero`。每次运行指定新的结果目录。

## Copyright

Copyright (c) 2026 Bingxue Li. Commercial use requires the copyright holder's
permission. Software licensing terms are pending.
