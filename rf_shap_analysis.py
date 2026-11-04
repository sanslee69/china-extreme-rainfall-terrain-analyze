"""Random forest and SHAP analysis of flow-conditioned terrain metrics.

Supports national and monsoon-domain analyses of daily rainfall maxima and
cumulative extreme-rainfall grid-days.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import shap
from matplotlib import font_manager
from matplotlib.lines import Line2D
from shapely.affinity import translate
from shapely import make_valid
from shapely.ops import unary_union
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =============================================================================
# 1. Analysis settings and command-line inputs
# =============================================================================

SHIFT_DISTANCE_M = 40000
BUFFER_DISTANCE_M = 30000
ANALYSIS_YEARS = 40.0


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Enter a positive integer.")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Enter a finite positive number.")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("intensity", "frequency"), required=True)
    parser.add_argument("--study-area", choices=("national", "monsoon"), default="national")
    parser.add_argument("--mountains", type=Path, required=True, help="Preprocessed mountain-segment vector.")
    parser.add_argument("--rainfall", type=Path, required=True, help="Intensity GeoTIFF or frequency CSV.")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory.")
    parser.add_argument("--monsoon-boundary", type=Path, help="Polygon vector required for monsoon runs.")
    parser.add_argument("--taiwan-boundary", type=Path, help="Taiwan exclusion polygon required for frequency runs.")
    parser.add_argument("--national-boundary", type=Path, help="Optional boundary overlay for the observed response map.")
    parser.add_argument("--inset-line", type=Path, help="Optional line layer that enables the observed-map inset.")
    parser.add_argument("--intensity-aggregation", choices=("max", "mean", "median"), default="max")
    parser.add_argument("--frequency-aggregation", choices=("sum", "max", "q90", "q95"), default="sum")
    parser.add_argument("--frequency-divisor", type=positive_float, default=1.0)
    parser.add_argument("--frequency-value-column", default="Frequency")
    parser.add_argument("--frequency-x-column", help="Longitude column. Defaults to the first column.")
    parser.add_argument("--frequency-y-column", help="Latitude column. Defaults to the second column.")
    parser.add_argument("--frequency-missing-is-zero", action="store_true",
                        help="Treat unmatched valid areas as zero when event-grid coverage is complete.")
    parser.add_argument("--trees", type=positive_int, default=1000)
    parser.add_argument("--permutations", type=positive_int, default=30)
    parser.add_argument("--shap-bootstraps", type=positive_int, default=1000)
    args = parser.parse_args(argv)
    if args.study_area == "monsoon" and args.monsoon_boundary is None:
        parser.error("--monsoon-boundary is required for monsoon runs.")
    if args.mode == "frequency" and args.taiwan_boundary is None:
        parser.error("--taiwan-boundary is required for frequency runs.")
    for name in ("mountains", "rainfall", "monsoon_boundary", "taiwan_boundary",
                 "national_boundary", "inset_line"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            parser.error(f"--{name.replace('_', '-')}: file not found: {path}")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory.")
    return args


def configure(args):
    global ANALYSIS_MODE, STUDY_AREA, OUTPUT_DIR, SHP_PATH, MONSOON_SHP_PATH
    global TAIWAN_SHP_PATH, BOUNDARY_SHP_PATH, NINE_DASH_SHP_PATH
    global TARGET_TIF_PATH, TARGET_CSV_PATH, RESPONSE_COL, RESPONSE_LABEL
    global RESPONSE_DISPLAY_NAME, OUTPUT_PREFIX, BAR_COLOR, SHAP_CMAP
    global INTENSITY_AGGREGATION, FREQUENCY_AGGREGATION, FREQUENCY_DIVISOR
    global FREQUENCY_VALUE_COLUMN, FREQUENCY_X_COLUMN, FREQUENCY_Y_COLUMN
    global FREQUENCY_MISSING_IS_ZERO, N_TREES, N_PERMUTATIONS, N_SHAP_BOOTSTRAPS
    global FIGURE_FONT

    ANALYSIS_MODE = args.mode.capitalize()
    STUDY_AREA = args.study_area.capitalize()
    OUTPUT_DIR, SHP_PATH = args.output, args.mountains
    MONSOON_SHP_PATH, TAIWAN_SHP_PATH = args.monsoon_boundary, args.taiwan_boundary
    BOUNDARY_SHP_PATH, NINE_DASH_SHP_PATH = args.national_boundary, args.inset_line
    TARGET_TIF_PATH = args.rainfall if args.mode == "intensity" else None
    TARGET_CSV_PATH = args.rainfall if args.mode == "frequency" else None
    INTENSITY_AGGREGATION = args.intensity_aggregation
    FREQUENCY_AGGREGATION, FREQUENCY_DIVISOR = args.frequency_aggregation, args.frequency_divisor
    FREQUENCY_VALUE_COLUMN = args.frequency_value_column
    FREQUENCY_X_COLUMN, FREQUENCY_Y_COLUMN = args.frequency_x_column, args.frequency_y_column
    FREQUENCY_MISSING_IS_ZERO = args.frequency_missing_is_zero
    N_TREES, N_PERMUTATIONS, N_SHAP_BOOTSTRAPS = args.trees, args.permutations, args.shap_bootstraps
    if ANALYSIS_MODE == "Intensity":
        RESPONSE_COL, OUTPUT_PREFIX = "rain_max", "TERPImax"
        RESPONSE_DISPLAY_NAME = "Extreme Rainfall Maxima (mm)"
        BAR_COLOR, SHAP_CMAP = "#94B6FF", "coolwarm"
    else:
        RESPONSE_COL, OUTPUT_PREFIX = "rain_freq", "TERPIpotential"
        BAR_COLOR, SHAP_CMAP = "#f1b6da", "PiYG_r"
        if FREQUENCY_AGGREGATION == "sum" and FREQUENCY_DIVISOR == 1:
            RESPONSE_DISPLAY_NAME = "Extreme-Rainfall Grid-Day Total"
        elif FREQUENCY_AGGREGATION == "sum" and FREQUENCY_DIVISOR == ANALYSIS_YEARS:
            RESPONSE_DISPLAY_NAME = "Annual Extreme-Rainfall Grid-Day Total"
        else:
            RESPONSE_DISPLAY_NAME = "Aggregated Extreme-Rainfall Grid-Day Count"
    RESPONSE_LABEL = RESPONSE_DISPLAY_NAME
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    FIGURE_FONT = "Times New Roman" if "Times New Roman" in installed_fonts else "DejaVu Serif"
    mpl.rcParams["font.serif"] = [FIGURE_FONT]


STUDY_AREA_AUDIT = {}
ACTIVE_STUDY_MASK = None

RAW_TO_FINGERPRINT = {
    "macro_len": "F1",
    "intersect_": "F2",
    "v_perp": "F3",
    "uplift_m": "F4",
    "dist_to_se": "F5",
    "block_cnt": "F6",
}
PRIMARY_FEATURES = ["F1", "F2", "F3", "F4", "F5", "F6"]
NATIONWIDE_FEATURES = ["F1", "F2", "F3", "F4", "F6"]

F5_SENTINEL_M = 6_000_000.0
RANDOM_HOLDOUT_TEST_SIZE = 0.20
SPATIAL_BLOCK_SIZE_M = 250_000
N_SPLITS = 5
N_TREES = 1000
MAX_DEPTH = 15
MIN_SAMPLES_LEAF = 3
N_PERMUTATIONS = 30
N_SHAP_BOOTSTRAPS = 1000
RANDOM_SEED = 42



# =============================================================================

# =============================================================================

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.linewidth": 0.5,
        "axes.spines.left": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.top": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
TICK_LENGTH_PT = 4.0
TICK_WIDTH_PT = 0.5
COLORBAR_OUTLINE_WIDTH_PT = 0.5
COLORBAR_TICK_LENGTH_PT = 4.0
COLORBAR_TICK_WIDTH_PT = 0.8
COLORBAR_TICK_PAD_PT = 3.0
COLORBAR_LABEL_PAD_PT = 8.0


FIGURE_FONT_SETTINGS = {
    "shap_importance": {
        "title": 9.0,
        "axis": 12,  #8
        "tick": 12,  #7
        "legend": 12,  #7
        "annotation": 9.0,  #7
    },
    "rf_importance": {
        "title": 14.0,
        "axis": 13.0,
        "tick": 12.0,
        "legend": 11.0,
        "annotation": 11.0,
    },
    "random_holdout": {
        "title": 13.0,
        "axis": 13.0,
        "tick": 13.0,
        "legend": 13.0,
        "annotation": 12.0,
    },
    "spatial_cv": {
        "title": 13.0,
        "axis": 13.0,
        "tick": 14.0,
        "legend": 13.0,
        "annotation": 12.0,
    },
    "f5_dependence": {
        "title": 9.0,
        "axis": 8.0,
        "tick": 7.0,
        "legend": 7.0,
        "annotation": 7.0,
    },
    "observed_map": {
        "title": 13.0,
        "axis": 11.0,
        "tick": 9.0,
        "legend": 9.0,
        "annotation": 9.0,
    },
    "interaction": {
        "title": 9.0,
        "axis": 10,
        "tick": 10,
        "legend": 10,
        "annotation": 7.0,
    },
    "shap_summary": {
        "title": 13.0,
        "axis": 12,
        "tick": 12,
        "legend": 12,
        "annotation": 9.0,
    },
}


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 300) -> None:
    """Save a PNG and close its figure."""
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def log_step(number: int, message: str) -> None:
    """Print a progress step immediately."""
    print(f"\nStep {number}: {message}", flush=True)


def log_detail(message: str) -> None:
    print(f"  -> {message}", flush=True)


def json_safe(value):
    """Convert NumPy scalars and non-finite floats for strict JSON output."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def apply_full_frame(ax: mpl.axes.Axes, figure_type: str) -> None:
    """Draw a full frame with inward ticks on the left and bottom."""
    font_setting = FIGURE_FONT_SETTINGS[figure_type]
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        bottom=True,
        left=True,
        top=False,
        right=False,
        length=TICK_LENGTH_PT,
        width=TICK_WIDTH_PT,
    )
    ax.title.set_fontfamily("serif")
    ax.title.set_fontsize(font_setting["title"])
    ax.xaxis.label.set_fontfamily("serif")
    ax.xaxis.label.set_fontsize(font_setting["axis"])
    ax.yaxis.label.set_fontfamily("serif")
    ax.yaxis.label.set_fontsize(font_setting["axis"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("serif")
        label.set_fontsize(font_setting["tick"])
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontfamily("serif")
            text.set_fontsize(font_setting["legend"])
    for text in ax.texts:
        text.set_fontfamily("serif")
        text.set_fontsize(font_setting["annotation"])


def apply_colorbar_style(
    colorbar_or_axis,
    figure_type: str,
    orientation: str = "vertical",
) -> None:
    """Style the colorbar with inward ticks and labels on the right."""
    font_setting = FIGURE_FONT_SETTINGS[figure_type]
    colorbar_axis = (
        colorbar_or_axis.ax
        if hasattr(colorbar_or_axis, "ax")
        else colorbar_or_axis
    )
    for spine in colorbar_axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(COLORBAR_OUTLINE_WIDTH_PT)
    if hasattr(colorbar_or_axis, "outline"):
        colorbar_or_axis.outline.set_visible(True)
        colorbar_or_axis.outline.set_linewidth(
            COLORBAR_OUTLINE_WIDTH_PT
        )
    if orientation == "vertical":
        colorbar_axis.yaxis.set_ticks_position("right")
        colorbar_axis.yaxis.set_label_position("right")
        colorbar_axis.tick_params(
            axis="y",
            which="both",
            direction="in",
            left=False,
            right=True,
            labelleft=False,
            labelright=True,
            length=COLORBAR_TICK_LENGTH_PT,
            width=COLORBAR_TICK_WIDTH_PT,
            pad=COLORBAR_TICK_PAD_PT,
        )
        colorbar_axis.tick_params(
            axis="x", bottom=False, top=False
        )
        colorbar_axis.yaxis.labelpad = COLORBAR_LABEL_PAD_PT
    else:
        colorbar_axis.xaxis.set_ticks_position("bottom")
        colorbar_axis.xaxis.set_label_position("bottom")
        colorbar_axis.tick_params(
            axis="x",
            which="both",
            direction="in",
            bottom=True,
            top=False,
            labelbottom=True,
            length=COLORBAR_TICK_LENGTH_PT,
            width=COLORBAR_TICK_WIDTH_PT,
            pad=COLORBAR_TICK_PAD_PT,
        )
        colorbar_axis.tick_params(
            axis="y", left=False, right=False
        )
        colorbar_axis.xaxis.labelpad = COLORBAR_LABEL_PAD_PT
    colorbar_axis.xaxis.label.set_fontfamily("serif")
    colorbar_axis.xaxis.label.set_fontsize(font_setting["axis"])
    colorbar_axis.yaxis.label.set_fontfamily("serif")
    colorbar_axis.yaxis.label.set_fontsize(font_setting["axis"])
    for label in (
        colorbar_axis.get_xticklabels()
        + colorbar_axis.get_yticklabels()
    ):
        label.set_fontfamily("serif")
        label.set_fontsize(font_setting["tick"])


def make_forest() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def pooled_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[finite]
    predicted = predicted[finite]
    if observed.size == 0:
        return {
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "bias_pred_minus_obs": np.nan,
            "correlation": np.nan,
        }
    r2 = (
        float(r2_score(observed, predicted))
        if observed.size >= 2 and not np.isclose(np.var(observed), 0.0)
        else np.nan
    )
    correlation = (
        float(np.corrcoef(observed, predicted)[0, 1])
        if observed.size >= 2
        and not np.isclose(np.std(observed), 0.0)
        and not np.isclose(np.std(predicted), 0.0)
        else np.nan
    )
    return {
        "r2": r2,
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "bias_pred_minus_obs": float(np.mean(predicted - observed)),
        "correlation": correlation,
    }


def build_spatial_groups(gdf: gpd.GeoDataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Build fixed 250 km spatial blocks in an equal-area CRS."""
    if gdf.crs is None:
        raise ValueError("The input vector has no CRS.")
    metric = gdf.to_crs("EPSG:6933")
    centroids = metric.geometry.centroid
    bx = np.floor(centroids.x / SPATIAL_BLOCK_SIZE_M).astype(int)
    by = np.floor(centroids.y / SPATIAL_BLOCK_SIZE_M).astype(int)
    groups = (bx.astype(str) + "_" + by.astype(str)).rename("spatial_block")
    coords = pd.DataFrame(
        {"x_m": centroids.x, "y_m": centroids.y}, index=gdf.index
    )
    return groups, coords


def nationwide_random_holdout_validation(
    X: pd.DataFrame,
    y: pd.Series,
    analysis_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Run a random 80/20 holdout within the selected domain.

    Nearby segments may be split across sets, so this is a baseline
    alongside the spatially blocked evaluation.
    """
    positions = np.arange(len(X), dtype=int)
    train_pos, test_pos = train_test_split(
        positions,
        test_size=RANDOM_HOLDOUT_TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    train_pos = np.sort(train_pos)
    test_pos = np.sort(test_pos)

    X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
    y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]
    log_detail(
        f"{analysis_name}: random holdout training rows: {len(train_pos)}, "
        f"test rows: {len(test_pos)}, test fraction: "
        f"{RANDOM_HOLDOUT_TEST_SIZE:.0%}."
    )

    model = make_forest()
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    metrics = pooled_metrics(y_test.to_numpy(), prediction)

    predictions = pd.DataFrame(
        {
            "analysis": analysis_name,
            "source_index": X.index[test_pos].to_numpy(),
            "observed": y_test.to_numpy(),
            "predicted": prediction,
            "residual_observed_minus_predicted": (
                y_test.to_numpy() - prediction
            ),
        }
    )

    subset = np.full(len(X), "train", dtype=object)
    subset[test_pos] = "test"
    split_assignment = pd.DataFrame(
        {
            "analysis": analysis_name,
            "source_index": X.index.to_numpy(),
            "subset": subset,
        }
    )

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=N_PERMUTATIONS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    permutation_table = pd.DataFrame(
        {
            "analysis": analysis_name,
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_sd": perm.importances_std,
        }
    )

    summary = {
        "analysis": analysis_name,
        "n_total": int(len(X)),
        "n_train": int(len(train_pos)),
        "n_test": int(len(test_pos)),
        "test_fraction": float(RANDOM_HOLDOUT_TEST_SIZE),
        "random_seed": int(RANDOM_SEED),
        **metrics,
    }
    if "F5" in X.columns:
        summary["f5_sentinel_fraction_train"] = float(
            np.isclose(
                X_train["F5"].to_numpy(),
                F5_SENTINEL_M,
                rtol=0.0,
                atol=0.5,
            ).mean()
        )
        summary["f5_sentinel_fraction_test"] = float(
            np.isclose(
                X_test["F5"].to_numpy(),
                F5_SENTINEL_M,
                rtol=0.0,
                atol=0.5,
            ).mean()
        )

    return predictions, split_assignment, permutation_table, summary


def spatial_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    coords: pd.DataFrame,
    analysis_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run grouped spatial cross-validation and held-out permutation importance.

    Keep every segment within a spatial block in the same fold.
    """
    if groups.nunique() < N_SPLITS:
        raise ValueError(f"{STUDY_AREA}: {groups.nunique()} spatial blocks available, "
                         f"but N_SPLITS={N_SPLITS}. Choose a justified smaller fold count or block size.")
    splitter = GroupKFold(n_splits=N_SPLITS)
    fold_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    perm_frames: list[pd.DataFrame] = []

    for fold, (train_pos, test_pos) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        log_detail(
            f"{analysis_name}: spatial fold {fold}/{N_SPLITS}, "
            f"training rows {len(train_pos)}, test rows {len(test_pos)}."
        )
        X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
        y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]

        model = make_forest()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        metrics = pooled_metrics(y_test.to_numpy(), prediction)

        fold_rows.append(
            {
                "analysis": analysis_name,
                "fold": fold,
                "n_train": len(train_pos),
                "n_test": len(test_pos),
                "n_train_blocks": groups.iloc[train_pos].nunique(),
                "n_test_blocks": groups.iloc[test_pos].nunique(),
                **metrics,
            }
        )

        pred_frames.append(
            pd.DataFrame(
                {
                    "analysis": analysis_name,
                    "fold": fold,
                    "source_index": X.index[test_pos],
                    "spatial_block": groups.iloc[test_pos].to_numpy(),
                    "x_m": coords.loc[X.index[test_pos], "x_m"].to_numpy(),
                    "y_m": coords.loc[X.index[test_pos], "y_m"].to_numpy(),
                    "observed": y_test.to_numpy(),
                    "predicted": prediction,
                    "residual": y_test.to_numpy() - prediction,
                }
            )
        )

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=N_PERMUTATIONS,
            random_state=RANDOM_SEED + fold,
            n_jobs=-1,
        )
        perm_frames.append(
            pd.DataFrame(
                {
                    "analysis": analysis_name,
                    "fold": fold,
                    "feature": X.columns,
                    "importance_mean": perm.importances_mean,
                    "importance_sd": perm.importances_std,
                }
            )
        )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    permutations = pd.concat(perm_frames, ignore_index=True)
    return folds, predictions, permutations


def bootstrap_mean_abs_shap(
    shap_values: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """
    Bootstrap mean absolute SHAP values by resampling mountain segments
    from the fitted model.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    n = shap_values.shape[0]
    estimates = np.empty((N_SHAP_BOOTSTRAPS, shap_values.shape[1]))
    for b in range(N_SHAP_BOOTSTRAPS):
        draw = rng.randint(0, n, size=n)
        estimates[b] = np.abs(shap_values[draw]).mean(axis=0)
        if (b + 1) % 200 == 0 or (b + 1) == N_SHAP_BOOTSTRAPS:
            log_detail(
                f"SHAP bootstrap progress: {b + 1}/{N_SHAP_BOOTSTRAPS}."
            )

    return pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "bootstrap_ci_low": np.quantile(estimates, 0.025, axis=0),
            "bootstrap_ci_high": np.quantile(estimates, 0.975, axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)


def plot_shap_importance(table: pd.DataFrame, output_stem: Path) -> None:
    table = table.sort_values("mean_abs_shap", ascending=True)
    xerr = np.vstack(
        [
            table["mean_abs_shap"] - table["bootstrap_ci_low"],
            table["bootstrap_ci_high"] - table["mean_abs_shap"],
        ]
    )
    fig, ax = plt.subplots(figsize=(6.0, 4.5))  #(4, 3))
    ax.barh(
        table["feature"],
        table["mean_abs_shap"],
        xerr=xerr,
        color=BAR_COLOR,
        alpha=0.9,
        capsize=2,
        # label="Mean absolute SHAP value with 95% bootstrap interval",
        label="Mean absolute SHAP value\nwith 95% bootstrap interval",
    )

    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    ax.legend(loc="lower right")
    apply_full_frame(ax, "shap_importance")
    save_figure(fig, output_stem)


def plot_rf_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str],
    output_stem: Path,
) -> None:
    """Plot impurity-based random forest feature importance."""
    importance = pd.Series(
        model.feature_importances_, index=feature_names
    ).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.barh(
        importance.index,
        importance.values,
        color=BAR_COLOR,
        align="center",
        label="RF feature importance",
    )
    ax.set_xlabel("Feature importance score", fontweight="bold")
    ax.set_ylabel("Terrain metrics", fontweight="bold")
    #ax.set_title(
       # f"Feature Importance Ranking for {RESPONSE_DISPLAY_NAME}",
       # fontweight="bold",
    #)
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    ax.legend(loc="lower right")
    apply_full_frame(ax, "rf_importance")
    fig.tight_layout()
    save_figure(fig, output_stem)


def plot_random_holdout_performance(
    predictions: pd.DataFrame,
    metrics: dict[str, float],
    output_stem: Path,
) -> None:
    """Plot observations against predictions for the random 20% holdout."""
    observed = predictions["observed"].to_numpy()
    predicted = predictions["predicted"].to_numpy()
    lower = float(min(np.min(observed), np.min(predicted)))
    upper = float(max(np.max(observed), np.max(predicted)))
    if np.isclose(lower, upper):
        upper = lower + 1.0

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(
        observed,
        predicted,
        s=16,
        alpha=0.55,
        color=BAR_COLOR,
        edgecolors="none",
        rasterized=True,
        label=f"Randomly held-out mountain segments (n={len(predictions)})",
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        color="0.25",
        linestyle="--",
        linewidth=1.2,
        label="1:1 reference",
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(f"Observed {RESPONSE_DISPLAY_NAME}")
    ax.set_ylabel(f"Random-holdout fitted {RESPONSE_DISPLAY_NAME}")
    #ax.set_title("Nationwide random holdout validation", fontweight="bold")
    ax.text(
        0.03,
        0.97,
        (
            f"Test R² = {metrics['r2']:.3f}\n"
            f"RMSE = {metrics['rmse']:.3f}\n"
            f"MAE = {metrics['mae']:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    ax.legend(loc="lower right")
    ax.grid(color="0.92", linewidth=0.6)
    apply_full_frame(ax, "random_holdout")
    fig.tight_layout()
    save_figure(fig, output_stem)


def plot_spatial_cv_performance(
    predictions: pd.DataFrame,
    metrics: dict[str, float],
    output_stem: Path,
) -> None:
    """Plot pooled out-of-fold predictions and a 1:1 reference line."""
    observed = predictions["observed"].to_numpy()
    predicted = predictions["predicted"].to_numpy()
    lower = float(min(np.min(observed), np.min(predicted)))
    upper = float(max(np.max(observed), np.max(predicted)))
    if np.isclose(lower, upper):
        upper = lower + 1.0

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(
        observed,
        predicted,
        s=16,
        alpha=0.55,
        color=BAR_COLOR,
        edgecolors="none",
        rasterized=True,
        label=f"Out-of-fold mountain segments (n={len(predictions)})",
    )
    ax.plot(
        [lower, upper],
        [lower, upper],
        color="0.25",
        linestyle="--",
        linewidth=1.2,
        label="1:1 reference",
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(f"Observed {RESPONSE_DISPLAY_NAME}")
    ax.set_ylabel(f"Spatial-CV predicted {RESPONSE_DISPLAY_NAME}")
    #ax.set_title("Spatial block cross-validation", fontweight="bold")
    ax.text(
        0.03,
        0.97,
        (
            f"Pooled R² = {metrics['r2']:.3f}\n"
            f"RMSE = {metrics['rmse']:.3f}\n"
            f"MAE = {metrics['mae']:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )
    ax.legend(loc="lower right")
    ax.grid(color="0.92", linewidth=0.6)
    apply_full_frame(ax, "spatial_cv")
    fig.tight_layout()
    save_figure(fig, output_stem)


def plot_f5_dependence(
    X: pd.DataFrame, shap_values: np.ndarray, output_stem: Path
) -> None:
    f5_pos = X.columns.get_loc("F5")
    distance_km = X["F5"].to_numpy() / 1000.0
    contribution = shap_values[:, f5_pos]
    sentinel_mask = np.isclose(
        X["F5"].to_numpy(),
        F5_SENTINEL_M,
        rtol=0.0,
        atol=0.5,
    )
    resolved_mask = ~sentinel_mask

    fig, ax = plt.subplots(figsize=(3.50, 2.76))
    ax.scatter(
        distance_km[resolved_mask],
        contribution[resolved_mask],
        s=8,
        alpha=0.35,
        color=BAR_COLOR,
        linewidths=0,
        rasterized=True,
        label="Resolved paths",
    )
    if sentinel_mask.any():
        ax.scatter(
            distance_km[sentinel_mask],
            contribution[sentinel_mask],
            s=11,
            alpha=0.45,
            facecolors="none",
            edgecolors=BAR_COLOR,
            linewidths=0.45,
            rasterized=True,
            label="Unresolved paths (6000-km sentinel)",
        )
    
    resolved_distance_km = distance_km[resolved_mask]
    resolved_contribution = contribution[resolved_mask]
    # Bin only resolved paths. An all-sentinel sample has no distance curve.
    bins = (
        pd.qcut(resolved_distance_km, q=min(20, len(resolved_distance_km)), duplicates="drop")
        if resolved_mask.any() else pd.Categorical([])
    )
    summary = (
        pd.DataFrame(
            {
                "distance_km": resolved_distance_km,
                "shap_f5": resolved_contribution,
                "bin": bins,
            }
        )
        .groupby("bin", observed=True)
        .agg(
            distance_km=("distance_km", "median"),
            shap_median=("shap_f5", "median"),
            shap_q25=("shap_f5", lambda z: np.quantile(z, 0.25)),
            shap_q75=("shap_f5", lambda z: np.quantile(z, 0.75)),
            n=("shap_f5", "size"),
        )
        .reset_index(drop=True)
    )
    summary.to_csv(
        output_stem.with_name(output_stem.name + "_binned_source.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    sentinel_summary = pd.DataFrame(
        [
            {
                "f5_encoded_km": F5_SENTINEL_M / 1000.0,
                "n": int(sentinel_mask.sum()),
                "shap_median": (
                    float(np.median(contribution[sentinel_mask]))
                    if sentinel_mask.any()
                    else np.nan
                ),
                "shap_q25": (
                    float(np.quantile(contribution[sentinel_mask], 0.25))
                    if sentinel_mask.any()
                    else np.nan
                ),
                "shap_q75": (
                    float(np.quantile(contribution[sentinel_mask], 0.75))
                    if sentinel_mask.any()
                    else np.nan
                ),
            }
        ]
    )
    sentinel_summary.to_csv(
        output_stem.with_name(output_stem.name + "_sentinel_source.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    if not summary.empty:
        ax.plot(
            summary["distance_km"],
            summary["shap_median"],
            color=BAR_COLOR,
            linewidth=1.5,
            label="Binned median",
        )
        ax.fill_between(
            summary["distance_km"],
            summary["shap_q25"],
            summary["shap_q75"],
            color=BAR_COLOR,
            alpha=0.28,
            linewidth=0,
            label="Interquartile range",
        )
    ax.axhline(0, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_xlabel("F5: coast-to-mountain airflow-path length (km)")
    ax.set_ylabel("SHAP contribution to model output")
    ax.legend(loc="best")
    apply_full_frame(ax, "f5_dependence")
    save_figure(fig, output_stem)


def plot_observed_study_domain(
    analysis_gdf: gpd.GeoDataFrame, output_stem: Path
) -> None:
    """
    Map observed responses with optional context layers and an inset.
    """
    boundary = (
        gpd.read_file(BOUNDARY_SHP_PATH).to_crs(analysis_gdf.crs)
        if BOUNDARY_SHP_PATH else None
    )
    values = analysis_gdf[RESPONSE_COL]
    norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = plt.get_cmap(SHAP_CMAP)

    fig, ax = plt.subplots(figsize=(7.20, 5.30))
    legend = []
    if boundary is not None:
        boundary.plot(ax=ax, facecolor="none", edgecolor="0.25", linewidth=0.5)
        legend.append(Line2D([0], [0], color="0.25", linewidth=0.8, label="National boundary"))
    if ACTIVE_STUDY_MASK is not None:
        gpd.GeoSeries([ACTIVE_STUDY_MASK], crs=analysis_gdf.crs).boundary.plot(
            ax=ax, color="#b35806", linewidth=0.8, label="Monsoon region")
    analysis_gdf.plot(
        ax=ax,
        column=RESPONSE_COL,
        cmap=cmap,
        norm=norm,
        linewidth=1.2,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Observed {RESPONSE_DISPLAY_NAME}", loc="left", fontweight="bold"
    )
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label(RESPONSE_DISPLAY_NAME)
    apply_colorbar_style(colorbar, "observed_map")

    if NINE_DASH_SHP_PATH:
        inset_line = gpd.read_file(NINE_DASH_SHP_PATH).to_crs(analysis_gdf.crs)
        bounds = inset_line.total_bounds
        if not np.isfinite(bounds).all():
            raise ValueError("The inset line layer has no finite bounds.")
        inset = inset_axes(ax, width="24%", height="30%", loc="lower right")
        if boundary is not None:
            boundary.plot(ax=inset, facecolor="none", edgecolor="0.35", linewidth=0.35)
        inset_line.plot(ax=inset, color="0.2", linewidth=0.7)
        analysis_gdf.plot(ax=inset, column=RESPONSE_COL, cmap=cmap, norm=norm, linewidth=0.8)
        inset.set_xlim(bounds[0] - 2, bounds[2] + 2)
        inset.set_ylim(bounds[1] - 2, bounds[3] + 2)
        inset.set_xticks([])
        inset.set_yticks([])
        apply_full_frame(inset, "observed_map")
        legend.append(Line2D([0], [0], color="0.2", linewidth=0.8, label="Reference line (inset)"))
    if legend:
        ax.legend(handles=legend, loc="lower left")
    apply_full_frame(ax, "observed_map")
    
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.02, top=0.92)
    save_figure(fig, output_stem, dpi=600)


def build_windward_polygon(geometry, wind_direction):
    """Shift the segment 40 km upwind and apply a 30 km buffer."""
    angle = np.radians(90.0 - float(wind_direction))
    dx = SHIFT_DISTANCE_M * np.cos(angle)
    dy = SHIFT_DISTANCE_M * np.sin(angle)
    return translate(geometry, xoff=dx, yoff=dy).buffer(BUFFER_DISTANCE_M)


def aggregate_intensity(values: np.ndarray) -> float:
    if INTENSITY_AGGREGATION == "max":
        return float(np.max(values))
    if INTENSITY_AGGREGATION == "mean":
        return float(np.mean(values))
    if INTENSITY_AGGREGATION == "median":
        return float(np.median(values))
    raise ValueError("INTENSITY_AGGREGATION must be max, mean or median.")


def aggregate_frequency(values: pd.Series) -> float:
    if FREQUENCY_AGGREGATION == "sum":
        count = values.sum()
    elif FREQUENCY_AGGREGATION == "max":
        count = values.max()
    elif FREQUENCY_AGGREGATION == "q90":
        count = values.quantile(0.90)
    elif FREQUENCY_AGGREGATION == "q95":
        count = values.quantile(0.95)
    else:
        raise ValueError(
            "FREQUENCY_AGGREGATION must be sum, max, q90 or q95."
        )
    return float(count / FREQUENCY_DIVISOR)

def polygonal_part(geometry):
    """Keep only polygon components after validity repair or intersection."""
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    if hasattr(geometry, "geoms"):
        parts = [polygonal_part(part) for part in geometry.geoms]
        parts = [part for part in parts if part is not None and not part.is_empty]
        return unary_union(parts) if parts else None
    return None


def linear_part(geometry):
    """Discard point-only boundary touches; retain one row per source segment."""
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type in {"LineString", "MultiLineString"}:
        return geometry if geometry.length > 0 else None
    if hasattr(geometry, "geoms"):
        parts = [linear_part(part) for part in geometry.geoms]
        parts = [part for part in parts if part is not None]
        return unary_union(parts) if parts else None
    return None


def clip_to_study_area(gdf):
    """Clip mountain geometry; preserve all precomputed terrain attributes."""
    global ACTIVE_STUDY_MASK, STUDY_AREA_AUDIT
    STUDY_AREA_AUDIT = {"study_area": STUDY_AREA, "source_segment_count": len(gdf)}
    ACTIVE_STUDY_MASK = None
    if STUDY_AREA == "National":
        return gdf
    if not gdf.crs.is_projected or any(
        not np.isclose(axis.unit_conversion_factor, 1.0) for axis in gdf.crs.axis_info
    ):
        raise ValueError("Monsoon sampling requires mountain coordinates projected in metres.")
    if not MONSOON_SHP_PATH.exists():
        raise FileNotFoundError(f"Monsoon boundary not found: {MONSOON_SHP_PATH}")
    region = gpd.read_file(MONSOON_SHP_PATH)
    if region.crs is None:
        raise ValueError("The monsoon boundary has no CRS.")
    invalid_count = int((~region.geometry.is_valid).sum())
    region.geometry = region.geometry.apply(
        lambda geom: polygonal_part(make_valid(geom)) if geom is not None else None
    )
    region = region.loc[region.geometry.notna() & ~region.geometry.is_empty].to_crs(gdf.crs)
    parts = [polygonal_part(make_valid(geom)) for geom in region.geometry]
    parts = [geom for geom in parts if geom is not None and not geom.is_empty]
    if not parts:
        raise ValueError("The monsoon boundary contains no usable polygon area.")
    mask = polygonal_part(make_valid(unary_union(parts)))
    if mask is None or mask.is_empty or mask.area <= 0:
        raise ValueError("The repaired monsoon mask is empty.")
    ACTIVE_STUDY_MASK = mask
    selected = gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    selected = selected.loc[selected.intersects(mask)].copy()
    original_lengths = selected.geometry.length
    selected.geometry = selected.geometry.intersection(mask).apply(linear_part)
    selected = selected.loc[selected.geometry.notna() & ~selected.geometry.is_empty].copy()
    if selected.empty:
        raise ValueError("No mountain line segments overlap the monsoon region.")
    selected["scope_source_index"] = selected.index
    selected["scope_original_length_m"] = original_lengths.loc[selected.index]
    selected["scope_clipped_length_m"] = selected.geometry.length
    selected["scope_retained_fraction"] = selected.scope_clipped_length_m / selected.scope_original_length_m
    STUDY_AREA_AUDIT.update({
        "boundary_path": str(MONSOON_SHP_PATH),
        "invalid_boundary_features_repaired": invalid_count,
        "clipped_segment_count": len(selected),
        "partially_clipped_segment_count": int((selected.scope_retained_fraction < 1 - 1e-8).sum()),
        "terrain_attributes": "Original values retained; not recalculated after clipping",
        "response_sampling": "Windward buffers of clipped lines intersected with the monsoon polygon",
        "frequency_zero_rule": (
            "Unmatched valid areas are zero"
            if FREQUENCY_MISSING_IS_ZERO else "Unmatched valid areas cause an error"
        ),
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame({"study_area": [STUDY_AREA]}, geometry=[mask], crs=gdf.crs).to_file(
        OUTPUT_DIR / "study_area_mask.gpkg", driver="GPKG", layer="study_area", index=False)
    selected.to_file(OUTPUT_DIR / "study_area_mountain_segments.gpkg", driver="GPKG", layer="segments", index=False)
    log_detail(f"Monsoon clipping: {len(gdf)} -> {len(selected)} mountain segments; "
               f"repaired boundary features: {invalid_count}.")
    return selected


def response_sampling_polygon(geometry, wind_direction):
    polygon = build_windward_polygon(geometry, wind_direction)
    if ACTIVE_STUDY_MASK is not None:
        polygon = polygonal_part(polygon.intersection(ACTIVE_STUDY_MASK))
    return polygon


def load_original_inputs_and_extract_response() -> gpd.GeoDataFrame:
    """
    Extract responses from a rainfall GeoTIFF or a point-count CSV.
    """
    gdf_source = gpd.read_file(SHP_PATH)
    crs = gdf_source.crs
    if crs is None or not crs.is_projected or any(
        not np.isclose(axis.unit_conversion_factor, 1.0) for axis in crs.axis_info
    ):
        raise ValueError("The mountain-segment vector requires a projected CRS in metres.")
    missing = (set(RAW_TO_FINGERPRINT) | {"wind_dir"}) - set(gdf_source.columns)
    if missing:
        raise ValueError(f"Missing mountain input fields: {', '.join(sorted(missing))}")
    gdf_source["wind_dir"] = pd.to_numeric(gdf_source["wind_dir"], errors="coerce")
    valid = (
        gdf_source.geometry.geom_type.isin(["LineString", "MultiLineString"])
        & gdf_source.geometry.is_valid
        & ~gdf_source.geometry.is_empty
        & (gdf_source.geometry.length > 0)
        & np.isfinite(gdf_source["wind_dir"])
    )
    if not valid.all():
        log_detail(f"Excluded {(~valid).sum()} segments with invalid geometry or wind direction.")
    gdf_source = gdf_source.loc[valid].copy()
    if gdf_source.empty:
        raise ValueError("No valid mountain segments are available.")

    gdf_source = clip_to_study_area(gdf_source)
    responses: list[float] = []
    sampling_area_valid: list[bool] = []
    if ANALYSIS_MODE == "Intensity":
        with rasterio.open(TARGET_TIF_PATH) as src:
            if src.crs is None:
                raise ValueError("The intensity raster has no CRS.")
            total_segments = len(gdf_source)
            for processed, (_, row) in enumerate(
                gdf_source.iterrows(), start=1
            ):
                if (
                    processed == 1
                    or processed % 200 == 0
                    or processed == total_segments
                ):
                    log_detail(
                        f"Intensity extraction progress: "
                        f"{processed}/{total_segments}."
                    )
                geometry = row.geometry
                wind_direction = row.get("wind_dir")
                if (
                    geometry is None
                    or geometry.is_empty
                    or pd.isna(wind_direction)
                ):
                    sampling_area_valid.append(False)
                    responses.append(np.nan)
                    continue

                polygon = response_sampling_polygon(geometry, wind_direction)
                valid_area = polygon is not None and not polygon.is_empty and polygon.area > 0
                sampling_area_valid.append(valid_area)
                if not valid_area:
                    responses.append(np.nan)
                    continue
                polygon_raster_crs = gpd.GeoSeries(
                    [polygon], crs=gdf_source.crs
                ).to_crs(src.crs).iloc[0]
                try:
                    image, _ = rasterio.mask.mask(
                        src, [polygon_raster_crs], crop=True,
                        filled=(STUDY_AREA == "National")
                    )
                except ValueError:
                    responses.append(np.nan)
                    continue

                array = image[0]
                if STUDY_AREA == "Monsoon":
                    array = array.compressed()
                valid = np.isfinite(array) & (array >= 0)
                if src.nodata is not None:
                    valid &= array != src.nodata
                values = array[valid]
                responses.append(
                    aggregate_intensity(values) if values.size else np.nan
                )
    else:
        points_table = pd.read_csv(TARGET_CSV_PATH)
        if len(points_table.columns) < 3:
            raise ValueError("The frequency CSV requires longitude, latitude and count columns.")
        x_col = (
            FREQUENCY_X_COLUMN
            if FREQUENCY_X_COLUMN is not None
            else points_table.columns[0]
        )
        y_col = (
            FREQUENCY_Y_COLUMN
            if FREQUENCY_Y_COLUMN is not None
            else points_table.columns[1]
        )
        columns = [x_col, y_col, FREQUENCY_VALUE_COLUMN]
        if len(set(columns)) != 3 or any(column not in points_table for column in columns):
            raise ValueError("Specify three distinct existing longitude, latitude and count columns.")
        values = points_table[columns].apply(pd.to_numeric, errors="coerce")
        valid = (
            np.isfinite(values).all(axis=1)
            & values[x_col].between(-180, 180)
            & values[y_col].between(-90, 90)
            & (values[FREQUENCY_VALUE_COLUMN] >= 0)
        )
        if not valid.all() or values.duplicated([x_col, y_col]).any():
            raise ValueError("Frequency rows require unique valid coordinates and finite nonnegative counts.")
        for column in columns:
            points_table[column] = values[column]
        points = gpd.GeoDataFrame(
            points_table,
            geometry=gpd.points_from_xy(
                points_table[x_col], points_table[y_col]
            ),
            crs="EPSG:4326",
        ).to_crs(gdf_source.crs)
        if ACTIVE_STUDY_MASK is not None:
            source_point_count = len(points)

            # Retain grid centres inside or on the monsoon boundary.
            points = points.loc[
                points.intersects(ACTIVE_STUDY_MASK)
            ].copy()

            # Export the original attributes and longitude/latitude.
            # Do not export the projected geometry column.
            filtered_rainfall = points_table.loc[points.index].copy()
            filtered_csv_path = (
                OUTPUT_DIR / "spatial_extremes_attr_Monsoon.csv"
            )
            filtered_rainfall.to_csv(
                filtered_csv_path,
                index=False,
                encoding="utf-8-sig",
            )

            STUDY_AREA_AUDIT.update({
                "frequency_source_points": source_point_count,
                "frequency_points_in_region": len(points),
                "filtered_rainfall_csv": str(filtered_csv_path),
            })

            log_detail(
                f"Monsoon rainfall data exported: {filtered_csv_path} "
                f"({len(filtered_rainfall)} grid cells)"
            )

        spatial_index = points.sindex

        total_segments = len(gdf_source)
        for processed, (_, row) in enumerate(
            gdf_source.iterrows(), start=1
        ):
            if (
                processed == 1
                or processed % 200 == 0
                or processed == total_segments
            ):
                log_detail(
                    f"Frequency extraction progress: "
                    f"{processed}/{total_segments}."
                )
            geometry = row.geometry
            wind_direction = row.get("wind_dir")
            if (
                geometry is None
                or geometry.is_empty
                or pd.isna(wind_direction)
            ):
                sampling_area_valid.append(False)
                responses.append(np.nan)
                continue

            polygon = response_sampling_polygon(geometry, wind_direction)
            valid_area = polygon is not None and not polygon.is_empty and polygon.area > 0
            sampling_area_valid.append(valid_area)
            if not valid_area:
                responses.append(np.nan)
                continue
            candidates = points.iloc[
                list(spatial_index.intersection(polygon.bounds))
            ]
            matches = candidates[candidates.intersects(polygon)]
            values_raw = pd.to_numeric(
                matches[FREQUENCY_VALUE_COLUMN], errors="coerce"
            )
            values = values_raw.loc[values_raw.notna()]
            responses.append(
                aggregate_frequency(values) if len(values) else np.nan
            )

    gdf_source[RESPONSE_COL] = responses
    if STUDY_AREA == "Monsoon":
        gdf_source["scope_valid_sampling_area"] = sampling_area_valid
        STUDY_AREA_AUDIT["excluded_empty_or_invalid_sampling_areas"] = int((~np.asarray(sampling_area_valid)).sum())
        gdf_source = gdf_source.loc[gdf_source.scope_valid_sampling_area].copy()
        if gdf_source.empty:
            raise ValueError("No valid windward sampling areas remain inside the monsoon region.")
        (OUTPUT_DIR / "study_area_audit.json").write_text(
            json.dumps(json_safe(STUDY_AREA_AUDIT), ensure_ascii=False, indent=2), encoding="utf-8")
    elif ANALYSIS_MODE == "Frequency":
        gdf_source = gdf_source.loc[sampling_area_valid].copy()
    return gdf_source



# =============================================================================

# =============================================================================

def main(argv=None):
    args = parse_args(argv)
    configure(args)
    mpl.use("Agg")
    print(
        f"--- Random forest and SHAP analysis | Mode: {ANALYSIS_MODE} ---",
        flush=True,
    )
    log_detail(f"Study area: {STUDY_AREA}; output: {OUTPUT_DIR}")
    log_detail(f"Frequency aggregation: {FREQUENCY_AGGREGATION}")
    log_detail(f"Plot color: {BAR_COLOR}; SHAP colormap: {SHAP_CMAP}")

    log_step(1, "Create the output directory.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_detail("Output directory ready.")

    log_step(2, "Read inputs and extract responses in windward sampling areas.")
    gdf = load_original_inputs_and_extract_response()
    log_detail(f"Response extraction complete for {len(gdf)} segments.")

    log_step(3, "Prepare samples for the primary and sensitivity models.")
    required = list(RAW_TO_FINGERPRINT) + [RESPONSE_COL, "geometry"]
    missing = [column for column in required if column not in gdf.columns]
    if missing:
        raise KeyError(f"Missing required input columns: {missing}")



    if ANALYSIS_MODE == "Frequency":
        if not TAIWAN_SHP_PATH.exists():
            raise FileNotFoundError(
                f"Taiwan boundary not found: {TAIWAN_SHP_PATH}"
            )
        taiwan = gpd.read_file(TAIWAN_SHP_PATH).to_crs(gdf.crs)
        taiwan_geometry = (
            taiwan.geometry.union_all()
            if hasattr(taiwan.geometry, "union_all")
            else taiwan.geometry.unary_union
        )
        outside_taiwan = ~gdf.geometry.intersects(taiwan_geometry)
        n_taiwan_removed = int((~outside_taiwan).sum())
        gdf = gdf.loc[outside_taiwan].copy()
        log_detail(f"Frequency mode excluded {n_taiwan_removed} segments intersecting Taiwan.")

    for column in ["macro_len", "intersect_", "v_perp"]:
        gdf[column] = pd.to_numeric(gdf[column], errors="coerce").fillna(0)
    gdf["uplift_m"] = pd.to_numeric(
        gdf["uplift_m"], errors="coerce"
    ).fillna(0)
    gdf["block_cnt"] = pd.to_numeric(
        gdf["block_cnt"], errors="coerce"
    ).fillna(99)
    if ANALYSIS_MODE == "Frequency":
    
        unmatched = gdf[RESPONSE_COL].isna()
        if unmatched.any() and not FREQUENCY_MISSING_IS_ZERO:
            raise ValueError(
                f"{unmatched.sum()} sampling areas have no frequency observations. "
                "Use --frequency-missing-is-zero only after confirming complete event-grid coverage."
            )
        gdf[RESPONSE_COL] = gdf[RESPONSE_COL].fillna(0)
    else:
        gdf = gdf[gdf[RESPONSE_COL] > 0].copy()

    gdf = gdf.rename(columns=RAW_TO_FINGERPRINT)
    for column in PRIMARY_FEATURES + [RESPONSE_COL]:
        gdf[column] = pd.to_numeric(gdf[column], errors="coerce")
    base_required = NATIONWIDE_FEATURES + [RESPONSE_COL]
    base_valid = gdf[base_required].notna().all(axis=1)
    base_valid &= np.isfinite(gdf[base_required]).all(axis=1)
    analysis_all = gdf.loc[base_valid].copy()


    f5_raw = pd.to_numeric(analysis_all["F5"], errors="coerce")
    unresolved_f5 = (
        f5_raw.isna()
        | (~np.isfinite(f5_raw))
        | (f5_raw <= 0)
        | np.isclose(
            f5_raw,
            F5_SENTINEL_M,
            rtol=0.0,
            atol=0.5,
        )
    )
    resolved_f5 = ~unresolved_f5
    maximum_resolved_f5 = np.nan
    if resolved_f5.any():
        maximum_resolved_f5 = float(f5_raw.loc[resolved_f5].max())
        if maximum_resolved_f5 >= F5_SENTINEL_M:
            raise ValueError(
                "The 6000 km sentinel must exceed all resolved F5 path lengths."
            )

    analysis_all.loc[unresolved_f5, "F5"] = F5_SENTINEL_M
    analysis_all.loc[resolved_f5, "F5"] = f5_raw.loc[resolved_f5]
    analysis_primary = analysis_all.copy()
    n_f5_sentinel = int(unresolved_f5.sum())
    n_f5_resolved = int(resolved_f5.sum())

    if not analysis_primary.index.equals(analysis_all.index):
        raise RuntimeError("The primary and sensitivity sample indices differ.")

    log_detail(
        f"Both models use "
        f"{len(analysis_primary)} segments. "
        f"Resolved F5 paths: {n_f5_resolved}. "
        f"Unresolved F5 paths: {n_f5_sentinel}."
    )


    sample_audit = pd.DataFrame(
        [
            {
                "dataset": "six_fingerprint_primary_all_valid_segments",
                "n": len(analysis_all),
                "features": "F1,F2,F3,F4,F5,F6",
            },
            {
                "dataset": "same_sample_five_fingerprint_sensitivity",
                "n": len(analysis_all),
                "features": "F1,F2,F3,F4,F6",
            },
        ]
    )
    sample_audit.to_csv(
        OUTPUT_DIR / "sample_audit.csv", index=False, encoding="utf-8-sig"
    )
    cleaned_model_data = analysis_primary[
        PRIMARY_FEATURES + [RESPONSE_COL]
    ].copy()
    cleaned_model_data.insert(
        0,
        "source_index",
        analysis_primary.index.to_numpy(),
    )
    cleaned_model_data.to_csv(
        OUTPUT_DIR / "cleaned_model_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    log_detail(
        f"Saved the primary model data: "
        f"{OUTPUT_DIR / 'cleaned_model_data.csv'}."
    )

    log_step(4, "Map the observed responses.")
    plot_observed_study_domain(
        analysis_all,
        OUTPUT_DIR / f"Fig_observed_study_domain_{OUTPUT_PREFIX}",
    )
    log_detail("Saved the observed response map.")


    # =============================================================================

    # =============================================================================

    X_primary = analysis_primary[PRIMARY_FEATURES]
    y_primary = analysis_primary[RESPONSE_COL]
    groups_primary, coords_primary = build_spatial_groups(analysis_primary)

    log_step(5, "Run the 80/20 random holdout for the F1-F6 model.")
    (
        random_holdout_predictions,
        random_holdout_assignment,
        random_holdout_permutation,
        random_holdout_summary,
    ) = nationwide_random_holdout_validation(
        X_primary,
        y_primary,
        analysis_name="six_fingerprint_nationwide_random_holdout",
    )
    random_holdout_predictions.to_csv(
        OUTPUT_DIR / "RF_random_holdout_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    random_holdout_assignment.to_csv(
        OUTPUT_DIR / "RF_random_holdout_split_assignment.csv",
        index=False,
        encoding="utf-8-sig",
    )
    random_holdout_permutation.to_csv(
        OUTPUT_DIR / "RF_random_holdout_permutation_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([random_holdout_summary]).to_csv(
        OUTPUT_DIR / "RF_random_holdout_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_random_holdout_performance(
        random_holdout_predictions,
        random_holdout_summary,
        OUTPUT_DIR / f"random_holdout_performance_{OUTPUT_PREFIX}",
    )
    log_detail(
        f"Random holdout complete. Test R2="
        f"{random_holdout_summary['r2']:.4f}, "
        f"RMSE={random_holdout_summary['rmse']:.4f}, "
        f"MAE={random_holdout_summary['mae']:.4f}."
    )

    log_step(6, "Run five-fold spatial cross-validation for the F1-F6 model.")
    cv_folds, cv_predictions, cv_permutation = spatial_cv(
        X_primary,
        y_primary,
        groups_primary,
        coords_primary,
        analysis_name="six_fingerprint_primary",
    )
    cv_folds.to_csv(
        OUTPUT_DIR / "RF_spatial_CV_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_predictions.to_csv(
        OUTPUT_DIR / "RF_spatial_CV_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_permutation.to_csv(
        OUTPUT_DIR / "RF_spatial_CV_permutation_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cv_pooled = pooled_metrics(
        cv_predictions["observed"].to_numpy(),
        cv_predictions["predicted"].to_numpy(),
    )
    cv_summary = {
        "analysis": "six_fingerprint_primary",
        "n": len(X_primary),
        "n_spatial_blocks": int(groups_primary.nunique()),
        "block_size_km": SPATIAL_BLOCK_SIZE_M / 1000,
        "fold_r2_mean": float(cv_folds["r2"].mean()),
        "fold_r2_sd": float(cv_folds["r2"].std(ddof=1)),
        "fold_r2_min": float(cv_folds["r2"].min()),
        "fold_r2_max": float(cv_folds["r2"].max()),
        **{f"pooled_{key}": value for key, value in cv_pooled.items()},
    }
    pd.DataFrame([cv_summary]).to_csv(
        OUTPUT_DIR / "RF_spatial_CV_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_spatial_cv_performance(
        cv_predictions,
        cv_pooled,
        OUTPUT_DIR / f"spatial_CV_performance_{OUTPUT_PREFIX}",
    )
    log_detail(
        f"Primary spatial CV complete. Pooled R2={cv_pooled['r2']:.4f}, "
        f"RMSE={cv_pooled['rmse']:.4f}, MAE={cv_pooled['mae']:.4f}."
    )

    log_step(7, "Fit the primary random forest and calculate SHAP values.")
    rf_primary = make_forest()
    rf_primary.fit(X_primary, y_primary)
    rf_importance_table = pd.DataFrame(
        {
            "feature": PRIMARY_FEATURES,
            "rf_impurity_importance": rf_primary.feature_importances_,
        }
    ).sort_values("rf_impurity_importance", ascending=False)
    rf_importance_table.to_csv(
        OUTPUT_DIR / "RF_feature_importance_primary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_rf_feature_importance(
        rf_primary,
        PRIMARY_FEATURES,
        OUTPUT_DIR / f"feature_importance_ranking_{OUTPUT_PREFIX}",
    )
    explainer = shap.TreeExplainer(rf_primary)
    shap_values = np.asarray(explainer.shap_values(X_primary))
    log_detail("Primary model fit and attribution complete.")

    log_step(8, "Bootstrap mean absolute SHAP importance.")
    shap_table = bootstrap_mean_abs_shap(shap_values, PRIMARY_FEATURES)
    shap_table.to_csv(
        OUTPUT_DIR / "RF_SHAP_importance_primary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    log_detail("Saved SHAP importance and bootstrap intervals.")


    log_step(9, "Calculate SHAP interactions and save attribution plots.")
    interaction_values = np.asarray(
        explainer.shap_interaction_values(X_primary)
    )
    interaction_matrix = np.abs(interaction_values).mean(axis=0)
    pd.DataFrame(
        interaction_matrix,
        index=PRIMARY_FEATURES,
        columns=PRIMARY_FEATURES,
    ).to_csv(
        OUTPUT_DIR / "RF_SHAP_interaction_matrix_primary.csv",
        encoding="utf-8-sig",
    )

    fig_interaction, ax_interaction = plt.subplots(
        figsize=(3.50, 3.07)
    )
    image = ax_interaction.imshow(
        interaction_matrix,
        cmap=SHAP_CMAP,
        aspect="equal",
    )
    ax_interaction.set_xticks(range(len(PRIMARY_FEATURES)))
    ax_interaction.set_yticks(range(len(PRIMARY_FEATURES)))
    ax_interaction.set_xticklabels(PRIMARY_FEATURES)
    ax_interaction.set_yticklabels(PRIMARY_FEATURES)
    ax_interaction.set_xlabel("Terrain metrics")
    ax_interaction.set_ylabel("Terrain metrics")
    colorbar = fig_interaction.colorbar(
        image, ax=ax_interaction, fraction=0.046, pad=0.04
    )
    colorbar.set_label("Mean |SHAP interaction value|")
    apply_full_frame(ax_interaction, "interaction")
    apply_colorbar_style(colorbar, "interaction")
    fig_interaction.tight_layout()
    save_figure(
        fig_interaction,
        OUTPUT_DIR / f"Fig_RF_SHAP_interactions_{OUTPUT_PREFIX}",
    )
    pd.concat(
        [
            X_primary.reset_index(drop=True),
            pd.DataFrame(
                shap_values, columns=[f"SHAP_{name}" for name in PRIMARY_FEATURES]
            ),
        ],
        axis=1,
    ).to_csv(
        OUTPUT_DIR / "RF_SHAP_values_primary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_shap_importance(
        shap_table,
        OUTPUT_DIR / f"Fig_RF_SHAP_importance_{OUTPUT_PREFIX}",
    )
    plot_f5_dependence(
        X_primary,
        shap_values,
        OUTPUT_DIR / f"shap_dependence_plot_distance_{OUTPUT_PREFIX}",
    )

    shap.summary_plot(
        shap_values,
        X_primary,
        show=False,
        cmap=plt.get_cmap(SHAP_CMAP),
        plot_size=(6.0, 4.5),
        axis_color="black",
    )
    summary_fig = plt.gcf()
    summary_fig.axes[0].set_xlabel("SHAP value (association with model output)")
    # summary_fig.axes[0].set_title(
    #     f"SHAP Summary Plot for {RESPONSE_DISPLAY_NAME}",
    #     fontweight="bold",
    # )
    apply_full_frame(summary_fig.axes[0], "shap_summary")
    for summary_axis in summary_fig.axes[1:]:
        apply_colorbar_style(summary_axis, "shap_summary")
    save_figure(
        summary_fig,
        OUTPUT_DIR / f"shap_summary_plot_{OUTPUT_PREFIX}",
    )
    log_detail("Saved the importance, F5 dependence, summary and interaction plots.")


    # =============================================================================

    # =============================================================================

    X_nationwide = analysis_all[NATIONWIDE_FEATURES]
    y_nationwide = analysis_all[RESPONSE_COL]
    groups_nationwide, coords_nationwide = build_spatial_groups(analysis_all)
    if not X_nationwide.index.equals(X_primary.index):
        raise RuntimeError("The primary and sensitivity sample order differs.")
    if not groups_nationwide.equals(groups_primary):
        raise RuntimeError("The primary and sensitivity spatial groups differ.")

    log_step(10, "Run spatial cross-validation for the five-feature sensitivity model.")
    (
        nationwide_folds,
        nationwide_predictions,
        nationwide_permutation,
    ) = spatial_cv(
        X_nationwide,
        y_nationwide,
        groups_nationwide,
        coords_nationwide,
        analysis_name="nationwide_five_fingerprint_sensitivity",
    )
    nationwide_folds.to_csv(
        OUTPUT_DIR / "RF_nationwide_sensitivity_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nationwide_predictions.to_csv(
        OUTPUT_DIR / "RF_nationwide_sensitivity_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nationwide_permutation.to_csv(
        OUTPUT_DIR / "RF_nationwide_sensitivity_permutation_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nationwide_pooled = pooled_metrics(
        nationwide_predictions["observed"].to_numpy(),
        nationwide_predictions["predicted"].to_numpy(),
    )
    nationwide_summary = {
        "analysis": "nationwide_five_fingerprint_sensitivity",
        "n": len(X_nationwide),
        "n_spatial_blocks": int(groups_nationwide.nunique()),
        "block_size_km": SPATIAL_BLOCK_SIZE_M / 1000,
        "fold_r2_mean": float(nationwide_folds["r2"].mean()),
        "fold_r2_sd": float(nationwide_folds["r2"].std(ddof=1)),
        "fold_r2_min": float(nationwide_folds["r2"].min()),
        "fold_r2_max": float(nationwide_folds["r2"].max()),
        **{
            f"pooled_{key}": value
            for key, value in nationwide_pooled.items()
        },
    }
    pd.DataFrame([nationwide_summary]).to_csv(
        OUTPUT_DIR / "RF_nationwide_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log_detail("Sensitivity CV complete. Fit the five-feature model.")
    rf_nationwide = make_forest()
    rf_nationwide.fit(X_nationwide, y_nationwide)
    nationwide_explainer = shap.TreeExplainer(rf_nationwide)
    nationwide_shap = np.asarray(
        nationwide_explainer.shap_values(X_nationwide)
    )
    nationwide_shap_table = bootstrap_mean_abs_shap(
        nationwide_shap, NATIONWIDE_FEATURES
    )
    nationwide_shap_table.to_csv(
        OUTPUT_DIR / "RF_SHAP_importance_nationwide_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    log_detail("Saved five-feature SHAP importance.")



    # =============================================================================

    # =============================================================================

    log_step(11, "Save validation summaries and the JSON run record.")


    validation_hierarchy = pd.DataFrame(
        [
            {
                "display_order": 1,
                "analysis": "six_fingerprint_nationwide_random_holdout",
                "role": "same_distribution_baseline",
                "features": ",".join(PRIMARY_FEATURES),
                "validation_scheme": "nationwide_random_80_20_holdout",
                "n_evaluated": random_holdout_summary["n_test"],
                "r2": random_holdout_summary["r2"],
                "rmse": random_holdout_summary["rmse"],
                "mae": random_holdout_summary["mae"],
                "bias_pred_minus_obs": random_holdout_summary[
                    "bias_pred_minus_obs"
                ],
                "correlation": random_holdout_summary["correlation"],
            },
            {
                "display_order": 2,
                "analysis": "six_fingerprint_primary",
                "role": "local_spatial_robustness",
                "features": ",".join(PRIMARY_FEATURES),
                "validation_scheme": "250_km_blocked_5_fold_cv",
                "n_evaluated": len(cv_predictions),
                "r2": cv_pooled["r2"],
                "rmse": cv_pooled["rmse"],
                "mae": cv_pooled["mae"],
                "bias_pred_minus_obs": cv_pooled["bias_pred_minus_obs"],
                "correlation": cv_pooled["correlation"],
            },
            {
                "display_order": 3,
                "analysis": "nationwide_five_fingerprint_sensitivity",
                "role": "feature_set_sensitivity",
                "features": ",".join(NATIONWIDE_FEATURES),
                "validation_scheme": "250_km_blocked_5_fold_cv",
                "n_evaluated": len(nationwide_predictions),
                "r2": nationwide_pooled["r2"],
                "rmse": nationwide_pooled["rmse"],
                "mae": nationwide_pooled["mae"],
                "bias_pred_minus_obs": nationwide_pooled[
                    "bias_pred_minus_obs"
                ],
                "correlation": nationwide_pooled["correlation"],
            },
        ]
    )
    validation_hierarchy.to_csv(
        OUTPUT_DIR / "RF_validation_hierarchy_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    top_shap = shap_table.iloc[0]
    top_nationwide_shap = nationwide_shap_table.iloc[0]
    off_diagonal_interactions = interaction_matrix.astype(float).copy()
    np.fill_diagonal(off_diagonal_interactions, np.nan)
    top_interaction_position = np.unravel_index(
        np.nanargmax(off_diagonal_interactions),
        off_diagonal_interactions.shape,
    )
    top_interaction_f1 = PRIMARY_FEATURES[top_interaction_position[0]]
    top_interaction_f2 = PRIMARY_FEATURES[top_interaction_position[1]]
    top_interaction_value = float(
        off_diagonal_interactions[top_interaction_position]
    )


    run_record = {
        "analysis_mode": ANALYSIS_MODE,
        "study_area": STUDY_AREA,
        "study_area_audit": STUDY_AREA_AUDIT,
        "legacy_nationwide_labels": "Refer to the selected study area; filenames retained for compatibility",
        "response": RESPONSE_COL,
        "response_label": RESPONSE_LABEL,
        "response_display_name": RESPONSE_DISPLAY_NAME,
        "frequency_aggregation": FREQUENCY_AGGREGATION,
        "frequency_divisor": FREQUENCY_DIVISOR,
        "output_prefix": OUTPUT_PREFIX,
        "figure_format": "PNG",
        "standard_figure_dpi": 300,
        "map_figure_dpi": 600,
        "bar_color": BAR_COLOR,
        "shap_colormap": SHAP_CMAP,
        "figure_font": FIGURE_FONT,
        "axes_frame": "four-sided",
        "tick_direction": "in",
        "tick_sides": ["left", "bottom"],
        "tick_length_pt": TICK_LENGTH_PT,
        "tick_width_pt": TICK_WIDTH_PT,
        "colorbar_tick_side": "right",
        "colorbar_tick_direction": "in",
        "colorbar_outline_width_pt": COLORBAR_OUTLINE_WIDTH_PT,
        "colorbar_tick_length_pt": COLORBAR_TICK_LENGTH_PT,
        "colorbar_tick_width_pt": COLORBAR_TICK_WIDTH_PT,
        "colorbar_tick_pad_pt": COLORBAR_TICK_PAD_PT,
        "colorbar_label_pad_pt": COLORBAR_LABEL_PAD_PT,
        "per_figure_font_settings": FIGURE_FONT_SETTINGS,
        "mountain_vector": str(SHP_PATH),
        "rainfall_input": str(TARGET_TIF_PATH or TARGET_CSV_PATH),
        "intensity_aggregation": INTENSITY_AGGREGATION,
        "frequency_missing_is_zero": FREQUENCY_MISSING_IS_ZERO,
        "forest_trees": N_TREES,
        "forest_max_depth": MAX_DEPTH,
        "forest_min_samples_leaf": MIN_SAMPLES_LEAF,
        "permutation_repeats": N_PERMUTATIONS,
        "shap_bootstrap_replicates": N_SHAP_BOOTSTRAPS,
        "packages": {name: version(name) for name in (
            "geopandas", "matplotlib", "numpy", "pandas", "rasterio", "shap",
            "shapely", "scikit-learn", "pyogrio", "pyproj",
        )},
        "national_boundary_vector": str(BOUNDARY_SHP_PATH) if BOUNDARY_SHP_PATH else None,
        "nine_dash_line_vector": str(NINE_DASH_SHP_PATH) if NINE_DASH_SHP_PATH else None,
        "Taiwan_boundary_vector": str(TAIWAN_SHP_PATH) if TAIWAN_SHP_PATH else None,
        "primary_features": PRIMARY_FEATURES,
        "primary_sample_n": len(analysis_primary),
        "nationwide_sensitivity_features": NATIONWIDE_FEATURES,
        "nationwide_sample_n": len(analysis_all),
        "f5_rule": (
            "All valid mountain segments in the selected study area entered the primary "
            "model. Nonfinite or nonpositive F5 values, indicating no resolved "
            "coast intersection, were encoded as a 6000000-m sentinel. The "
            "sentinel is a computational category and not a physical path length."
        ),
        "f5_sentinel_m": F5_SENTINEL_M,
        "f5_sentinel_n": n_f5_sentinel,
        "f5_resolved_n": n_f5_resolved,
        "maximum_resolved_f5_m": maximum_resolved_f5,
        "same_samples_and_spatial_groups_for_both_models": True,
        "cleaned_model_data_csv": str(
            OUTPUT_DIR / "cleaned_model_data.csv"
        ),
        "random_holdout_test_fraction": RANDOM_HOLDOUT_TEST_SIZE,
        "nationwide_random_holdout": random_holdout_summary,
        "spatial_block_size_km": SPATIAL_BLOCK_SIZE_M / 1000,
        "spatial_cv_folds": N_SPLITS,
        "primary_spatial_cv": cv_summary,
        "nationwide_sensitivity_spatial_cv": nationwide_summary,
        "validation_hierarchy_csv": str(
            OUTPUT_DIR / "RF_validation_hierarchy_summary.csv"
        ),
        "top_primary_shap_feature": str(top_shap["feature"]),
        "top_primary_mean_abs_shap": float(top_shap["mean_abs_shap"]),
        "top_exploratory_interaction": {
            "feature_1": top_interaction_f1,
            "feature_2": top_interaction_f2,
            "mean_abs_shap_interaction": top_interaction_value,
        },
        "top_nationwide_sensitivity_shap_feature": str(
            top_nationwide_shap["feature"]
        ),
        "random_seed": RANDOM_SEED,
        "interpretation_boundary": (
            "SHAP values quantify fitted multivariable associations and are not "
            "causal effects. In the full-sample model, F5 attribution combines "
            "variation among resolved path lengths with the unresolved-path "
            "sentinel category."
        ),
    }
    with (OUTPUT_DIR / "RF_SHAP_run_record.json").open("w", encoding="utf-8") as file:
        json.dump(
            json_safe(run_record),
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    log_step(12, "Analysis complete.")
    log_detail(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
