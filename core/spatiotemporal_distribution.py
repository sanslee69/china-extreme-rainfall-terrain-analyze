
import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import IdentityTransform
# from matplotlib.colors import TwoSlopeNorm, Normalize
import rioxarray
import pyproj
import shapely.ops as ops
from shapely.geometry import box
import logging
import config


def set_sci_plot_style():
    """
    Apply the configured Matplotlib style.
    """
    plt.rcParams.update({
        'font.family': config.FONT_FAMILY,
        'axes.titlesize': config.FONT_SIZE_TITLE,
        'axes.labelsize': config.FONT_SIZE_LABEL,
        'xtick.labelsize': config.FONT_SIZE_TICK,
        'ytick.labelsize': config.FONT_SIZE_TICK,
        'legend.fontsize': config.FONT_SIZE_TICK,
        'figure.dpi': config.PLOT_DPI,
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'svg.fonttype': 'none',
        'pdf.fonttype': 42
    })


def format_lon(x, pos):
    """Format longitude tick labels."""
    if x < 0:
        return f"{-x:g}°W"
    elif x > 0:
        return f"{x:g}°E"
    return "0°"


def format_lat(y, pos):
    """Format latitude tick labels."""
    if y < 0:
        return f"{-y:g}°S"
    elif y > 0:
        return f"{y:g}°N"
    return "0°"


def plot_mountain_with_halo(ax, mountain_gdf, color, linewidth):
    """Draw mountain lines with a light outline."""
    if mountain_gdf.empty:
        return

    collection_count = len(ax.collections)
    mountain_gdf.plot(
        ax=ax,
        color=color,
        linewidth=linewidth,
        alpha=config.MOUNTAIN_LINE_ALPHA,
        zorder=7
    )


    for collection in ax.collections[collection_count:]:
        collection.set_path_effects([
            path_effects.Stroke(
                linewidth=linewidth + config.MOUNTAIN_HALO_EXTRA_WIDTH,
                foreground=config.MOUNTAIN_HALO_COLOR,
                alpha=config.MOUNTAIN_HALO_ALPHA
            ),
            path_effects.Normal()
        ])


class MountainCurveLabel(Artist):
    """Place letters along the display-space line at render time."""

    def __init__(self, ax, lines, name):
        super().__init__()
        self._ax = ax
        self._lines = lines
        self._name = str(name)
        self.set_zorder(8)
        self.set_in_layout(False)

    def draw(self, renderer):
        if not self.get_visible():
            return
        paths = []
        for line in self._lines:
            xy = self._ax.transData.transform(np.asarray(line.coords)[:, :2])
            keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
            xy = xy[keep]
            if len(xy) > 1:
                s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
                paths.append((s[-1], xy, s))
        if not paths:
            return
        length, xy, s = max(paths, key=lambda item: item[0])


        direction = xy[-1] - xy[0]
        if direction[0] < 0 or (abs(direction[0]) < 1e-6 and direction[1] < 0):
            xy = xy[::-1]
            s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]

        fontsize = config.MOUNTAIN_LABEL_FONT_SIZE
        font = FontProperties(family=config.FONT_FAMILY, size=fontsize)
        spacing = config.MOUNTAIN_LABEL_LETTER_SPACING

        def advances():
            return np.array([
                renderer.get_text_width_height_descent(char, font, False)[0] * spacing
                for char in self._name
            ])

        widths = advances()
        total = widths.sum()
        if total <= 0:
            return
        if total > 0.9 * length:
            fontsize = min(fontsize, max(config.MOUNTAIN_LABEL_MIN_FONT_SIZE,
                                        fontsize * 0.9 * length / total))
            font.set_size(fontsize)
            widths = advances()
            total = widths.sum()
        centers = np.cumsum(widths) - widths / 2 + (length - total) / 2

        def at(distance):
            return np.array([np.interp(distance, s, xy[:, 0]),
                             np.interp(distance, s, xy[:, 1])])


        window = renderer.points_to_pixels(fontsize * 0.75)
        middle = at(length / 2)
        midpoint_tangent = at(min(length, length / 2 + window)) - at(max(0, length / 2 - window))
        if np.linalg.norm(midpoint_tangent) < 1e-9:
            midpoint_tangent = direction
        if np.linalg.norm(midpoint_tangent) < 1e-9:
            midpoint_tangent = np.array([1.0, 0.0])
        midpoint_tangent /= np.linalg.norm(midpoint_tangent)
        offset = renderer.points_to_pixels(config.MOUNTAIN_LABEL_OFFSET_PT)

        for char, distance in zip(self._name, centers):
            if total > length:

                tangent = midpoint_tangent
                anchor = middle + (distance - length / 2) * tangent
            else:
                anchor = at(distance)
                tangent = at(min(length, distance + window)) - at(max(0, distance - window))
                norm = np.linalg.norm(tangent)
                tangent = tangent / norm if norm > 1e-9 else midpoint_tangent
            normal = np.array([-tangent[1], tangent[0]])
            position = anchor + offset * normal
            if char.isspace():
                continue
            text = Text(
                x=position[0], y=position[1], text=char,
                fontproperties=font, color=config.MOUNTAIN_LABEL_COLOR,
                rotation=np.degrees(np.arctan2(tangent[1], tangent[0])),
                rotation_mode='anchor', ha='center', va='center',
                transform=IdentityTransform(), clip_on=True,
            )
            text.set_figure(self.figure)
            text.axes = self._ax
            text.set_clip_path(self._ax.patch)
            text.set_path_effects([
                path_effects.Stroke(
                    linewidth=config.MOUNTAIN_LABEL_HALO_WIDTH,
                    foreground=config.MOUNTAIN_LABEL_HALO_COLOR,
                    alpha=config.MOUNTAIN_LABEL_HALO_ALPHA,
                ),
                path_effects.Normal(),
            ])
            text.draw(renderer)
        self.stale = False


def label_mountain_names(ax, mountains, xlim, ylim):
    """Label each mountain name on its longest visible line."""
    if not getattr(config, "MOUNTAIN_LABEL_ENABLED", True):
        return

    field = getattr(config, "MOUNTAIN_LABEL_FIELD", "Ename")
    if field not in mountains.columns:
        raise KeyError(f"Mountain layer lacks the name field: {field}")

    mountains = mountains.copy()
    mountains["_label"] = mountains[field].astype("string").str.strip()
    mountains = mountains.loc[
        mountains["_label"].notna() & mountains["_label"].ne("")
    ]
    viewport = box(xlim[0], ylim[0], xlim[1], ylim[1])

    def line_parts(geometry):
        """Yield lines from multipart and clipped geometries."""
        if geometry is None or geometry.is_empty:
            return []
        if geometry.geom_type == "LineString":
            return [geometry]
        if hasattr(geometry, "geoms"):
            return [line for part in geometry.geoms for line in line_parts(part)]
        return []

    for name, group in mountains.groupby("_label", sort=True):
        visible_lines = []
        for geometry in group.geometry:
            if geometry is not None and not geometry.is_empty:
                visible_lines.extend(line_parts(geometry.intersection(viewport)))
        if not visible_lines:
            continue


        merged = ops.unary_union(visible_lines)
        if merged.geom_type == 'MultiLineString':
            merged = ops.linemerge(merged)
        ax.add_artist(MountainCurveLabel(ax, line_parts(merged), name))


def resolve_background_tif_path():
    """Resolve the required background raster."""
    candidates = [
        getattr(config, 'BACKGROUND_TIF_PATH', None),
        getattr(config, 'BACKGROUND_TIF_LOCAL_FALLBACK', None),
    ]
    checked = []
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(os.path.expandvars(candidate))
        if normalized in checked:
            continue
        checked.append(normalized)
        if os.path.isfile(normalized):
            return normalized

    checked_text = "\n  - ".join(checked) if checked else "No paths configured"
    raise FileNotFoundError(
        "Satellite background raster was not found.\n"
        f"Checked:\n  - {checked_text}\n"
        "Set RAINFALL_BACKGROUND_TIF "
        "or supply data/background.tif."
    )


def _normalize_image_band(data):
    """Normalize a band to the 0–1 display range."""
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("The clipped background contains only nodata.")

    max_value = float(np.nanmax(data))
    if max_value <= 1.0:
        scale = 1.0
    elif max_value <= 255.0:
        scale = 255.0
    elif max_value <= 65535.0:
        scale = 65535.0
    else:
        scale = max_value
    return np.clip(data / scale, 0.0, 1.0)


def _percentile_stretch(data, valid_mask):
    """Stretch bands with common percentile limits."""
    valid_values = data[:, valid_mask] if data.ndim == 3 else data[valid_mask]
    if valid_values.size == 0:
        raise ValueError("The background has no valid pixels for stretching.")

    low_percentile = getattr(config, 'BACKGROUND_PERCENTILE_LOW', 2.0)
    high_percentile = getattr(config, 'BACKGROUND_PERCENTILE_HIGH', 98.0)
    low_value, high_value = np.nanpercentile(
        valid_values, [low_percentile, high_percentile]
    )
    if not np.isfinite(low_value) or not np.isfinite(high_value):
        raise ValueError("Background percentile limits are invalid.")
    if high_value <= low_value:
        logging.warning("Skipping stretch because the background range is too narrow.")
        return np.clip(data, 0.0, 1.0)

    return np.clip((data - low_value) / (high_value - low_value), 0.0, 1.0)


def _enhance_rgb_background(rgb_bands, valid_mask):
    """Adjust RGB stretch, gamma, contrast and saturation."""
    rgb_normalized = _normalize_image_band(rgb_bands)
    rgb_stretched = _percentile_stretch(rgb_normalized, valid_mask)
    gamma = getattr(config, 'BACKGROUND_GAMMA', 0.95)
    contrast = getattr(config, 'BACKGROUND_CONTRAST', 1.15)
    saturation = getattr(config, 'BACKGROUND_SATURATION', 1.08)

    rgb = np.moveaxis(np.power(rgb_stretched, gamma), 0, -1)
    rgb = (rgb - 0.5) * contrast + 0.5


    luminance = (
        0.2126 * rgb[..., 0]
        + 0.7152 * rgb[..., 1]
        + 0.0722 * rgb[..., 2]
    )
    rgb = luminance[..., np.newaxis] + saturation * (
        rgb - luminance[..., np.newaxis]
    )
    return np.clip(rgb, 0.0, 1.0)


def _enhance_gray_background(gray_band, valid_mask):
    """Apply the corresponding enhancement to a single band."""
    gray = _normalize_image_band(gray_band)
    gray = _percentile_stretch(gray, valid_mask)
    gamma = getattr(config, 'BACKGROUND_GAMMA', 0.95)
    contrast = getattr(config, 'BACKGROUND_CONTRAST', 1.15)
    gray = np.power(gray, gamma)
    return np.clip((gray - 0.5) * contrast + 0.5, 0.0, 1.0)


def plot_satellite_background(ax, xlim, ylim):
    """Clip, reproject and render the background raster."""
    bg_path = resolve_background_tif_path()
    logging.info("Loading the background raster: %s", bg_path)

    bg_da = None
    try:

        bg_da = rioxarray.open_rasterio(bg_path, masked=True)
        tif_crs = bg_da.rio.crs
        if tif_crs is None:
            raise ValueError("Background GeoTIFF has no CRS.")


        bg_xlim = [xlim[0] - 1.0, xlim[1] + 1.0]
        bg_ylim = [ylim[0] - 1.0, ylim[1] + 1.0]
        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", tif_crs, always_xy=True
        )
        extent_tif = ops.transform(
            transformer.transform,
            box(bg_xlim[0], bg_ylim[0], bg_xlim[1], bg_ylim[1])
        )
        minx, miny, maxx, maxy = extent_tif.bounds

        bg_clipped = bg_da.rio.clip_box(
            minx=minx, miny=miny, maxx=maxx, maxy=maxy
        )
        bg_4326 = bg_clipped.rio.reproject("EPSG:4326")
        raster = np.asarray(bg_4326.values, dtype=np.float32)
        if raster.ndim != 3 or raster.shape[0] == 0:
            raise ValueError(f"Unexpected background array shape: {raster.shape}")

        x_values = np.asarray(bg_4326['x'].values, dtype=float)
        y_values = np.asarray(bg_4326['y'].values, dtype=float)
        extent = [
            float(np.nanmin(x_values)), float(np.nanmax(x_values)),
            float(np.nanmin(y_values)), float(np.nanmax(y_values))
        ]
        origin = 'upper' if y_values[0] > y_values[-1] else 'lower'
        alpha = getattr(config, 'BACKGROUND_ALPHA', 0.88)
        interpolation = getattr(config, 'BACKGROUND_INTERPOLATION', 'none')

        if raster.shape[0] >= 3:
            rgb_bands = raster[:3]
            valid_mask = np.all(np.isfinite(rgb_bands), axis=0)
            rgb = _enhance_rgb_background(rgb_bands, valid_mask)
            rgb = np.nan_to_num(rgb, nan=1.0, posinf=1.0, neginf=0.0)

            if raster.shape[0] >= 4:
                source_alpha = _normalize_image_band(raster[3])
                alpha_band = np.where(valid_mask, source_alpha, 0.0)
            else:
                alpha_band = valid_mask.astype(float)
            rgba = np.dstack([rgb, alpha_band])
            ax.imshow(
                rgba, extent=extent, origin=origin,
                zorder=1, alpha=alpha, interpolation=interpolation
            )
        else:
            valid_mask = np.isfinite(raster[0])
            gray = _enhance_gray_background(raster[0], valid_mask)
            gray = np.ma.masked_invalid(gray)
            ax.imshow(
                gray, extent=extent, origin=origin, cmap='gray',
                zorder=1, alpha=alpha, interpolation=interpolation
            )

        logging.info(
            "Background rendered. bands=%d, shape=%s, "
            "alpha=%.2f, contrast=%.2f, saturation=%.2f, gamma=%.2f",
            raster.shape[0], raster.shape, alpha,
            getattr(config, 'BACKGROUND_CONTRAST', 1.15),
            getattr(config, 'BACKGROUND_SATURATION', 1.08),
            getattr(config, 'BACKGROUND_GAMMA', 0.95)
        )
    except Exception as exc:
        raise RuntimeError(f"Could not load or render the background: {bg_path}") from exc
    finally:
        if bg_da is not None:
            bg_da.close()


def thin_spatial_points(spatial_summary, cell_degrees=0.2):
    """Keep the intensity and frequency peaks per display cell.

    Break ties by the other metric, distance to cell centre and coordinates.
    Retain the original attributes of the selected points.
    """
    if not np.isfinite(cell_degrees) or cell_degrees <= 0:
        raise ValueError("Display thinning requires a finite positive cell size.")
    if spatial_summary.empty:
        return spatial_summary.copy()
    coords = spatial_summary[['Lon', 'Lat']].to_numpy(dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Display thinning requires finite coordinates.")

    cells = np.floor(np.round(coords / cell_degrees, 10)).astype(np.int64)
    centers = (cells + 0.5) * cell_degrees
    ranked = spatial_summary.copy()
    ranked['_cell_lon'] = cells[:, 0]
    ranked['_cell_lat'] = cells[:, 1]
    ranked['_distance'] = np.round(((coords - centers) ** 2).sum(axis=1), 12)
    attributes = ranked[['Max_Precipitation', 'Frequency']].to_numpy(dtype=float)
    if not np.isfinite(attributes).all():
        raise ValueError("Display thinning requires finite intensity and count values.")
    cell_keys = ['_cell_lon', '_cell_lat']
    tie_keys = ['_distance', 'Lon', 'Lat']
    rain_peaks = ranked.sort_values(
        cell_keys + ['Max_Precipitation', 'Frequency'] + tie_keys,
        ascending=[True, True, False, False, True, True, True], kind='mergesort'
    ).drop_duplicates(cell_keys)
    frequency_peaks = ranked.sort_values(
        cell_keys + ['Frequency', 'Max_Precipitation'] + tie_keys,
        ascending=[True, True, False, False, True, True, True], kind='mergesort'
    ).drop_duplicates(cell_keys)
    selected = pd.concat([rain_peaks, frequency_peaks], ignore_index=True)
    selected = selected.drop_duplicates(['Lon', 'Lat']).sort_values(
        ['Lon', 'Lat'], kind='mergesort'
    )
    return selected[spatial_summary.columns].reset_index(drop=True)


def read_reference_layer(path, layer_name, geometry_types):
    """Validate a reference layer and project it to map coordinates."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found for {layer_name}: {path}")
    layer = gpd.read_file(path)
    if layer.crs is None:
        raise ValueError(f"{layer_name} has no CRS: {path}")
    valid = layer.geometry.notna() & ~layer.geometry.is_empty
    layer = layer.loc[valid].copy()
    if layer.empty:
        raise ValueError(f"{layer_name} has no usable geometry: {path}")
    if not layer.geom_type.isin(geometry_types).all():
        raise ValueError(f"{layer_name} has unexpected geometry types: {layer.geom_type.unique().tolist()}")
    layer = layer.to_crs('EPSG:4326')
    logging.info('Loaded %s: %d features in EPSG:4326.', layer_name, len(layer))
    return layer


def plot_south_china_sea_inset(
        ax, boundaries, dash_lines, study_boundary, monsoons, provinces,
        mountain_groups, mountains_to_label, capitals, plot_summary,
        size_factor, main_scatter):
    """Draw an inset with the same plotting data and colour scale."""
    extent = tuple(config.SOUTH_CHINA_SEA_EXTENT)
    rect = tuple(config.SOUTH_CHINA_SEA_INSET_RECT)
    if len(extent) != 4 or not all(math.isfinite(v) for v in extent):
        raise ValueError('SOUTH_CHINA_SEA_EXTENT needs four finite values: west, east, south and north.')
    west, east, south, north = extent
    if not (-180 <= west < east <= 180 and -90 < south < north < 90):
        raise ValueError('Invalid inset extent.')
    if (len(rect) != 4 or not all(math.isfinite(v) for v in rect)
            or rect[0] < 0 or rect[1] < 0 or rect[2] <= 0 or rect[3] <= 0
            or rect[0] + rect[2] > 1 or rect[1] + rect[3] > 1):
        raise ValueError('Inset rectangle must fit inside the parent axes.')
    land = boundaries.cx[west:east, south:north]
    lines = dash_lines.cx[west:east, south:north]
    if land.empty or lines.empty:
        raise ValueError('The inset extent requires intersecting polygon and line features.')

    inset_ax = ax.inset_axes(rect, zorder=15)
    inset_ax.set_in_layout(False)
    inset_ax.set_facecolor(ax.get_facecolor())
    xlim, ylim = (west, east), (south, north)


    plot_satellite_background(inset_ax, xlim, ylim)
    if provinces is not None:
        plot_province_boundaries(inset_ax, provinces, xlim, ylim)


    land.dissolve().boundary.plot(
        ax=inset_ax, color='black', linewidth=1.2, zorder=5,
    )
    visible_study = study_boundary.cx[west:east, south:north]
    if not visible_study.empty:
        visible_study.plot(
            ax=inset_ax, facecolor='none', edgecolor='black', linewidth=1.2, zorder=5,
        )
    if monsoons is not None:
        plot_monsoon_region(inset_ax, monsoons, xlim, ylim)
    visible_rain = plot_summary.loc[
        plot_summary['Lon'].between(west, east)
        & plot_summary['Lat'].between(south, north)
    ]

    inset_ax.scatter(
        visible_rain['Lon'], visible_rain['Lat'],
        c=visible_rain['Max_Precipitation'],
        s=visible_rain['Frequency'] * size_factor,
        norm=main_scatter.norm, cmap=main_scatter.cmap,
        alpha=main_scatter.get_alpha(), edgecolors='white', linewidth=0.2, zorder=4,
    )
    for mountains, color, width in mountain_groups:
        visible_mountains = mountains.cx[west:east, south:north]
        plot_mountain_with_halo(inset_ax, visible_mountains, color, width)
    inset_ax.set_xlim(xlim)
    inset_ax.set_ylim(ylim)
    label_mountain_names(inset_ax, mountains_to_label, xlim, ylim)
    city_texts, city_positions = [], []
    if capitals is not None:
        city_texts, city_positions = plot_capital_cities(inset_ax, capitals, xlim, ylim)

    lines.plot(
        ax=inset_ax, color=config.SOUTH_CHINA_SEA_DASH_COLOR,
        linewidth=config.SOUTH_CHINA_SEA_DASH_WIDTH, linestyle='-', zorder=6,
    )
    inset_ax.set_xlim(west, east)
    inset_ax.set_ylim(south, north)

    inset_ax.set_aspect(1.0 / math.cos(math.radians((south + north) / 2)))
    inset_ax.set_anchor('SE')
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_xlabel('')
    inset_ax.set_ylabel('')
    inset_ax.grid(False)
    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#222222')
        spine.set_linewidth(config.SOUTH_CHINA_SEA_FRAME_WIDTH)
    if config.SOUTH_CHINA_SEA_TITLE:
        inset_ax.text(
            0.5, 0.98, config.SOUTH_CHINA_SEA_TITLE,
            transform=inset_ax.transAxes, ha='center', va='top',
            fontfamily=config.FONT_FAMILY, fontsize=config.SOUTH_CHINA_SEA_TITLE_SIZE,
            color='#222222', zorder=3,
            bbox=dict(facecolor='white',
                      edgecolor='none', alpha=0.85, pad=1),
        )
    logging.info('Created South China Sea inset for %s with %d reference lines.', extent, len(lines))
    adjust_city_labels(inset_ax, city_texts, city_positions, [])
    return inset_ax


def plot_monsoon_region(ax, monsoons, xlim, ylim):
    """Draw the monsoon region above the study outline and beneath the mountains."""
    visible = monsoons.cx[xlim[0]:xlim[1], ylim[0]:ylim[1]]
    if visible.empty:
        return


    visible.plot(
        ax=ax, facecolor=config.MONSOON_FILL_COLOR, edgecolor='none',
        linewidth=0, alpha=config.MONSOON_FILL_ALPHA, zorder=5.2,
    )
    visible.boundary.plot(
        ax=ax, color=config.MONSOON_EDGE_COLOR,
        linewidth=config.MONSOON_EDGE_WIDTH, linestyle='-',
        alpha=config.MONSOON_EDGE_ALPHA, zorder=5.5,
    )
    logging.info('Added %d monsoon polygons with fill opacity %.2f.',
                 len(visible), config.MONSOON_FILL_ALPHA)


def plot_province_boundaries(ax, provinces, xlim, ylim):
    """Draw provincial boundaries above the raster."""
    visible = provinces.cx[xlim[0]:xlim[1], ylim[0]:ylim[1]]
    if not visible.empty:
        visible.boundary.plot(
            ax=ax, color=config.PROVINCE_BOUNDARY_COLOR,
            linewidth=config.PROVINCE_BOUNDARY_WIDTH,
            alpha=config.PROVINCE_BOUNDARY_ALPHA, zorder=1.5,
        )


def plot_capital_cities(ax, capitals, xlim, ylim):
    """Draw capital symbols above the map layers."""
    city_texts = []
    city_positions = []
    points = capitals.explode(index_parts=False)
    visible = points.cx[xlim[0]:xlim[1], ylim[0]:ylim[1]]
    # label_field = config.CAPITAL_CITY_LABEL_FIELD
    # if config.CAPITAL_CITY_LABEL_ENABLED and label_field not in capitals.columns:

    label_field = config.CAPITAL_CITY_LABEL_FIELD


    if label_field not in capitals.columns:
        raise KeyError(f"Capital layer lacks the field: {label_field}")

    if not visible.empty:
        names = visible[label_field].astype("string").str.strip()
        is_beijing = names.str.casefold().eq("beijing").fillna(False)


        for mask, marker, size in [
            (~is_beijing, "o", config.PROVINCIAL_CITY_SIZE),
            (is_beijing, "*", config.CAPITAL_CITY_SIZE),
        ]:
            cities = visible.loc[mask]
            if cities.empty:
                continue

            ax.scatter(
                cities.geometry.x,
                cities.geometry.y,
                marker=marker,
                s=size,
                facecolors="none",
                edgecolors=config.CAPITAL_CITY_EDGE_COLOR,
                linewidths=config.CAPITAL_CITY_EDGE_WIDTH,
                zorder=10,
                clip_on=True,
            )

    # if not visible.empty:
    #     ax.scatter(
    #         visible.geometry.x, visible.geometry.y, marker='*',
    #         s=config.CAPITAL_CITY_SIZE, facecolors=config.CAPITAL_CITY_COLOR,
    #         edgecolors=config.CAPITAL_CITY_EDGE_COLOR,
    #         linewidths=config.CAPITAL_CITY_EDGE_WIDTH, zorder=10,
    #         clip_on=True,
    #     )
        if config.CAPITAL_CITY_LABEL_ENABLED:
            for name, point in zip(visible[label_field], visible.geometry):
                if pd.isna(name) or not str(name).strip():
                    continue

                label = ax.text(
                    point.x, point.y, str(name).strip(),
                    fontsize=config.CAPITAL_CITY_LABEL_FONT_SIZE,
                    fontfamily=config.FONT_FAMILY,
                    color=config.CAPITAL_CITY_LABEL_COLOR,
                    ha='left', va='bottom', zorder=11, clip_on=True,
                )
                label.set_path_effects([
                    path_effects.Stroke(
                        linewidth=config.CAPITAL_CITY_LABEL_HALO_WIDTH,
                        foreground=config.CAPITAL_CITY_LABEL_HALO_COLOR,
                    ),
                    path_effects.Normal(),
                ])
                city_texts.append(label)
                city_positions.append((point.x, point.y))

    logging.info('Drew %d capital symbols.', len(visible))
    return city_texts, city_positions


def adjust_city_labels(ax, texts, positions, legends):
    """Adjust capital labels to avoid symbols and legends."""
    if not texts:
        return
    try:
        from adjustText import adjust_text
    except ImportError as exc:
        raise ImportError(
            'Capital label adjustment requires adjustText>=1.3.0. Install it with '
            'python -m pip install "adjustText>=1.3.0"'
        ) from exc
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()

    obstacles = [
        legend.get_window_extent(renderer).transformed(ax.transData.inverted())
        for legend in legends if legend is not None and legend.get_visible()
    ]
    city_x, city_y = zip(*positions)
    adjust_text(
        texts, x=city_x, y=city_y, target_x=city_x, target_y=city_y,
        objects=obstacles or None, ax=ax,
        expand=(1.15, 1.25), force_text=(0.2, 0.3),
        force_static=(0.2, 0.3), force_pull=(0.01, 0.01),
        ensure_inside_axes=True, expand_axes=False,
        iter_lim=300, min_arrow_len=5,
        arrowprops=dict(
            arrowstyle='-', color=config.CAPITAL_CITY_LABEL_COLOR,
            linewidth=0.5, alpha=0.7, zorder=10.5,
        ),
    )
    logging.info('Adjusted %d capital labels.', len(texts))


def resolve_spatial_colorbar(values):
    """Derive the shared colour scale from all event-grid maxima."""
    samples = [float(value) for value in values]
    if not samples or not all(math.isfinite(value) for value in samples):
        raise ValueError('The colour scale requires nonempty finite rainfall data.')
    data_min, data_max = min(samples), max(samples)
    mode = str(config.SPATIAL_COLORBAR_MODE).strip().lower()
    if mode == 'fixed':
        vmin = float(config.SPATIAL_COLORBAR_FIXED_MIN)
        vmax = float(config.SPATIAL_COLORBAR_FIXED_MAX)
        if not (math.isfinite(vmin) and math.isfinite(vmax) and vmin < vmax):
            raise ValueError('Fixed colour limits must be finite and increasing.')
        ticks = [float(value) for value in config.SPATIAL_COLORBAR_FIXED_TICKS]
        if (not ticks or not all(math.isfinite(t) and vmin <= t <= vmax for t in ticks)
                or any(a >= b for a, b in zip(ticks, ticks[1:]))):
            raise ValueError('Fixed ticks must increase within the colour limits.')
    elif mode == 'auto':
        low, high = data_min, data_max
        if low == high:

            pad = max(abs(low) * 0.05, 1.0)
            low, high = max(0.0, low - pad), high + pad
        raw_step = (high - low) / 6.0
        magnitude = 10.0 ** math.floor(math.log10(raw_step))
        step = next(factor * magnitude for factor in (1, 2, 2.5, 5, 10)
                    if factor * magnitude >= raw_step)
        first = math.floor(low / step)
        last = math.ceil(high / step)
        ticks = [float(f'{index * step:.12g}') for index in range(first, last + 1)]
        vmin, vmax = ticks[0], ticks[-1]
    else:
        raise ValueError('SPATIAL_COLORBAR_MODE must be "fixed" or "auto".')
    under, over = data_min < vmin, data_max > vmax
    extend = ('both' if under and over else 'min' if under else
              'max' if over else 'neither')
    logging.info('Colour scale %s. Data range %.3f–%.3f mm. Display range %.3f–%.3f mm.',
                 mode, data_min, data_max, vmin, vmax)
    return vmin, vmax, ticks, extend


def plot_spatial_extremes():
    """
    Aggregate event-grid maxima and counts, then draw the mountain overlay map.
    """
    if not os.path.exists(config.EXTREMES_CSV_OUT):
        raise FileNotFoundError(f"Event CSV not found: {config.EXTREMES_CSV_OUT}")
    if not os.path.exists(config.SHP_FILE_PATH):
        raise FileNotFoundError(f"Study boundary not found: {config.SHP_FILE_PATH}")

    logging.info("Aggregating extreme-rainfall grid-days.")
    df = pd.read_csv(config.EXTREMES_CSV_OUT)

    if df.empty:
        logging.warning("No events available for the spatial map.")
        return


    spatial_summary = df.groupby(['Lon', 'Lat']).agg(
        Max_Precipitation=('Precipitation_mm', 'max'),
        Frequency=('Precipitation_mm', 'count')
    ).reset_index()


    logging.info("Loading the study boundary.")
    gdf = gpd.read_file(config.SHP_FILE_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")


    if not os.path.exists(config.MOUNTAIN_SHP_PATH):
        raise FileNotFoundError(f"Mountain vector not found: {config.MOUNTAIN_SHP_PATH}")

    logging.info("Loading and reprojecting mountain lines.")
    gdf_mountains = gpd.read_file(config.MOUNTAIN_SHP_PATH)
    if gdf_mountains.crs is None:
        raise ValueError("The mountain layer has no CRS.")
    if 'Rank' not in gdf_mountains.columns:
        raise KeyError(
            "The mountain layer lacks Rank. Available fields: "
            f"{list(gdf_mountains.columns)}"
        )
    gdf_mountains = gdf_mountains.to_crs("EPSG:4326")
    gdf_mountains['Rank'] = pd.to_numeric(gdf_mountains['Rank'], errors='coerce')

    invalid_rank_count = int(gdf_mountains['Rank'].isna().sum())
    if invalid_rank_count:
        logging.warning("Skipping %d mountains with invalid Rank values.", invalid_rank_count)

    primary_mountains = gdf_mountains[gdf_mountains['Rank'].isin([0, 1])]
    secondary_mountains = gdf_mountains[gdf_mountains['Rank'] == 2]
    tertiary_mountains = gdf_mountains[gdf_mountains['Rank'] == 3]

    provinces = None
    capitals = None
    inset_boundaries = None
    inset_dash_lines = None
    monsoons = None
    if config.MONSOON_ENABLED:
        monsoons = read_reference_layer(
            config.MONSOON_SHP_PATH, 'monsoon region', ['Polygon', 'MultiPolygon']
        )
    if config.PROVINCE_BOUNDARY_ENABLED:
        provinces = read_reference_layer(
            config.PROVINCE_SHP_PATH, 'provincial boundaries', ['Polygon', 'MultiPolygon']
        )
    if config.SOUTH_CHINA_SEA_INSET_ENABLED:
        if (provinces is not None and
                os.path.normcase(os.path.abspath(config.SOUTH_CHINA_SEA_BOUNDARY_SHP_PATH)) ==
                os.path.normcase(os.path.abspath(config.PROVINCE_SHP_PATH))):
            inset_boundaries = provinces
        else:
            inset_boundaries = read_reference_layer(
                config.SOUTH_CHINA_SEA_BOUNDARY_SHP_PATH,
                'inset boundaries', ['Polygon', 'MultiPolygon']
            )
        inset_dash_lines = read_reference_layer(
            config.SOUTH_CHINA_SEA_DASH_SHP_PATH,
            'inset reference lines', ['LineString', 'MultiLineString']
        )
    if config.CAPITAL_CITY_ENABLED:
        capitals = read_reference_layer(
            config.CAPITAL_CITY_SHP_PATH, 'provincial capitals', ['Point', 'MultiPoint']
        )


    set_sci_plot_style()

    # ==========================================

    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 9))


    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    buffer_x = (bounds[2] - bounds[0]) * 0.05
    buffer_y = (bounds[3] - bounds[1]) * 0.05
    xlim = [bounds[0] - buffer_x, bounds[2] + buffer_x]
    ylim = [bounds[1] - buffer_y, bounds[3] + buffer_y]


    plot_satellite_background(ax, xlim, ylim)

    if monsoons is not None:
        plot_monsoon_region(ax, monsoons, xlim, ylim)

    if provinces is not None:
        plot_province_boundaries(ax, provinces, xlim, ylim)


    gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.2, zorder=5)


    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # ==========================================

    # ==========================================
    if getattr(config, 'SPATIAL_THIN_ENABLED', True):
        cell_degrees = getattr(config, 'SPATIAL_THIN_CELL_DEGREES', 0.2)
        plot_summary = thin_spatial_points(spatial_summary, cell_degrees)
        logging.info(
            "Display thinning on %.2f° by %.2f° cells: %d to %d points. "
            "Kept intensity and count peaks for display. The CSV contains all event grids.",
            cell_degrees, cell_degrees, len(spatial_summary), len(plot_summary)
        )
    else:
        plot_summary = spatial_summary.copy()
        logging.info("Plotting all %d event grids.", len(plot_summary))
    size_factor = 12


    # rain_values = spatial_summary['Max_Precipitation'].to_numpy(dtype=float)
    # rain_values = rain_values[np.isfinite(rain_values)]
    # if rain_values.size == 0:

    # color_min = float(rain_values.min())
    # color_max = float(rain_values.max())

    # if color_min < color_center < color_max:
    #     color_norm = TwoSlopeNorm(
    #         vmin=color_min, vcenter=color_center, vmax=color_max
    #     )
    #     logging.info(

    #         color_min, color_max, color_center
    #     )
    # else:
    #     color_norm = Normalize(vmin=color_min, vmax=color_max)


    # scatter = ax.scatter(
    #     plot_summary['Lon'], plot_summary['Lat'],
    #     c=plot_summary['Max_Precipitation'],
    #     s=plot_summary['Frequency'] * size_factor,
    #     norm=color_norm,
    #     cmap=config.COLOR_CMAP,
    #     alpha=0.9,
    #     edgecolors='white',
    #     linewidth=0.2,
    #     zorder=4
    # )

    color_min, color_max, color_ticks, colorbar_extend = resolve_spatial_colorbar(
        spatial_summary['Max_Precipitation']
    )
    scatter = ax.scatter(
        plot_summary['Lon'],
        plot_summary['Lat'],
        c=plot_summary['Max_Precipitation'],
        s=plot_summary['Frequency'] * size_factor,
        vmin=color_min,
        vmax=color_max,
        cmap=config.COLOR_CMAP,
        alpha=0.8,
        edgecolors='white',
        linewidth=0.2,
        zorder=5.3   #4
    )

    # ==========================================

    # ==========================================


    plot_mountain_with_halo(
        ax, primary_mountains,
        config.MOUNTAIN_PRIMARY_COLOR, config.MOUNTAIN_PRIMARY_WIDTH
    )
    plot_mountain_with_halo(
        ax, secondary_mountains,
        config.MOUNTAIN_SECONDARY_COLOR, config.MOUNTAIN_SECONDARY_WIDTH
    )
    plot_mountain_with_halo(
        ax, tertiary_mountains,
        config.MOUNTAIN_TERTIARY_COLOR, config.MOUNTAIN_TERTIARY_WIDTH
    )


    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


    label_mask = gdf_mountains['Rank'].isin(config.MOUNTAIN_LABEL_RANKS)
    min_length = config.MOUNTAIN_LABEL_MIN_LENGTH
    if config.MOUNTAIN_LABEL_ENABLED and min_length is not None:
        length_field = config.MOUNTAIN_LABEL_LENGTH_FIELD
        if length_field not in gdf_mountains.columns:
            raise KeyError(f"Mountain layer lacks the length field: {length_field}")
        if not np.isfinite(min_length):
            raise ValueError("MOUNTAIN_LABEL_MIN_LENGTH must be finite or None.")
        lengths = pd.to_numeric(gdf_mountains[length_field], errors='coerce')
        label_mask &= np.isfinite(lengths) & (lengths > min_length)
    mountains_to_label = gdf_mountains.loc[label_mask]
    label_mountain_names(ax, mountains_to_label, xlim, ylim)

    city_texts = []
    city_positions = []
    if capitals is not None:
        city_texts, city_positions = plot_capital_cities(ax, capitals, xlim, ylim)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # ==========================================

    # ==========================================
    ax.grid(True, linestyle='--', linewidth=0.5, color='white', alpha=0.6, zorder=2)


    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.tick_params(axis='both', labelsize=config.FONT_SIZE_TICK)


    ax.set_xlabel("")
    ax.set_ylabel("")


    min_freq = spatial_summary['Frequency'].min()
    max_freq = spatial_summary['Frequency'].max()
    mid_freq = int((min_freq + max_freq) / 2)
    legend_freqs = sorted(list(set([min_freq, mid_freq, max_freq])))

    handles = []
    for freq in legend_freqs:
        h = plt.scatter([], [], s=freq * size_factor, c='none', edgecolors='black', label=f'{freq}')   #label=f'{freq} days'
        handles.append(h)

    # ==========================================================

    # ==========================================================
    frequency_legend = ax.legend(
        handles=handles, title="Days", loc='upper left', frameon=True,     #Extreme-rainfall days
        title_fontsize=config.FONT_SIZE_TICK, fontsize=config.FONT_SIZE_TICK,
        labelspacing=1.1,
        borderpad=0.5,
        handletextpad=0.5
    )

    frequency_legend.set_zorder(20)

    ax.add_artist(frequency_legend)

    mountain_legends = [
        Line2D(
            [0], [0], color=config.MOUNTAIN_PRIMARY_COLOR,
            linewidth=config.MOUNTAIN_PRIMARY_WIDTH,
            label='Primary Mountain Ranges'
        ),
        Line2D(
            [0], [0], color=config.MOUNTAIN_SECONDARY_COLOR,
            linewidth=config.MOUNTAIN_SECONDARY_WIDTH,
            label='Secondary Mountain Ranges'
        ),
        Line2D(
            [0], [0], color=config.MOUNTAIN_TERTIARY_COLOR,
            linewidth=config.MOUNTAIN_TERTIARY_WIDTH,
            label='Tertiary Mountain Ranges'
        )
    ]
    for legend_line in mountain_legends:
        legend_line.set_path_effects([
            path_effects.Stroke(
                linewidth=legend_line.get_linewidth() + config.MOUNTAIN_HALO_EXTRA_WIDTH,
                foreground=config.MOUNTAIN_HALO_COLOR,
                alpha=config.MOUNTAIN_HALO_ALPHA
            ),
            path_effects.Normal()
        ])
    # if capitals is not None:
    #     mountain_legends.append(Line2D(
    #         [0], [0], linestyle='none', marker='*',
    #         markersize=np.sqrt(config.CAPITAL_CITY_SIZE),
    #         markerfacecolor=config.CAPITAL_CITY_COLOR,
    #         markeredgecolor=config.CAPITAL_CITY_EDGE_COLOR,
    #         markeredgewidth=config.CAPITAL_CITY_EDGE_WIDTH,
    #         label='Provincial capitals',
    #     ))
    if capitals is not None:
        for marker, size, label in [
            ("*", config.CAPITAL_CITY_SIZE, "National capital"),
            ("o", config.PROVINCIAL_CITY_SIZE, "Other provincial capitals"),
        ]:
            mountain_legends.append(Line2D(
                [0], [0],
                linestyle="none",
                marker=marker,
                markersize=np.sqrt(size),
                markerfacecolor="none",
                markeredgecolor=config.CAPITAL_CITY_EDGE_COLOR,
                markeredgewidth=config.CAPITAL_CITY_EDGE_WIDTH,
                label=label,
            ))

    if monsoons is not None:
        mountain_legends.append(Line2D(
            [0], [0], color=config.MONSOON_EDGE_COLOR,
            linewidth=config.MONSOON_EDGE_WIDTH, linestyle='-',
            alpha=config.MONSOON_EDGE_ALPHA, label='Monsoon-influenced region',
        ))
    mountain_legend = ax.legend(
        handles=mountain_legends, loc='lower left', frameon=True,         #lower right  left
        facecolor='white', framealpha=0.9, edgecolor='#cccccc', fontsize=16
    )
    mountain_legend.set_zorder(20)


    # plt.title("Spatial Distribution of Extreme Rainfall Events", fontweight='bold', pad=20)
    ax.set_title("")
    plt.tight_layout()

    # ==========================================================

    # ==========================================================


    fig.canvas.draw()
    map_box = ax.get_position()
    figure_width_inches = fig.get_figwidth()
    colorbar_pad = 0.18 / figure_width_inches
    colorbar_width = 0.22 / figure_width_inches
    colorbar_ax = fig.add_axes([
        map_box.x1 + colorbar_pad,
        map_box.y0,
        colorbar_width,
        map_box.height
    ])

    cbar = fig.colorbar(
        scatter, cax=colorbar_ax, orientation='vertical',
        ticks=color_ticks,
        extend=colorbar_extend,
        extendrect=True
    )
    cbar.set_label(
        'Max Extreme Daily Precipitation (mm)',
        fontweight='bold', size=config.FONT_SIZE_LABEL
    )
    cbar.ax.tick_params(labelsize=config.FONT_SIZE_TICK)


    label_obstacles = [frequency_legend, mountain_legend]
    if config.SOUTH_CHINA_SEA_INSET_ENABLED:
        inset_ax = plot_south_china_sea_inset(
            ax, inset_boundaries, inset_dash_lines, gdf, monsoons, provinces,
            [
                (primary_mountains, config.MOUNTAIN_PRIMARY_COLOR, config.MOUNTAIN_PRIMARY_WIDTH),
                (secondary_mountains, config.MOUNTAIN_SECONDARY_COLOR, config.MOUNTAIN_SECONDARY_WIDTH),
                (tertiary_mountains, config.MOUNTAIN_TERTIARY_COLOR, config.MOUNTAIN_TERTIARY_WIDTH),
            ],
            mountains_to_label, capitals, plot_summary, size_factor, scatter,
        )

        label_obstacles.append(inset_ax)


    adjust_city_labels(
        ax, city_texts, city_positions, label_obstacles
    )


    fig.savefig(config.FIG_SPATIAL_MAP, dpi=config.PLOT_DPI, bbox_inches='tight')
    svg_out = getattr(
        config, 'FIG_SPATIAL_MAP_SVG',
        os.path.join(config.OUTPUT_DIR, 'spatial_extremes_map.svg')
    )
    pdf_out = getattr(
        config, 'FIG_SPATIAL_MAP_PDF',
        os.path.join(config.OUTPUT_DIR, 'spatial_extremes_map.pdf')
    )
    tiff_out = getattr(
        config, 'FIG_SPATIAL_MAP_TIFF',
        os.path.join(config.OUTPUT_DIR, 'spatial_extremes_map.tiff')
    )
    fig.savefig(svg_out, bbox_inches='tight')
    fig.savefig(pdf_out, bbox_inches='tight')
    fig.savefig(tiff_out, dpi=600, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved the spatial map to {config.FIG_SPATIAL_MAP}")

    # ==========================================

    # ==========================================
    geo_csv_out = config.SPATIAL_GEO_OUT.replace(".tif", ".csv")
    spatial_summary.to_csv(geo_csv_out, index=False)
    logging.info(f"Saved grid-day counts and maxima to {geo_csv_out}")
