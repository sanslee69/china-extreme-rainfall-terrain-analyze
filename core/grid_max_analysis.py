import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import rioxarray
import logging
import warnings
import config


def set_sci_plot_style():
    """Apply the configured Matplotlib style."""
    plt.rcParams.update({
        'font.family': config.FONT_FAMILY,
        'axes.titlesize': config.FONT_SIZE_TITLE,
        'axes.labelsize': config.FONT_SIZE_LABEL,
        'xtick.labelsize': config.FONT_SIZE_TICK,
        'ytick.labelsize': config.FONT_SIZE_TICK,
        'legend.fontsize': config.FONT_SIZE_TICK,
        'figure.dpi': config.PLOT_DPI
    })


def format_lon(x, pos):
    if x < 0:
        return f"{-x:g}°W"
    elif x > 0:
        return f"{x:g}°E"
    return "0°"


def format_lat(y, pos):
    if y < 0:
        return f"{-y:g}°S"
    elif y > 0:
        return f"{y:g}°N"
    return "0°"


def calculate_and_plot_grid_max(masked_ds):
    """
    Export the maximum and its date per grid cell from the supplied dataset.
    """
    logging.info("Computing grid maxima and their dates.")

    prec_data = masked_ds[config.VAR_NAME]


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        max_vals = prec_data.max(dim='time', skipna=True).compute()
        max_dates = prec_data.idxmax(dim='time', skipna=True).compute()

    # ==========================================================

    # ==========================================================
        logging.info(f"Filtering grid maxima at or below {config.GRID_MAX_THRESHOLD} mm.")


        max_vals_filtered = max_vals.where(max_vals > config.GRID_MAX_THRESHOLD, other=np.nan)


        max_vals_filtered = max_vals_filtered.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        max_vals_filtered = max_vals_filtered.rio.write_crs("EPSG:4326")


        max_vals_filtered = max_vals_filtered.astype(np.float32)

        logging.info(f"Writing grid-maximum GeoTIFF to {config.GRID_MAX_TIF_OUT}")
        max_vals_filtered.rio.to_raster(config.GRID_MAX_TIF_OUT, nodata=-9999.0)


    df_max = max_vals.to_dataframe(name='Max_Precipitation_mm').reset_index()
    df_dates = max_dates.to_dataframe(name='Date').reset_index()


    df_grid_max = df_max.copy()
    df_grid_max['Date'] = df_dates['Date']

    # ==========================================================

    # ==========================================================
    logging.info(f"Retaining grid maxima above {config.GRID_MAX_THRESHOLD} mm.")


    df_grid_max = df_grid_max[df_grid_max['Max_Precipitation_mm'] > config.GRID_MAX_THRESHOLD]


    df_grid_max = df_grid_max.rename(columns={'lat': 'Lat', 'lon': 'Lon'})

    if not df_grid_max.empty:
        df_grid_max['Date'] = pd.to_datetime(df_grid_max['Date']).dt.strftime('%Y-%m-%d')

        df_grid_max = df_grid_max.sort_values(by='Max_Precipitation_mm', ascending=False)
    else:
        logging.warning(
            f"No grid cells exceed {config.GRID_MAX_THRESHOLD} mm.")


    output_dir = os.path.dirname(config.GRID_MAX_CSV_OUT)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_grid_max.to_csv(config.GRID_MAX_CSV_OUT, index=False, encoding='utf-8-sig')
    logging.info(f"Saved {len(df_grid_max)} grid maxima to {config.GRID_MAX_CSV_OUT}")

    # ==========================================

    # ==========================================
    logging.info("Plotting grid-maximum rainfall.")

    gdf = gpd.read_file(config.SHP_FILE_PATH)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")

    set_sci_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))


    gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=3)


    bounds = gdf.total_bounds
    buffer_x = (bounds[2] - bounds[0]) * 0.05
    buffer_y = (bounds[3] - bounds[1]) * 0.05
    ax.set_xlim([bounds[0] - buffer_x, bounds[2] + buffer_x])
    ax.set_ylim([bounds[1] - buffer_y, bounds[3] + buffer_y])


    scatter = ax.scatter(
        df_grid_max['Lon'], df_grid_max['Lat'],
        c=df_grid_max['Max_Precipitation_mm'],
        cmap=config.COLOR_CMAP,
        s=15,
        marker='s',
        alpha=0.9,
        edgecolors='none',
        zorder=2
    )


    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5, zorder=1)
    ax.xaxis.set_major_formatter(FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(format_lat))
    ax.tick_params(axis='both', labelsize=config.FONT_SIZE_TICK)


    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    cbar.set_label('Historical Maximum Daily Precipitation (mm)', fontweight='bold', size=config.FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=config.FONT_SIZE_TICK)

    plt.title("Spatial Distribution of Historical Maximum Daily Rainfall", fontweight='bold', pad=20)
    plt.tight_layout()

    fig.savefig(config.FIG_GRID_MAX_MAP, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved the grid-maximum map to {config.FIG_GRID_MAX_MAP}")
