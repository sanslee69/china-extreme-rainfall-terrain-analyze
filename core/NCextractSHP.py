import os
import glob
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray
import logging
import config


def extract_and_mask_nc(output_nc=None):
    """
    Read daily NetCDF files, clip to the study region and filter wet values.

    If output_nc is provided, also save the masked dataset.
    """

    # =========================

    # =========================

    search_pattern = os.path.join(config.NC_DATA_DIR, "*.nc")
    nc_files = glob.glob(search_pattern)

    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {config.NC_DATA_DIR}.")

    nc_files.sort()
    logging.info(f"Reading {len(nc_files)} NetCDF files.")

    # =========================

    # =========================
    logging.info(f"Reading the study boundary: {config.SHP_FILE_PATH}")
    gdf = gpd.read_file(config.SHP_FILE_PATH)
    shp_crs = gdf.crs

    if shp_crs is None:
        logging.warning("Study boundary has no declared CRS. Assuming EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")
        shp_crs = gdf.crs

    # =========================

    # =========================
    block_days = getattr(config, 'PROCESS_TIME_CHUNK_DAYS', 30)
    if isinstance(block_days, bool) or not isinstance(block_days, int) or block_days <= 0:
        raise ValueError('PROCESS_TIME_CHUNK_DAYS must be a positive integer.')
    ds = xr.open_mfdataset(
        nc_files, combine='by_coords',
        chunks={'time': block_days, 'lat': 128, 'lon': 128}, engine='netcdf4'
    )

    if config.VAR_NAME not in ds.variables:
        raise KeyError(f"Variable '{config.VAR_NAME}' was not found. Available variables: {list(ds.variables.keys())}")


    if float(ds["lat"][0]) < float(ds["lat"][-1]):
        ds = ds.sortby("lat", ascending=False)

    ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    ds = ds.rio.write_crs("EPSG:4326")

    # =========================

    # =========================
    if ds.rio.crs != shp_crs:
        logging.info(f"Reprojecting the boundary from {shp_crs} to EPSG:4326.")
        gdf = gdf.to_crs(ds.rio.crs)

    # =========================

    # =========================
    logging.info("Clipping precipitation to the study boundary.")
    clipped_ds = ds.rio.clip(gdf.geometry, gdf.crs, drop=True)

    # =========================

    # =========================
    logging.info(f"Retaining precipitation above {config.WET_DAY_THRESHOLD} mm.")
    prec_data = clipped_ds[config.VAR_NAME]

    missing_value = prec_data.attrs.get("_FillValue", np.nan)
    if not np.isnan(missing_value):
        prec_data = prec_data.where(prec_data != missing_value)


    valid_wet_data = prec_data.where(prec_data > config.WET_DAY_THRESHOLD)

    clipped_ds[config.VAR_NAME] = valid_wet_data

    # =========================

    # =========================
    if output_nc:
        logging.info(f"Saving the clipped dataset to {output_nc}")
        clipped_ds.to_netcdf(output_nc)
        logging.info("Clipping and filtering complete.")

    return clipped_ds
