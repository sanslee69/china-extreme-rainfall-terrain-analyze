"""Command-line entry point for the rainfall and mountain mapping workflow."""

import argparse
import logging
import os
from pathlib import Path

import config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--map-only", action="store_true",
                       help="Draw the spatial map from the event CSV in the output directory.")
    stage.add_argument("--grid-max", action="store_true",
                       help="Also export grid maxima from the clipped and wet-filtered dataset.")
    args = parser.parse_args(argv)

    output = Path(config.OUTPUT_DIR)
    inputs = {
        "RAINFALL_BOUNDARY": config.SHP_FILE_PATH,
        "RAINFALL_MOUNTAINS": config.MOUNTAIN_SHP_PATH,
        "RAINFALL_BACKGROUND_TIF": config.BACKGROUND_TIF_PATH,
    }
    for variable, value in (
        ("RAINFALL_MONSOON", config.MONSOON_SHP_PATH),
        ("RAINFALL_PROVINCES", config.PROVINCE_SHP_PATH),
        ("RAINFALL_CAPITALS", config.CAPITAL_CITY_SHP_PATH),
    ):
        if value:
            inputs[variable] = value
    if config.SOUTH_CHINA_SEA_INSET_ENABLED:
        inputs["RAINFALL_INSET_LINES"] = config.SOUTH_CHINA_SEA_DASH_SHP_PATH
        inputs["RAINFALL_INSET_BOUNDARY or RAINFALL_PROVINCES"] = config.SOUTH_CHINA_SEA_BOUNDARY_SHP_PATH
    for name, filename in inputs.items():
        if not Path(filename).is_file():
            parser.error(f"{name} requires an existing file. Current path: {filename or '(unset)'}")
    if args.map_only:
        if not Path(config.EXTREMES_CSV_OUT).is_file():
            parser.error(f"Event CSV not found: {config.EXTREMES_CSV_OUT}")
    else:
        if not any(Path(config.NC_DATA_DIR).glob("*.nc")):
            parser.error(f"RAINFALL_NC_DIR contains no .nc files: {config.NC_DATA_DIR}")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            parser.error("Set RAINFALL_OUTPUT_DIR to a new or empty directory for a full run.")

    os.environ.setdefault("MPLBACKEND", "Agg")
    from core import (
        calculate_and_plot_grid_max,
        calculate_r99p_and_extract_events,
        extract_and_mask_nc,
        plot_spatial_extremes,
        plot_temporal_distributions,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.map_only:
        plot_spatial_extremes()
        return

    output.mkdir(parents=True, exist_ok=True)
    logging.info("Step 1/4. Clip and filter daily precipitation.")
    with extract_and_mask_nc() as masked:
        logging.info("Step 2/4. Calculate the domain percentile and extract extreme grid-days.")
        threshold, events = calculate_r99p_and_extract_events(masked)
        logging.info("Threshold %.6f mm. Extracted %d grid-days.", threshold, len(events))
        del events
        logging.info("Step 3/4. Plot annual and monthly distributions.")
        plot_temporal_distributions()
        logging.info("Step 4/4. Plot extreme rainfall with mountain overlays.")
        plot_spatial_extremes()
        if args.grid_max:
            calculate_and_plot_grid_max(masked)
    logging.info("Results saved in %s.", output)


if __name__ == "__main__":
    main()
