import os
import tempfile
import logging
import numpy as np
import pandas as pd
import config


def calculate_r99p_and_extract_events(masked_ds):
    """Compute a pooled linear percentile and extract grid-days at or above it.

    Store valid samples in a temporary disk file to limit memory use.
    """
    block_days = getattr(config, 'PROCESS_TIME_CHUNK_DAYS', 30)
    if isinstance(block_days, bool) or not isinstance(block_days, int) or block_days <= 0:
        raise ValueError('PROCESS_TIME_CHUNK_DAYS must be a positive integer.')
    percentile = config.PERCENTILE
    if not np.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError('PERCENTILE must be between 0 and 100.')
    data = masked_ds[config.VAR_NAME]
    if set(data.dims) != {'time', 'lat', 'lon'}:
        raise ValueError(f'Expected precipitation dimensions time, lat and lon. Found {data.dims}')
    data = data.transpose('time', 'lat', 'lon')
    output_dir = os.path.dirname(os.path.abspath(config.EXTREMES_CSV_OUT))
    os.makedirs(output_dir, exist_ok=True)
    total_days = data.sizes['time']
    total_blocks = (total_days + block_days - 1) // block_days

    def blocks():
        for start in range(0, total_days, block_days):
            block = data.isel(time=slice(start, start + block_days))

            values = np.asarray(block.compute(scheduler='single-threaded').values)
            yield start, block, values

    logging.info('Computing domain percentile %s in %d-day chunks across %d chunks.',
                 percentile, block_days, total_blocks)

    with tempfile.TemporaryDirectory(prefix='rainfall_percentile_', dir=output_dir) as scratch:
        values_path = os.path.join(scratch, 'valid_values.bin')
        sample_count = 0
        value_dtype = np.dtype(data.dtype)
        with open(values_path, 'wb') as sink:
            for start, block, values in blocks():
                valid = values[~np.isnan(values)]
                valid.tofile(sink)
                sample_count += valid.size
                del valid, values, block
                number = start // block_days + 1
                if number == 1 or number % 20 == 0 or number == total_blocks:
                    logging.info('Threshold collection chunk %d/%d. Valid samples: %d.',
                                 number, total_blocks, sample_count)
        if not sample_count:
            raise ValueError('No valid precipitation samples. Check the boundary, wet threshold and NetCDF inputs.')
        logging.info('Computing the exact percentile from a %.2f GiB temporary file.',
                     os.path.getsize(values_path) / 1024 ** 3)
        samples = np.memmap(values_path, dtype=value_dtype, mode='r+', shape=(sample_count,))
        try:

            threshold = float(np.percentile(samples, percentile, axis=0, overwrite_input=True))
        finally:
            samples._mmap.close()
            del samples
    if not np.isfinite(threshold):
        raise ValueError('The extreme threshold is non-finite. Check input values.')
    logging.info('Domain percentile %s threshold: %.2f mm', percentile, threshold)

    logging.info('Extracting grid-days at or above the threshold.')
    frames = []
    event_count = 0
    for start, block, values in blocks():
        ti, yi, xi = np.nonzero(values >= threshold)
        if ti.size:
            frame = pd.DataFrame({
                'Date': block['time'].values[ti],
                'Lat': block['lat'].values[yi],
                'Lon': block['lon'].values[xi],
                'Precipitation_mm': values[ti, yi, xi],
            })
            if pd.api.types.is_datetime64_any_dtype(frame['Date']):
                frame['Date'] = frame['Date'].dt.strftime('%Y-%m-%d')
            frames.append(frame)
            event_count += len(frame)
        del values, block, ti, yi, xi
        number = start // block_days + 1
        if number == 1 or number % 20 == 0 or number == total_blocks:
            logging.info('Event extraction chunk %d/%d. Grid-days: %d.', number, total_blocks, event_count)
    columns = ['Date', 'Lat', 'Lon', 'Precipitation_mm']
    extremes_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    del frames
    extremes_df = extremes_df.sort_values(
        by=['Date', 'Precipitation_mm'], ascending=[True, False]
    ).reset_index(drop=True)

    fd, temp_csv = tempfile.mkstemp(prefix='extreme_events_', suffix='.csv', dir=output_dir)
    os.close(fd)
    try:
        extremes_df.to_csv(temp_csv, index=False, encoding='utf-8-sig')
        os.replace(temp_csv, config.EXTREMES_CSV_OUT)
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
    logging.info('Saved %d extreme-rainfall grid-days to %s.', len(extremes_df), config.EXTREMES_CSV_OUT)
    return threshold, extremes_df
