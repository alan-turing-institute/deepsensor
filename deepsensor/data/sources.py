import tqdm

from deepsensor.plot import extent_str_to_tuple

import urllib.request
import multiprocessing
from functools import partial

from typing import Optional, List, Union, Tuple
import os
import time
import xarray as xr
import pandas as pd

from joblib import Memory


def get_ghcnd_station_data(
    var_IDs: Optional[List[str]] = None,
    extent: Union[Tuple[float, float, float, float], str] = "global",
    date_range: Optional[Tuple[str, str]] = None,
    subsample_frac: float = 1.0,
    num_processes: Optional[int] = None,
    verbose: bool = False,
    cache: bool = False,
    cache_dir: str = ".datacache",
) -> pd.DataFrame:  # pragma: no cover
    """Download Global Historical Climatology Network Daily (GHCND) station data from NOAA
    into a pandas DataFrame.
    Source: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily.

    .. note::
        Requires the `scotthosking/get-station-data` repository to be installed
        manually in your Python environment with:
        ``pip install git+https://github.com/scott-hosking/get-station-data.git``

    .. note::
        Example key variable IDs:
        - ``"TAVG"``: Average temperature (degrees Celsius)
        - ``"TMAX"``: Maximum temperature (degrees Celsius)
        - ``"TMIN"``: Minimum temperature (degrees Celsius)
        - ``"PRCP"``: Precipitation (mm)
        - ``"SNOW"``: Snowfall
        - ``"AWND"``: Average wind speed (m/s)
        - ``"AWDR"``: Average wind direction (degrees)

        The full list of variable IDs can be found here:
        https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt

    .. warning::
        If this function is updated, the cache will be invalidated and the data will need
        to be re-downloaded. To avoid this risk, set ``cache=False`` and save the data to disk
        manually.

    Args:
        var_IDs: list
            List of variable IDs to download. If None, all variables are downloaded.
            See the list of available variable IDs above.
        extent: tuple[float, float, float, float] | str
            Tuple of (lon_min, lon_max, lat_min, lat_max) or string of region name.
            Options are: "global", "north_america", "uk", "europe".
        date_range: tuple[str, str]
            Tuple of (start_date, end_date) in format "YYYY-MM-DD".
        subsample_frac: float
            Fraction of available stations to download (useful for reducing download size).
            Default is 1.0 (download all stations).
        num_processes: int, optional
            Number of CPUs to use for downloading station data in parallel. If not specified, will
            use 75% of all available CPUs.
        verbose: bool
            Whether to print status messages. Default is ``False``.
        cache: bool
            Whether to cache the station metadata and data locally. If ``True``, calling the
            function again with the same arguments will load the data from the cache instead
            of downloading it again. Default is ``False``.
        cache_dir: str
            Directory to store the cached data. Default is ``".datacache"``.

    Returns:
        :class:`pandas.DataFrame`
            Station data with indexes ``time``, ``lat``, ``lon``, ``station`` and columns
            ``var1``, ``var2``, etc.
    """
    try:
        from get_station_data import ghcnd
    except ImportError:
        raise ImportError(
            "Must manually pip-install get-station-data with: `pip install git+https://github.com/scott-hosking/get-station-data.git`"
        )
    if not cache:
        cache_dir = None
    memory = Memory(cache_dir, verbose=0)

    @memory.cache
    def _get_ghcnd_station_data_cached(
        var_IDs=None,
        extent: Union[Tuple[float, float, float, float], str] = "global",
        date_range=None,
        subsample_frac=1.0,
        verbose=False,
    ) -> pd.DataFrame:
        if verbose:
            print(
                f"Downloading GHCND station data from NOAA...",
                end=" ",
                flush=True,
            )
        tic = time.time()

        stn_md = ghcnd.get_stn_metadata(verbose=verbose, cache=False)  # Already caching

        if isinstance(extent, str):
            extent = extent_str_to_tuple(extent)
        else:
            extent = tuple([float(x) for x in extent])
        lon_min, lon_max, lat_min, lat_max = extent

        stn_md = stn_md[
            (lat_min <= stn_md.lat)
            & (stn_md.lat <= lat_max)
            & (lon_min <= stn_md.lon)
            & (stn_md.lon <= lon_max)
        ]
        stn_md = stn_md.sample(frac=subsample_frac, random_state=43)
        if date_range:
            # Filter out stations with no data in the date range
            start_year = int(date_range[0][:4])
            end_year = int(date_range[1][:4])
            stn_md = stn_md[stn_md.end_year >= start_year]
            stn_md = stn_md[stn_md.start_year <= end_year]
        station_df = ghcnd.get_data(
            stn_md,
            include_flags=False,
            date_range=date_range,
            element_types=var_IDs,
            num_processes=num_processes,
            verbose=verbose,
            cache=False,  # Already caching
        )
        station_df = station_df.rename({"date": "time"}, axis=1)
        station_df = station_df.pivot_table(
            index=["time", "lat", "lon", "station"], columns="element", values="value"
        )
        station_df = station_df.dropna(how="all")
        station_df.columns.name = ""
        if verbose:
            print(
                f"{station_df.memory_usage(deep=True).sum() / 1e6:.2f} MB downloaded in {time.time() - tic:.2f} s"
            )
        return station_df

    return _get_ghcnd_station_data_cached(
        var_IDs=var_IDs,
        extent=extent,
        date_range=date_range,
        subsample_frac=subsample_frac,
        verbose=verbose,
    )


def get_era5_reanalysis_data(
    var_IDs: Optional[List[str]] = None,
    extent: Union[Tuple[float, float, float, float], str] = "global",
    date_range: Optional[Tuple[str, str]] = None,
    freq: str = "D",
    num_processes: Optional[int] = 1,
    verbose: bool = False,
    cache: bool = False,
    cache_dir: str = ".datacache",
) -> xr.Dataset:  # pragma: no cover
    """Download ERA5 reanalysis data from Google Cloud Storage into an xarray Dataset.
    Source: https://cloud.google.com/storage/docs/public-datasets/era5.

    Supports parallelising downloads into monthly chunks across multiple CPUs.
    Supports caching the downloaded data locally to avoid re-downloading when calling
    the function again with the same arguments.
    The data is cached on a per-month basis, so if you call the function again with
    a different date range, data will only be downloaded if the new date range includes
    months that have not already been downloaded.

    .. note::
        See the list of available variable IDs here: https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false

    .. note::
        The aggregation method for when freq = "D" is "mean" (which may not be
        appropriate for accumulated variables like precipitation).

    .. warning::
        If this function is updated, the cache will be invalidated and the data will need
        to be re-downloaded. To avoid this risk, set ``cache=False`` and save the data to disk
        manually.

    Args:
        var_IDs: list
            List of variable IDs to download. If None, all variables are downloaded.
            See the list of available variable IDs above.
        extent: tuple[float, float, float, float] | str
            Tuple of (lon_min, lon_max, lat_min, lat_max) or string of region name.
            Options are: "global", "north_america", "uk", "europe".
        date_range: tuple
            Tuple of (start_date, end_date) in format "YYYY-MM-DD".
        freq: str
            Frequency of data to download. Options are: "D" (daily) or "H" (hourly).
            If "D", the data is downloaded from the 1-hourly dataset and then resampled
            to daily averages. If "H", the 1-hourly data is returned as-is.
        num_processes Optional[int]:
            Number of CPUs to use for downloading years of ERA5 data in parallel.
            Defaults to 1 (i.e. no parallelisation). 75% of all available CPUs or 8 CPUs, whichever is smaller.
        verbose: bool
            Whether to print status messages. Default is ``False``.
        cache: bool
            Whether to cache the station metadata and data locally. If ``True``, calling the
            function again with the same arguments will load the data from the cache instead
            of downloading it again. Default is ``False``.
        cache_dir: str
            Directory to store the cached data. Default is ``".datacache"``.


    Returns:
        :class:`xarray.Dataset`
            ERA5 reanalysis data with dimensions ``time``, ``lat``, ``lon`` and variables
            ``var1``, ``var2``, etc.
    """
    if verbose:
        print(
            f"Downloading ERA5 data from Google Cloud Storage...",
            end=" ",
            flush=True,
        )
    tic = time.time()

    if date_range is None:
        date_range = ("1959-01-01", "2021-01-01")

    # Derive monthly chunks to download in parallel
    #   Uses calendar month boundaries to ensure repeat calls with different but overlapping
    #   date ranges use as many cached months as possible
    date_range = pd.to_datetime(date_range)
    start_date = date_range[0]
    # End of month at 1 minute before midnight
    end_date = (
        start_date
        + pd.offsets.MonthEnd()
        + pd.offsets.MonthBegin()
        - pd.DateOffset(seconds=1)
    )
    date_ranges = []
    while True:
        if end_date > date_range[1]:
            end_date = date_range[1].replace(hour=23, minute=59, second=59)
            stop = True
        else:
            stop = False
        date_ranges.append((start_date, end_date))
        # Start of next month
        start_date = (end_date + pd.offsets.MonthBegin()).replace(
            hour=0, minute=0, second=0
        )
        end_date = (
            start_date
            + pd.offsets.MonthEnd()
            + pd.offsets.MonthBegin()
            - pd.DateOffset(seconds=1)
        )
        if stop:
            break

    max_num_processes = 8
    if num_processes is None:
        # If user hasn't specified num CPUs, use 75% of available CPUs
        num_processes = max(1, int(0.75 * multiprocessing.cpu_count()))
        num_processes = min(num_processes, len(date_ranges))
        num_processes = min(num_processes, max_num_processes)

    if num_processes == 1:
        # Just download in one go
        if verbose:
            print("Downloading ERA5 data without parallelisation... ")
        era5_da = _get_era5_reanalysis_data_parallel(
            date_range=date_range,
            var_IDs=var_IDs,
            freq=freq,
            extent=extent,
            cache=cache,
            cache_dir=cache_dir,
        )
    elif num_processes > 1:
        if verbose:
            print(
                f"Using {num_processes} CPUs out of {multiprocessing.cpu_count()}... "
            )
        with multiprocessing.Pool(num_processes) as pool:
            partial_era5 = partial(
                _get_era5_reanalysis_data_parallel,
                var_IDs=var_IDs,
                freq=freq,
                extent=extent,
                cache=cache,
                cache_dir=cache_dir,
            )

            era5_das = list(
                tqdm.tqdm(
                    pool.imap(partial_era5, date_ranges),
                    total=len(date_ranges),
                    smoothing=0,
                    disable=not verbose,
                )
            )

        era5_da = xr.concat(era5_das, dim="time")

    if verbose:
        print(f"{era5_da.nbytes / 1e9:.2f} GB loaded in {time.time() - tic:.2f} s")
    return era5_da


def _get_era5_reanalysis_data_parallel(
    date_range,
    var_IDs=None,
    freq="D",
    extent="global",
    cache=False,
    cache_dir=".datacache",
):  # pragma: no cover
    """Helper function for downloading ERA5 data in parallel with caching.

    For documentation, see get_era5_reanalysis_data()
    """
    if not cache:
        cache_dir = None
    memory = Memory(cache_dir, verbose=0)

    @memory.cache
    def _get_era5_reanalysis_data_parallel_cached(
        date_range, var_IDs=None, freq="D", extent="global"
    ):
        if isinstance(extent, str):
            extent = extent_str_to_tuple(extent)
        else:
            extent = tuple([float(x) for x in extent])
        lon_min, lon_max, lat_min, lat_max = extent

        if freq == "D":
            # Need to download hourly data and then resample to daily
            #   See https://github.com/google-research/arco-era5/issues/62
            source = (
                "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/"
            )
        elif freq == "H":
            source = (
                "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/"
            )
        else:
            raise ValueError(f"Invalid freq: {freq}")

        era5_zarr = xr.open_zarr(source, consolidated=True, chunks={"time": 48})
        if var_IDs is not None:
            era5_zarr = era5_zarr[var_IDs]
        era5_da = era5_zarr.sel(time=slice(*date_range))
        # Replace longitude 0 to 360 with -180 to 180
        era5_da = era5_da.assign_coords(
            longitude=(era5_da.longitude + 180) % 360 - 180
        ).sortby("longitude")
        era5_da = era5_da.sel(
            latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)
        )
        if freq == "D":
            era5_da = era5_da.resample(time="1D").mean()
        era5_da = era5_da.load()
        era5_da = era5_da.rename({"latitude": "lat", "longitude": "lon"})
        return era5_da

    return _get_era5_reanalysis_data_parallel_cached(date_range, var_IDs, freq, extent)


def get_gldas_land_mask(
    extent: Union[Tuple[float, float, float, float], str] = "global",
    verbose: bool = False,
    cache: bool = False,
    cache_dir: str = ".datacache",
) -> xr.DataArray:  # pragma: no cover
    """Get GLDAS land mask at 0.25 degree resolution.
    Source: https://ldas.gsfc.nasa.gov/gldas/vegetation-class-mask.

    .. warning::
        If this function is updated, the cache will be invalidated and the data will need
        to be re-downloaded. To avoid this risk, set ``cache=False`` and save the data to disk
        manually.

    Args:
        extent: tuple[float, float, float, float] | str
            Tuple of (lon_min, lon_max, lat_min, lat_max) or string of region name.
            Options are: "global", "north_america", "uk", "europe".
        verbose: bool
            Whether to print status messages. Default is ``False``.
        cache: bool
            Whether to cache the station metadata and data locally. If ``True``, calling the
            function again with the same arguments will load the data from the cache instead
            of downloading it again. Default is ``False``.
        cache_dir: str
            Directory to store the cached data. Default is ``".datacache"``.

    Returns:
        :class:`xarray.DataArray`
            Land mask (1 = land, 0 = water) with dimensions ``lat``, ``lon``.
    """
    if not cache:
        cache_dir = None
    memory = Memory(cache_dir, verbose=0)

    @memory.cache
    def _get_gldas_land_mask_cached(
        extent: Union[Tuple[float, float, float, float], str] = "global",
        verbose: bool = False,
    ) -> xr.DataArray:
        if verbose:
            print(
                f"Downloading GLDAS land mask from NASA...",
                end=" ",
                flush=True,
            )
        tic = time.time()

        fname = "GLDASp5_landmask_025d.nc4"
        url = "https://ldas.gsfc.nasa.gov/sites/default/files/ldas/gldas/VEG/" + fname
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            with open(fname, "wb") as f:
                f.write(response.read())
        da = xr.open_dataset(fname)["GLDAS_mask"].isel(time=0).drop_vars("time").load()

        if isinstance(extent, str):
            extent = extent_str_to_tuple(extent)
        else:
            extent = tuple([float(x) for x in extent])
        lon_min, lon_max, lat_min, lat_max = extent

        # Reverse latitude to match ERA5
        da = da.reindex(lat=da.lat[::-1])
        da = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
        da.attrs = {}

        os.remove(fname)

        if verbose:
            print(f"{da.nbytes / 1e6:.2f} MB loaded in {time.time() - tic:.2f} s")

        return da

    return _get_gldas_land_mask_cached(extent, verbose)


def get_earthenv_auxiliary_data(
    var_IDs: Tuple[str] = ("elevation", "tpi"),
    extent: Union[Tuple[float, float, float, float], str] = "global",
    resolution: str = "1KM",
    verbose: bool = False,
    cache: bool = False,
    cache_dir: str = ".datacache",
) -> xr.Dataset:  # pragma: no cover
    """Download global static auxiliary data from EarthEnv into an xarray DataArray.
    See: https://www.earthenv.org/topography.

    .. note::
        Requires the `rioxarray` package to be installed. e.g. via ``pip install rioxarray``.
        See the ``rioxarray`` pages for more installation options:
        https://corteva.github.io/rioxarray/stable/installation.html

    .. note::
        This method downloads the data from EarthEnv to disk, then reads it into memory,
        and then deletes the file from disk. This is because EarthEnv does not support
        OpenDAP, so we cannot read the data directly into memory.

    .. note::
        At 1KM resolution, the global data is ~3 GB per variable.

    .. note::
        Topographic Position Index (TPI) is a measure of the local topographic position of a cell
        relative to its surrounding landscape. It is calculated as the difference between the
        elevation of a cell and the mean elevation of its surrounding landscape. This highlights
        topographic features such as mountains (positive TPI) and valleys (negative TPI).

    .. todo::
        support land cover data: https://www.earthenv.org/landcover

    .. warning::
        If this function is updated, the cache will be invalidated and the data will need
        to be re-downloaded. To avoid this risk, set ``cache=False`` and save the data to disk
        manually.

    Args:
        var_IDs: tuple
            List of variable IDs. Options are: "elevation", "tpi".
        extent: tuple[float, float, float, float] | str
            Tuple of (lon_min, lon_max, lat_min, lat_max) or string of region name.
            Options are: "global", "north_america", "uk", "europe".
        resolution: str
            Resolution of data. Options are: "1KM", "5KM", "10KM", "50KM", "100KM".
        verbose: bool
            Whether to print status messages. Default is ``False``.
        cache: bool
            Whether to cache the station metadata and data locally. If ``True``, calling the
            function again with the same arguments will load the data from the cache instead
            of downloading it again. Default is ``False``.
        cache_dir: str
            Directory to store the cached data. Default is ``".datacache"``.

    Returns:
        :class:`xarray.DataArray`
            Auxiliary data with dimensions ``lat``, ``lon`` and variable ``var_ID``.
    """
    if not cache:
        cache_dir = None
    memory = Memory(cache_dir, verbose=0)

    # Check for rioxarray and raise error if not present
    import importlib.util

    if importlib.util.find_spec("rioxarray") is None:
        raise ImportError(
            "The rioxarray is required to run this function, it was not found. Install with `pip install rioxarray`."
        )

    @memory.cache
    def _get_auxiliary_data_cached(
        var_IDs: str,
        extent: Union[Tuple[float, float, float, float], str] = "global",
        resolution: str = "1KM",
        verbose: bool = False,
    ) -> xr.Dataset:
        if verbose:
            print(
                f"Downloading EarthEnv data...",
                end=" ",
                flush=True,
            )
        tic = time.time()

        valid_var_IDs = ["elevation", "tpi"]
        valid_resolutions = ["1KM", "5KM", "10KM", "50KM", "100KM"]
        for var_ID in var_IDs:
            if var_ID not in valid_var_IDs:
                raise ValueError(
                    f"Invalid var_ID: {var_ID}. Options are: {valid_var_IDs}"
                )
        if resolution not in valid_resolutions:
            raise ValueError(
                f"Invalid resolution: {resolution}. Options are: {valid_resolutions}"
            )

        if isinstance(extent, str):
            extent = extent_str_to_tuple(extent)
        else:
            extent = tuple([float(x) for x in extent])
        lon_min, lon_max, lat_min, lat_max = extent

        da_dict = {}
        for var_ID in var_IDs:
            # Download data
            if var_ID == "elevation":
                suffix = "mn"
            elif var_ID == "tpi":
                suffix = "md"
            fname = f"{var_ID}_{resolution}mn_GMTED{suffix}.tif"
            url = "https://data.earthenv.org/topography/" + fname
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response:
                with open(fname, "wb") as f:
                    f.write(response.read())

            # Read data
            da = xr.open_dataset(fname).to_array().squeeze().load()
            da = da.rename({"y": "lat", "x": "lon"})
            da = da.drop_vars(["band", "spatial_ref", "variable"])
            da.name = var_ID
            da = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
            da_dict[var_ID] = da

            # Remove file
            os.remove(fname)

        ds = xr.Dataset(da_dict)
        if verbose:
            print(f"{ds.nbytes / 1e9:.2f} GB loaded in {time.time() - tic:.2f} s")

        return ds

    return _get_auxiliary_data_cached(var_IDs, extent, resolution, verbose)


if __name__ == "__main__":  # pragma: no cover
    # Using the same settings allows use to use pre-downloaded cached data
    data_range = ("2015-06-25", "2015-06-30")
    extent = "europe"
    era5_var_IDs = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]
    cache_dir = "tmp/"

    era5_raw_ds = get_era5_reanalysis_data(
        era5_var_IDs,
        extent,
        date_range=data_range,
        cache=True,
        cache_dir=cache_dir,
        verbose=True,
    )
