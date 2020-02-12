import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import eoxmagmod


def _datetime_to_mjd2000(t: dt.datetime) -> float:
    "Convert a datetime object to MJD2000."
    # Convert to datetime64 ns
    t = np.datetime64(t).astype("M8[ns]")
    # Get offset to year 2000
    t = (t - np.datetime64('2000')).astype('int64')
    # Convert from ns to days
    NS2DAYS = 1.0/(24*60*60*1e9)
    return t * NS2DAYS


def _generate_latlon_grid(resolution=2, min_lat=-90, max_lat=90):
    "Generate a grid of positions over the Earth at a given degree resolution."
    lat, lon = np.meshgrid(
        np.arange(min_lat, max_lat, resolution),
        np.arange(-180, 180, resolution))
    REFRAD_KM = 6371.200
    rad = np.ones_like(lat)*REFRAD_KM
    return lat, lon, rad


def _eval_model_on_grid(
        lat: np.ndarray, lon: np.ndarray, rad: np.ndarray,
        times: np.ndarray, model=None, shc_model=eoxmagmod.data.IGRF12,
        **opts):
    """Use eoxmagmod to evaluate a model over a grid.
    
    Evaluate the B_NEC vector at each point in a grid, for a range of times.
    
    Args:
        lat (ndarray): Latitude in degrees (Spherical geocentric)
        lon (ndarray): Longitude in degrees
        rad (ndarray): Geocentric radius in kilometres
        times (ndarray): 1-D array of datetimes
        model (magnetic_model): Model loaded with eoxmagmod.load_model_<x>
        shc_model (str): Path to a shc model

    Returns:
        ndarray: B_NEC values at each point

    """
    if shc_model and not model:
        model = eoxmagmod.load_model_shc(shc_model)
    times_mjd2000 = [_datetime_to_mjd2000(t) for t in times]
    # Reshape the input coordinates to use eoxmagmod
    orig_shape = lat.shape
    _lat, _lon, _rad = map(lambda x: x.flatten(), (lat, lon, rad))
    coords = np.stack((_lat, _lon, _rad), axis=1)
    coords = np.stack([coords for i in range(len(times_mjd2000))])
    timestack = np.stack([np.ones_like(_lat)*t for t in times_mjd2000])
    # Use the model and do the computation
    b_nec = model.eval(timestack, coords, scale=[1, 1, -1], **opts)
    # Reshape output back to original grid
    b_nec = b_nec.reshape(times.shape + orig_shape + (3, ))
    return b_nec


def build_model_xarray(model=None, times=None, resolution=2):
    lat, lon, rad = _generate_latlon_grid(resolution=resolution)
    # Evaluate the model
    # b_nec = eval_model_on_grid(lat, lon, rad, times, shc_model=eoxmagmod.data.IGRF12)
    b_nec = _eval_model_on_grid(lat, lon, rad, times, shc_model=model)
    # Assign to an xarray.Dataset
    ds = xr.Dataset({'B_NEC': (['time', 'x', 'y', 'NEC'],  b_nec)},
                 coords={'lon': (['x', 'y'], lon),
                         'lat': (['x', 'y'], lat),
                         'time': times})
    # Add some columns https://intermagnet.github.io/faq/10.geomagnetic-comp.html
    # Intensity
    ds["F"] = np.sqrt(np.sum(ds["B_NEC"]**2, axis=3))
    # Horizontal intensity
    ds["H"] = np.sqrt(ds["B_NEC"][:, :, :, 0]**2 + ds["B_NEC"][:, :, :, 1]**2)
    # Declination, arctan(B_E / B_N)
    ds["DEC"] = np.rad2deg(np.arctan(
        ds["B_NEC"][:, :, :, 1] / ds["B_NEC"][:, :, :, 0]))
    # Inclination, arctan(B_C / H)
    ds["INC"] = np.rad2deg(np.arctan(
        ds["B_NEC"][:, :, :, 2] / ds["H"][:, :, :]))
    # Separate components
    ds["B_N"] = ds["B_NEC"][:, :, :, 0]
    ds["B_E"] = ds["B_NEC"][:, :, :, 1]
    ds["B_C"] = ds["B_NEC"][:, :, :, 2]
    return ds