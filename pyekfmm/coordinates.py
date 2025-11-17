import math
import numpy as np

def deg_to_km_factors(lat_deg):
    """
    Return (km_per_deg_lat, km_per_deg_lon) at the given latitude (degrees).
    Uses mean Earth radius approximation.
    Accurate to ~0.1% for most latitudes.
    """
    # mean radius of Earth in km (WGS84 mean)
    R = 6371.0088  # km

    # 1 deg in radians
    deg_rad = math.pi / 180.0

    # km per degree latitude: arc length = R * dtheta
    km_per_deg_lat = R * deg_rad

    # km per degree longitude: depends on cosine(latitude)
    km_per_deg_lon = R * math.cos(math.radians(lat_deg)) * deg_rad

    return km_per_deg_lat, km_per_deg_lon


def lonlat_to_local_km_equirectangular(lon0, lat0, lon, lat):
    """
    Convert lon/lat (deg) to local Cartesian coordinates (km) relative to origin (lon0, lat0).
    lon, lat may be scalars or iterables (lists/arrays) of same shape.
    Uses equirectangular approximation with cos(lat0) scaling for longitude.
    Accurate for areas up to a few hundred km from origin.
    Returns (x_km, y_km) where x east, y north.
    """
    # ensure scalar inputs become floats
    lon0 = float(lon0); lat0 = float(lat0)

    def _to_iterable(v):
        try:
            iter(v)
            return list(v)
        except TypeError:
            return [v]

    lon_list = _to_iterable(lon)
    lat_list = _to_iterable(lat)

    km_per_deg_lat, km_per_deg_lon_at_lat0 = deg_to_km_factors(lat0)
    x_km = []
    y_km = []
    for lo, la in zip(lon_list, lat_list):
        dx_deg = lo - lon0
        dy_deg = la - lat0
        x_km.append(dx_deg * km_per_deg_lon_at_lat0)
        y_km.append(dy_deg * km_per_deg_lat)
    # return scalar if input scalar
    if len(x_km) == 1:
        return x_km[0], y_km[0]
    return x_km, y_km

def local_km_to_lonlat_equirectangular(lon0, lat0, x, y):
    """
    Convert local Cartesian coordinates (km) back to geographic coordinates (lon, lat).
    lon0, lat0: reference point (degrees)
    x, y: local coordinates in km
    """
    
    R = 6371.0  # km

    # Convert reference lat0 to radians
    lat0_rad = np.deg2rad(lat0)
    
    # Compute latitude
    lat = lat0 + (y / R) * (180/np.pi)
    
    # Compute longitude
    lon = lon0 + (x / (R * np.cos(lat0_rad))) * (180/np.pi)
    
    return lon, lat
