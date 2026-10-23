
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape

from config import CITIES_URL


def get_closest_city(tile):
    response = requests.get(CITIES_URL)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    tile_gdf = gpd.GeoDataFrame(geometry=[shape(tile.geometry)], crs="EPSG:4326")

    # Reproject to local UTM coordinates
    utm = tile_gdf.estimate_utm_crs()

    cities_proj = gdf.to_crs(utm)
    tile_proj = tile_gdf.to_crs(utm)

    # Find the nearest city to the center of the tile
    center = tile_proj.geometry.centroid.iloc[0]

    distances = cities_proj.geometry.distance(center)

    closest_city = cities_proj.loc[distances.idxmin()]
    dist = distances.min()

    return closest_city, dist

