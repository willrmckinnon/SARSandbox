# Tile Analysis
import requests
import pandas as pd
import pystac_client
from PIL import Image
from io import BytesIO
import geopandas as gpd
import planetary_computer
import matplotlib.pyplot as plt
from shapely.geometry import shape
from IPython.display import display

# Data Anslysis
import rasterio
import numpy as np
import planetary_computer
from pyproj import Transformer
import matplotlib.pyplot as plt

# Local Imports
from get_city import get_closest_city



def multichannel_data_analysis(items, total_sqkm = 100):


    xmin = -5.690798188840799
    ymin = 35.17138623212788
    xmax = -2.6076774506808538
    ymax = 37.07037914207646



    y_perc = 0.4
    x_perc = 0.1

    x_cent = xmin+(x_perc*(xmax-xmin))
    y_cent = ymin+(y_perc*(ymax-ymin))


    print(f'Total Tiles Collected {len(items)}\n')

    # Tile Analysis
    tile = items[0]
    print('Each Tile Contains the following Assets:')
    for key in tile.assets.keys(): print(key)



    # Geometry Analysis
    polygon = shape(tile.geometry)
    #centroid = polygon.centroid
    #lon, lat = centroid.x, centroid.y #Centerpoint
    lon, lat = x_cent, y_cent
    print(f'This scene is centered on {lat:.4f}, {lon:.4f}\n')

    # Find Nearest City
    city, dist = get_closest_city(tile)
    print(f"The closest city to this tile is {city['name']}, {city['country']} which is {dist/1000:.2f} km from scene center")

    # Fetch image
    response = requests.get(tile.assets["rendered_preview"].href)
    response.raise_for_status()

    # Open image
    img = Image.open(BytesIO(response.content))

    # Display
    display(img)




    # Sign the item to pull the data
    signed = planetary_computer.sign(tile)

    # Define center area
    half_width = (np.sqrt(total_sqkm)/2)*1000
    # Project center from geographic coordinates to raster CRS
    with rasterio.open(signed.assets["vv"].href) as src:
        transformer = Transformer.from_crs(
            "EPSG:4326", src.crs, always_xy=True
        )
        cx, cy = transformer.transform(lon, lat)

        # Define 10 km x 10 km bounding box
        bounds = (
            cx - half_width,
            cy - half_width,
            cx + half_width,
            cy + half_width
        )

        # Convert geographic bounds to pixel window
        window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()

    # Read in both polarizations
        vv = src.read(1, window=window, boundless=True, fill_value=np.nan)

    with rasterio.open(signed.assets["vh"].href) as src:
        vh = src.read(1, window=window, boundless=True, fill_value=np.nan)

    # Convert positive linear backscatter to dB
    vv_db = np.where(vv > 0, 10 * np.log10(np.maximum(vv, 1e-10)), np.nan)
    vh_db = np.where(vh > 0, 10 * np.log10(np.maximum(vh, 1e-10)), np.nan)

    # Display the different polarizations
    print('The Following images display a comparision between VH and VV polarization of the same tile.')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(vh_db)
    axes[0].set_title(rf'VH Polarization at {total_sqkm} $km^2$')
    axes[0].axis('off')

    axes[1].imshow(vv_db)
    axes[1].set_title(rf'VV Polarization at {total_sqkm} $km^2$')
    axes[1].axis('off')