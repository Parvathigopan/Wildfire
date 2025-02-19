import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Load fire station shapefile
fire_stations = gpd.read_file("dataset/fire_stations.shp")

# Function to calculate vegetation risk based on NDVI
def calculate_vegetation_risk(ndvi_data):
    return np.where(ndvi_data > 0.6, 5, np.where(ndvi_data > 0.3, 3, 1))

# Function to calculate proximity risk
def calculate_proximity_risk(window, transform, fire_stations, tile_shape):
    cols, rows = np.meshgrid(
        np.arange(tile_shape[1]) * transform[0] + transform[2],
        np.arange(tile_shape[0]) * transform[4] + transform[5]
    )
    risk_reduction = np.zeros(tile_shape, dtype=float)
    for _, station in fire_stations.iterrows():
        station_point = Point(station.geometry.x, station.geometry.y)
        distances = np.sqrt((cols - station_point.x) ** 2 + (rows - station_point.y) ** 2)
        risk_reduction += np.where(distances < 0.01, 1, 0)  # Adjust radius as needed
    return risk_reduction

# Process raster in chunks
tile_size = 512  # Tile size for memory-safe processing
output_filename = "fire_risk_map.tif"

with rasterio.open("ndvi.tif") as src:
    transform = src.transform
    crs = src.crs
    raster_shape = src.shape
    profile = src.profile
    profile.update(dtype="float32", count=1)

    with rasterio.open(output_filename, "w", **profile) as dst:
        for row_off in tqdm(range(0, raster_shape[0], tile_size), desc="Processing rows"):
            for col_off in range(0, raster_shape[1], tile_size):
                # Define window
                window = Window(col_off, row_off, min(tile_size, raster_shape[1] - col_off), min(tile_size, raster_shape[0] - row_off))
                window_transform = rasterio.windows.transform(window, transform)
                tile_shape = (window.height, window.width)

                # Read NDVI data for the window
                ndvi_data = src.read(1, window=window)

                # Calculate vegetation risk
                vegetation_risk = calculate_vegetation_risk(ndvi_data)

                # Calculate proximity risk
                proximity_risk = calculate_proximity_risk(window, window_transform, fire_stations, tile_shape)

                # Combine risks and clip to range [0, 5]
                risk_scores = vegetation_risk - proximity_risk
                risk_scores = np.clip(risk_scores, 0, 5)

                # Write to output raster
                dst.write(risk_scores.astype("float32"), 1, window=window)

# Downsample the raster for visualization
def plot_downsampled_raster(raster_path, downsample_factor, output_image_path):
    with rasterio.open(raster_path) as src:
        downsampled_data = src.read(
            out_shape=(1, src.height // downsample_factor, src.width // downsample_factor),
            resampling=rasterio.enums.Resampling.average,
        )[0]
        bounds = rasterio.transform.array_bounds(
            downsampled_data.shape[0], downsampled_data.shape[1], src.transform
        )
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

    plt.figure(figsize=(10, 8))
    plt.title("Fire Risk Map (Downsampled for Visualization)")
    plt.imshow(downsampled_data, cmap="hot", extent=extent)
    plt.colorbar(label="Risk Level")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(output_image_path, dpi=300)
    plt.show()

# Visualize downsampled fire risk map
plot_downsampled_raster(output_filename, downsample_factor=10, output_image_path="fire_risk_map_downsampled.png")