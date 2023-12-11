from pystac_client import Client
import satsearch
import os
import rasterio as rio
from pyproj import Transformer
from json import load
import matplotlib.pyplot as plt

def main():
    file_path = "geo.geojson"
    file_content = load(open(file_path))
    geometry = file_content["features"][0]["geometry"]

    time_range = '2022-11-01/2022-11-30'

    SentinelSearch = satsearch.Search.search(
        url="https://earth-search.aws.element84.com/v0",
        intersects=geometry,
        datetime=time_range,
        collections=['sentinel-s2-l2a-cogs'],
        query={"eo:cloud_cover": {"lt": 40}})

    Sentinel_items = SentinelSearch.items()
    print(Sentinel_items.summary())

    bbox = rio._features._bounds(geometry)

    def get_subset(geotiff_file, bbox):
        with rio.open(geotiff_file) as geo_fp:
            transf = Transformer.from_crs("epsg:4326", geo_fp.crs)
            lat_north, lon_west = transf.transform(bbox[3], bbox[0])
            lat_south, lon_east = transf.transform(bbox[1], bbox[2])
            x_top, y_top = geo_fp.index(lat_north, lon_west)
            x_bottom, y_bottom = geo_fp.index(lat_south, lon_east)
            window = rio.windows.Window.from_slices((x_top, x_bottom), (y_top, y_bottom))
            subset = geo_fp.read(1, window=window)
        return subset

    def plot_ndvi(nir, red, filename):
        ndvi = (nir - red) / (nir + red)
        ndvi[ndvi > 1] = 1
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.imshow(ndvi)
        plt.savefig(filename)
        plt.close()

    for i, item in enumerate(Sentinel_items):
        red_s3 = item.assets['B04']['href']
        nir_s3 = item.assets['B08']['href']
        date = item.properties['datetime'][0:10]
        print("Sentinel item number " + str(i) + "/" + str(len(Sentinel_items)) + " " + date)
        red = get_subset(red_s3, bbox)
        nir = get_subset(nir_s3, bbox)
        
        # Create an RGB image and save it as GeoTIFF
        # rgb_filename = f"sentinel/{date}_rgb.tif"
        # with rio.open(rgb_filename, 'w', driver='GTiff', width=red.shape[1], height=red.shape[0], count=3, dtype=str(red.dtype), crs='EPSG:4326'):
        #     pass

        # # Save the NDVI as GeoTIFF
        # ndvi_filename = f"sentinel/{date}_ndvi.tif"
        # with rio.open(ndvi_filename, 'w', driver='GTiff', width=red.shape[1], height=red.shape[0], count=1, dtype=str(red.dtype), crs='EPSG:4326'):
        #     pass

        # plot_ndvi(nir, red, ndvi_filename)

if __name__ == "__main__":
    main()

