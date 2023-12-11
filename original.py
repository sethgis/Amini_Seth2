from pystac_client import Client
import satsearch
import os
import boto3
import rasterio as rio
from pyproj import Transformer
from json import load
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
import numpy as np
from rasterio.transform import from_origin
import rasterio
from rasterio.transform import from_origin
# import gdal
from osgeo import gdal
# from rasterio.transform import from_origin
from rasterio.features import bounds

from geo.Geoserver import Geoserver



# origins = [
#     "http://localhost:8080",
#     "*",
  
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,  # Set this to True if your frontend app includes credentials (cookies, HTTP authentication)
#     allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#     allow_headers=["*"],  # Set this to the list of allowed HTTP headers, or use "*" to allow all headers
# )

def main():
    file_path = "geo.geojson"
    file_content = load(open(file_path))
    geometry = file_content["features"][0]["geometry"]

    timeRange = '2022-11-27/2022-11-30'

    SentinelSearch = satsearch.Search.search(
        url="https://earth-search.aws.element84.com/v0",
        intersects=geometry,
        datetime=timeRange,
        # collections=['sentinel-s2-l2a'],
        # collections=['sentinel-s2-l1c'],
        # sentinel-s2-l2a
        collections=['sentinel-s2-l2a-cogs'],
        query={"eo:cloud_cover": {"lt": 20}})

    Sentinel_items = SentinelSearch.items()
    print(Sentinel_items.summary())

    aws_session = boto3.Session()
    
    bbox = rio._features._bounds(geometry)

    def getSubset(geotiff_file, bbox):
        with rio.open(geotiff_file) as geo_fp:
            # Calculate pixels with PyProj
            Transf = Transformer.from_crs("epsg:4326", geo_fp.crs)
            lat_north, lon_west = Transf.transform(bbox[3], bbox[0])
            lat_south, lon_east = Transf.transform(bbox[1], bbox[2])
            x_top, y_top = geo_fp.index(lat_north, lon_west)
            x_bottom, y_bottom = geo_fp.index(lat_south, lon_east)
            # Define window in RasterIO
            window = rio.windows.Window.from_slices((x_top, x_bottom), (y_top, y_bottom))
            # Actual HTTP range request
            subset = geo_fp.read(1, window=window)
        return subset
    def plotNDVI(nir, red, filename):
        ndvi = (nir - red) / (nir + red)
        ndvi[ndvi > 1] = 1

        output = f"output_{filename}.tif"
        os.makedirs(os.path.dirname(output), exist_ok=True)

        arr = ndvi
        if arr.dtype == np.float32:
            arr_type = gdal.GDT_Float32
        else:
            arr_type = gdal.GDT_Int32

        # outpu = f"output{filename}.tif"

        
        file_path = "geo.geojson"
        file_content = load(open(file_path))
        geometry = file_content["features"][0]["geometry"]
        bbox = bounds(geometry)
        output_filename = "ndvi_" + filename + ".tif"

        final_filename = "ndvi_proj" + filename + ".tif"
        os.makedirs(os.path.dirname(final_filename), exist_ok=True)


        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        crs = 'EPSG:4326'

        with rasterio.open(
        output_filename,
        'w',
        driver='GTiff',
        width=len(arr[0]),  # Width is the number of columns
        height=len(arr),     # Height is the number of rows
        count=1,              # Number of bands (single-band)
        dtype=str(arr.dtype), 
        # crs=crs,
        # Data type of the array
        # Set the coordinate reference system as needed
        transform=from_origin(bbox[0], bbox[3], (bbox[2] - bbox[0]) / len(arr[0]), (bbox[1] - bbox[3]) / len(arr))) as dst:
            dst.write(arr, 1) 


        # # Reference GeoTIFF file with the correct projection
        # reference_filename = "Wemast_wetland/wetland2/sentinel/PROJECTION.tiff"

        # # New GeoTIFF file that needs the same projection
        # output_filename = "Wemast_wetland/wetland2/sentinel/PROJECTION.tiff"

        # Open the reference GeoTIFF to get its CRS

        # File paths
        geotiff_without_crs_path = output_filename
        file_path = "//Users/sethnyawacha/Desktop/FAST_API/Wemast_wetland/wetland2/sentinel/real_projection.tiff"
        if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
        reference_geotiff_path = file_path
        output_geotiff_path = output_filename

        # Open the reference GeoTIFF to get its CRS and transform
        with rasterio.open(reference_geotiff_path) as reference_ds:
            crs = reference_ds.crs
            transform = reference_ds.transform

    #Assign projection using an existing file

        # Open the GeoTIFF without CRS
        with rasterio.open(geotiff_without_crs_path) as input_ds:
            # Read the data and other metadata
            data_without = input_ds.read()
            dtype = input_ds.dtypes[0]
            count = input_ds.count

            # crs = 'EPSG:4326'
            # Save the new GeoTIFF with the same CRS and transform
            with rasterio.open(
                    final_filename,
                    'w',
                    driver='GTiff',
                    width=input_ds.width,
                    height=input_ds.height,
                    count=count,
                    dtype=dtype,
                    crs=crs,
                    transform=transform
            ) as output_ds:
                output_ds.write(data_without)

        # access geoserver

        # geo = Geoserver('http://66.42.65.87:8080/geoserver', username='', password='')

        # #publish the created layer in geoserver and retrieve the API for sharing as a json to fornt end

        # geo.create_coveragestore(layer_name=final_filename, path=f"{final_filename}", workspace='STATS')

        # API_structure = "http://66.42.65.87:8080/geoserver/STATS/wms?service=WMS&version=1.1.0&request=GetMap&layers=STATS%3A{final_filename}&bbox=34.725705586%2C0.170183264%2C34.781436957%2C0.230932724&width=704&height=768&srs=EPSG%3A4326&styles=&format=image%2Fpng"


        # # print(API_structure)

        # return {"The API for generated layer is": API_structure}

       

        plt.imshow(ndvi)
        plt.savefig(filename)
        plt.close()


    def merge_bands(red, green, blue, bbox):
        # Read the individual bands
        with rio.open(red) as red_fp, rio.open(green) as green_fp, rio.open(blue) as blue_fp:
            # Calculate pixels with PyProj
            transf = Transformer.from_crs("epsg:4326", red_fp.crs)
            lat_north, lon_west = transf.transform(bbox[3], bbox[0])
            lat_south, lon_east = transf.transform(bbox[1], bbox[2])
            x_top, y_top = red_fp.index(lat_north, lon_west)
            x_bottom, y_bottom = red_fp.index(lat_south, lon_east)

            # Define window in RasterIO
            window = rio.windows.Window.from_slices((x_top, x_bottom), (y_top, y_bottom))

            # Read and resample the bands
            red_band = red_fp.read(1, window=window, out_shape=(red_fp.height, red_fp.width), resampling=Resampling.bilinear)
            green_band = green_fp.read(1, window=window, out_shape=(green_fp.height, green_fp.width), resampling=Resampling.bilinear)
            blue_band = blue_fp.read(1, window=window, out_shape=(blue_fp.height, blue_fp.width), resampling=Resampling.bilinear)

        # Stack the bands to create an RGB image
        rgb_image = np.stack([red_band, green_band, blue_band], axis=0)

        return rgb_image

    for i, item in enumerate(Sentinel_items):
        red_s3 = item.assets['B04']['href']
        nir_s3 = item.assets['B08']['href']
        blue_s3 = item.assets['B03']['href']
        green_s3 = item.assets['B02']['href']
        date = item.properties['datetime'][0:10]
        print("Sentinel item number " + str(i) + "/" + str(len(Sentinel_items)) + " " + date)
        red = getSubset(red_s3, bbox)
        nir = getSubset(nir_s3, bbox)
        plotNDVI(nir, red, "sentinel/" + date + "_ndvi.png")
        rgb_image = merge_bands(red_s3, green_s3, blue_s3, bbox)
        rgb_filename = f"sentinel/{date}_rgb.tif"

    with rio.open(
        rgb_filename,
        'w',
        driver='GTiff',
        width=red.shape[1],
        height=red.shape[0],
        count=3,
        dtype=str(red.dtype),
        # crs=rio.crs.CRS.from_epsg(4326),
        transform=rio.transform.from_bounds(*bbox, width=red.shape[1], height=red.shape[0])
    ) as dst:
        dst.write(rgb_image)

if __name__ == "__main__":
    main()


# /Users/sethnyawacha/Desktop/FAST_API/Wemast_wetland/wetland2/output1.tif

# /Users/sethnyawacha/Desktop/FAST_API/Wemast_wetland/wetland2/gpt.py

# /Users/sethnyawacha/Desktop/FAST_API/Wemast_wetland/wetland2/sentinel/