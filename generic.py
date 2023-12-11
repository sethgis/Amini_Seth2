from satsearch import Search
import fiona
import rasterio
import rioxarray
from shapely.geometry import mapping
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.features import shapes
from rasterio.merge import merge
import geopandas as gpd
import os
import numpy as np
from osgeo import gdal

# DOWNLOAD AND RESAMPLING IMAGES

def main():
    # Set your input parameters
    polygon_path = 'geo.geojson'
    start_date = '2017-01-01'
    end_date = '2023-03-31'
    band = ["thumbnail"]

    # band = ["thumbnail","visual"]
    # band = ["red","green","blue","visual"]
    max_cloud_cover = 10

    # def search_sentinel(polygon_path: str,
    #             start_date: '2017-01-01',
    #             end_date:'2023-03-31',
#             band: str,
#             max_cloud_cover: int)->list:
    """
    Searches the Sentinel images in the S3 bucket according to the 
    defined search parameters
    Parameters
    -----------
        polygon_path: str
            Path to the AOIpolygon.
        start_date: str
            Start date in 'YYYY-MM-DD' format.
        end_date: str
            End date in 'YYYY-MM-DD' format.
        band: str
            Name of the required band
        max_cloud_cover: int
            Percentage of maximum cloud cover for the images
    Returns
    -----------
        The list of download urls for the requested images.
    """
    # Read the polygon
    gdf = gpd.read_file(polygon_path)
    print(gdf.crs)
    # Get the total bounds
    bounding_box = gdf.total_bounds.tolist()
    print('The bounds of the selected polygon are:\n',bounding_box)
    # Do the search
    search = Search(bbox=bounding_box,
                datetime=f'{start_date}/{end_date}',
                collections=['sentinel-s2-l2a-cogs'],
                query={'eo:cloud_cover': {'lt': max_cloud_cover}},
                url='https://earth-search.aws.element84.com/v0')


    print('bbox search: %s items' % search.found())
    
    # List of items
    items = search.items()
    #print(items.summary(['date', 'id', 'eo:cloud_cover'])) 
    
    result = items.summary(['date', 'id', 'eo:cloud_cover'])
    # Save the bands URL's into a list
    # list_urls = [e.asset(band)['href'] for e in items]

    # print(list_urls)
    
    # return list_urls


    list_urls = []
    for band in band:
        list_urls.extend([e.asset(band)['href'] for e in items])
    
    print("List of Download URLs:")
    for url in list_urls:
        xds = rioxarray.open_rasterio(url,masked=True)

        print(url)

    

    
    return list_urls
if __name__ == "__main__":
     main()


    # # def resample(path_input:str,
    # #             file_name:str,
    # #             path_output:str):
    #     """
    #     Resamples the input raster from 20 m pixel resolution to 10 m.
    #     Parameters
    #     -----------
    #         path_input: str
    #             Path to to the folder where the files are.
    #         file_name: str
    #             Name of the file to be resampled.
    #         path_output: str
    #             Path to to the folder where the files will be stored.
    #     Returns
    #     -----------
    #         A new resampled file with 10 m pixel resolution.
    #     """
        
    #     xds = rioxarray.open_rasterio(path_input+'\\'+file_name,
    #     masked=True,
    #     )
    #     upscale_factor = 2
    #     new_width = xds.rio.width * upscale_factor
    #     new_height = xds.rio.height * upscale_factor

    #     xds_upsampled = xds.rio.reproject(
    #         xds.rio.crs,
    #         shape=(new_height, new_width),
    #         resampling=Resampling.bilinear,
    #     )
    #     xds_upsampled.rio.to_raster(path_output+'\\'+file_name)

    # def clip_by_polygon_sub(geoms: list,
    #                         input_folder:str,
    #                         file_name:str,
    #                         path_to_output: str):
    #     """
    #     Function to crop an image by the polygon bounds. The image is saved to disk.
    #     Parameters
    #     -----------
    #         geoms: list
    #             Geometry of the AOI polygon.
    #         input_folder: str
    #             Path to the folder where the images to be clipped are.
    #         file_name: str
    #             Name of the file.
    #         path_to_output: str
    #             Path to the folder where the clipped images will be stored.
    #     Returns
    #     -----------
    #         A new cropped file. If there are some exceptions, the lis with the files
    #         which haven't been cropped is returned.        
    #     """    
    #     # Exception list
    #     exceptions = []
    
    #     try:
    #         # Crop the image
    #         with rasterio.open(input_folder+'\\'+file_name) as src:
    #             out_image, out_transform = mask(src, geoms, crop=True)
    #             out_meta = src.meta
    #             # Update metadata
    #             out_meta.update({"driver": "GTiff",
    #                 "height": out_image.shape[1],
    #                 "width": out_image.shape[2],
    #                 "transform": out_transform})
    #             src.close()
    #         with rasterio.open(path_to_output+file_name, "w", **out_meta) as dest:
    #             dest.write(out_image)
    #             dest.close()
    #             print(f'[INFO]: Cropped {file_name} image saved at {path_to_output}')  
    #     except Exception as error:
    #         print(error)
    #         exceptions.append(file_name)
    #     return exceptions

    # def geom_polygon(path_polygon:str, list_links:list) -> dict:
    #     """
    #     Function to reproject the input polygon into the same CRS
    #     as the imagery.
    #     Parameters
    #     -----------
    #         path_polygon: str
    #             Path to the input polygon.
    #         list_links: str
    #             List with the image download url links to get the CRS of the images.
    #     Returns
    #     -----------
    #         Geometry of the polygon.
    #     """
    #     # Prepare the polygon
    #     # Get the SRC of one image
    #     image = rasterio.open(list_links[0])
    #     print(image.crs)
    #     # Read the polygon
    #     shapefile = gpd.read_file(path_polygon)
    #     shape_prj = shapefile.to_crs(image.crs)
    #     geoms = shape_prj.geometry.values
    #     geoms = [mapping(geoms[0])] 
    #     return geoms

    # def clip_by_polygon(geoms: list,
    #                     path_to_image:str,
    #                     path_to_output: str):
    #     """
    #     Function to crop an image by the polygon bounds. The image is saved to disk.
    #     Parameters
    #     -----------
    #         geoms: list
    #             Geometry of the AOI polygon.
    #         path_to_image: str
    #             Url of the image.
    #         path_to_output: str
    #             Path to the folder where the clipped images will be stored.
    #     Returns
    #     -----------
    #         A new cropped file. If there are some exceptions, the lis with the files
    #         which haven't been cropped is returned.        
    #     """       
    #     # Exception list
    #     exceptions = []
    #     # Define the output image name
    #     out_image_name = path_to_image.split("/")[-2]
    #     try:
    #         # Crop the image
    #         with rasterio.open(path_to_image) as src:
    #             out_image, out_transform = mask(src, geoms, crop=True)
    #             out_meta = src.meta
    #             # Update metadata
    #             out_meta.update({"driver": "GTiff",
    #                 "height": out_image.shape[1],
    #                 "width": out_image.shape[2],
    #                 "transform": out_transform})
    #             src.close()
    #         with rasterio.open(path_to_output+f'{out_image_name}.tif', "w", **out_meta) as dest:
    #             dest.write(out_image)
    #             dest.close()
    #             print(f'[INFO]: Cropped {out_image_name} image saved at {path_to_output}')  
    #     except Exception as error:
    #         print(error)
    #         exceptions.append(out_image_name)
        
    #     return exceptions       
        

    # # CLOUD MASKING

    # def create_scl_mask(input_scl_path:str,
    #                     scl_file:str):
    #     """
    #     Function to create the mask for clouds and shadows and other features
    #     that could introduce noise in the image analysis from the SCL image.
    #     Parameters
    #     -----------
    #         input_scl_path: str
    #             Path to the SCL files folder
    #         scl_file: str
    #             Name of the SCL file from which creating the mask.
    #     Returns
    #     -----------
    #         A GeoJson file with all the features that will be used as a mask for
    #         the rest of images
    #     """       
        
    #     # Create the look up table to reclassify the scl band
    #     lookup = np.arange(12, dtype=np.uint8)
    #     lookup[1] = 100 # Saturated or deffective
    #     lookup[2] = 100 # Dark areas
    #     lookup[3] = 100 # Cloud shadows
    #     lookup[6] = 100 # Water
    #     lookup[7] = 100 # Unclassified
    #     lookup[8] = 100 # Cloud medium probability
    #     lookup[9] = 100 # Cloud high probability
    #     lookup[10] = 100 # Thin cirrus
    #     lookup[11] = 100 # Snow
    #     print(scl_file)
        
    #     try:
    #         with rasterio.open(input_scl_path+'\\'+scl_file) as src:
    #             # Read as numpy array
    #             array = src.read()
    #             profile = src.profile

    #             # Reclassify in a single operation using broadcasting
    #             array = lookup[array]
    #             mask = array == 100
    #             # Convert to geodataframe
    #             results = (
    #                     {'properties': {'type': v}, 'geometry': s}
    #                     for i, (s, v) 
    #                     in enumerate(
    #                             shapes(array, mask=mask, transform=src.transform)))
    #             geoms = list(results)

    #             # Transform to geodaframe
    #             poly = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)
    #             # Create a folder to store the mask
    #             path_to_mask = input_scl_path+'\\MASKS\\'
    #             if not os.path.exists(path_to_mask):
    #                 os.mkdir(path_to_mask)        
    #             poly.to_file(path_to_mask+f'{scl_file.split(".tif")[0]}.geojson', driver='GeoJSON')
    #     except Exception as error:
    #             print(error)

    # def apply_mask(path_to_band:str,
    #             input_band:str,
    #             band_name:str,
    #             path_to_scl:str):
    #     """
    #     Function to apply the corresponding cloud mask to an image.
    #     Parameters
    #     -----------
    #         path_to_band: str
    #             Path to the folder where the files to be masked are.
    #         input_band: str
    #             Name of the file to be masked.
    #         band_name: str
    #             Name of the band ("B04", "B08", "B11", "B12")
    #     Returns
    #     -----------
    #         A new image where the masked values are equal to NoData.
    #     """  
    #     # Create a folder to store the masked bands
    #     path_to_masked = f'{path_to_band}\\MASKED\\'
    #     if not os.path.exists(path_to_masked):
    #         os.mkdir(path_to_masked) 
        
    #     try:
    #         # Define the path to the masks
    #         path_to_masks= f"{path_to_scl}\\MASKS\\{input_band.replace('.tif', '.geojson')}" 
    #         print(path_to_masks)
    #         with fiona.open(path_to_masks, "r") as layer:
    #             shapes = [feature["geometry"] for feature in layer]
    #         with rasterio.open(path_to_band+'\\'+input_band) as src:
    #             out_image, out_transform = rasterio.mask.mask(src, shapes, crop=False, invert=True)
    #             out_meta = src.meta

    #             out_meta.update({"driver": "GTiff",
    #                         "height": out_image.shape[1],
    #                         "width": out_image.shape[2],
    #                         "transform": out_transform})

    #         with rasterio.open(f'{path_to_masked}{input_band.replace(".tif",f"_{band_name}_mskd.tif")}', "w", **out_meta) as dest:
    #             dest.write(out_image)       
            
    #     except Exception as error:
    #         print(error)

    # # VEGETATION INDICES CALCULATION

    # def ndmi(b11_mskd_folder: str,
    #         b8_mskd_folder: str,
    #         b11_file: str,
    #         ouput_path:str):
    #     """
    #     Function to calculate the Normalized Difference Moisture Index.
    #     Parameters
    #     -----------
    #         b11_mskd_folder: str
    #             Path to the folder where the B11 band files are.
    #         b8_mskd_folder: str
    #             Path to the folder where the B08 band files are.
    #         b11_file: str
    #             Name of the B11 band file.
    #         ouput_path: str
    #             Path to the folder where the NDMI images will be saved.
    #     Returns
    #     -----------
    #         A NDMI image.
    #     """  
    #     with rasterio.open(b11_mskd_folder+'\\'+b11_file) as src:
    #         b11 = src.read(1, masked=True).astype('float64')
    #         #print("SHAPE", b11.shape)
    #     # The name of the B8 file will be the same as the b4
    #     b8_file = b11_file.replace('B11', 'B08')
    #     #print("B8",b8_mskd_folder+'\\'+b8_file)
    #     with rasterio.open(b8_mskd_folder+'\\'+b8_file) as src:
    #         b8 = src.read(1, masked=True).astype('float64')
    #         #print("SHAPE", b8.shape)
    #         profile = src.meta # To get the metadata
        
    #     # Calculate the NDVI
    #     # Create a empty float32 array
    #     ndmi = np.zeros(src.shape, dtype=rasterio.float32)
    #     ndmi = (b8-b11)/(b8+b11)
        
    #     # Update the metada
    #     profile.update(
    #         dtype=rasterio.float32,
    #         count=1,
    #         compress='lzw')
        
    #     # Save to disk
    #     with rasterio.open(f'{ouput_path}//{b11_file.replace("B11_mskd","NDMI")}',
    #                         "w",
    #                         **profile) as dest:
    #         dest.write(ndmi, indexes=1) 

    # def ndvi(b4_mskd_folder: str,
    #         b8_mskd_folder: str,
    #         b4_file: str,
    #         ouput_path:str):
    #     """
    #     Function to calculate the Normalized Difference Vegetation Index.
    #     Parameters
    #     -----------
    #         b4_mskd_folder: str
    #             Path to the folder where the B04 band files are.
    #         b8_mskd_folder: str
    #             Path to the folder where the B08 band files are.
    #         b4_file: str
    #             Name of the B04 band file.
    #         ouput_path: str
    #             Path to the folder where the NDMI images will be saved.
    #     Returns
    #     -----------
    #         A NDVI image.
    #     """
    #     with rasterio.open(b4_mskd_folder+'\\'+b4_file) as src:
    #         b4 = src.read(1, masked=True).astype('float64')
    #     # The name of the B8 file will be the same as the b4
    #     b8_file = b4_file.replace('B04', 'B08')
    #     #print("B8",b8_mskd_folder+'\\'+b8_file)
    #     with rasterio.open(b8_mskd_folder+'\\'+b8_file) as src:
    #         b8 = src.read(1, masked=True).astype('float64')
    #         profile = src.meta # To get the metadata
        
    #     # Calculate the NDVI
    #     # Create a empty float32 array
    #     ndvi = np.zeros(src.shape, dtype=rasterio.float32)
    #     ndvi = (b8-b4)/(b8+b4)
        
    #     # Update the metada
    #     profile.update(
    #         dtype=rasterio.float32,
    #         count=1,
    #         compress='lzw')
        
    #     # Save to disk
    #     with rasterio.open(f'{ouput_path}//{b4_file.replace("B04_mskd","NDVI")}',
    #                     "w",
    #                     **profile) as dest:
    #         dest.write(ndvi, indexes=1)      

    # # COMPOSITING FUNCTIONS

    # def get_nd_proportion(path_folder:str,
    #                     file: str):
    #     """
    #     Function to calculate proportion of NoData values in an image.
    #     Parameters
    #     -----------
    #         path_folder: str
    #             Path to the folder where the images are.
    #         file: str
    #             Name of the file.
    #     Returns
    #     -----------
    #         The proportion of NoData values from 0 to 1.
    #     """
    #     # Check the percentage of no data
    #     with rasterio.open(path_folder+'\\'+file) as src:
    #         array = src.read(1) 
    #     # Get an array with the no data pixels
    #     array_nd = array[array==0.0]  
    #     # Get the proportion
    #     p = array_nd.size/array.size
    #     #print(p)
    #     return p    

    # def mosaic_rasters(input_folder:str,
    #                 input_list: str,
    #                 output_folder:str):
    #     """
    #     Function to mosaic the tiles of the same date into a single tile.
    #     Parameters
    #     -----------
    #         input_folder: str
    #             Path to the folder where the images to be mosaicked are.
    #         input_list: str
    #             List with the names of the files to be mosaicked.
    #         output_folder: str
    #             Path to the folder where the mosaicked images will be saved.
    #     Returns
    #     -----------
    #         A mosaicked image.
    #     """
    #     src_files_to_mosaic = []
    #     for fp in input_list:
    #         src = rasterio.open(input_folder+'\\'+fp, masked=True)
    #         src_files_to_mosaic.append(src)
        
    #     mosaic, out_trans = merge(src_files_to_mosaic)

    #     # Copy the metadata
    #     out_meta = src.meta.copy()
    #     # Update the metadata
    #     out_meta.update({"driver": "GTiff",
    #                 "height": mosaic.shape[1],
    #                 "width": mosaic.shape[2],
    #                 "transform": out_trans               
    #                 }
    #                 )
    #     # Save to disk
    #     nf = input_list[0]      
    #     with rasterio.open(mosaic_path+nf, "w", **out_meta) as dest:
    #         dest.write(mosaic)
    #     print(f"{nf} saved at {mosaic_path}")

    # def resize_extent(largest_image_path:str,
    #                 image_to_resize:str,
    #                 input_folder_path:str,
    #                 output_folder_path:str):
    #     """
    #     Function to resize the images to the size of the mosaicked images
    #     in order to make possible performing the composites.
    #     Parameters
    #     -----------
    #         largest_image_path: str
    #             Path to the image which will be used as reference to resize the rest
    #             of images.
    #         image_to_resize: str
    #             Name of the file to be resized.
    #         input_folder_path: str
    #             Path to the folder where the images to be resized are.
    #         output_folder: str
    #             Path to the folder where the resized images will be saved.
    #     Returns
    #     -----------
    #         A mosaicked image.
    #     """
    #     # Get the extent of a full size image
    #     info = gdal.Info(largest_image_path, format='json')
    #     #print(info['cornerCoordinates'])

    #     # Get the extent
    #     xmax = info['cornerCoordinates']['lowerRight'][0]
    #     xmin = info['cornerCoordinates']['upperLeft'][0]
    #     ymax = info['cornerCoordinates']['upperLeft'][1]
    #     ymin = info['cornerCoordinates']['lowerRight'][1]
        
    #     # Warp the image
    #     # # Create a folder to store the resize image
    #     # path_to_resize = output_path+'\\'+"mosaic"
    #     # if not os.path.exists(path_to_resize):
    #     #     os.mkdir(path_to_resize)
    #     kwargs = {'outputBounds': [xmin, ymin, xmax, ymax]}
    #     gdal.Warp(destNameOrDestDS=output_folder_path+f"\\{image_to_resize}",
    #             srcDSOrSrcDSTab=input_folder_path+'\\'+image_to_resize,
    #             **kwargs)
    #     print(f"{image_to_resize} saved at {output_folder_path}")
        
    # def median_composite(month:list,
    #                     input_path:str,
    #                     name_output:str,
    #                     output_path:str):
    #     """
    #     Function to create a monthly composite by calculating the median of all the
    #     images that belongs to the same month.
    #     Parameters
    #     -----------
    #         month: str
    #             List of the images belonging to the same month.
    #         input_path: str
    #             Path to the folder where the images to be resized are.
    #         name_output: str
    #             Name for the output file.
    #         output_path: str
    #             Path to the folder where the composited images will be saved.
    #     Returns
    #     -----------
    #         The composite image for a certain month.
    #     """
    #     array_files_composite = []
    #     array_size = []
    #     for fp in month:
    #         src = rasterio.open(input_path+'\\'+fp)
    #         img = src.read()
    #         profile = src.profile
    #         array_size.append(src.shape)
    #         array_files_composite.append(img)
        
    #     # Get the maximum size of the files
    #     max_size = max(array_size,key=lambda item:item[1])
    #     #print(len(array_files_composite))
            
    #     # Create an empty array an fill it with each band within the month
    #     composite_array = np.zeros(shape=max_size + (len(month),))
    #     # Fill the empty array 
    #     for index,img in enumerate(array_files_composite):
    #         try:
    #             composite_array[:,:,index] = img
    #         except Exception as error:
    #             print(error+' at file: '+ fp[index])
                
    #     # The 0 values are NoData in the input images
    #     composite_array[composite_array==0] = np.nan 
    #     #print("composite array", composite_array.shape)
    #     # Calculate the median along the time axis
    #     median = np.nanmedian(composite_array, axis=2)
    #     # Save to disk
    #     with rasterio.open(output_path+f'\\{name_output}_{str(i).zfill(2)}.tif',
    #                     'w',
    #                     **profile) as dest:
    #         dest.write(median, indexes=1)  
            
    # def stack_rasters(input_folder:str,
    #                 files_to_stack:list,
    #                 name_output:str,
    #                 output_folder:str):
    #     """
    #     Function to stack the tiles of the same folder into a single file.
    #     Parameters
    #     -----------
    #         input_folder: str
    #             Path to the folder where the images to be stacked are.
    #         files_to_stack: str
    #             List with the names of the files to be stacked.
    #         name_output: str
    #             Name of the output file.
    #         output_folder: str
    #             Path to the folder where the stack will be saved.
    #     Returns
    #     -----------
    #         A stacked image with many bands as number of images in the input folder.
    #         Each band is tagged with its date.
    #     """
    #     # Read metadata of first file
    #     with rasterio.open(input_folder+'\\'+files_to_stack[0][1]) as src0: # for tuples list
    #         meta = src0.meta

    #     # Update meta to reflect the number of layers
    #     meta.update(count = len(files_to_stack))

    #     # Read each layer and write it to stack
    #     with rasterio.open(output_folder+f'\\{name_output}_stack.tif',
    #                     'w',
    #                     **meta) as dst:
    #         for id, fp in enumerate(files_to_stack, start=1):
    #             with rasterio.open(input_folder+'\\'+fp[1]) as src1: # for tuples list
    #                 dst.write_band(id, src1.read(1))
    #                 # Add date tag to each band
    #                 dst.update_tags(id, TIFFTAG_DATETIME=fp[0])

                    
    # def stack_monthly_rasters(input_folder:str,
    #                         files_to_stack:list,
    #                         name_output:str,
    #                         output_folder:str):
    #     """
    #     Function to stack the tiles of the same folder into a single file.
    #     Parameters
    #     -----------
    #         input_folder: str
    #             Path to the folder where the images to be stacked are.
    #         files_to_stack: str
    #             List with the names of the files to be stacked.
    #         name_output: str
    #             Name of the output file.
    #         output_folder: str
    #             Path to the folder where the stack will be saved.
    #     Returns
    #     -----------
    #         A stacked image with many bands as number of images in the input folder.
    #         Each band is tagged with its corresponding month number.
    #     """
        
    #     # Read metadata of first file
    #     with rasterio.open(input_folder+'\\'+files_to_stack[0]) as src0:
    #         meta = src0.meta

    #     # Update meta to reflect the number of layers
    #     meta.update(count = len(files_to_stack))

    #     # Read each layer and write it to stack
    #     with rasterio.open(output_folder+f'\\{name_output}_monthly_stack.tif',
    #                     'w',
    #                     **meta) as dst:
    #         for id, fp in enumerate(files_to_stack, start=1):
    #             with rasterio.open(input_folder+'\\'+fp) as src1:
    #                 dst.write_band(id, src1.read(1))
    #                 # Add date tag to each band
    #                 dst.update_tags(id, MONTH=fp.split('_')[1][:2])    

    # year = ""
    # # Input polygon for the search
    # file = ""
    # # Buffered input polygon for the 20m bands
    # file20 = ""
    # site=""
    # root_path = f"C:\\PATH\\TO\\FILE\\{site}"

    # # Create the folders
    # year_path = f"{root_path}\\{year}"
    # b4_path= f"{root_path}\\{year}\\B04"
    # b8_path= f"{root_path}\\{year}\\B08"
    # b11_path= f"{root_path}\\{year}\\B11"
    # b11rs_path= f"{root_path}\\{year}\\B11rs"
    # b11b_path= f"{root_path}\\"

