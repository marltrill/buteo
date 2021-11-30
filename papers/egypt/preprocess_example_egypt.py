# %%
import sys, os
sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

from buteo.earth_observation.s2_utils import (
    get_tiles_from_geom,
    get_tile_geom_from_name,
    unzip_files_to_folder,
    get_tile_files_from_safe_zip,
)
from buteo.earth_observation.download import download_s2_tile, download_s1_tile
from buteo.earth_observation.s2_mosaic import mosaic_tile_s2, join_s2_tiles
from buteo.earth_observation.s1_mosaic import mosaic_s1, mosaic_s1
from buteo.earth_observation.s1_preprocess import backscatter
from buteo.utils import delete_files_in_folder, make_dir_if_not_exists
from glob import glob
from shutil import rmtree

# %%
# Project folder
folder = "C:/Users/MALT/Desktop/ICZM_sentinel/"

folder_s1_raw = make_dir_if_not_exists(folder + "S1_raw/")
folder_s1_mosaic = make_dir_if_not_exists(folder + "S1_mosaic/")
folder_s2_raw = make_dir_if_not_exists(folder + "S2_raw/")
folder_s2_mosaic = make_dir_if_not_exists(folder + "S2_mosaic/")
folder_raster = make_dir_if_not_exists(folder + "raster/")  # final folder
folder_tmp = make_dir_if_not_exists(folder + "tmp/")

# project area in project EPSG
project_area = folder + "project_area.gpkg"
project_epsg = 32635

# Intersecting S2 tiles
project_tiles = get_tiles_from_geom(project_area)  # names

project_start = "20211001"  # yyyy-mm-dd
project_end = "20211101"  # yyyy-mm-dd
project_start_s1 = "20211004"  # yyyy-mm-dd
project_end_s1 = "20211018"  # yyyy-mm-dd

scihub_username = "marltrill"
scihub_password = "429Marltrill201188"
onda_user = "mtrill20@student.aau.dk"
onda_pass = "ONDA!Marltrill29"

# # Download raw sentinel 2 files
# for tile in project_tiles:
#     download_s2_tile(
#         scihub_username,
#         scihub_password,
#         onda_user,
#         onda_pass,
#         folder_s2_raw,
#         tile,
#         date=(project_start, project_end),
#         clouds=20,
#     )

# Download raw sentinel 1 files
# for tile in project_tiles:
#     download_s1_tile(
#         scihub_username,
#         scihub_password,
#         onda_user,
#         onda_pass,
#         folder_s1_raw,
#         get_tile_geom_from_name(tile),
#         date=(project_start_s1, project_end_s1),
#     )

# Create tile mosaics of the sentinel 2 images
# for tile in project_tiles:

#     # if tile in ["35RLQ", "35RMQ", "35RNQ"]:
#     #     continue

#     # Unzip files
#     unzip_files_to_folder(
#         get_tile_files_from_safe_zip(folder_s2_raw, tile),
#         folder_tmp,
#     )

#     # Create mosaic
#     mosaic_tile_s2(
#         folder_tmp,
#         tile,
#         folder_s2_mosaic,
#         process_bands=[
#             {"size": "10m", "band": "B02"},
#             {"size": "10m", "band": "B03"},
#             {"size": "10m", "band": "B04"},
#             {"size": "20m", "band": "B05"},
#             {"size": "20m", "band": "B06"},
#             {"size": "20m", "band": "B07"},
#             {"size": "20m", "band": "B8A"},
#             {"size": "10m", "band": "B08"},
#             {"size": "20m", "band": "B11"},
#             {"size": "20m", "band": "B12"},
#         ],
#     )

#     # remove the unzipped s2 files
#     tmp_files = glob(folder_tmp + "*.SAFE")
#     for f in tmp_files:
#         rmtree(f)

# # Merge the sentinel 2 mosaics 10m and 20m seperately
# join_s2_tiles(
#     folder_s2_mosaic,
#     folder_raster,
#     folder_s2_mosaic,
#     harmonisation=True,
#     pixel_height=10.0,
#     pixel_width=10.0,
#     nodata_value=None,
#     bands_to_process=[
#         "B02_10m",
#         "B03_10m",
#         "B04_10m",
#         "B08_10m",
#     ],
#     projection_to_match=project_epsg,
# )

# join_s2_tiles(
#     folder_s2_mosaic,
#     folder_raster,
#     folder_s2_mosaic,
#     harmonisation=True,
#     pixel_height=20.0,
#     pixel_width=20.0,
#     nodata_value=None,
#     bands_to_process=[
#         "B05_20m",
#         "B06_20m",
#         "B07_20m",
#         "B8A_20m",
#         "B11_20m",
#         "B12_20m",
#     ],
#     projection_to_match=project_epsg,
# )

# exit()
# delete_files_in_folder(folder_tmp)
# # #Fix: we lost the step 2 .tif files, so we had to create them again from the .dim and convert to dB
# from buteo.earth_observation.s1_preprocess import convert_to_tiff

# dims = glob(folder_tmp + "*step_2.dim")

# for dim in dims:
#     convert_to_tiff(dim, folder_tmp, True)

# #s2_mosaic_B12 = folder_raster + "B12_20m.tif"


# # Preprocess the sentinel 1 images
# zip_files_s1 = glob(folder_s1_raw + "*.zip")
# for idx, image in enumerate(zip_files_s1):
#     try:
#         backscatter(
#             image,
#             folder_tmp,
#             folder_s1_mosaic,
#             extent=s2_mosaic_B12,
#             epsg=project_epsg,
#             decibel=True,
#         )
#     except Exception as e:
#         raise Exception(f"Error with image: {image}, {e}")

#     print(f"Completed {idx+1}/{len(zip_files_s1)}")
folder_in= "C:/Users/MALT/Desktop/Gamma0/"
s2_mosaic_B04 = folder_raster + "B04_10_left.tif"
vv_paths = glob(folder_in + "*_Gamma0_VV.tif")
vh_paths = glob(folder_in + "*_Gamma0_VH.tif")

# Mosaic the sentinel 1 images
mosaic_s1(
    vv_paths,
    vh_paths,
    folder_raster,
    folder_in,
    s2_mosaic_B04,
    chunks=3,
)
