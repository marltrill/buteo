# %%
import sys, os

sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

from glob import glob
from buteo.machine_learning.patch_extraction_v2 import extract_patches
# %%

# 10 m resolution raster files (s1 + s2), tile_size= 32
# 20 m resolution raster files (s2), tile_size= 16
# I had to create different output directories, otherwise it will rename it for the 20m 

folder= "C:/Users/MALT/Desktop/set3_patches/"
m10 = glob(folder + "*10m*.tif")
m20 = glob(folder + "*20m*.tif")
out_path10= "C:/Users/MALT/Desktop/set3_patches/out_patches_10m/"
out_path20= "C:/Users/MALT/Desktop/set3_patches/out_patches_20m/"
training_sites= "C:/Users/MALT/Desktop/set3_patches/trainingsites.gpkg"
buildings= "C:/Users/MALT/Desktop/set3_patches/allbuildings_.gpkg" 

extract_patches(
    m20,
    out_path20,
    tile_size=16,
    zones=training_sites,
    options=
        { "label_geom": buildings },
)
# %%