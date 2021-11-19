#%%
buteo_follow = "D:/NIRAS/pilot_test/buteo/"
import os
import sys 
sys.path.append(buteo_follow)
sys.path.append(os.path.join(buteo_follow, "buteo/"))


import math
import time
import numpy as np
from glob import glob
from osgeo import gdal, ogr
import pandas as pd
import matplotlib.pyplot as plt

# import ml_utils 
from machine_learning.patch_extraction import extract_patches
from raster.io import raster_to_array, raster_to_metadata


#%%

folder = "D:/NIRAS/pilot_test/Sentinel/dB_rasters/"
out = "D:/NIRAS/pilot_test/Sentinel/patches/"
raster = folder + "Subset_S1A_IW_GRDH_1SDV_20210927T040103_20210927T040128_039862_04B743_8EDC_Orb_NR_Cal_TF_TC_SRTM_Stack_dB.tif"
vector = out + "trainingsites.gpkg"

# ##check raster
# raster1 = gdal.Open(raster)
# band = raster1.GetRasterBand(1)

# min = band.GetMinimum()
# max = band.GetMaximum()

#%%
# raster1 = None
# band = None

# ##check vector
# geom = ogr.Open(vector, 1)
# layer = geom.GetLayer()
# featureCount = layer.GetFeatureCount()

#%%
# geom = None
# layer = None
# featureCount = None
#%%

###extract geometry patches

path_np, path_geom = extract_patches(
        raster,
        out_dir=out,
        prefix="",
        postfix="_patches_dsm",
        size=64,
        offsets=None,
        generate_grid_geom=True,
        generate_border_patches=True,
        clip_geom=vector,
        verify_output=True,
        verification_samples=100,
        verbose=1,
    )

##Visualize array to double check
arr = np.load(out + "test_area_dtm_tests.npy")

for i in range(5):
    plt.subplot(330+1+i)
    plt.imshow(np.reshape(arr[i], (64,64)))

# %%