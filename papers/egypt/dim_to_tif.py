#%%
import sys, os
sys.path.append("../../")
from buteo.filters.convolutions import hood_sigma_lee
##%%

from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.reproject import reproject_raster
from buteo.raster.clip import clip_raster
from buteo.raster.align import rasters_are_aligned
from buteo.utils import progress

import numpy as np
from osgeo import gdal
from glob import glob

#this line is just for GDAL to work, set my environment variable here
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

# folder = "D:/NIRAS/pilot_test/Sentinel/S1_set1/S1_PROCESSED/Subset_S1B_IW_GRDH_1SDV_20211003T040027_20211003T040051_028966_0374E5_9A4B_Cal_TF_TC_Stack.data/"
folder = "C:/Users/MALT/Desktop/Subset_S1B_2021Sept09/"
proj =  "D:/NIRAS/pilot_test/pilotarea.gpkg"

images = glob(folder + "*VV*.img") #read multiple files that have the same format

#"*.img")
for idx, img in enumerate(images):
    progress(idx, len(images), "converting")
    name = os.path.splitext(os.path.basename(img))[0] #split text removes the last part of the name, the 1st name is 0, the last name is the extension

    print("reprojecting raster.")
    reprojected_raster = reproject_raster(img, proj)

    print("clipping raster.")
    clipped = clip_raster(reprojected_raster, proj, dst_nodata=0)

    print("converting to db.")
    img_arr = np.abs(np.nan_to_num(raster_to_array(clipped).filled(0))) #set all values to absolute values and zero, so that the log function doesn't get negative values and crashes
    with np.errstate(divide='ignore'):
        img_db = np.where(img_arr > 0.000001, 10 * np.log10(img_arr), 0)

    print("writing raster.")
    array_to_raster(
        img_db,
        clipped,
        out_path=folder + name + "_db.tif",
        creation_options=["compress=none"],
    )
# #this unlinks those variables, if I don't need them, so that I'm not using a lot of RAM (commented them bc of an error)
#     gdal.unlink(reprojected_raster)
#     gdal.unlink(clipped)

    progress(idx + 1, len(images), "converting")

#THIS DIDN'T WORK, SO WE DID IT IN QGIS

# s2_path = "D:/NIRAS/pilot_test/Sentinel/S2A_MSIL2A_20211013T083921_N0301_R064_T35RQQ_20211013T113537.SAFE/GRANULE/L2A_T35RQQ_A032951_20211013T084318/IMG_DATA/R10m/convT35RQQ_20211013T083921_B04_10m.tif"
# print(rasters_are_aligned(glob(folder + "*_aligned.tif")))


# %%
#Are the rasters aligned? (after the QGIS alignment step?)
folder2 = "C:/Users/MALT/Desktop/Subset_S1B_2021Sept09/"
images = glob(folder2 + "*_aligned*.tif")
rasters_are_aligned(images, same_extent=False, same_dtype=False, same_nodata=False)
# %%