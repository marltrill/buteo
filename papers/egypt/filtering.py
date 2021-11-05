# %%
import sys, os

sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

from buteo.raster.io import raster_to_array, array_to_raster
from buteo.filters.convolutions import filter_array
from glob import glob

folder = "C:/Users/MALT/Desktop/Subset_S1B_2021Sept09/"
proj =  "D:/NIRAS/pilot_test/pilotarea.gpkg"

images = glob(folder + "*VH*_aligned*.tif")
arr = raster_to_array(images)

filtered = filter_array(
    arr,
    (3, 3, 3),
    distance_calc=None,
    operation="median",
)

array_to_raster(filtered, images[0], out_path=folder + "VH_2021Sept09_10m.tif")

# %%
