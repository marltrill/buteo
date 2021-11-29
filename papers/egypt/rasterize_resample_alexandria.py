# %%
import sys, os

sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

from osgeo import gdal
from glob import glob
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.filters.convolutions import filter_array
from buteo.vector.rasterize import rasterize_vector
from buteo.raster.resample import internal_resample_raster
from buteo.machine_learning.patch_extraction import extract_patches
# %%
# folder = "C:/Users/MALT/Desktop/Subset_S1B_2021Sept09/"
# proj =  "D:/NIRAS/pilot_test/pilotarea.gpkg"

# images = glob(folder + "*VH*_aligned*.tif")

extent_Alex= "D:/NIRAS/Buteo_EgyptData/Extent_Alexandria.gpkg"
buildings = "D:/NIRAS/Buteo_EgyptData/Mixed_1.gpkg"
buildings_rasterized= "D:/NIRAS/Buteo_EgyptData/M_buildings_rasterized_1.tif"
#buildings_rasterized_out= "D:/NIRAS/Buteo_EgyptData/buildings_rasterized_50cm.tif"


print("rasterizing vector.")

#Make sure that the "Class" field supports the data type, in this case dtype="uint8"
rasterize_vector(
    buildings,
    0.5,
    out_path= buildings_rasterized,
    attribute= "Class",
    extent=extent_Alex,
    )

print("writing rasterized 50cm final output.")
array_to_raster(
        (raster_to_array(buildings_rasterized)).astype(
            "float32"
        ),
        reference=buildings_rasterized,
        # out_path=folder + f"fid_{number}_rasterized.tif",
        #out_path=buildings_rasterized_out,
    )
# %%
buildings_rasterized= "D:/NIRAS/Buteo_EgyptData/M_buildings_rasterized_1.tif"
buildings_resampled= "D:/NIRAS/Buteo_EgyptData/buildings_rasterized_resampled.tif"
buildings_rr_final2= "D:/NIRAS/Buteo_EgyptData/M_buildings_resampled_1.tif"

print("resampling.")
internal_resample_raster(
        buildings_rasterized,
        10.0,
        resample_alg="average",
        out_path= buildings_resampled,
    )
print("writing final output.")
array_to_raster(
        (raster_to_array(buildings_resampled)*100).astype(
            "float32"
        ),
        reference=buildings_resampled,
        # out_path=folder + f"fid_{number}_rasterized.tif",
        out_path=buildings_rr_final2,
    )

gdal.Unlink(buildings_rasterized)
gdal.Unlink(buildings_resampled)

# %%