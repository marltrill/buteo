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

trainingsites= "D:/NIRAS/pilot_test/trainingsites.gpkg"
buildings = "D:/NIRAS/pilot_test/allbuildings_.gpkg"
buildings_rasterized= "D:/NIRAS/pilot_test/preprocessing_steps/rasterized/buildings_rasterized.tif"
buildings_resampled= "D:/NIRAS/pilot_test/preprocessing_steps/rasterized/buildings_rasterized_resampled.tif"
buildings_rr_final2= "D:/NIRAS/pilot_test/preprocessing_steps/rasterized/buildings_rr_final.tif"

print("rasterizing vector.")
# rasterize_vector(
#         buildings,
#         2,
#         out_path= buildings_rasterized,
#         extent=trainingsites,
#     )
try:
        rasterize_vector(
            buildings,
            0.5,
            out_path= buildings_rasterized,
            extent=trainingsites,
        )
except:
        rasterize_vector(
            trainingsites,
            0.5,
            out_path=buildings_rasterized,
            extent=trainingsites,
            fill_value=0,
            burn_value=0,
        )
print("resampling.")
internal_resample_raster(
        buildings_rasterized,
        10.0,
        resample_alg="average",
        out_path= buildings_resampled,
    )
print("writing final output.")
array_to_raster(
        (raster_to_array(buildings_resampled) * 100).astype(
            "float32"
        ),
        reference=buildings_resampled,
        # out_path=folder + f"fid_{number}_rasterized.tif",
        out_path=buildings_rr_final2,
    )

gdal.Unlink(buildings_rasterized)
gdal.Unlink(buildings_resampled)

# %%