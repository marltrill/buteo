# %%
import sys, os

sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

from token import DOUBLESTAR
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.raster.clip import clip_raster
from glob import glob

import numpy as np

# %%

predictions = "C:/Users/MALT/Desktop/set3_patches/predictions_pilotarea/"

folder= "D:/NIRAS/pilot_test/vector_data/rasterized_aa/"
truth_alexandria11 = raster_to_array(folder + "buildings_11aa_rasterized_clipped.tif")
truth_alexandria12 = raster_to_array(folder + "buildings_12aa_rasterized_clipped.tif")

for pred_path in glob(predictions + f"model_clipped*.tif"):
    name = os.path.splitext(os.path.basename(pred_path))[0]
    pred = raster_to_array(pred_path)

    if "11aa" in name:
        truth = truth_alexandria11
    elif "12aa" in name:
        truth = truth_alexandria12

    sum_dif = ((np.sum(pred) - np.sum(truth)) / np.sum(truth)) * 100
    mae = np.mean(np.abs(pred - truth))
    mse = np.mean(np.power(pred - truth, 2))
    binary = (((np.rint(pred) == 0) == (np.rint(truth) == 0)).sum() / truth.size) * 100

    # import pdb

    # pdb.set_trace()

    print(name)
    print("TSUM: " + str(np.sum(truth)))
    print("PSUM: " + str(np.sum(pred)))
    print("MAE:  " + str(round(mae, 4)))
    print("MSE:  " + str(round(mse, 4)))
    print("sum:  " + str(round(sum_dif, 4)))
    print("BIN:  " + str(round(binary, 4)))
    print("")

# ("aarhus_area@1" > 0 AND "aarhus_area_32x32_9_overlaps@1" > 0) *
# ((("aarhus_area_32x32_9_overlaps@1" - "aarhus_area@1") + 0.00000001) / ("aarhus_area@1" + 0.00000001)) + ("aarhus_area_32x32_9_overlaps@1" <  0.00000001 and "aarhus_area@1" >  0.00000001) * "aarhus_area@1" * -1

# %%
