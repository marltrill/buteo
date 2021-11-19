# %%
import sys, os
sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

from osgeo import gdal
from glob import glob
from buteo.raster.align import rasters_are_aligned
from buteo.raster.io import array_to_raster, raster_to_array, raster_to_metadata
from buteo.raster.resample import resample_raster
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)

import numpy as np
# %%

def round_to_decimals(x):
    return f"{np.round(x, 4):.4f}"


def metrics(truth, pred, name, resample=False, target=None):
    if not isinstance(truth, list):
        truth = [truth]
    if not isinstance(pred, list):
        pred = [pred]

    if len(truth) != len(pred):
        raise ValueError("Length of truth and pred must be equal")

    processed_truth = []
    processed_pred = []

    for idx in range(len(truth)):
        if (
            raster_to_metadata(truth[idx])["size"]
            != raster_to_metadata(pred[idx])["size"]
        ):
            print(f"{name} rasters are not the same size")
            return

        if resample:
            truth[idx] = resample_raster(truth[idx], 100, resample_alg="sum")
            pred[idx] = resample_raster(pred[idx], 100, resample_alg="sum")

        arr_truth = raster_to_array(truth[idx])
        arr_pred = raster_to_array(pred[idx])

        mask = np.logical_or(arr_truth == -9999.0, arr_pred == -9999.0)

        arr_truth.mask = mask
        arr_pred.mask = mask

        arr_truth = arr_truth.compressed()
        arr_pred = arr_pred.compressed()

        processed_truth.append(arr_truth.ravel())
        processed_pred.append(arr_pred.ravel())

    tarr = np.concatenate(processed_truth)
    tarr = tarr.ravel()
    parr = np.concatenate(processed_pred)
    parr = parr.ravel()

    mae = round_to_decimals(mean_absolute_error(tarr, parr))
    mse = round_to_decimals(mean_squared_error(tarr, parr))
    tpe = round_to_decimals(((np.sum(parr) - np.sum(tarr)) / np.sum(tarr)) * 100)

    if target == "people": #"area" ?
        tarr = np.array(tarr > 0.01, dtype=np.uint8)
        parr = np.array(parr > 0.01, dtype=np.uint8)
    else:
        tarr = np.array(tarr >= 1.0, dtype=np.uint8)
        parr = np.array(parr >= 1.0, dtype=np.uint8)

    acc = round_to_decimals(accuracy_score(tarr, parr))
    bacc = round_to_decimals(balanced_accuracy_score(tarr, parr))
    prec = round_to_decimals(precision_score(tarr, parr))
    rec = round_to_decimals(recall_score(tarr, parr))
    f1 = round_to_decimals(f1_score(tarr, parr))

    adjust_name = name.ljust(10, " ")

    print(f"{adjust_name} (reg) - MAE: {mae}, MSE: {mse}, TPE: {tpe}")
    print(
        f"{adjust_name} (bin) - ACC: {acc}, BACC: {bacc}, PREC: {prec}, REC: {rec}, F1: {f1}"
    )

target= "area"
resample = True

base_truth= "D:/NIRAS/pilot_test/vector_data/rasterized_aa/Aligned/"
truth_alexandria11 = base_truth + f"11aa_label_{target}.tif"
truth_alexandria12 = base_truth + f"12aa_label_{target}.tif"

base_predictions = "C:/Users/MALT/Desktop/set3_patches/predictions_pilotarea/Aligned/"
pred_alexandria11 = base_predictions + f"11aa_prediction_{target}.tif"
pred_alexandria12 = base_predictions + f"12aa_prediction_{target}.tif"

# Are rasters aligned?
base_truth_11aa= "D:/NIRAS/pilot_test/vector_data/rasterized_aa/Aligned/11aa/"
images = glob(base_truth_11aa + "*.tif")
rasters_are_aligned(images, same_extent=False, same_dtype=False, same_nodata=False)

base_truth_12aa= "D:/NIRAS/pilot_test/vector_data/rasterized_aa/Aligned/12aa/"
images = glob(base_truth_12aa + "*.tif")
rasters_are_aligned(images, same_extent=False, same_dtype=False, same_nodata=False)

# %%

#Pixel Size= 10m (We are not resampling)
resample = False #If we want to resample to 100m pixels, set resample= True
metrics(truth_alexandria11, pred_alexandria11, "Site 11a", resample=resample, target=target)
metrics(truth_alexandria12, pred_alexandria12, "Site 12a", resample=resample, target=target)
metrics(
    [
        truth_alexandria11,
        truth_alexandria12,
    ],
    [
        pred_alexandria11,
        pred_alexandria12,
    ],
    "All",
    resample=resample, #test sites not combined: resample= resample
    target=target, #test sites not combined: target= target, or "area", its the same
)

# %%

# if target == "area":
#     print("")
#     truth_odense = folder + f"odense_label_{target}.tif"
#     truth_bornholm = folder + f"bornholm_label_{target}.tif"

#     pred_odense = folder + f"odense_prediction_{target}.tif"
#     pred_bornholm = folder + f"bornholm_prediction_{target}.tif"

#     metrics(truth_odense, pred_odense, "Odense", resample=resample, target=target)
#     metrics(truth_bornholm, pred_bornholm, "Bornholm", resample=resample, target=target)
#     metrics(
#         [
#             truth_odense,
#             truth_bornholm,
#         ],
#         [
#             pred_odense,
#             pred_bornholm,
#         ],
#         "All",
#         resample= resample, 
#         target=target, 
#     )

