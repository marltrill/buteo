# %%
import sys, os

sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

import numpy as np
from osgeo import gdal
from glob import glob
from buteo.machine_learning.ml_utils import (
    preprocess_optical,
    preprocess_sar,
)
# %%

#define folder for set1
folder_10m= "C:/Users/MALT/Desktop/set3_patches/out_patches_10m/"
folder_20m= "C:/Users/MALT/Desktop/set3_patches/out_patches_20m/"
out_path= "C:/Users/MALT/Desktop/set3_patches/normalisation_output/"

def preprocess(
    folder_10m,
    folder_20m,
    out_path,
    low=0,
    high=1,
    optical_top=8000,
):
    b02 = folder_10m + "B02_0918_10m.npy"
    b03 = folder_10m + "B03_0918_10m.npy"
    b04 = folder_10m + "B04_0918_10m.npy"
    b08 = folder_10m + "B08_0918_10m.npy"

    b05 = folder_20m + "B05_0918_20m.npy"
    b06 = folder_20m + "B06_0918_20m.npy"
    b07 = folder_20m + "B07_0918_20m.npy"
    b11 = folder_20m + "B11_0918_20m.npy"
    b12 = folder_20m + "B12_0918_20m.npy"

    vv = folder_10m + "VV_0909_10m.npy"
    vh = folder_10m + "VH_0909_10m.npy"

    target = "area"

    label_area = folder_10m + f"label_{target}_10m.npy"

    area = np.load(label_area)
    shuffle_mask = np.random.permutation(area.shape[0])

    label_out = out_path + f"label_{target}_10m.npy"

    np.save(label_out, area[shuffle_mask])

    rgbn = preprocess_optical(
        np.stack(
            [
                np.load(b02),
                np.load(b03),
                np.load(b04),
                np.load(b08),
            ],
            axis=3,
        )[:, :, :, :, 0],
        target_low=low,
        target_high=high,
        cutoff_high=optical_top,
    )

    np.save(out_path + "RGBN.npy", rgbn[shuffle_mask])

    reswir = preprocess_optical(
        np.stack(
            [
                np.load(b05),
                np.load(b06),
                np.load(b07),
                np.load(b11),
                np.load(b12),
            ],
            axis=3,
        )[:, :, :, :, 0],
        target_low=low,
        target_high=high,
        cutoff_high=optical_top,
    )

    np.save(out_path + "RESWIR.npy", reswir[shuffle_mask])

    sar_stacked = np.stack(
        [
            np.load(vv),
            np.load(vh),
        ],
        axis=3,
    )[:, :, :, :, 0]
    sar = preprocess_sar(
        sar_stacked,
        target_low=low,
        target_high=high,
        convert_db=False, #my s1 rasters are already converted to dB
    )

    np.save(out_path + "SAR.npy", sar[shuffle_mask])

preprocess(folder_10m, folder_20m, out_path)

# %%
