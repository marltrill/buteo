import sys, os

sys.path.append("../../")
sys.path.append("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin/")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

import numpy as np
from glob import glob
from osgeo import gdal
from buteo.raster.io import stack_rasters, raster_to_array
from buteo.machine_learning.patch_extraction_v2 import predict_raster
from buteo.raster.io import (
    raster_to_array,
    array_to_raster,
    stack_rasters_vrt,
)
from buteo.raster.clip import internal_clip_raster, clip_raster
from buteo.machine_learning.ml_utils import (
    preprocess_optical,
    preprocess_sar,
    tpe,
    get_offsets,
)

raster_folder = "C:/Users/MALT/Desktop/set1_patches_input/"
model_folder = "C:/Users/MALT/Desktop/inputs_training/models/"
model = model_folder + "egypt_areaclone5"
outdir = raster_folder + "predictions_pilotarea/"
image_id = "1013"

for region in glob(raster_folder + "grid/*.gpkg"):
    region_name = os.path.splitext(os.path.basename(region))[0]

    print(f"Processing region: {region_name}")

    # if region_name in ["id_1", "id_10", "id_11"]:
    #     continue

    print("Clipping RESWIR.")
    b20m_clip = internal_clip_raster(
        raster_folder + f"B05_{image_id}_20m.tif",
        region,
        adjust_bbox=False,
        all_touch=False,
        out_path="/vsimem/20m_clip.tif",
    )

    reswir = clip_raster(
        [
            raster_folder + f"B05_{image_id}_20m.tif",
            raster_folder + f"B06_{image_id}_20m.tif",
            raster_folder + f"B07_{image_id}_20m.tif",
            raster_folder + f"B11_{image_id}_20m.tif",
            raster_folder + f"B12_{image_id}_20m.tif",
        ],
        clip_geom=region,
        adjust_bbox=False,
        all_touch=False,
    )

    print("Stacking RESWIR.")
    reswir_stack = []
    for idx, raster in enumerate(reswir):
        reswir_stack.append(
            array_to_raster(
                preprocess_optical(
                    raster_to_array(reswir[idx]),
                    target_low=0,
                    target_high=1,
                    cutoff_high=8000,
                ),
                reference=reswir[idx],
            ),
        )
    reswir_stacked = stack_rasters(reswir_stack, dtype="float32")
    for raster in reswir:
        gdal.Unlink(raster)

    print("Clipping RGBN.")
    b10m_clip = internal_clip_raster(
        raster_folder + f"B04_{image_id}_10m.tif",
        b20m_clip,
        adjust_bbox=False,
        all_touch=False,
        out_path="/vsimem/10m_clip.tif",
    )
    rgbn = clip_raster(
        [
            raster_folder + f"B02_{image_id}_10m.tif",
            raster_folder + f"B03_{image_id}_10m.tif",
            raster_folder + f"B04_{image_id}_10m.tif",
            raster_folder + f"B08_{image_id}_10m.tif",
        ],
        clip_geom=b20m_clip,
        adjust_bbox=False,
        all_touch=False,
    )

    print("Stacking RGBN.")
    rgbn_stack = []
    for idx, raster in enumerate(rgbn):
        rgbn_stack.append(
            array_to_raster(
                preprocess_optical(
                    raster_to_array(rgbn[idx]),
                    target_low=0,
                    target_high=1,
                    cutoff_high=8000,
                ),
                reference=rgbn[idx],
            ),
        )
    rgbn_stacked = stack_rasters(rgbn_stack, dtype="float32")
    for raster in rgbn:
        gdal.Unlink(raster)

    print("Clipping SAR.")
    sar = clip_raster(
        [
            raster_folder + f"VV_{image_id}_10m.tif",
            raster_folder + f"VH_{image_id}_10m.tif",
        ],
        clip_geom=b20m_clip,
        adjust_bbox=False,
        all_touch=False,
    )

    print("Stacking SAR.")
    sar_stack = []
    for idx, raster in enumerate(sar):
        sar_stack.append(
            array_to_raster(
                preprocess_sar(raster_to_array(sar[idx]), target_low=0, target_high=1, convert_db=False),
                reference=sar[idx],
            ),
        )
    sar_stacked = stack_rasters(sar_stack, dtype="float32")
    for raster in sar:
        gdal.Unlink(raster)

    print("Ready for predictions.")

    outname = os.path.splitext(os.path.basename(region))[0]

    predict_raster(
        [rgbn_stacked, sar_stacked, reswir_stacked],
        tile_size=[32, 32, 16],
        output_tile_size=32,
        model_path=model,
        reference_raster=b10m_clip,
        out_path=outdir + f"{outname}.tif",
        offsets=[
            get_offsets(32),
            get_offsets(32),
            get_offsets(16),
        ],
        batch_size=1024,
        output_channels=1,
        scale_to_sum=False,
        method="median",
    )

    try:
        for raster in reswir_stack:
            gdal.Unlink(raster)

        for raster in rgbn_stack:
            gdal.Unlink(raster)

        for raster in sar_stack:
            gdal.Unlink(raster)

        gdal.Unlink(reswir_stacked)
        gdal.Unlink(rgbn_stacked)
        gdal.Unlink(sar_stacked)
        gdal.Unlink(b10m_clip)
    except:
        pass

    #exit() #uncomment if I'm predicting on a particular extent, not a mosaic

#exit() #uncomment if I'm predicting on a particular extent, not a mosaic

print("Creating prediction mosaic.") #predicting for all s2 tiles (the pilot area split into 9)
mosaic = stack_rasters_vrt(
    glob(raster_folder + f"predictions_pilotarea/id_*.tif"),
    "/vsimem/vrt_predictions.vrt",
    seperate=False,
)
mosaic = "/vsimem/vrt_predictions.vrt"

rounded = array_to_raster(
    np.clip(np.rint(raster_to_array(mosaic)), 0, 8000).astype("uint16"), mosaic
)
# internal_clip_raster(
#     rounded,
#     folder + "vector/ghana_buffered_1k.gpkg",
#     out_path=folder + "predictions/Ghana_volume_uint16_v5.tif",
#     dst_nodata=65535,
# )

# internal_clip_raster(
#     mosaic,
#     folder + "vector/ghana_buffered_1k.gpkg",
#     out_path=folder + "predictions/Ghana_volume_float32_v5.tif",
#     dst_nodata=-9999.9,
# )