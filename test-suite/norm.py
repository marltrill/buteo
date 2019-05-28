import sys
import os
import numpy as np
import time
from glob import glob

sys.path.append('../lib')
from raster_stats import raster_stats
from raster_to_array import raster_to_array
from array_to_raster import array_to_raster

before = time.time()

base = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\MSI\\'
out_folder = 'D:\\PhD\\Projects\\SavingsAtTheFrontiers\\pilot_analysis\\data\\_temp_\\'

bands = glob(f"{base}*_B*.tif")

for path in bands:
    band_name = path.rsplit('\\')[-1]
    stats = raster_stats(path, statistics=['q98', 'q02', 'min'])
    ratio = 1000 / (stats['q98'] - stats['q02'])
    band = raster_to_array(path)
    zeroed = np.subtract(band, stats['min'])
    scaled_band = np.multiply(zeroed, ratio)
    array_to_raster(scaled_band, reference_raster=path, out_raster=f"{out_folder}scaled_{band_name}")


after = time.time()
print((after - before) / 60)
