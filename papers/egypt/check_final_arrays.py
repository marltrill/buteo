# %%
import sys, os

sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

import numpy as np
from osgeo import gdal
from glob import glob
# %%
# CHECK NORMALISATION OUTPUT ARRAYS
folder= "C:/Users/MALT/Desktop/set3_patches/normalisation_output/"

checkfile= folder + "RGBN_0918.npy"

array= np.load(checkfile)
print (array)

max_value= np.max(array)
print('Maximum value of the array is',max_value)
min_value= np.min(array)
print('Minimum value of the array is',min_value)

# %%
# CONCATENATE ARRAYS: SAR
# Merge SAR images
folder= "C:/Users/MALT/Desktop/inputs_training/"

SAR1_path= folder + "SAR_0909.npy"
SAR1= np.load(SAR1_path)

SAR2_path= folder + "SAR_0927.npy"
SAR2= np.load(SAR2_path)

SAR3_path= folder + "SAR_1003.npy"
SAR3= np.load(SAR3_path)

SAR_all= np.concatenate((SAR1, SAR2, SAR3), dtype=float)
SAR_out = folder + "SAR_all"
SAR_save= np.save(SAR_out, SAR_all)

#Check output array
print (SAR_all)
max_value= np.max(SAR_all)
print('Maximum value of SAR images is',max_value)
min_value= np.min(SAR_all)
print('Minimum value of SAR images is',min_value)
# %%
# CONCATENATE ARRAYS: RGBN
# Merge RGBN images
folder= "C:/Users/MALT/Desktop/inputs_training/"
RGBN1_path= folder + "RGBN_0908.npy"
RGBN1= np.load(RGBN1_path)

RGBN2_path= folder + "RGBN_0918.npy"
RGBN2= np.load(RGBN2_path)

RGBN3_path= folder + "RGBN_1013.npy"
RGBN3= np.load(RGBN3_path)

RGBN_all= np.concatenate((RGBN1, RGBN2, RGBN3), dtype=float)
RGBN_out = folder + "RGBN_all"
RGBN_save= np.save(RGBN_out, RGBN_all)

#Check output array
print (RGBN_all)
max_value= np.max(RGBN_all)
print('Maximum value of RGBN images is',max_value)
min_value= np.min(RGBN_all)
print('Minimum value of RGBN images is',min_value)


# %%
# %%
# CONCATENATE ARRAYS: RESWIR
# Merge RESWIR images
folder= "C:/Users/MALT/Desktop/inputs_training/"

RESWIR1_path= folder + "RESWIR_0908.npy"
RESWIR1= np.load(RESWIR1_path)
#print(RESWIR1.shape)

RESWIR2_path= folder + "RESWIR_0918.npy"
RESWIR2= np.load(RESWIR2_path)
#print(RESWIR2.shape)

RESWIR3_path= folder + "RESWIR_1013.npy"
RESWIR3= np.load(RESWIR3_path)
#print(RESWIR3.shape)

RESWIR_all= np.concatenate([RESWIR1, RESWIR2, RESWIR3], dtype="float32")
RESWIR_out = folder + "RESWIR_all"
RESWIR_save= np.save(RESWIR_out, RESWIR_all)

#Check output array
print (RESWIR_all)
max_value= np.max(RESWIR_all)
print('Maximum value of RESWIR images is',max_value)
min_value= np.min(RESWIR_all)
print('Minimum value of RESWIR images is',min_value)
print(RESWIR_all.shape)
# %%
# %%
# CONCATENATE ARRAYS: label_area
# Merge label area images
folder= "C:/Users/MALT/Desktop/inputs_training/"
LA1_path= folder + "label_area1_10m.npy"
LA1= np.load(LA1_path)

LA2_path= folder + "label_area2_10m.npy"
LA2= np.load(LA2_path)

LA3_path= folder + "label_area3_10m.npy"
LA3= np.load(LA3_path)

LA_all= np.concatenate((LA1, LA2, LA3), dtype=float)
LA_out = folder + "label_area_all"
LA_save= np.save(LA_out, LA_all)

#Check output array
print (LA_all)
max_value= np.max(LA_all)
print('Maximum value of LA images is',max_value)
min_value= np.min(LA_all)
print('Minimum value of LA images is',min_value)

# %%
# # TRY 
# folder= "C:/Users/MALT/Desktop/inputs_training/"
# SAR_images= glob(folder + "*SAR*.npy")
# SAR_load= np.load(SAR_images)

# #SAR_read= np.read(SAR_path)
# for idx, img in enumerate(SAR_load):
#     SAR_merge= np.concatenate(SAR_load, dtype=float)


# %%
# CHECK INPUT TRAINING ARRAYS
folder= "C:/Users/MALT/Desktop/inputs_training/"

checkfile= folder + "RGBN_0918.npy"

array= np.load(checkfile)
#print (array)


shape= array.shape
print(shape)
dim= array.ndim
print(dim)
size= array.size
print(size)
length= len(array)
print(length)

max_value= np.max(array)
print('Maximum value of the array is',max_value)
min_value= np.min(array)
print('Minimum value of the array is',min_value)
# %%
