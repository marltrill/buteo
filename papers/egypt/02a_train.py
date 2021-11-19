# %%
import sys, os

sys.path.append("../../")
sys.path.append("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin/")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

import os
import time
import numpy as np

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    EarlyStopping,
)
from buteo.machine_learning.ml_utils import create_step_decay, tpe, SaveBestModel
from buteo.utils import timing
from buteo.machine_learning.augmentation import image_augmentation
from test_site_predict import predict_model_test

# %%
np.set_printoptions(suppress=True)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

model_name = "egypt_areaclone6"

folder = "C:/Users/MALT/Desktop/inputs_training/"
outdir = folder + "models/"

add_noise = True

x_train = [
    np.concatenate([
        np.load(folder + f"RGBN_1013.npy"),
        np.load(folder + f"RGBN_0918.npy"),
        np.load(folder + f"RGBN_0908.npy"),
    ]),
    np.concatenate([
        np.load(folder + f"SAR_1013.npy"),
        np.load(folder + f"SAR_0918.npy"),
        np.load(folder + f"SAR_0908.npy"),
    ]),
    np.concatenate([
        np.load(folder + f"RESWIR_1013.npy"),
        np.load(folder + f"RESWIR_0918.npy"),
        np.load(folder + f"RESWIR_0908.npy"),
    ]),
]

y_train = np.concatenate([
    np.load(folder + f"label_area_1013.npy"),
    np.load(folder + f"label_area_0918.npy"),
    np.load(folder + f"label_area_0908.npy"),
])

test_set = 2000

x_test = [
    x_train[0][-test_set:],
    x_train[1][-test_set:],
    x_train[2][-test_set:],
]

y_test = y_train[-test_set:]

x_train = [
    x_train[0][:-test_set],
    x_train[1][:-test_set],
    x_train[2][:-test_set],
]

y_train = y_train[:-test_set]

if add_noise:
    x_train, y_train = image_augmentation(x_train, y_train, options={
        "scale": 0.035,
        "band": 0.01,
        "contrast": 0.01,
        "pixel": 0.01,
        "drop_threshold": 0.00,
        "clamp": True,
        "clamp_max": 1,
        "clamp_min": 0,
    })

lr = 0.0001
min_delta = 0.005

with tf.device("/device:GPU:0"):
    # epochs = [5]
    # bs = [64]
    epochs = [10, 10, 5, 5]
    bs = [16, 32, 64, 128]
  
    donor_model_path = outdir + "egypt_areaclone3"

    # version 1 Without momentum from the Ghana model
    donor_model = tf.keras.models.load_model(donor_model_path, custom_objects={"tpe": tpe})
    model = tf.keras.models.clone_model(donor_model)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mse", "mae", tpe],  # tpe
    )

    model.set_weights(donor_model.get_weights())

    # # version 2 With momentum from the Ghana model

    # model = tf.keras.models.load_model(donor_model_path, custom_objects={"tpe": tpe})

    model.evaluate(x=x_test, y=y_test, batch_size=1024) #it first evaluates the accuracy of the donor model

    start = time.time()

    save_best_model = SaveBestModel()

   
    for phase in range(len(bs)):
        use_epoch = np.cumsum(epochs)[phase]
        use_bs = bs[phase]
        initial_epoch = np.cumsum(epochs)[phase - 1] if phase != 0 else 0

        model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_test, y_test),
            shuffle=True,
            epochs=use_epoch,
            initial_epoch=initial_epoch,
            verbose=1,
            batch_size=use_bs,
            use_multiprocessing=True,
            workers=0,
            callbacks=[
                EarlyStopping(            #it gives the model 3 epochs to improve results based on val_loss value, if it doesnt improve-drops too much, the model running
                    monitor="val_loss",   #is stopped. If this continues, it would be overfitting (refer to notes)
                    patience=3,
                    min_delta=min_delta,
                ),
                save_best_model,
            ],
        )
                # LearningRateScheduler(
                #     create_step_decay(
                #         learning_rate=lr,
                #         drop_rate=0.75,
                #         epochs_per_drop=3,
                #     )
                # ),
                # ModelCheckpoint(
                #     filepath=f"{outdir}{model_name.lower()}_" + "{epoch:02d}",
                #     save_best_only=True,
                # ),

    model.set_weights(save_best_model.best_weights)
    
    model.evaluate(x=x_test, y=y_test, batch_size=1024) #it evaluates the accuracy of the model we just created here

    model.save(f"{outdir}{model_name.lower()}")

    predict_model_test(model_name, model_name) #this function saves the output predicted raster into a new folder, for each training run

    timing(start)

# teacher: mse: 104.7100 - mae: 3.4287 - tpe: 1.0075
# student: mse: 97.1954 - mae: 3.3744 - tpe: -0.0915

# from tensorflow.keras.callbacks import History
# import matplotlib.pyplot as plt
# model = egypt_areaclone5
# history = model.fit(...)
# plot_history(history)
# plt.show()
# plot_history(history, path="standard.png")
# plt.close()

# %%
