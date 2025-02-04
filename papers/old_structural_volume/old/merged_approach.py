from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy

import os
import ml_utils
import numpy as np

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\analysis\\"
size = 320
seed = 42
kfolds = 5
batches = 32
validation_split = 0.3
rotation = False
noise = False
noise_amount = 0.01
msg = f"{str(size)} - rgbn + backscatter + coherence"

def learning_rate_decay(epoch):
  if epoch < 4:
    return 1e-3
  elif epoch >= 3 and epoch < 8:
    return 1e-4
  else:
    return 1e-5

# ***********************************************************************
#                   LOADING DATA
# ***********************************************************************

blue = 0
green = 1
red = 2
nir = 0

# Load and scale RGB channels
X_rgb = np.load(folder + f"{str(int(size))}_rgb.npy").astype('float32')
X_rgb[:, :, :, blue] = ml_utils.scale_to_01(np.clip(X_rgb[:, :, :, blue], 0, 4000))
X_rgb[:, :, :, green] = ml_utils.scale_to_01(np.clip(X_rgb[:, :, :, green], 0, 5000))
X_rgb[:, :, :, red] = ml_utils.scale_to_01(np.clip(X_rgb[:, :, :, red], 0, 6000))

# Load and scale NIR channel (Add additional axis to match RGB)
X_nir = np.load(folder + f"{str(int(size))}_nir.npy").astype('float32')
X_nir = X_nir[:, :, :, np.newaxis]
X_nir[:, :, :, nir] = ml_utils.scale_to_01(np.clip(X_nir[:, :, :, nir], 0, 11000))

# Merge RGB and NIR
X = np.concatenate([X_rgb, X_nir], axis=3)

# Load Backscatter (asc + desc), remove the largest outliers (1% - 99%)
bs = np.load(folder + f"{str(int(size))}_bs.npy")[:, :, :, [ml_utils.sar_class("asc"), ml_utils.sar_class("desc")]]
bs = ml_utils.scale_to_01(np.clip(bs, np.quantile(bs, 0.01), np.quantile(bs, 0.99)))
bs = np.concatenate([
    bs.mean(axis=(1,2)),
    bs.std(axis=(1,2)),
    bs.min(axis=(1,2)),
    bs.max(axis=(1,2)),
    np.median(bs, axis=(1,2)),
], axis=1)

# Load coherence
coh = np.load(folder + f"{str(int(size))}_coh.npy")[:, :, :, [ml_utils.sar_class("asc"), ml_utils.sar_class("desc")]]
coh = np.concatenate([
    coh.mean(axis=(1,2)),
    coh.std(axis=(1,2)),
    coh.min(axis=(1,2)),
    coh.max(axis=(1,2)),
    np.median(coh, axis=(1,2)),
], axis=1)

sar = np.concatenate([bs, coh], axis=1)

X_rgb = None
X_nir = None
bs = None
coh = None

y = np.load(folder + f"{str(int(size))}_y.npy")[:, ml_utils.y_class("volume")]

y = (y * (100 * 100)) / 400 # Small house (100m2 * 4m avg. height)
y = (y >= 1.0).astype('int64')

# ***********************************************************************
#                   PREPARING DATA
# ***********************************************************************

# Rotate and add all images, add random noise to images to reduce overfit.
if rotation is True:
    X = ml_utils.add_rotations(X)
    sar = ml_utils.add_rotations(sar)
    y = np.concatenate([y, y, y, y])

if noise is True:
    X = ml_utils.add_noise(X, noise_amount)

# Find minority class
frequency = ml_utils.count_freq(y)
minority = frequency.min(axis=0)[1]

# Undersample
mask = ml_utils.minority_class_mask(y, minority)
y = y[mask]
X = X[mask]
sar = sar[mask]

# Shuffle
shuffle = np.random.permutation(len(y))
y = y[shuffle]
X = X[shuffle]
sar = sar[shuffle]


mask = None
shuffle = None

# ***********************************************************************
#                   ANALYSIS
# ***********************************************************************

if size == 80:
    kernel_start = (3, 3)
    kernel_mid = (2, 2)
    kernel_end = (2, 2)
elif size == 160:
    kernel_start = (5, 5)
    kernel_mid = (5, 5)
    kernel_end = (3, 3)
else:
    kernel_start = (7, 7)
    kernel_mid = (5, 5)
    kernel_end = (3, 3)


def create_mlp_model(shape, name):
    model_input = Input(shape=shape, name=name)
    model = Dense(512, activation='swish', kernel_initializer='he_normal')(model_input)
    model = BatchNormalization()(model)
    model = Dense(256, activation='swish', kernel_initializer='he_normal')(model)
    model = BatchNormalization()(model)
    model = Dense(128, activation='swish', kernel_initializer='he_normal')(model)
    model = BatchNormalization()(model)

    return (model, model_input)


def create_cnn_model(shape, name):
    model_input = Input(shape=shape, name=name)
    model = Conv2D(64, kernel_size=kernel_start, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model_input)
    model = Conv2D(64, kernel_size=kernel_start, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)

    model = Conv2D(128, kernel_size=kernel_mid, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = Conv2D(128, kernel_size=kernel_mid, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = MaxPooling2D(pool_size=(2, 2))(model)
    model = BatchNormalization()(model)

    model = Conv2D(256, kernel_size=kernel_end, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = Conv2D(256, kernel_size=kernel_end, padding='same', activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = GlobalAveragePooling2D()(model)
    model = BatchNormalization()(model)

    model = Flatten()(model)

    model = Dense(512, activation='swish', kernel_initializer='he_uniform', kernel_constraint=max_norm(3))(model)
    model = BatchNormalization()(model)

    return (model, model_input)


skf = StratifiedKFold(n_splits=kfolds)

scores = []

for train_index, test_index in skf.split(np.zeros(len(y)), y):
    X_train_1, X_test_1 = X[train_index], X[test_index]
    X_train_2, X_test_2 = sar[train_index], sar[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model_graph_1, input_graph_1 = create_cnn_model(ml_utils.get_shape(X_train_1), "sentinel_2")
    model_graph_2, input_graph_2 = create_mlp_model((X_train_2.shape[1],), "sentinel_1")

    model = Concatenate()([
        model_graph_1,
        model_graph_2,
    ])

    model = Dense(512, activation='swish', kernel_initializer='he_uniform')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)

    predictions = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=[
        input_graph_1,
        input_graph_2,
    ], outputs=predictions)

    model.compile(optimizer=Adam(name='Adam'), loss='binary_crossentropy', metrics=[BinaryAccuracy()])

    model.fit(
        x=[
            X_train_1,
            X_train_2,
        ],
        y=y_train,
        epochs=500,
        verbose=1,
        batch_size=batches,
        validation_split=validation_split,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                min_delta=0.01,
                restore_best_weights=True,
            ),
            LearningRateScheduler(learning_rate_decay),
        ]
    )

    loss, acc = model.evaluate([
        X_test_1,
        X_test_2,
    ], y_test, verbose=1)
    print('Test Accuracy: %.3f' % acc)

    scores.append(acc)

mean = np.array(scores).mean()
std = np.array(scores).std()

print(mean, std)

from playsound import playsound; playsound(folder + "alarm.wav")
import pdb; pdb.set_trace()
