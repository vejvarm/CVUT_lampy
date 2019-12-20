import os

import numpy as np
import tensorflow as tf

from datetime import datetime

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import InputLayer, Layer, Conv1D, Dense, Softmax, Flatten, Dropout, BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau

# global settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')


def prepare_dataset(ds, ndata, batch_size, nclasses):
    ds = ds.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), depth=nclasses)), num_parallel_calls=4)
    ds = ds.batch(batch_size)
    ds = ds.shuffle(ndata)
    ds = ds.repeat()
    return ds


def load_dataset(paths):
    # load first part of data (it has shape (ndata, nfft/2, naccs)
    X = np.load(paths[0] + "/X.npy")
    y = np.load(paths[0] + "/y.npy")
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    ndata = X.shape[0]

    for i in range(1, len(paths)):
        X = np.load(paths[i] + "/X.npy")
        y = np.load(paths[i] + "/y.npy")
        ndata += X.shape[0]
        ds = ds.concatenate(tf.data.Dataset.from_tensor_slices((X, y)))

    data_shape = (ndata, X.shape[1], X.shape[2])

    return data_shape, ds


def build_model(input_shape, nfilters, kernel_sizes, strides, drop_rates, activation="relu", nclasses=2,
                loss=CategoricalCrossentropy(), optimizer=Adam(), metrics=("accuracy", )):

    # initialize sequetial model
    model = Sequential()

    # define input shapes
    model.add(InputLayer(input_shape=input_shape))

    # add convolutional layers with activation and dropout
    for f, k, s, d in zip(nfilters, kernel_sizes, strides, drop_rates):
        model.add(Conv1D(f, k, s, activation=activation))
        model.add(Dropout(d))

    # flatten the conv outputs
    model.add(Flatten())

    # add final Dense layer with softmax
    model.add(Dense(nclasses, activation='softmax'))

    # define loss, optimizer and calculated metrics and compile the model
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    return model


if __name__ == '__main__':
    # PARAMS
    nclasses = 2
    nepochs = 50
    batch_size = 32
    init_lr = 0.001
    min_lr = 0.0000001

    # CONV layers
    nfilters = [8, 8, 8, 8, 8]
    kernel_sizes = [8]*len(nfilters)
    strides = [2]*len(nfilters)
    bn = False
    drop_rates = [0.1]*len(nfilters)

    # LOGGING
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")
    root = ".."
    result_dir = f"{root}\\results\\models\\{timestamp}\\"
    log_dir = f"{result_dir}\\logs"

    # PREPARE TRAINING DATA ---------------------------------------
    folder = "trening"
    dataset = ["neporuseno", "neporuseno2", "poruseno"]
    period = "2months"  # week or 2months
    paths = [f"{root}/data/{folder}/{d}/{period}" for d in dataset]

    (ndata_train, nfft, naccs), ds_train = load_dataset(paths)  # load training dataset
    ds_train = prepare_dataset(ds_train, ndata_train, batch_size, nclasses)  # prepare dataset object for training
    # -----------------------------------------------------

    # PREPARE VALIDATION DATA -------------------------------------
    folder = "validace"
    dataset = ["neporuseno", "poruseno"]
    paths = [f"{root}/data/{folder}/{d}" for d in dataset]

    (ndata_valid, _, _), ds_valid = load_dataset(paths)  # load validation dataset
    ds_valid = prepare_dataset(ds_valid, ndata_valid, batch_size, nclasses)  # prepare dataset object for validation

    # -----------------------------------------------------

    # MODEL ------------------------------------------------
    # define loss, optimizer and metrics
    loss = CategoricalCrossentropy()  # define cross entropy loss
    optimizer = Adam(learning_rate=init_lr, amsgrad=True)  # define optimizer
    metrics = ["accuracy"]

    # build and compile the model
    model = build_model((nfft, naccs), nfilters, kernel_sizes, strides, drop_rates, nclasses=nclasses,
                        loss=loss, optimizer=optimizer, metrics=metrics)

    model.summary()

    # save model architecture for future loading
    model.save(result_dir+"model")


    # define callbacks
    reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", min_lr=min_lr)
    early_stopping = EarlyStopping("accuracy", patience=5, mode="max")
    checkpoint = ModelCheckpoint(filepath=result_dir+"checkpoint-{epoch:04d}.ckpt", monitor="val_accuracy",
                                 save_best_only=True, save_weights_only=True, mode="max")
    tensor_board = TensorBoard(log_dir=log_dir)

    # train model
    model.fit(ds_train,
              epochs=nepochs,
              steps_per_epoch=ndata_train//batch_size + 1,
              verbose=2,
              validation_data=ds_valid,
              validation_steps=ndata_valid//batch_size + 1,
              callbacks=[reduce_lr, early_stopping, checkpoint, tensor_board])

    # save model and it's trained weights (saved using callbacks)
#    tf.keras.models.save_model(model, f"{root}/results/models")