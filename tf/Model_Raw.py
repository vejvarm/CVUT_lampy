import os

import numpy as np
import tensorflow as tf

from datetime import datetime

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import InputLayer, Layer, Conv1D, Dense, Softmax, Flatten, Dropout, BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from scipy.io import loadmat

# global settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.keras.backend.set_floatx('float64')


def get_paths(path_list, ext):

    paths = []

    for pth in path_list:
        if ext in pth:
            paths.append(pth)
        else:
            path_gen = os.walk(pth)
            for p, sub, files in path_gen:
                paths.extend([os.path.join(p, file) for file in files if ext in file])
    return paths


def _mat2ds(path):
    """
    expected structure of data:
        :key 'Acc1': 2D float array [15360, 144] (1st lamp, 1st direction)
        :key 'Acc2': 2D float array [15360, 144] (1st lamp, 2nd direction)
        ...
        :key 'Acc6': 2D float array [15360, 144] (3rd lamp, 2nd direction)
        :key 'FrekvenceSignalu': 1D uint16 array [1] (512 Hz)
        :key 'WindDirection': 1D string array [144] (dir. of wind described by characters N, S, E, W)
        :key 'WindSpeed': 1D float array [144] (speed of the wind)

    :return:
    """
    naccs = 6
    nlamps = 3

    df = loadmat(path)

    accs = np.array([df[f'Acc{i}'] for i in range(1, naccs+1)])  # [naccs, nsamples, nmeas]
    naccs, nsamples, nmeas = accs.shape

    accs = np.transpose(accs, (2, 1, 0))  # [nmeas, nsamples, naccs]
    accs = np.split(accs, nlamps, axis=-1)  # split to 'nlamps' arrays of shape [nmeas, nsamples, naccs//nlamps]
    accs = np.vstack(accs)  # stack split arrays on top of each other (axis 0) [nmeas*nlamps, nsamples, naccs//nlamps]

    # make labels
    if "neporuseno" in path:
        labels = np.zeros((nmeas*nlamps, ), dtype=np.float32)
    else:
        labels = np.concatenate((np.zeros((nmeas, ), dtype=np.float32), np.ones((nmeas*(nlamps-1), ), dtype=np.float32)))

    ds = tf.data.Dataset.from_tensor_slices((accs, labels))

    return accs.shape, ds


def load_dataset(folder_or_file_path, ext=".mat"):

    paths = get_paths(folder_or_file_path, ext)

    ds = None
    nsamples = None
    nsamples = None
    naccs = None
    for path in paths:
        if ds:
            (ns, _, _), ds2 = _mat2ds(path)
            ds = ds.concatenate(ds2)  # concatenate existing ds with newly acquired ds
            nsamples += ns
        else:
            (nsamples, nsamples, naccs), ds = _mat2ds(path)  # create new ds from path

    data_shape = (nsamples, nsamples, naccs)

    return data_shape, ds


def prepare_dataset(ds, ndata, batch_size, nclasses):
    ds = ds.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), depth=nclasses)), num_parallel_calls=4)
    ds = ds.batch(batch_size)
    ds = ds.shuffle(ndata)
    ds = ds.repeat()
    return ds


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


# TODO: TFRecord datasets
# TODO: Implement GridSearchCV


if __name__ == '__main__':
    # PARAMS
    nclasses = 2
    nepochs = 50
    batch_size = 32
    init_lr = 0.001
    min_lr = 0.0000001

    # CONV layers
    nfilters = [8, 8, 8, 8, 8, 8, 8]
    kernel_sizes = [8]*len(nfilters)
    strides = [2]*len(nfilters)
    bn = False
    drop_rates = [0.1]*len(nfilters)

    # LOGGING
    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")
    root = "D:\\!private\\Lord\\Git\\CVUT_lampy"
    result_dir = f"{root}\\results\\models\\{timestamp}\\"
    log_dir = f"{result_dir}\\logs"

    # PREPARE TRAINING DATA ---------------------------------------
    folder = "trening"
    dataset = ["neporuseno", "neporuseno2", "poruseno"]
    period = "week"  # week or 2months
    paths = [f"{root}/data/{folder}/{d}/{period}" for d in dataset]

    (ndata_train, nsamples, naccs), ds_train = load_dataset(paths)  # load training dataset
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
    model = build_model((nsamples, naccs), nfilters, kernel_sizes, strides, drop_rates, nclasses=nclasses,
                        loss=loss, optimizer=optimizer, metrics=metrics)

    model.summary()

    # save model architecture for future loading
    model.save(result_dir+"model")

    # TODO: GRID SEARCH

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