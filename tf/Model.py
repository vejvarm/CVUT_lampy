import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Conv1D, Dense, Softmax, Flatten, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

class ConvDropout(Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 drop_rate=0.0,
                 **kwargs):
        super(ConvDropout, self).__init__()

        self.conv = Conv1D(filters, kernel_size, strides, padding, data_format, dilation_rate, activation,
                           use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer,
                           activity_regularizer, kernel_constraint, bias_constraint, **kwargs)
        self.drop = Dropout(drop_rate)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        if training:
            x = self.drop(x)
        return x


class CNNModel(Model):
    # 1D deep CNN model
    def __init__(self, nfilters, kernel_size, strides=1, padding="valid",
                 dilation_rate=1, activation="relu", nclasses=2, drop_rate=0.1):
        super(CNNModel, self).__init__()

        self.nclasses = nclasses
        self.nlayers = len(nfilters)
        self.dlayers = dict()

        self.conv_drop = list()

        for i in range(self.nlayers):
            self.conv_drop.append(ConvDropout(nfilters[i], kernel_size=kernel_size, strides=strides, padding=padding,
                                              data_format="channels_last", dilation_rate=dilation_rate,
                                              activation=activation, drop_rate=drop_rate))

        # flatten to two dimensions
        self.flatten = Flatten()

        # FF Softmax layer
        self.out = Dense(self.nclasses, activation='softmax')

    def call(self, x, training=False):
        # Apply Conv layers with dropout
        for i in range(self.nlayers):
            # noinspection PyCallingNonCallable
            x = self.conv_drop[i](x)
        # Flatten conv output
        x = self.flatten(x)
        # Apply FF layer with softmax
        return self.out(x)


def prepare_dataset(ds, ndata, batch_size, nclasses):
    ds = ds.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), depth=nclasses)), num_parallel_calls=4)
    ds = ds.batch(batch_size)
    ds = ds.shuffle(ndata)
    ds = ds.repeat()
    return ds


def load_dataset(paths):
    # load first part of data (is has shape (ndata, nfft/2, naccs)
    X = np.load(paths[0] + "/X.npy")
    y = np.load(paths[0] + "/y.npy")
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    ndata = X.shape[0]

    for i in range(1, len(paths)):
        X = np.load(paths[i] + "/X.npy")
        y = np.load(paths[i] + "/y.npy")
        ndata += X.shape[0]
        ds = ds.concatenate(tf.data.Dataset.from_tensor_slices((X, y)))

    return ndata, ds

# TODO: Implement GridSearchCV


if __name__ == '__main__':
    # PARAMS
    nclasses = 2
    nepochs = 100
    batch_size = 64

    # CONV layers
    nfilters = [8, 16, 32, 64]
    kernel_size = 16
    strides = 2
    drop_rate = 0.1

    nfft = 2560
    naccs = 2

    # TensorBoard
    log_dir = "..\\results\\models\\tb\\logs"

    # PREPARE TRAINING DATA ---------------------------------------
    tf.keras.backend.set_floatx('float64')

    root = "D:/!private/Lord/Git/CVUT_lampy"
    folder = "trening"
    dataset = ["neporuseno", "neporuseno2", "poruseno"]
    period = "2months"  # week or 2months
    paths = [f"{root}/data/{folder}/{d}/{period}" for d in dataset]

    ndata_train, ds_train = load_dataset(paths)  # load training dataset
    ds_train = prepare_dataset(ds_train, ndata_train, batch_size, nclasses)  # prepare dataset object for training
    # -----------------------------------------------------

    # PREPARE VALIDATION DATA -------------------------------------
    folder = "validace"
    dataset = ["neporuseno", "poruseno"]
    paths = [f"{root}/data/{folder}/{d}" for d in dataset]

    ndata_valid, ds_valid = load_dataset(paths)  # load validation dataset
    ds_valid = prepare_dataset(ds_valid, ndata_valid, batch_size, nclasses)  # prepare dataset object for validation

    # -----------------------------------------------------

    # MODEL ------------------------------------------------
    # instantiate model
    model = CNNModel(nfilters, kernel_size, strides, nclasses=nclasses, drop_rate=drop_rate)

    loss = CategoricalCrossentropy()  # define cross entropy loss
    optimizer = Adam()  # define optimizer

    # compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # build model for summary
    model.build((ndata_train, nfft, naccs))

    model.summary()

    # TODO: GRID SEARCH

    # define callbacks
    early_stopping = EarlyStopping("accuracy", patience=10, mode="max")
    tensor_board = TensorBoard(log_dir=log_dir)

    # train model
    model.fit(ds_train,
              epochs=nepochs,
              steps_per_epoch=ndata_train//batch_size + 1,
              verbose=1,
              validation_data=ds_valid,
              validation_steps=ndata_valid//batch_size + 1,
              callbacks=[early_stopping, tensor_board])

    # save model and it's trained weights
    tf.keras.models.save_model(model, f"{root}/results/models")