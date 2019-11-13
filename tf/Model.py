import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, Dense, Softmax, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy

class CNNModel(Model):
    # 1D deep CNN model
    def __init__(self):
        super(CNNModel, self).__init__()

        self.nclasses = 2

        # CONV layers
        nfilters = [8, 16, 32]
        self.nlayers = len(nfilters)
        kernel_size = 16
        strides = 2
        padding = "valid"
        dilation_rate = 1
        activation = 'relu'
        self.conv = list()
        for i in range(self.nlayers):
            self.conv.append(Conv1D(nfilters[i],
                                    kernel_size=kernel_size,
                                    strides=strides,
                                    padding=padding,
                                    data_format="channels_last",
                                    dilation_rate=dilation_rate,
                                    activation=activation))
        # flatten to two dimensions
        self.flatten = Flatten()

        # FF Softmax layer
        self.out = Dense(self.nclasses, activation='softmax')


    def call(self, x):
        # Apply Conv layers
        for i in range(self.nlayers):
            x = self.conv[i](x)
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

if __name__ == '__main__':
    # PARAMS
    nclasses = 2
    nepochs = 5
    batch_size = 32

    # TRAINING DATA ---------------------------------------
    root = "D:/!private/Lord/Git/CVUT_lampy"
    dataset = ["neporuseno", "neporuseno2", "poruseno"]
    period = "2months"  # week or 2months
    paths = [f"{root}/data/{d}/{period}" for d in dataset]

    # load first part of data (is has shape (ndata, nfft/2, naccs)
    X = np.load(paths[0]+"/X.npy")
    y = np.load(paths[0]+"/y.npy")
    ds_train = tf.data.Dataset.from_tensor_slices((X, y))

    ndata_train = X.shape[0]

    for i in range(1, len(paths)):
        X = np.load(paths[i]+"/X.npy")
        y = np.load(paths[i]+"/y.npy")
        ndata_train += X.shape[0]
        ds_train = ds_train.concatenate(tf.data.Dataset.from_tensor_slices((X, y)))

    # prepare dataset object for training
    ds_train = prepare_dataset(ds_train, ndata_train, batch_size, nclasses)
    # -----------------------------------------------------

    print(ds_train)

    # VALIDATION DATA -------------------------------------
    folder = "validace"
    dataset = ["neporuseno", "poruseno"]
    paths = [f"{root}/data/{folder}/{d}" for d in dataset]

    # neporuseno
    X = np.load(paths[0]+"/X.npy")
    y = np.load(paths[0]+"/y.npy")
    ndata_valid = X.shape[0]
    ds_valid = tf.data.Dataset.from_tensor_slices((X, y))

    # add poruseno
    X = np.load(paths[1]+"/X.npy")
    y = np.load(paths[1]+"/y.npy")
    ndata_valid += X.shape[0]
    ds_valid = ds_valid.concatenate(tf.data.Dataset.from_tensor_slices((X, y)))

    # prepare dataset object for validation
    ds_valid = prepare_dataset(ds_valid, ndata_valid, batch_size, nclasses)
    # -----------------------------------------------------

    # MODEL ------------------------------------------------
    # instantiate model
    model = CNNModel()

    # define cross entropy loss
    loss = CategoricalCrossentropy()

    # compile model
    model.compile(
        optimizer="adam",
        loss=loss,
        metrics=['accuracy'],
    )

    # train model
    model.fit(ds_train,
              epochs=nepochs,
              steps_per_epoch=ndata_train//batch_size + 1,
              verbose=1,
              validation_data=ds_valid,
              validation_steps=ndata_valid//batch_size + 1)

    # save model and it's trained weights
    tf.keras.models.save_model(model, f"{root}/results/models")