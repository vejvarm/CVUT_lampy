import numpy as np
import tensorflow as tf

from tf.Model import prepare_dataset

if __name__ == '__main__':
    path_to_model = "d:/!private/Lord/Git/CVUT_lampy/results/models/08340acc/"
    model = tf.keras.models.load_model(path_to_model)

    batch_size = 100

    state = "neporuseno"
    validation_path = f"d:/!private/Lord/Git/CVUT_lampy/data/validace/{state}/"

    X = np.load(validation_path+"X.npy")
    y = np.load(validation_path+"y.npy")

    ndata, nfft, nacc = X.shape

    ds = tf.data.Dataset.from_tensor_slices((X, y))

    ds = prepare_dataset(ds, ndata=ndata, batch_size=batch_size, nclasses=2)

    predictions = model.predict(X, batch_size=batch_size)

    loss, accuracy = model.evaluate(ds, verbose=0, steps=ndata//batch_size+1)

    print(f"loss: {loss} \naccuracy: {accuracy}")

