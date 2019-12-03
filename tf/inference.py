import numpy as np
import tensorflow as tf

from tf.Model import load_dataset, prepare_dataset

if __name__ == '__main__':
    path_to_folder = "../results/models/2019-12-03_14-44/"
    checkpoint = tf.train.latest_checkpoint(path_to_folder)

    # load model architecture (with last trained weights)
    model = tf.keras.models.load_model(path_to_folder+"model/")

    # load desired weights from specified checkpoint
    model.load_weights(checkpoint)

    batch_size = 256

    states = ["neporuseno", "poruseno"]
    for state in states:
        validation_path = f"../data/validace/{state}/"

        X = np.load(validation_path+"X.npy")
        y = np.load(validation_path+"y.npy")

        (ndata, nfft, nacc), ds = load_dataset([validation_path])
        ds = prepare_dataset(ds, ndata=ndata, batch_size=batch_size, nclasses=2)

        predictions = model.predict(X, batch_size=batch_size)

        loss, accuracy = model.evaluate(ds, verbose=0, steps=ndata//batch_size+1)

        print(f"{state}\nloss: {loss:.4f} \naccuracy: {accuracy*100:.2f} %")

    model.build((ndata, nfft, nacc))
    model.summary()

