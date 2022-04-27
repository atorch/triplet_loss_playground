import io
import imageio
import numpy as np
from scipy.spatial import distance_matrix
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds


def get_model(embedding_dim=128):

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=64,
                kernel_size=2,
                padding="same",
                activation="relu",
                input_shape=(28, 28, 1),
            ),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=2, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                embedding_dim, activation=None
            ),  # No activation on final dense layer
            tf.keras.layers.Lambda(
                lambda x: tf.math.l2_normalize(x, axis=1)
            ),  # L2 normalize embeddings
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss(),
    )

    return model


def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    return (img, label)


def main(n_epochs=10):

    train_dataset, test_dataset = tfds.load(
        name="mnist", split=["train", "test"], as_supervised=True
    )

    train_dataset = train_dataset.shuffle(1024).batch(32)
    train_dataset = train_dataset.map(_normalize_img)

    test_dataset = test_dataset.batch(32)
    test_dataset = test_dataset.map(_normalize_img)

    model = get_model()

    history = model.fit(train_dataset, epochs=n_epochs,)

    # TODO Save a few of the MNIST inputs as PNGs, to compare to new_images

    # Maps images of digits into embedding_dim
    # TODO Would be interesting to get a column with class labels for test dataset
    predictions_on_digits = model.predict(test_dataset)

    # These are Xs, i.e. they're a type of image that was never seen during training
    new_image1 = imageio.imread("new_images/x1.png")
    new_image2 = imageio.imread("new_images/x2.png")
    new_image3 = imageio.imread("new_images/x3.png")
    new_images = np.stack([new_image1, new_image2, new_image3])

    # Match the model's expected input shape
    new_images = np.expand_dims(new_images, axis=-1)

    new_predictions = model.predict(_normalize_img(new_images, label=None)[0])

    ## Expect these to be nearby (small distance)
    print("Distances between new images (images of Xs):")
    print(np.linalg.norm(new_predictions[0] - new_predictions[1]))
    print(np.linalg.norm(new_predictions[0] - new_predictions[2]))
    print(np.linalg.norm(new_predictions[1] - new_predictions[2]))

    ## Expect these to be far (larger distance)
    print("Distances between new images and digits (min, mean, max):")
    # print(np.linalg.norm(new_predictions[0] - predictions_on_digits[0]))  # Should match distances[0, 0]

    distances = distance_matrix(new_predictions, predictions_on_digits)
    print(f"{distances.min()}, {distances.mean()}, {distances.max()}")

    # TODO /usr/local/lib/python3.6/dist-packages/tensorflow_addons/utils/ensure_tf_install.py:67: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.4.0 and strictly below 2.7.0 (nightly versions are not supported).
    # The versions of TensorFlow you are currently using is 2.3.0 and is not supported.


if __name__ == "__main__":
    main()
