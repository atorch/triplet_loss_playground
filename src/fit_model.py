import io
import imageio
import numpy as np
from scipy.spatial import distance_matrix
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds


def get_model(embedding_dim, n_filters1=128, n_filters2=64, dropout=0.2):

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                filters=n_filters1,
                kernel_size=2,
                padding="same",
                activation="relu",
                input_shape=(28, 28, 1),
            ),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Conv2D(
                filters=n_filters2, kernel_size=2, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling2D(pool_size=2),
            tf.keras.layers.Dropout(dropout),
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

    print(model.summary())

    return model


def _normalize_img(img, label):

    # Normalize values that are originally in [0, 1, ..., 2^8 - 1]
    img = tf.cast(img, tf.float32) / 255.0
    return (img, label)


def describe_distances(predictions_on_new_images, new_image_description, predictions_on_digits, test_classes):

    print("***")
    print(f"Distances between {new_image_description}:")
    print(distance_matrix(predictions_on_new_images, predictions_on_new_images))

    distances = distance_matrix(predictions_on_new_images, predictions_on_digits)
    print(f"Distances between mnist digits and {new_image_description}:")
    print(f"min={distances.min()}, mean={distances.mean()}, median={np.median(distances)}, max={distances.max()}")
    print(f"Min distances between mnist digits and {new_image_description}:")
    print(np.min(distances, axis=1))
    print(f"Classes of the mnist digits closest to {new_image_description}:")
    print(test_classes[np.argmin(distances, axis=1)])


def main(n_epochs=200, embedding_dim=64):

    train_dataset, test_dataset = tfds.load(
        name="mnist", split=["train", "test"], as_supervised=True
    )

    test_as_np = tfds.as_numpy(test_dataset)
    for image, label in test_as_np:
        imageio.imwrite("example_mnist_digit.png", image)
        break  # TODO Save a few more?  Not the full training set, maybe the first batch

    # TODO Try adding some random shifts and small rotations to the training set?  Visualize them
    train_dataset = train_dataset.shuffle(1024).batch(32)
    train_dataset = train_dataset.map(_normalize_img)

    test_dataset = test_dataset.batch(32)
    test_dataset = test_dataset.map(_normalize_img)
    test_classes = np.concatenate([y for x, y in test_dataset], axis=0)

    model = get_model(embedding_dim=embedding_dim)

    history = model.fit(train_dataset, epochs=n_epochs,)

    # Maps images of digits into embedding_dim
    predictions_on_digits = model.predict(test_dataset)

    print("Sanity check: are all model outputs unit vectors, as expected?")
    print(np.allclose(np.linalg.norm(predictions_on_digits, axis=1), 1.0))

    # These are Xs, a shape that was never seen during training
    new_image1 = imageio.imread("new_images/x1.png")
    new_image2 = imageio.imread("new_images/x2.png")
    new_image3 = imageio.imread("new_images/x3.png")
    new_images = np.stack([new_image1, new_image2, new_image3])
    # Match the model's expected input shape
    new_images = np.expand_dims(new_images, axis=-1)
    predictions_on_Xs = model.predict(_normalize_img(new_images, label=None)[0])

    describe_distances(
        predictions_on_new_images=predictions_on_Xs,
        new_image_description="my hand drawn Xs",
        predictions_on_digits=predictions_on_digits,
        test_classes=test_classes,
    )

    # These are 8s that I drew by hand, so they should be close to the 8s from mnist, but let's check
    my_eight1 = imageio.imread("new_images/an_eight_not_from_mnist1.png")
    my_eight2 = imageio.imread("new_images/an_eight_not_from_mnist2.png")
    my_eight3 = imageio.imread("new_images/an_eight_not_from_mnist3.png")
    my_eights = np.stack([my_eight1, my_eight2, my_eight3])
    # Match the model's expected input shape
    my_eights = np.expand_dims(my_eights, axis=-1)
    predictions_on_eights = model.predict(_normalize_img(my_eights, label=None)[0])

    describe_distances(
        predictions_on_new_images=predictions_on_eights,
        new_image_description="my hand drawn 8s",
        predictions_on_digits=predictions_on_digits,
        test_classes=test_classes,
    )

    print("Distances between my hand-drawn 8s and Xs:")
    print(distance_matrix(predictions_on_eights, predictions_on_Xs))

    # TODO /usr/local/lib/python3.6/dist-packages/tensorflow_addons/utils/ensure_tf_install.py:67: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.4.0 and strictly below 2.7.0 (nightly versions are not supported).
    # The versions of TensorFlow you are currently using is 2.3.0 and is not supported.


if __name__ == "__main__":
    main()
