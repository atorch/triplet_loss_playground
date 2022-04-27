import io
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds


def get_model(embedding_dim=128):

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(embedding_dim, activation=None), # No activation on final dense layer
        tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss(),
    )

    return model


def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)


def main(n_epochs=20):

    train_dataset, test_dataset = tfds.load(name="mnist", split=['train', 'test'], as_supervised=True)

    train_dataset = train_dataset.shuffle(1024).batch(32)
    train_dataset = train_dataset.map(_normalize_img)

    test_dataset = test_dataset.batch(32)
    test_dataset = test_dataset.map(_normalize_img)

    model = get_model()

    history = model.fit(
        train_dataset,
        epochs=n_epochs,
    )

    results = model.predict(test_dataset)
    import pdb; pdb.set_trace()

    # TODO Try drawing something that was never seen during training
    # A Greek letter, for example
    # Draw it twice, and see whether model puts both nearby in embedding space

    # TODO /usr/local/lib/python3.6/dist-packages/tensorflow_addons/utils/ensure_tf_install.py:67: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.4.0 and strictly below 2.7.0 (nightly versions are not supported). 
    # The versions of TensorFlow you are currently using is 2.3.0 and is not supported.


if __name__ == "__main__":
    main()

