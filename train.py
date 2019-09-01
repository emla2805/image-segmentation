import logging
import os
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_datasets as tfds

from networks import unet_model
from utils import load_image_train, load_image_test

logging.basicConfig(level=logging.INFO)
AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1000
NUM_SAMPLES = 5


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log-dir", default="models/1")
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--image-size", default=128, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    args = parser.parse_args()

    dataset, info = tfds.load("oxford_iiit_pet:3.0.0", with_info=True)

    ds_train = (
        dataset["train"]
        .map(load_image_train, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(args.batch_size)
        .prefetch(AUTOTUNE)
    )
    ds_test = dataset["test"].map(load_image_test).batch(args.batch_size)
    ds_sample = dataset["test"].map(load_image_test).batch(NUM_SAMPLES)
    sample_images, sample_masks = next(iter(ds_sample))

    model = unet_model(3)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, args.log_dir, max_to_keep=1)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="train_accuracy"
    )

    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name="test_accuracy"
    )

    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(args.log_dir, "train")
    )
    test_summary_writer = tf.summary.create_file_writer(
        os.path.join(args.log_dir, "test")
    )

    @tf.function
    def train_step(image, label):
        with tf.GradientTape() as tape:
            predictions = model(image)
            loss = loss_object(label, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(label, predictions)

    @tf.function
    def test_step(image, label):
        predictions = model(image)
        t_loss = loss_object(label, predictions)

        test_loss(t_loss)
        test_accuracy(label, predictions)

    for epoch in range(args.epochs):
        for image, label in ds_train:
            train_step(image, label)

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", train_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", train_accuracy.result(), step=epoch)

        for test_image, test_label in ds_test:
            test_step(test_image, test_label)

        if (epoch + 1) % 5 == 0:
            save_path = manager.save()
            print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

        with test_summary_writer.as_default():
            tf.summary.scalar("loss", test_loss.result(), step=epoch)
            tf.summary.scalar("accuracy", test_accuracy.result(), step=epoch)
            tf.summary.image(
                "Sample Image", sample_images * 0.5 + 0.5, step=epoch
            )
            tf.summary.image(
                "True Mask", sample_masks * 0.5 + 0.5, step=epoch
            )
            tf.summary.image(
                "Predicted Mask", model(sample_images) * 0.5 + 0.5, step=epoch
            )

        template = "Epoch {}, Train Loss: {}, Train Accuracy: {:.2f}, Test Loss: {}, Test Accuracy: {:.2f}"
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result() * 100,
                test_loss.result(),
                test_accuracy.result() * 100,
            )
        )

        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
