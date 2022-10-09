from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow_examples.models.pix2pix import pix2pix
from dataclasses import dataclass, field
from pyspark.context import SparkContext
from tensorflowonspark import TFCluster
import tensorflow_datasets as tfds
from pyspark.conf import SparkConf
import tensorflow as tf
import argparse


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 128.0 - 1
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint["image"], (128, 128))
    input_mask = tf.image.resize(datapoint["segmentation_mask"], (128, 128))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint):
    input_image = tf.image.resize(datapoint["image"], (128, 128))
    input_mask = tf.image.resize(datapoint["segmentation_mask"], (128, 128))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


@dataclass
class DataHandler:
    dataset: any = field(default=None)
    info: dict = field(default=None)

    def __post_init__(self):
        self.dataset, self.info = tfds.load("oxford_iiit_pet:3.2.0", with_info=True)

    def get_train_dataset(self, buffer_size: int, batch_size: int):
        train = self.dataset["train"].map(
            load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        train_dataset = train.cache().shuffle(buffer_size).batch(batch_size).repeat()
        train_dataset = train_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        return train_dataset

    def get_val_dataset(self, batch_size: int):
        test = self.dataset["test"].map(load_image_test)
        test_dataset = test.batch(batch_size)

        return test_dataset


@dataclass
class UNet:
    model: any = field(default=None)
    output_channels: int = field(default=3)

    def __post_init__(self):
        self.base_model = self.get_base_model()
        self.fe_model = self.get_feature_extraction_model()

        self.up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),  # 32x32 -> 64x64
        ]

        self.model = self.get_model()
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def get_base_model(self):
        return tf.keras.applications.MobileNetV2(
            input_shape=[128, 128, 3], include_top=False
        )

    def get_feature_extraction_model(self):
        layer_names = [
            "block_1_expand_relu",  # 64x64
            "block_3_expand_relu",  # 32x32
            "block_6_expand_relu",  # 16x16
            "block_13_expand_relu",  # 8x8
            "block_16_project",  # 4x4
        ]
        layers = [self.base_model.get_layer(name).output for name in layer_names]

        fe_model = tf.keras.Model(inputs=self.base_model.input, outputs=layers)

        fe_model.trainable = False

        return fe_model

    def get_model(self):
        last = tf.keras.layers.Conv2DTranspose(
            self.output_channels, 3, strides=2, padding="same", activation="softmax"
        )

        inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        x = inputs

        # Downsampling through the model
        skips = self.fe_model(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

@dataclass
class SparkIntegration:
    spark_context: SparkContext = field(default=None)
    handler_func: any = field(default=None)
    num_executors: int = field(default=1)
    args: any = field(default=None)

    def __post_init__(self):
        self.spark_context = SparkContext(conf=SparkConf().setAppName("segmentation"))
        executors = self.spark_context._conf.get("spark.executor.instances")
        self.num_executors = int(executors) if executors is not None else 1
        self.args = self.get_args()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--batch_size", help="number of records per batch", type=int, default=64
        )
        parser.add_argument(
            "--buffer_size", help="size of shuffle buffer", type=int, default=1000
        )
        parser.add_argument(
            "--cluster_size",
            help="number of nodes in the cluster",
            type=int,
            default=self.num_executors,
        )
        parser.add_argument("--epochs", help="number of epochs", type=int, default=3)
        parser.add_argument(
            "--model_dir",
            help="path to save model/checkpoint",
            default="segmentation_model",
        )
        parser.add_argument(
            "--export_dir",
            help="path to export saved_model",
            default="segmentation_export",
        )
        parser.add_argument(
            "--tensorboard", help="launch tensorboard process", action="store_true"
        )

        return parser.parse_args()

    def run(self):
        cluster = TFCluster.run(
            self.spark_context,
            self.handler_func,
            self.args,
            self.args.cluster_size,
            num_ps=0,
            tensorboard=self.args.tensorboard,
            input_mode=TFCluster.InputMode.TENSORFLOW,
            master_node="chief",
        )
        cluster.shutdown(grace_secs=30)


def handler(args, ctx):
    import tensorflow as tf

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    data_handler = DataHandler()

    train_data = data_handler.get_train_dataset(
        batch_size=args.batch_size, buffer_size=args.buffer_size
    )
    val_data = data_handler.get_val_dataset(batch_size=args.batch_size)

    train_lenght = data_handler.info.splits["train"].num_examples
    steps_per_epoch = train_lenght // args.batch_size
    validation_splits = (
        data_handler.info.splits["test"].num_examples // args.batch_size // 5
    )

    with strategy.scope():
        unet = UNet()

    tf.io.gfile.makedirs(args.model_dir)
    filepath = args.model_dir + "/weights-{epoch:04d}"
    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath, verbose=1, save_weights_only=True
    )

    unet.model.fit(
        train_data,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[ckpt_callback],
        validation_steps=validation_splits,
        validation_data=val_data,
    )

    unet.model.save(args.export_dir, save_format="tf")


if __name__ == "__main__":
    spark_integration = SparkIntegration(handler_func=handler)
    spark_integration.run()
