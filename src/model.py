import tensorflow as tf

from .constants import IMAGE_DIMENSIONS

image_dimensions = IMAGE_DIMENSIONS["model"]


class Model(object):
    color_image_input: tf.keras.Input = None
    depth_image_input: tf.keras.Input = None
    segmentation_image_output: tf.keras.layers.Dense = None
    model: tf.keras.models.Model = None

    def __init__(self):
        super(Model, self).__init__()

        # define inputs
        self.color_image_input = tf.keras.Input(
            shape=image_dimensions["color_image"], name="color_image_input"
        )
        self.depth_image_input = tf.keras.Input(
            shape=image_dimensions["depth_image"], name="depth_image_input"
        )

        color_layer = self.color_image_input
        color_layer = tf.keras.layers.Conv2D(3, 3, 1, dilation_rate=1)(color_layer)
        color_layer = tf.keras.layers.Activation("relu")(color_layer)

        # depth_layer = self.depth_image_input
        depth_layer = tf.keras.layers.Conv2D(3, 3, 1, dilation_rate=1)(
            self.depth_image_input
        )

        merge_layer = tf.keras.layers.Add()([color_layer, depth_layer])
        merge_layer = tf.keras.layers.Conv2D(3, 3, 1, dilation_rate=1)(merge_layer)
        # merge_layer = tf.keras.layers.Activation("relu")(merge_layer)
        merge_layer = tf.keras.layers.Conv2D(3, 3, 1, dilation_rate=1)(merge_layer)
        # merge_layer = tf.keras.layers.Activation("relu")(merge_layer)
        merge_layer = tf.keras.layers.Conv2D(3, 3, 1, dilation_rate=1)(merge_layer)
        # merge_layer = tf.keras.layers.Activation("relu")(merge_layer)

        # define output
        self.segmentation_image_output = tf.keras.layers.Dense(
            units=1,
            input_shape=image_dimensions["segmentation_image"],
            name="segmentation_image_output",
        )(merge_layer)

        inputs = [self.color_image_input, self.depth_image_input]
        output = [self.segmentation_image_output]
        self.model = tf.keras.models.Model(inputs=inputs, outputs=output)

    def compile(self):
        self.model.compile(
            optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"]
        )

    def train(self, color_image_array, depth_image_array, segmentation_image_array):
        inputs = {
            "color_image_input": color_image_array,
            "depth_image_input": depth_image_array,
        }
        outputs = {"segmentation_image_output": segmentation_image_array}
        self.model.fit(x=inputs, y=outputs)

    # def evaluate(self):
    #     score = model.evaluate(x_test, y_test, batch_size=16)
    #     pass

    def predict(self, color_image_array, depth_image_array):
        inputs = {
            "color_image_input": color_image_array,
            "depth_image_input": depth_image_array,
        }
        return self.model.predict(inputs, batch_size=16)

    def summary(self):
        return self.model.summary()
