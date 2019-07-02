import tensorflow as tf
import coremltools
import keras

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
        self.color_image_input = keras.layers.Input(
            shape=image_dimensions["color_image"], name="color_image_input"
        )
        self.depth_image_input = keras.layers.Input(
            shape=image_dimensions["depth_image"], name="depth_image_input"
        )

        color_layer = self.color_image_input
        color_layer = keras.layers.Conv2D(3, 3, 1, dilation_rate=1)(color_layer)
        color_layer = keras.layers.Activation("relu")(color_layer)

        depth_layer = keras.layers.Conv2D(3, 3, 1, dilation_rate=1)(
            self.depth_image_input
        )

        merge_layer = keras.layers.Add()([color_layer, depth_layer])
        
        layer_stack = keras.layers.Conv2D(3, 3, 1, dilation_rate=1)(merge_layer)
        layer_stack = keras.layers.Activation("relu")(layer_stack)

        # merge_layer = keras.layers.Conv2D(3, 3, 1, dilation_rate=1)(merge_layer)
        # merge_layer = keras.layers.Conv2D(1, 3, 1, dilation_rate=1)(merge_layer)
        # merge_layer = keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x))(merge_layer)

        # define output
        conv_layer = keras.layers.Conv2D(1, 3, 1, dilation_rate=1, name="segmentation_image_output")(layer_stack)
        self.segmentation_image_output = conv_layer

        inputs = [self.color_image_input, self.depth_image_input]
        output = [self.segmentation_image_output]
        self.model = keras.models.Model(inputs=inputs, outputs=output)

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

    def predict(self, color_image_array, depth_image_array):
        inputs = {
            "color_image_input": color_image_array,
            "depth_image_input": depth_image_array,
        }
        return self.model.predict(inputs, batch_size=16)

    def print_summary(self):
        return self.model.summary()

    def save_h5(self, h5_model_path):
        self.model.save(h5_model_path)

    def save_coreml(self, mld_model_path):
        input_names = ["color_image_input", "depth_image_input"]
        output_name = "segmentation_image_output"
        coreml_model = coremltools.converters.keras.convert(
            self.model,
            input_names=input_names,
            output_names=output_name,
            image_input_names=input_names,
            add_custom_layers=True,
        )
        coreml_model.save(mld_model_path)
