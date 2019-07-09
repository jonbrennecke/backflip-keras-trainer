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

        channel_order = "channels_last"
        color_channels = 1
        default_conv2d_args = dict(
            kernel_size=(3, 3),
            dilation_rate=1,
            padding="same",
            data_format=channel_order,
            kernel_initializer='he_normal'
        )

        # color input layer
        self.color_image_input = keras.layers.Input(
            shape=image_dimensions["color_image"], name="color_image_input"
        )
        color_layer = self.color_image_input
        color_layer = keras.layers.Conv2D(32, activation="relu", **default_conv2d_args)(color_layer)
        color_layer = keras.layers.Conv2D(32, activation="relu", **default_conv2d_args)(color_layer)
        color_layer = keras.layers.MaxPool2D(pool_size=(2,2))(color_layer)
        color_layer = keras.layers.Conv2D(64, activation="relu", **default_conv2d_args)(color_layer)
        color_layer = keras.layers.Conv2D(128, activation="relu", **default_conv2d_args)(color_layer)
        color_layer = keras.layers.MaxPool2D(pool_size=(2,2))(color_layer)
        # color_layer = keras.layers.Conv2D(128, activation="relu", **default_conv2d_args)(color_layer)
        # color_layer = keras.layers.Conv2D(256, activation="relu", **default_conv2d_args)(color_layer)
        # color_layer = keras.layers.MaxPool2D(pool_size=(2,2))(color_layer)
        color_layer = keras.layers.Dropout(0.5)(color_layer)
        # color_layer = keras.layers.Conv2D(128, activation="relu", **default_conv2d_args)(color_layer)
        # color_layer = keras.layers.UpSampling2D(size=(2,2), interpolation="bilinear", data_format=channel_order)(color_layer)
        # color_layer = keras.layers.Conv2D(128, activation="relu", **default_conv2d_args)(color_layer)
        # color_layer = keras.layers.UpSampling2D(size=(2,2), interpolation="bilinear", data_format=channel_order)(color_layer)
        
        # depth input layer
        self.depth_image_input = keras.layers.Input(
            shape=image_dimensions["depth_image"], name="depth_image_input"
        )
        depth_layer = self.depth_image_input
        depth_layer = keras.layers.Conv2D(32, activation="relu", **default_conv2d_args)(depth_layer)
        depth_layer = keras.layers.Conv2D(32, activation="relu", **default_conv2d_args)(depth_layer)
        depth_layer = keras.layers.MaxPool2D(pool_size=(2,2))(depth_layer)
        depth_layer = keras.layers.Conv2D(64, activation="relu", **default_conv2d_args)(depth_layer)
        depth_layer = keras.layers.Conv2D(128, activation="relu", **default_conv2d_args)(depth_layer)
        depth_layer = keras.layers.MaxPool2D(pool_size=(2,2))(depth_layer)
        # depth_layer = keras.layers.Conv2D(128, activation="relu", **default_conv2d_args)(depth_layer)
        # depth_layer = keras.layers.Conv2D(256, activation="relu", **default_conv2d_args)(depth_layer)
        # depth_layer = keras.layers.MaxPool2D(pool_size=(2,2))(depth_layer)
        depth_layer = keras.layers.Dropout(0.5)(depth_layer)
        # depth_layer = keras.layers.Conv2D(64, activation="relu", **default_conv2d_args)(depth_layer)
        # depth_layer = keras.layers.UpSampling2D(size=(2,2), interpolation="bilinear", data_format=channel_order)(depth_layer)
        # depth_layer = keras.layers.Conv2D(128, activation="relu", **default_conv2d_args)(depth_layer)
        # depth_layer = keras.layers.UpSampling2D(size=(2,2), interpolation="bilinear", data_format=channel_order)(depth_layer)

        # combine inputs paths
        # layer_stack = color_layer
        layer_stack = keras.layers.Add()([color_layer, depth_layer])
        layer_stack = keras.layers.Conv2D(128, activation="relu", **default_conv2d_args)(layer_stack)
        # layer_stack = keras.layers.Conv2D(128, activation="relu", **default_conv2d_args)(layer_stack)


        # final block on the combined inputs; ends with a sigmoid activation layer so that output is
        # the probability of being in the foreground or background
        layer_stack = keras.layers.Conv2D(1, **default_conv2d_args)(layer_stack)
        layer_stack = keras.layers.Activation("sigmoid", name="segmentation_image_output")(layer_stack)
        
        self.segmentation_image_output = layer_stack
    
        inputs = [self.color_image_input, self.depth_image_input]
        output = [self.segmentation_image_output]
        self.model = keras.models.Model(inputs=inputs, outputs=output)

    def compile(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr = 1e-4), loss="binary_crossentropy", metrics=["accuracy"]
        )

    def train(self, color_image_array, depth_image_array, segmentation_image_array):
        inputs = {
            "color_image_input": color_image_array,
            "depth_image_input": depth_image_array,
        }
        outputs = {"segmentation_image_output": segmentation_image_array}
        self.model.fit(x=inputs, y=outputs, epochs=1)

    def train_generator(self, generator):
        def arrange_items():
            for gen_data in generator:
                color_image_array, depth_image_array, segmentation_image_array = (
                    gen_data
                )
                inputs = {
                    "color_image_input": color_image_array,
                    "depth_image_input": depth_image_array,
                }
                outputs = {"segmentation_image_output": segmentation_image_array}
                yield (inputs, outputs)

        self.model.fit_generator(
            arrange_items(), steps_per_epoch=10, epochs=1, shuffle=True
        )

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
            add_custom_layers=False,
            is_bgr=False,
            # image_scale=1 / 255.0,  # expect normalized output in range of [0, 1]
        )
        coreml_model.save(mld_model_path)
