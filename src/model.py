import tensorflow as tf
import coremltools
import keras

from .constants import IMAGE_DIMENSIONS, NUMBER_OF_STEPS_PER_EPOCH, NUMBER_OF_EPOCHS

image_dimensions = IMAGE_DIMENSIONS["model"]


class Model(object):
    model: keras.models.Model

    def __init__(self, model: keras.models.Model):
        super(Model, self).__init__()
        self.model = model

    def create_model():
        default_conv2d_args = dict(
            padding="same",
            data_format="channels_last",
            kernel_initializer='he_normal'
        )

        # color input layer
        color_image_input = keras.layers.Input(
            shape=image_dimensions["color_image"], name="color_image_input"
        )
        color_layer = color_image_input

        # depth input layer
        depth_image_input = keras.layers.Input(
            shape=image_dimensions["depth_image"], name="depth_image_input"
        )
        depth_layer = depth_image_input

        # block 1
        block1 = keras.layers.Concatenate(axis=3)([color_layer, depth_layer])
        block1 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block1)
        block1 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block1)

        # block 2
        block2 = keras.layers.MaxPool2D(pool_size=(2,2))(block1)
        block2 = keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block2)
        block2 = keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block2)
        
        # block 3
        block3 = keras.layers.MaxPool2D(pool_size=(2,2))(block2)
        block3 = keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block3)
        block3 = keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block3)

        # block 4
        block4 = keras.layers.MaxPool2D(pool_size=(2,2))(block3)
        block4 = keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block4)
        block4 = keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block4)
        # block4 = keras.layers.Dropout(0.5)(block4)

        # block 5
        block5 = keras.layers.MaxPool2D(pool_size=(2,2))(block4)
        block5 = keras.layers.Conv2D(1024, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block5)
        block5 = keras.layers.Conv2D(1024, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block5)
        # block5 = keras.layers.Dropout(0.5)(block5) for some reason, this extra dropout breaks everything
        
        # block 6
        block6 = keras.layers.UpSampling2D(size=(2,2))(block5)
        block6 = keras.layers.Conv2D(512, kernel_size=(2, 2), activation="relu", **default_conv2d_args)(block6)
        block6 = keras.layers.Concatenate(axis=3)([block4, block6])
        block6 = keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block6)
        block6 = keras.layers.Conv2D(512, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block6)

        # block 7
        block7 = keras.layers.UpSampling2D(size=(2,2))(block6)
        block7 = keras.layers.Conv2D(256, kernel_size=(2, 2), activation="relu", **default_conv2d_args)(block7)
        block7 = keras.layers.Concatenate(axis=3)([block3, block7])
        block7 = keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block7)
        block7 = keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block7)

        # block 8
        block8 = keras.layers.UpSampling2D(size=(2,2))(block7)
        block8 = keras.layers.Conv2D(128, kernel_size=(2, 2), activation="relu", **default_conv2d_args)(block8)
        block8 = keras.layers.Concatenate(axis=3)([block2, block8])
        block8 = keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block8)
        block8 = keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block8)

        # block 9
        block9 = keras.layers.UpSampling2D(size=(2,2))(block8)
        block9 = keras.layers.Conv2D(64, kernel_size=(2, 2), activation="relu", **default_conv2d_args)(block9)
        block9 = keras.layers.Concatenate(axis=3)([block1, block9])
        block9 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block9)
        block9 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block9)
        block9 = keras.layers.Conv2D(2, kernel_size=(3, 3), activation="relu", **default_conv2d_args)(block9)

        # final block on the combined inputs; ends with a sigmoid activation layer so that output is
        # the probability of being in the foreground or background
        output = keras.layers.Conv2D(
            1,
            kernel_size=(1, 1),
            activation="sigmoid",
            name="segmentation_image_output"
        )(block9)
        segmentation_image_output = output
    
        inputs = [color_image_input, depth_image_input]
        output = [segmentation_image_output]
        return keras.models.Model(inputs=inputs, outputs=output)

    def compile(self):
        self.model.compile(
            optimizer="sgd", loss="mean_squared_error", metrics=["accuracy"]
        )

    def train_generator(self, generator):
        def arrange_items():
            for gen_data in generator:
                color_image_array, depth_image_array, segmentation_image_array = (
                    gen_data
                )
                inputs = {
                    "color_image_input": color_image_array / 255,
                    "depth_image_input": depth_image_array / 255,
                }
                outputs = {"segmentation_image_output": segmentation_image_array / 255 }
                yield (inputs, outputs)

        self.model.fit_generator(
            arrange_items(), steps_per_epoch=NUMBER_OF_STEPS_PER_EPOCH, epochs=NUMBER_OF_EPOCHS, shuffle=True
        )

    def predict(self, color_image_array, depth_image_array):
        inputs = {
            "color_image_input": color_image_array,
            "depth_image_input": depth_image_array,
        }
        return self.model.predict(inputs)

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
        )
        coreml_model.save(mld_model_path)
