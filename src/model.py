import numpy as np
np.random.seed(1337)

import tensorflow as tf
import coremltools
import keras

from keras import backend as K

from .constants import IMAGE_DIMENSIONS, NUMBER_OF_STEPS_PER_EPOCH, NUMBER_OF_EPOCHS

image_dimensions = IMAGE_DIMENSIONS["model"]


class Model(object):
    model: keras.models.Model

    def __init__(self, model: keras.models.Model):
        super(Model, self).__init__()
        self.model = model

    def create_model():
        default_conv2d_args = dict(
            padding="same", data_format="channels_last", kernel_initializer="he_normal"
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

        def make_convolution_block(filters, kernel_size=(3, 3)):
            def fn(input):
                layers = keras.layers.Conv2D(
                    filters, kernel_size=kernel_size, **default_conv2d_args
                )(input)
                layers = keras.layers.BatchNormalization(epsilon=1e-4)(layers)
                return keras.layers.Activation("relu")(layers)
            return fn

        def make_downsample_block(filters, kernel_size=(3,3), pool_size=(2, 2), convolutions=3):
            def fn(input):
                layers = keras.layers.MaxPool2D(pool_size=pool_size)(input)
                for _ in range(0, convolutions):
                    layers = make_convolution_block(filters, kernel_size)(layers)
                return layers
            return fn

        def make_upsample_block(filters, kernel_size=(3,3), upsample_size=(2, 2), convolutions=3):
            def fn(concat_layer, upsample_layer):
                layers = keras.layers.UpSampling2D(size=upsample_size)(upsample_layer)
                layers = keras.layers.Concatenate(axis=3)([concat_layer, layers])
                for _ in range(0, convolutions):
                    layers = make_convolution_block(filters, kernel_size)(layers)
                return layers
            return fn

        concat = keras.layers.Concatenate(axis=3)([color_layer, depth_layer])
        block1 = make_convolution_block(64)(concat)
        block1 = make_convolution_block(64)(block1)
        block2 = make_downsample_block(128)(block1)
        block5 = make_upsample_block(64)(block1, block2)

        # final block on the combined inputs; ends with a sigmoid activation layer so that output is
        # the probability of being in the foreground or background
        output = keras.layers.Conv2D(
            1,
            kernel_size=(1, 1),
            activation="sigmoid",
            name="segmentation_image_output",
        )(block5)
        segmentation_image_output = output

        inputs = [color_image_input, depth_image_input]
        output = [segmentation_image_output]
        return keras.models.Model(inputs=inputs, outputs=output)

    def compile(self):
        def dice_coef(y_true, y_pred, smooth = 1):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        def dice_coef_loss(y_true, y_pred):
            return 1 - dice_coef(y_true, y_pred)

        def loss(y_true, y_pred):
            return keras.losses.binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)
        
        self.model.compile(
            loss=loss, optimizer=keras.optimizers.Adam(lr=1e-4), metrics=[dice_coef]
        )

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
            arrange_items(),
            steps_per_epoch=NUMBER_OF_STEPS_PER_EPOCH,
            epochs=NUMBER_OF_EPOCHS,
            shuffle=False,
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
            image_scale=1./255
        )
        coreml_model.save(mld_model_path)
