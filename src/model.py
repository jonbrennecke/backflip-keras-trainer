import tensorflow as tf

# TODO: load these constants from constants file
COLOR_IMAGE_SHAPE = (2320, 3088, 3)
DEPTH_IMAGE_SHAPE = (480, 640, 1)
SEGMENTATION_IMAGE_SHAPE = (1160, 1544, 1)


class Model(object):
    color_image_input: tf.keras.Input = None
    depth_image_input: tf.keras.Input = None
    segmentation_image_output: tf.keras.layers.Dense = None
    model: tf.keras.models.Model = None

    def __init__(self):
        super(Model, self).__init__()

        # K.variable

        # define inputs
        self.color_image_input = tf.keras.Input(
            shape=COLOR_IMAGE_SHAPE, name="color_image_input"
        )
        # self.depth_image_input = tf.keras.Input(
        #     shape=DEPTH_IMAGE_SHAPE, name="depth_image_input"
        # )

        layer = tf.keras.layers.Convolution2D(8, 3, 3)(self.color_image_input)

        # define output
        self.segmentation_image_output = tf.keras.layers.Dense(
            units=32,
            input_shape=SEGMENTATION_IMAGE_SHAPE,
            name="segmentation_image_output",
        )(layer)

        inputs = [self.color_image_input]  # TODO: add depth_image_input
        output = [self.segmentation_image_output]
        self.model = tf.keras.models.Model(inputs=inputs, outputs=output)

    def compile(self):
        self.model.compile(optimizer=None)  # TODO

    def train(self):
        # self.model.fit
        pass

    def summary(self):
        return self.model.summary()
