
import tensorflow as tf
import numpy as np
import model.loss as l
import config.train_config as tc
import config.model_config as mc
import os
import yolo.box as b
import config.config as cfg

class yolo_model:

    def __init__(self, model, output_shape):
        """
        Constructor of the yolo_model class.
        :param model: The model to be used.
        """
        self.output_shape = output_shape
        self.model = model

    @classmethod
    def initialize(cls, grid_wh, ab_count, class_count):
        """
        Initiates the YOLO model.
        :param grid_wh: Tuple that contains the widht and height of the grid.
        :param ab_count: Number of anchor boxes.
        :param class_count: Number of classes.
        """
        output_count = grid_wh[0] * grid_wh[1] * ab_count * (class_count + 5)

        shape = (480, 640, 3)

        inputs = tf.keras.layers.Input(shape=shape)

        darknet_model = cls.darknet53(shape)

        (x_large, x_medium, x_small) = darknet_model(inputs)

        y_small, sample_x = cls.yolo_detector(x_small, name='small_scale')

        y_medium, sample_x = cls.yolo_detector(x_medium, name='medium_scale', detector_input=sample_x)

        y_large, _ = cls.yolo_detector(x_large, name='large_scale', detector_input=sample_x)

        yolo_model = tf.keras.Model(inputs, (y_small, y_medium, y_large))

        return cls(yolo_model, (grid_wh[0], grid_wh[1], ab_count, class_count + 5))

    @classmethod
    def darknet53(cls, input_shape):

        inputs = tf.keras.layers.Input(shape=input_shape)

        x = cls.darknet_conv(inputs, 32, kernel_size=3, stride=1, name='conv2d_0')

        x = cls.darknet_conv(x, 64, kernel_size=3, stride=2, name='conv2d_1')

        x = cls.darknet_residual(x, 32, 64, 'residual_0_0')

        x = cls.darknet_conv(x, 128, kernel_size=3, stride=2, name='conv2d_2')

        for i in range(2):
            x = cls.darknet_residual(x, 64, 128, 'residual_1_' + str(i))

        x = cls.darknet_conv(x, 256, kernel_size=3, stride=2, name='conv2d_3')

        for i in range(8):
            x = cls.darknet_residual(x, 128, 256, 'residual_2_' + str(i))

        y0 = x

        x = cls.darknet_conv(x, 512, kernel_size=3, stride=2, name='conv2d_4')

        for i in range(8):
            x = cls.darknet_residual(x, 256, 512, 'residual_3_' + str(i))

        y1 = x

        x = cls.darknet_conv(x, 1024, kernel_size=3, stride=2, name='conv2d_5')

        for i in range(4):
            x = cls.darknet_residual(x, 512, 1024, 'residual_4_' + str(i))

        y2 = x

        return tf.keras.Model(inputs, (y0, y1, y2), name='darknet_53')

    @classmethod
    def darknet_conv(cls, inputs, filters, kernel_size, stride, name):

        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding='same',
            name=name + '_conv2d',
            use_bias=False
        )(inputs)

        x = tf.keras.layers.BatchNormalization(name=name + '_bn')(x)

        x = tf.keras.layers.LeakyReLU(alpha=0.01, name=name + '_leakyrelu')(x)

        return x

    @classmethod
    def darknet_residual(cls, inputs, filters1, filters2, name):

        shortcut = inputs

        x = cls.darknet_conv(inputs, filters1, kernel_size=1, stride=1, name=name + '_1x1')

        x = cls.darknet_conv(x, filters2, kernel_size=3, stride=1, name=name + '_3x3')

        x = tf.keras.layers.add(inputs=[shortcut, x], name=name + '_add')

        return x

    @classmethod
    def yolo_detector(cls, inputs, name, detector_input=None):
        """
        Implementes a YOLO detector with a certain grid output.
        The dimensions of the output solely depends on the input shape.
        :param inputs: The input tensor.
        :param name: Name of the detector.
        :param detector_input: The output of the previous YOLO detector.
        :return: The grid and the output of the fourth convolutional layer.
        """
        name = 'yolo_det_' + name

        if detector_input is not None:
            detector_input = tf.keras.layers.UpSampling2D(size=(2, 2), name=name + '_upsampling')(detector_input)
            inputs = tf.keras.layers.Concatenate(name=name + '_concat')([inputs, detector_input])

        x = cls.darknet_conv(inputs, 512, kernel_size=1, stride=1, name=name + '_conv2d_0_1x1')

        x = cls.darknet_conv(x, 1024, kernel_size=3, stride=1, name=name + '_conv2d_1_3x3')

        x = cls.darknet_conv(x, 512, kernel_size=1, stride=1, name=name + '_conv2d_2_1x1')

        x = cls.darknet_conv(x, 1024, kernel_size=3, stride=1, name=name + '_conv2d_3_3x3')

        x = cls.darknet_conv(x, 512, kernel_size=1, stride=1, name=name + '_conv2d_4_1x1')

        # Save the value of x for the next YOLO detector.
        sample_x = x

        x = cls.darknet_conv(x, 1024, kernel_size=3, stride=1, name=name + '_conv2d_5_3x3')

        out_filters = 3 * (5 + cfg.class_count)

        x = cls.darknet_conv(x, filters=out_filters, kernel_size=1, stride=1, name=name + '_conv1d_6_1x1')

        return x, sample_x

    @classmethod
    def load_from_file(cls, filename, output_shape):
        """
        Loads the model from the configured file.
        :return: -
        """
        file = os.path.join(mc.model_folder, filename)

        loss = l.yolo_loss_fn

        loaded_model = tf.keras.models.load_model(file, custom_objects={loss.__name__: loss })

        return cls(loaded_model, output_shape)

    def fit(self, train_ds, validation_ds=None):
        """
        Trains the model.
        :param train_ds: The training examples dataset.
        :param validation_ds: The validation examples dataset.
        :return: -
        """
        self.model.compile(optimizer='Adam', loss=l.yolo_loss_fn, metrics=['accuracy'])
        self.model.fit(train_ds, epochs=tc.epochs, validation_data=validation_ds)

    def evaluate(self, test_ds):
        """
        Evaluates the model based on the given test data.
        :param test_ds: The test data.
        :return: The evaluation of the model.
        """
        return self.model.evaluate(test_ds)

    def print_summary(self):
        """
        Prints the summary of the model into the console.
        :return: -
        """
        print(self.model.summary())

    def save_to_file(self, filename):
        """
        Saves the model to the configured file.
        :return: -
        """
        file = os.path.join(mc.model_folder, filename)

        self.model.save(file)

    def non_max_supression(self, detections):
        """
        Performs non-max supression on the detections.
        :param detections: The list of detections for each class.
        :return: The detection array without the
        """
        suppressed_detections = [[] * (self.output_shape[-1] - 5)]

        yolo_boxes = []

        for i in range(len(detections)):

            yolo_boxes.append([])

            for j in range(len(detections[i])):

                x = detections[i][j][1]
                y = detections[i][j][2]
                w = detections[i][j][3]
                h = detections[i][j][4]

                yolo_boxes[i].append(b.box(x, y, w, h))

        for i in range(len(detections)):

            if len(detections[i]) != 0:
                processed_elements = np.zeros(len(detections[i]), dtype=bool)

                while not np.all(processed_elements):
                    max_ind = -1

                    for j in range(len(detections[i])):
                        if not processed_elements[j] and detections[i][j][0] > detections[i][max_ind][0]:
                            max_ind = j

                    for j in range(len(detections[i])):
                        if not processed_elements[j]:
                            if yolo_boxes[i][j].calculate_iou(yolo_boxes[i][max_ind]) > mc.non_max_supression_threhold:
                                processed_elements[j] = True

                    suppressed_detections[i].append(yolo_boxes[i][max_ind])

        return suppressed_detections

    def predict(self, input_image):
        """
        Predicts based on the given input.
        :param input_image: The image to run the prediction on.
        :return: Prediction.
        """

        prediction = self.model.predict(input_image)

        prediction = np.reshape(prediction, self.output_shape)

        detections = [[] * (self.output_shape[-1] - 5)]

        for i in range(8):
            for j in range(8):
                for k in range(6):
                    if prediction[i][j][k][0] > mc.detection_threshold:

                        w = prediction[i][j][k][3]
                        h = prediction[i][j][k][4]

                        if w > 0 and h > 0:
                            max_class = 0

                            for l in range(1, self.output_shape[-1] - 5):
                                if prediction[i][j][k][max_class + 5] < prediction[i][j][k][l + 5]:
                                    max_class = l

                            detections[max_class].append(prediction[i][j][k][0:5])

        suppressed_detections = self.non_max_supression(detections)

        return suppressed_detections
