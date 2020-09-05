
import tensorflow as tf
import model.loss as l
import config.train_config as tc
import config.model_config as mc
import os


class yolo_model:

    def __init__(self, model):
        """
        Constructor of the yolo_model class.
        :param model: The model to be used.
        """
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

        init_model = tf.keras.models.Sequential()

        init_model.add(tf.keras.layers.Conv2D(32, 3, activation='relu',
                                              input_shape=(480, 640, 3), data_format="channels_last"))
        init_model.add(tf.keras.layers.MaxPool2D())

        init_model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
        init_model.add(tf.keras.layers.MaxPool2D())

        init_model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
        init_model.add(tf.keras.layers.MaxPool2D())

        init_model.add(tf.keras.layers.Conv2D(256, 3, activation='relu'))
        init_model.add(tf.keras.layers.MaxPool2D())

        init_model.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))
        init_model.add(tf.keras.layers.MaxPool2D())

        init_model.add(tf.keras.layers.Flatten())
        init_model.add(tf.keras.layers.Dense(128, activation='relu'))
        init_model.add(tf.keras.layers.Dense(output_count, activation='sigmoid'))

        return cls(init_model)

    @classmethod
    def load_from_file(cls, filename):
        """
        Loads the model from the configured file.
        :return: -
        """
        file = os.path.join(mc.model_folder, filename)

        loss = l.yolo_loss_function

        loaded_model = tf.keras.models.load_model(file, custom_objects={loss.__name__: loss})

        return cls(loaded_model)

    def fit(self, train_ds, validation_ds=None):
        """
        Trains the model.
        :param train_ds: The training examples dataset.
        :param validation_ds: The validation examples dataset.
        :return: -
        """
        self.model.compile(optimizer='Adam', loss=l.yolo_loss_function, metrics=['accuracy'])
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

