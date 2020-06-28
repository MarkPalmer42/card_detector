
import tensorflow as tf
import model.loss as l
import config.train_config as tc


class yolo_model:

    model = tf.keras.models.Sequential()

    def __init__(self, grid_wh, ab_count, class_count):
        """
        Initiates the YOLO model.
        :param grid_wh: Tuple that contains the widht and height of the grid.
        :param ab_count: Number of anchor boxes.
        :param class_count: Number of classes.
        """
        output_count = grid_wh[0] * grid_wh[1] * ab_count * (class_count + 5)

        self.model.add(tf.keras.layers.Conv2D(32, 3, activation='relu',
                                              input_shape=(480, 640, 3), data_format="channels_last"))
        self.model.add(tf.keras.layers.MaxPool2D())

        self.model.add(tf.keras.layers.Conv2D(64, 3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D())

        self.model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D())

        self.model.add(tf.keras.layers.Conv2D(256, 3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D())

        self.model.add(tf.keras.layers.Conv2D(512, 3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D())

        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dense(output_count, activation='sigmoid'))

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
