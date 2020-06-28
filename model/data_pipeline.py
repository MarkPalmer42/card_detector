
import utilities.load_dataset as ld
import tensorflow as tf
import config.train_config as tc


def process_path(file_path, label):
    """
    Loads the image file associated with the given path and normalizes it.
    :param file_path: The path of the image file.
    :param label: Label associated with the image.
    :return: The normalized image and the label.
    """
    image = tf.io.decode_image(tf.io.read_file(file_path), dtype=tf.float32)

    normalized_image = tf.divide(image, 255.0)

    return tf.reshape(normalized_image, [480, 640, 3]), label


def load_yolo_dataset(source_folder, extension='.jpg'):
    """
    Loads a tensorflow dataset containing the yolo data.
    :param source_folder: The source of the data-
    :param extension: The extemsion of the images.
    :return: A tensorflow dataset.
    """
    # Load all labels at once.
    labels = ld.load_all_labels(source_folder, True)

    labels_set = tf.data.Dataset.from_tensor_slices(labels)

    file_pattern = source_folder + '*' + extension

    # Load the image paths.
    image_set = tf.data.Dataset.list_files(file_pattern, shuffle=False)

    # Zip the image dataset with the labels dataset
    full_ds = tf.data.Dataset.zip((image_set, labels_set))

    # Load and normalize images.
    full_ds = full_ds.map(process_path)

    # Shuffle and batch the dataset
    # Setting reshuffle_each_iteration to False is important to avoid overlapping in the split dataset.
    full_ds = full_ds.shuffle(len(labels), reshuffle_each_iteration=False).batch(tc.batch_size)

    # Split the dataset into train, validation and test datasets.
    split_dataset = full_ds.enumerate().filter(lambda x, y: x % tc.test_val_shard == 0).map(lambda x, y: y)
    train_dataset = full_ds.enumerate().filter(lambda x, y: x % tc.test_val_shard != 0).map(lambda x, y: y)

    test_dataset = split_dataset.enumerate().filter(lambda x, y: x % 2 == 0).map(lambda x, y: y)
    val_dataset = split_dataset.enumerate().filter(lambda x, y: x % 2 != 0).map(lambda x, y: y)

    return train_dataset, val_dataset, test_dataset
