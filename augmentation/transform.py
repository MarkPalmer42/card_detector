
import numpy as np
import config.transform_config as tc
import scipy.ndimage as nd
import utilities.console_util as cu


def transform_image_batch(dataset, verbose=True):
    """
    Applies flipping and rotation to the given data set.
    :param dataset: The data set to be augmented.
    :param verbose: Whether the process status should be written to the console.
    :return: The augmented data set as well as its labels.
    """
    transormed_data = []

    for i in range(len(dataset)):

        cu.print_status('Transforming', i, len(dataset), verbose)

        transormed_data.append(dataset[i])

        if tc.flip_horizontally:
            transormed_data.append(np.flip(dataset[i], axis=0))

        if tc.flip_vertically:
            transormed_data.append(np.flip(dataset[i], axis=1))

        for rotation in tc.rotations:
            transormed_data.append(nd.rotate(dataset[i], rotation, reshape=False))

    return np.array(transormed_data)


def transform_label_batch(labels):
    """
    Applies transformation to a whole batch of labels.
    :param labels:
    :return:
    """
    transformed_labels = []

    copies_to_be_made = len(tc.rotations) + 1

    if tc.flip_vertically:
        copies_to_be_made = copies_to_be_made + 1

    if tc.flip_horizontally:
        copies_to_be_made = copies_to_be_made + 1

    for i in range(len(labels)):

        for j in range(copies_to_be_made):
            transformed_labels.append(labels[i])

    return np.array(transformed_labels)
