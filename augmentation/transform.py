
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

        transormed_data.append(np.flip(dataset[i], axis=0))
        transormed_data.append(np.flip(dataset[i], axis=1))

        for rotation in tc.rotations:
            transormed_data.append(nd.rotate(dataset[i], rotation, reshape=False))

    return np.array(transormed_data)


def transform_label_batch(labels):

    transformed_labels = []

    for i in range(len(labels)):

        for j in range(len(tc.rotations) + 3):
            transformed_labels.append(labels[i])

    return np.array(transformed_labels)
