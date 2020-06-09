
import numpy as np
import config.transform_config as tc
import scipy.ndimage as nd
import utilities.console_util as cu


def transform_image(dataset, labels, verbose=True):
    """
    Applies flipping and rotation to the given data set.
    :param dataset: The data set to be augmented.
    :param labels: The labels for the data set.
    :param verbose: Whether the process status should be written to the console.
    :return: The augmented data set as well as its labels.
    """
    transormed_data = []
    transformed_labels = []

    for i in range(len(dataset)):

        cu.print_status('Transforming', i, len(dataset), verbose)

        transormed_data.append(dataset[i])

        transormed_data.append(np.flip(dataset[i], axis=0))
        transormed_data.append(np.flip(dataset[i], axis=1))

        for rotation in tc.rotations:
            transormed_data.append(nd.rotate(dataset[i], rotation, reshape=False))

        for j in range(len(tc.rotations) + 3):
            transformed_labels.append(labels[i])

    return np.array(transormed_data), np.array(transformed_labels)

