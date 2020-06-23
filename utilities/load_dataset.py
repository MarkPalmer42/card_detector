
import cv2
import utilities.folder_utils as fu
import os
import numpy as np

def load_dataset(path, extension='.jpg'):

    file_list = fu.list_files(path, extension)

    dataset = []

    for file in file_list:
        dataset.append(cv2.imread(file))

    return np.array(dataset)


def load_batch(path, file_list, a, b):
    """
    Loads the batch from the given file lists.
    :param file_list: The file list to load the batch from.
    :return: The loaded batch.
    """
    batch = []
    labels = load_all_labels(path)

    for file in file_list[a: b]:

        batch.append(cv2.imread(file))

    return np.array(batch), np.array(labels[a: b])


def load_all_labels(path, squeeze_dims=False):
    """
    Loads all the labels from the given path.
    :param path: The directory to load the labels from.
    :return: The labels.
    """
    label_file = os.path.join(path, 'labels.txt')

    dims = []
    labels = []

    with open(label_file, 'r') as handle:

        dims = [int(x) for x in next(handle).split('\t')]

        next_line = next(handle, '')

        while not next_line == '':
            labels.append([float(x) if not x == 'None' else None for x in next_line.split()])

            next_line = next(handle, '')

    if squeeze_dims:
        dims = [dims[0],  int(np.prod(dims[1:]))]

    return np.reshape(labels, dims)


def load_batch_images(path, batch_size, extension='.jpg'):
    """
    Loads the batch of images as well as the labels for the batch.
    :param path: The directory to load from.
    :param batch_size: The size of the batch.
    :param extension: The extension of the files to be loaded.
    :return: A generator that creates the batches.
    """
    file_list = fu.list_files(path, extension)

    batches = [(x, x + batch_size) for x in range(0, len(file_list), batch_size)]

    return ((load_batch(path, file_list, batches[i][0], batches[i][1]), i, len(batches)) for i in range(len(batches)))
