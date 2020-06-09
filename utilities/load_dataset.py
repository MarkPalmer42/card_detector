
import cv2
import utilities.folder_utils as fu
import os
import numpy as np


def load_batch(file_list):
    """
    Loads the batch from the given file lists.
    :param file_list: The file list to load the batch from.
    :return: The loaded batch.
    """
    batch = []
    labels = []

    for file in file_list:

        batch.append(cv2.imread(file[0]))
        labels.append(file[1])

    return np.array(batch), np.array(labels)


def load_all_labels(path):
    """
    Loads all the labels from the given path.
    :param path: The directory to load the labels from.
    :return: The labels.
    """
    label_file = os.path.join(path, 'labels.txt')

    dims = []
    labels = []

    with open(label_file, 'r') as handle:

        dims = [int(x) for x in next(handle)].split()

        next_line = next(handle, '')

        while not next_line == '':
            labels.append([int(x) for x in next_line.split()])

            next_line = next(handle, '')

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

    batches = [x for x in range(0, len(file_list), batch_size)]

    return ((load_batch(file_list[batches[i]: batches[i] + batch_size]), i, len(batches)) for i in range(len(batches)))
