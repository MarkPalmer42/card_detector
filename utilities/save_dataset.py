
import cv2
import os
import config.dataset_config as dc
import utilities.folder_utils as fu
import numpy as np


def is_number(s):
    """
    Decides whether the given input is a number or not.
    :param s: String that might be a number.
    :return: True if the given string can be casted to int, false otherwise.
    """
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_num_list(directory_listing):
    """
    Retrieves the file names from a directory listing that only contain decimal digits before the dot.
    :param directory_listing: List of file names.
    :return: List of file names only containing decimal digits without their extensions.
    """
    result = []

    for f in directory_listing:

        filename = f.split('.')[0]

        if is_number(filename):
            result.append(int(filename))

    return result


def save_dataset(path, dataset, image_extension='jpg', use_config_switch=True):
    """
    Saves the data set to the given path.
    :param path: The target path to save the
    :param dataset: The data set to be saved.
    :param image_extension: The extension to save, only cv2 image extensions are supported.
    :param use_config_switch: Whether the config switch of 'save_all_data' should be considered or not.
    :return: -
    """
    if not use_config_switch or dc.save_all_data:

        dirlist = os.listdir(path)

        if len(dirlist) == 0:
            index = 0
        else:
            index = next(reversed(sorted(get_num_list(dirlist))), 0)

        for i in range(len(dataset)):

            cv2.imwrite(os.path.join(path, str(index) + '.' + image_extension), dataset[i])

            index = index + 1


def get_saved_dims(label_file_path):
    """
    Retrieves the dimensions of the labels in the given label file.
    :param label_file_path:
    :return: Dimensions of the saved labels.
    """
    dims = []

    if os.path.exists(label_file_path):

        # The first line of the file should contain the dimensions of the labels.
        with open(label_file_path, 'r') as handle:
            dims = [int(x) for x in next(handle).split()]

    return dims


def save_labels(path, labels, use_config_switch=True):
    """
    Appends the labels.txt in the target folder with the given labels.
    :param path: The path of the folder to save labels.txt in.
    :param labels: The labels to be saved.
    :param use_config_switch: Whether the config switch of 'save_all_data' should be considered or not.
    :return: -
    """
    if not use_config_switch or dc.save_all_data:

        label_path = os.path.join(path, 'labels.txt')

        label_text = []

        dims = get_saved_dims(label_path)

        for label in labels:
            label_text.append('\t'.join(str(x) for x in np.ndarray.flatten(np.array(label))))

        label_text = ['\t'.join(str(x) for x in labels.shape)]

        handle = open(os.path.join(path, 'labels.txt'), 'a+')

        if not dims == []:
            handle.write('\t'.join([str(x) for x in dims]) + '\n')

        handle.write('\n'.join(label_text))

        handle.close()


def save_all(path, dataset, labels, image_extension='jpg', use_config_switch=True):
    """
    Saves both the given data set and the labels to the given folder path.
    :param path: The directory to save into.
    :param dataset: The data set to be saved.
    :param labels: The labels to be saved.
    :param image_extension: Image extension to save the data set in. Only cv2 image extensions are supported.
    :param use_config_switch: Whether the config switch of 'save_all_data' should be considered or not.
    :return: -
    """
    save_dataset(path, dataset, image_extension, use_config_switch)
    save_labels(path, labels, use_config_switch)


def truncate_folders(path_list):
    """
    Truncates the given list of directories.
    :param path_list: The list of directories to be truncated.
    :return: -
    """
    for path in path_list:
        fu.create_truncated_folder(path)
