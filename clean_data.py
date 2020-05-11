
import os
import folder_utils as fu
import cv2
import numpy as np
import config as cfg


def color_keying(array, color):
    """
    Color keying function used for the blue boxing technique.
    It calculates the r + b - g function given that the blue boxing color is green.
    :param array: The array to calculate color keying on.
    :param color: The color used for blue boxing.
    :return: The color keys of the array.
    """
    mask = np.ones(3) - np.array(color) * 2.0

    return np.sum(array * mask, axis=-1)


def normalize_array(array):
    """
    Normalizes the given array on axis 0 and 1 to be in the [0, 1] interval.
    :param array: The array to be normalized.
    :return: The normalized array.
    """
    minval = np.min(array, axis=(0, 1))
    maxval = np.max(array, axis=(0, 1))

    return np.divide(np.subtract(array, minval), maxval - minval)


def clean_vectorized(array):
    """
    A vectorized version of cleaning an image for bluebox technique.
    :param array: The input array.
    :param clean_threshold: The threshold for cleaning.
    :return: The cleaning image.
    """
    # Normalize the input array for convenience.
    normalized_array = normalize_array(array)

    # Calculate the color keys. (Note: in the image, RGB values are reversed: BGR)
    color_keys = color_keying(normalized_array, list(reversed(cfg.bluebox_color)))

    # Normalize the color key array.
    color_diff = normalize_array(color_keys)

    # Calculate the mask for blue boxing.
    mask = color_diff > cfg.bluebox_threshold

    # Create numpy arrays for masking purposes.
    ones = np.ones(normalized_array.shape)
    cleaned_image = np.zeros(normalized_array.shape)
    mask_image = np.zeros(normalized_array.shape)

    # Mask out the unnecessary parts.
    cleaned_image[mask] = array[mask]
    mask_image[mask] = ones[mask] * 255

    return cleaned_image, mask_image


def clean_image(source_path, target_path, mask_path, image_extension='.jpg', verbose=True):
    """
    Cleans all the images in the given source path and saves them to target_path.
    :param source_path: The path containing the images, traversed recursively.
    :param target_path: The path to save the cleaned images to.
    :param threshold: The threshold used for cleaning the image.
    :param image_extension: The extension of the images.
    :param verbose: Writes to the consoles which file is being converted.
    :return: -
    """
    # Create target path if not exists, truncate otherwise
    fu.create_truncated_folder(target_path)
    fu.create_truncated_folder(mask_path)

    # Loop over files in the source_path folder
    for image_file in fu.list_files(source_path, image_extension):

        if verbose:
            print("Cleaning " + image_file)

        # Retrieve the file name
        file_name = image_file.split(os.sep)[-2]
        file_full_name = image_file.split(os.sep)[-1]

        # Create directory for the image file
        cleaned_folder_path = os.path.join(target_path, file_name)
        mask_folder_path = os.path.join(mask_path, file_name)

        # Create directory if not exists
        if not os.path.exists(cleaned_folder_path):
            os.mkdir(cleaned_folder_path)

        if not os.path.exists(mask_folder_path):
            os.mkdir(mask_folder_path)

        # Open image file
        image = cv2.imread(image_file)

        # Clean the image background with the given threshold
        cleaned_image, mask = clean_vectorized(image)

        # Write cleaned image to file
        cv2.imwrite(os.path.join(cleaned_folder_path, file_full_name), cleaned_image)
        cv2.imwrite(os.path.join(mask_folder_path, file_full_name), mask)
