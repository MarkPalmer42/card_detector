
import os
import folder_utils as fu
import cv2
import numpy as np


def clean_numpy_array(array, clean_threshold):
    """
    Cleans the background of the input image based on the threshold given.
    The background must be dark, the result of the cleaning is that the dark background
    is converted into a completely black (RGB (0, 0, 0)) background.
    :param array: The input image as a numpy array.
    :param clean_threshold: The threshold to separate the dark background from the content.
    :return: The cleaned image.
    """

    # Calculate the threshold based on the average of the input image.
    avg_threshold = np.average(array) * clean_threshold

    # Traverse the columns of the image.
    for i in range(array.shape[0]):
        first_block_end = -1
        second_block_start = -1

        # Find the first pixel in the row that is not part of the background
        for j in range(array.shape[1]):
            if np.average(array[i][j]) > avg_threshold:
                first_block_end = j
                break

        # Find the last pixel in the row that is not part of the background
        for j in reversed(range(array.shape[1])):
            if np.average(array[i][j]) > avg_threshold:
                second_block_start = j
                break

        # If the entire row is background, set it to perfect black
        if first_block_end is -1 and second_block_start is -1:
            for j in range(array.shape[1]):
                array[i][j] = [0, 0, 0]
        else:
            # Set the background to black at the left side of the actual content.
            if first_block_end is not -1:
                for j in range(first_block_end):
                    array[i][j] = [0, 0, 0]

            # Set the background to black at the right side of the actual content.
            if second_block_start is not -1:
                for j in range(array.shape[1] - 1, second_block_start, -1):
                    array[i][j] = [0, 0, 0]

    return array


def clean_image(source_path, target_path, threshold, image_extension='.jpg', verbose=True):
    """
    Cleans all the images in the given source path and saves them to target_path.
    :param source_path: The path containing the images, traversed recursively.
    :param target_path: The path to save the cleaned images to.
    :param image_extension: The extension of the images.
    :param verbose: Writes to the consoles which file is being converted.
    :return: -
    """
    # Create target path if not exists, truncate otherwise
    fu.created_truncated_folder(target_path)

    # Loop over files in the source_path folder
    for image_file in fu.list_files(source_path, image_extension):

        if verbose:
            print("Cleaning " + image_file)

        # Retrieve the file name
        file_name = image_file.split(os.sep)[-2]
        file_full_name = image_file.split(os.sep)[-1]

        # Create directory for the image file
        folder_path = os.path.join(target_path, file_name)

        # Create directory if not exists
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        # Open image file
        image = cv2.imread(image_file)

        # Clean the image background with the given threshold
        cleaned_image = clean_numpy_array(image, threshold)

        # Write cleaned image to file
        cv2.imwrite(os.path.join(folder_path, file_full_name), cleaned_image)
