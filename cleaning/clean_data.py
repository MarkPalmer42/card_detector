
import numpy as np
import utilities.console_util as cu


def clean_vectorized(image, mask):
    """
    A vectorized version of cleaning an image for bluebox technique.
    :param array: The input array.
    :param clean_threshold: The threshold for cleaning.
    :return: The cleaning image.
    """

    # Create numpy array for masking purposes.
    cleaned_image = np.zeros(image.shape)

    # Retrieve the boolean mask from the mask image.
    boolean_mask = mask > 0.5

    # Mask out the unnecessary parts.
    cleaned_image[boolean_mask] = image[boolean_mask]

    return cleaned_image


def clean_dataset(dataset, masks, verbose=True):
    """
    Cleans the given data set.
    :param dataset: The input data set.
    :param masks: The masks to be applied to the data set.
    :param verbose: Whether the process status should be written into the console.
    :return: The cleaned data set.
    """
    cleaned_dataset = []

    for i in range(len(dataset)):

        cu.print_status('Cleaning', i, len(dataset), verbose)

        cleaned_dataset.append(clean_vectorized(dataset[i], masks[i]))

    return np.array(cleaned_dataset)

