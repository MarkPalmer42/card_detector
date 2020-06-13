
import numpy as np
import config.cleaning_config as cfg


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


def get_mask_for_bluebox_image(image):
    """
    Calculates a boolean mask for a bluebox image.
    An input image of dimensions (x, y, 3) will yield a (x, y) dimensional boolean mask,
    indicating which pixels should be kept and which ones should be discarded.
    :param image: The input image.
    :return: The bluebox boolean mask.
    """
    # Normalize the input array for convenience.
    normalized_array = normalize_array(image)

    # Calculate the color keys. (Note: in the image, RGB values are reversed: BGR)
    color_keys = color_keying(normalized_array, list(reversed(cfg.bluebox_color)))

    # Normalize the color key array.
    color_diff = normalize_array(color_keys)

    # Calculate and return the mask for blue boxing.
    boolean_mask = color_diff > cfg.bluebox_threshold

    white = np.ones(image.shape) * 255
    black = np.zeros(image.shape)

    black[boolean_mask] = white[boolean_mask]

    return black, boolean_mask


def get_masked_dataset(dataset):
    """
    Retrieves the blue box masks for the data set.
    :param dataset: The input data set.
    :return: The numpy array of the masks.
    """
    masked_dataset = []
    boolean_masks = []

    for image in dataset:

        mask, bool_mask = get_mask_for_bluebox_image(image)

        masked_dataset.append(mask)
        boolean_masks.append(bool_mask)

    return np.array(masked_dataset), np.array(boolean_masks)
