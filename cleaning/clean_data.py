
import numpy as np
import utilities.console_util as cu
import config.cleaning_config as cc


def clean_shadow(image):
    """
    Cleans the noise cause by shadows around the perimeter of the card.
    :param image: The image that went trough the vectorized cleaning process.
    :return: The cleaned image with the newly created mask.
    """
    cleaned_image = np.zeros(image.shape)

    threshold_vals = np.array([255, 255, 255]) * cc.shadow_threshold

    new_mask = image < threshold_vals

    new_mask = np.all(new_mask, axis=-1)

    new_flipped_mask = np.flip(new_mask, axis=[0, 1])

    left_boundaries = np.argmin(new_mask, axis=1)
    right_boundaries = image.shape[1] - np.flip(np.argmin(new_flipped_mask, axis=1))

    horizontal_boundaries = np.array((left_boundaries, right_boundaries)).T

    cleaned_boundary_lists = np.array([np.arange(x + 2, y - 2) if not (x == 0 and y == image.shape[1])
                                       else np.arange(0, 0) for x, y in horizontal_boundaries])

    v = [i for i in range(len(cleaned_boundary_lists)) for j in range(len(cleaned_boundary_lists[i]))]

    t = np.concatenate(cleaned_boundary_lists)

    boundary_tuple = (v, t)

    mask = np.zeros(image.shape, dtype=bool)

    mask[boundary_tuple] = True

    cleaned_image[mask] = image[mask]

    return cleaned_image, mask


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

    # Clean the shadows around the image if necessary.
    if cc.clean_shadows:
        cleaned_image, mask = clean_shadow(cleaned_image)

        white = np.ones(image.shape) * 255
        black = np.zeros(image.shape)

        black[mask] = white[mask]

        mask = black

    return cleaned_image, mask


def clean_dataset(dataset, masks, verbose=True):
    """
    Cleans the given data set.
    :param dataset: The input data set.
    :param masks: The masks to be applied to the data set.
    :param verbose: Whether the process status should be written into the console.
    :return: The cleaned data set.
    """
    cleaned_dataset = []
    shadow_masks = []

    for i in range(len(dataset)):

        cu.print_status('Cleaning', i, len(dataset), verbose)

        cleaned_image, shadow_mask = clean_vectorized(dataset[i], masks[i])
        cleaned_dataset.append(cleaned_image)
        shadow_masks.append(shadow_mask)

    return np.array(cleaned_dataset), np.array(shadow_masks)

