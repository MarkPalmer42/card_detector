
import numpy as np
import config.transform_config as tc


def modify_background_color(image, mask, color):
    """
    Modifies the background of the given image.
    :param image: The image to be modified.
    :param mask: The mask of the image.
    :param color: The color to be used.
    :return: The modified image.
    """
    new_image = np.ones(image.shape) * color

    boolean_mask = mask > 10

    new_image[boolean_mask] = image[boolean_mask]

    return new_image


def batch_augment_background(batch, masks, labels):
    """
    Augments the background of the given batch of examples.
    :param batch: The batch to be augmented.
    :param masks: The masks of the batch examples.
    :param labels: The labels of the examples.
    :return: The augmented batch and labels.
    """
    augmented_batch = []
    augmented_labels = []

    for i in range(len(batch)):

        for j in range(tc.background_augmentation_count):

            crange = tc.bg_color_ranges
            rand = np.random.rand(len(crange))
            color = []

            for k in range(len(crange)):
                color.append(int((crange[k][1] - crange[k][0]) * rand[k] + crange[k][0]))

            augmented_batch.append(modify_background_color(batch[i], masks[i], tuple(color)))

            augmented_labels.append(labels[i])

    return np.array(augmented_batch), np.array(augmented_labels)
