
import numpy as np
import config.transform_config as tc


def modify_background_color(image, mask, color):

    new_image = np.ones(image.shape) * color

    boolean_mask = mask > 10

    new_image[boolean_mask] = image[boolean_mask]

    return new_image


def batch_augment_background(batch, masks, labels):

    augmented_batch = []
    augmented_labels = []

    for i in range(len(batch)):

        for color in tc.background_colors:

            augmented_batch.append(modify_background_color(batch[i], masks[i], color))

            augmented_labels.append(labels[i])

    return np.array(augmented_batch), np.array(augmented_labels)
