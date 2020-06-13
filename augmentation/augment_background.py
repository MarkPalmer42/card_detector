
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

        for j in range(tc.background_augmentation_count):

            crange = tc.bg_color_ranges
            rand = np.random.rand(len(crange))
            color = []

            for k in range(len(crange)):
                color.append(int((crange[k][1] - crange[k][0]) * rand[k] + crange[k][0]))

            augmented_batch.append(modify_background_color(batch[i], masks[i], tuple(color)))

            augmented_labels.append(labels[i])

    return np.array(augmented_batch), np.array(augmented_labels)
