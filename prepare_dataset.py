"""
    This source file prepares the data set for the training.
    The input is a folder containing video files of a given extension.
    In order for the cleaning to work, the video files must have a blue box
    background color of a pre defined color.

    The output is the cleaned and augmented data set based on the given input.
"""

import os
import config.dataset_config as dc
import utilities.convert_video as cv
import utilities.save_dataset as sd
import cleaning.bluebox_mask as bm
import cleaning.clean_data as cd
import augmentation.transform as tf
import utilities.load_dataset as ld

dataset_paths = {}

dataset_paths['converted'] = os.path.join(dc.dataset_folder, dc.converted_folder)
dataset_paths['cleaned'] = os.path.join(dc.dataset_folder, dc.cleaned_folder)
dataset_paths['mask'] = os.path.join(dc.dataset_folder, dc.mask_folder)
dataset_paths['transformed'] = os.path.join(dc.dataset_folder, dc.transformed_folder)

sd.truncate_folders(dataset_paths.values())

dataset_paths['original'] = os.path.join(dc.dataset_folder, dc.original_folder)

cv.convert_video_to_images(dataset_paths['original'], dataset_paths['converted'])

for batch in ld.load_batch_images(dataset_paths['converted'], 128):

    (dataset, labels), index, len_batches = batch

    print('Batch #' + str(index + 1) + '/' + str(len_batches))

    masks = bm.get_masked_dataset(dataset)

    sd.save_all(dataset_paths['mask'], masks, labels)

    clean_data = cd.clean_dataset(dataset, masks, verbose=dc.verbose)

    sd.save_all(dataset_paths['cleaned'], clean_data, labels)

    transformed_data, transformed_labels = tf.transform_image(clean_data, labels, verbose=dc.verbose)

    sd.save_all(dataset_paths['transformed'], transformed_data, transformed_labels)
