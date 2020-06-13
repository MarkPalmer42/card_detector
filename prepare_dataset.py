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
import label.label_images as li
import yolo.anchor_box_configuration as abc
import config.config as cfg
import augmentation.augment_background as ab

dataset_paths = {}

dataset_paths['converted'] = os.path.join(dc.dataset_folder, dc.converted_folder)
dataset_paths['cleaned'] = os.path.join(dc.dataset_folder, dc.cleaned_folder)
dataset_paths['mask'] = os.path.join(dc.dataset_folder, dc.mask_folder)
dataset_paths['transformed'] = os.path.join(dc.dataset_folder, dc.transformed_folder)
dataset_paths['transformed_mask'] = os.path.join(dc.dataset_folder, dc.transformed_mask_folder)
dataset_paths['labeled'] = os.path.join(dc.dataset_folder, dc.labeled_folder)
dataset_paths['augmented'] = os.path.join(dc.dataset_folder, dc.augmented_folder)

sd.truncate_folders(dataset_paths.values())

dataset_paths['original'] = os.path.join(dc.dataset_folder, dc.original_folder)

cv.convert_video_to_images(dataset_paths['original'], dataset_paths['converted'])

for batch in ld.load_batch_images(dataset_paths['converted'], 16):

    (dataset, labels), index, len_batches = batch

    print('Batch #' + str(index + 1) + '/' + str(len_batches))

    masks, boolean_masks = bm.get_masked_dataset(dataset)

    sd.save_all(dataset_paths['mask'], masks, labels)

    dataset = cd.clean_dataset(dataset, masks, verbose=dc.verbose)

    sd.save_all(dataset_paths['cleaned'], dataset, labels)

    dataset = tf.transform_image_batch(dataset, verbose=dc.verbose)
    masks = tf.transform_image_batch(masks, verbose=dc.verbose)
    boolean_masks = tf.transform_image_batch(boolean_masks, verbose=dc.verbose)
    labels = tf.transform_label_batch(labels)

    sd.save_all(dataset_paths['transformed'], dataset, labels)


    abc_config = abc.anchor_box_config(cfg.image_dims, cfg.yolo_grids, cfg.ab_heights, cfg.ab_aspect_ratios)

    #masks, boolean_masks = bm.get_masked_dataset(dataset)

    yolo_batch, labels = li.label_yolo_image_batch(dataset, labels, masks, abc_config)

    sd.save_all(dataset_paths['labeled'], yolo_batch, labels)

    sd.save_all(dataset_paths['transformed_mask'], masks, labels)

    dataset, labels = ab.batch_augment_background(dataset, masks, labels)

    sd.save_all(dataset_paths['augmented'], dataset, labels)
