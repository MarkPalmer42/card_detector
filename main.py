
import os
import config as cfg
import anchor_box_configuration as abc
import label_images as lab
import convert_video as cv
import clean_data as cd
import transform as t

original_dataset = os.path.join('dataset', 'original_dataset')
converted_dataset = os.path.join('dataset', 'converted_dataset')
cleaned_dataset = os.path.join('dataset', 'cleaned_dataset')
mask_dataset = os.path.join('dataset', 'mask_dataset')

transformed_dataset = os.path.join('dataset', 'transformed_dataset')

to_be_labeled_dataset = os.path.join('dataset', 'to_be_labeled')
label_path = os.path.join('dataset', 'labels')

# Convert videos to frame images
#cv.convert_video_to_frames(original_dataset, converted_dataset)

# Clean images
#cd.clean_image(converted_dataset, cleaned_dataset, mask_dataset)


# abconfig = abc.anchor_box_config(cfg.input_width, cfg.input_height, cfg.yolo_grids,
#                                     cfg.anchor_box_heights, cfg.anchor_box_aspect_ratios)
#
# lab.get_boundary_box(cleaned_dataset)

t.transform_image(cleaned_dataset, transformed_dataset)
