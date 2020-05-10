
import os
import config as cfg
import anchor_box_configuration as abc
import label_images as lab

original_dataset = os.path.join('dataset', 'original_dataset')
converted_dataset = os.path.join('dataset', 'converted_dataset')
cleaned_dataset = os.path.join('dataset', 'cleaned_dataset')

to_be_labeled_dataset = os.path.join('dataset', 'to_be_labeled')
label_path = os.path.join('dataset', 'labels')

# Convert videos to frame images
#cv.convert_video_to_frames(original_dataset, converted_dataset)

# Clean images
#cd.clean_image(converted_dataset, cleaned_dataset, 4.5)


abconfig = abc.anchor_box_config(cfg.input_width, cfg.input_height, cfg.yolo_grids,
                                 cfg.anchor_box_heights, cfg.anchor_box_aspect_ratios)

lab.label_images(to_be_labeled_dataset, label_path, abconfig)
