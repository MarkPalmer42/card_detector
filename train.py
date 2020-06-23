
import model.model as m
import utilities.load_dataset as ld
import config.dataset_config as cd
import os
import numpy as np
import tensorflow as tf
import config.config as cfg

source_folder = os.path.join(cd.dataset_folder, cd.training_folder)

train_x = ld.load_dataset(source_folder)
train_y = ld.load_all_labels(source_folder, True)

train_x = train_x / 255.

dl_model = m.yolo_model(cfg.yolo_grids, cfg.anchor_box_count, 1)

dl_model.print_summary()

dl_model.fit(train_x, train_y)
