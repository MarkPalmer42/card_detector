"""
    This source file trains the yolo model for object recognititon.
"""
import model.model as m
import config.dataset_config as cd
import os
import config.config as cfg
import model.data_pipeline as dp

source_folder = os.path.join(cd.dataset_folder, cd.training_folder)

train_ds, val_ds, test_ds = dp.load_yolo_dataset(source_folder)

dl_model = m.yolo_model(cfg.yolo_grids, cfg.anchor_box_count, 1)

dl_model.fit(train_ds, val_ds)

dl_model.evaluate(test_ds)
