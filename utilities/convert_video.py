
import cv2
import numpy as np
from utilities import folder_utils as fu
import os
import utilities.save_dataset as sd
import config.dataset_config as dc


def convert_video_to_images(source_path, target_path, video_extension='mov', image_extension='jpg', verbose=True):
    """
    Converts the data set from the input videos to images.
    :param source_path: The path of the input videos.
    :param target_path: The path to save the images into.
    :param video_extension: The extension of the input video files.
    :param verbose: Whether or not write the current state to the console.
    :return: The data set and the labels
    """

    fu.create_truncated_folder(target_path)

    labels = []
    index = 0

    # Loop over files in the video_path folder
    for video_file, cls in fu.list_video_files_with_class(source_path, video_extension):

        if verbose:
            print("Loading " + video_file)

        # Open video file
        vidcap = cv2.VideoCapture(video_file)
        success, image = vidcap.read()
        counter = 0

        # Loop over the frames of the video
        while success:

            if counter % dc.keep_images == 0:
                cv2.imwrite(os.path.join(target_path, str(index) + '.' + image_extension), image)
                labels.append(cls)
                index = index + 1

            success, image = vidcap.read()
            counter = counter + 1

    sd.save_labels(target_path, np.array(labels))

