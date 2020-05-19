
import numpy as np
import folder_utils as fu
import cv2
import os

def transform_image(source_path, target_path, image_extension='.jpg', verbose=True):

    # Create target path if not exists, truncate otherwise
    fu.create_truncated_folder(target_path)

    # Loop over files in the source_path folder
    for image_file in fu.list_files(source_path, image_extension):

        if verbose:
            print('Transforming ' + image_file)

        # Retrieve the file name
        file_name = image_file.split(os.sep)[-2]
        file_full_name = image_file.split(os.sep)[-1]

        # Open image file
        image = cv2.imread(image_file)

        vflipped_image = np.flip(image, axis=0)
        hflipped_image = np.flip(image, axis=1)

        # Write transformed images to file
        cv2.imwrite(os.path.join(target_path, 'vflipped_' + file_full_name), vflipped_image)
        cv2.imwrite(os.path.join(target_path, 'hflipped_' + file_full_name), hflipped_image)

