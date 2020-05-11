
import os
import folder_utils as fu
import cv2
import numpy as np
import box as ab
import config as cfg


def determine_boundary_box(image):
    """
    Determines the boundary box of an image.
    Only works with black backgrounds and one object in the image.
    :param image: The input numpy array.
    :return: The boundary box matching the object.
    """
    threshold = 10

    # Determine the boundaries of the object in the image.
    boundaries_x = []
    boundaries_y = []

    for i in range(image.shape[0]):

        for j in range(image.shape[1]):
            if np.all(image[i][j] > threshold):
                boundaries_x.append(j)
                boundaries_y.append(i)
                break

        for j in range(image.shape[1] - 1, 0, -1):
            if np.all(image[i][j] > threshold):
                boundaries_x.append(j)
                boundaries_y.append(i)
                break

    # Determine the corners of the boundary box.
    min_x = np.ndarray.min(np.array(boundaries_x))
    max_x = np.ndarray.max(np.array(boundaries_x))
    min_y = np.ndarray.min(np.array(boundaries_y))
    max_y = np.ndarray.max(np.array(boundaries_y))

    # Determine the upper left and lower right corners of the boundary box.
    cord1_x = min_x / image.shape[1]
    cord1_y = min_y / image.shape[0]

    cord2_x = max_x / image.shape[1]
    cord2_y = max_y / image.shape[0]

    # Determine the midpoint coordinates.
    midcord_x = (cord1_x + cord2_x) / 2.0
    midcord_y = (cord1_y + cord2_y) / 2.0

    # Determine the width and height of the boundary box.
    width = cord2_x - cord1_x
    height = cord2_y - cord1_y

    return ab.box(midcord_x, midcord_y, width, height)


def get_yolo_output_from_image(image, anchor_box_conf, class_count, current_class):
    """
    Determines the yolo output of a numpy image.
    :param image: The numpy array of the image.
    :param anchor_box_conf: The anchor box configuration
    :param class_count: The number of classes.
    :param current_class: The class represented in the image.
    :return:
    """
    box = determine_boundary_box(image)

    # Determine the matching anchor box.
    (idi, idj, idk) = anchor_box_conf.get_matching_anchor_box_indices(box)

    anchor_box_per_grid = anchor_box_conf.get_anchor_boxes_per_grid_cell()
    (grid_v, grid_h) = anchor_box_conf.get_grid_dimensions()

    # Create an array of Nones with the correct dimensions.
    output = np.empty((grid_v, grid_h, anchor_box_per_grid, 5 + class_count), dtype=object)

    # Set the object probability to zero.
    for i in range(grid_v):
        for j in range(grid_h):
            for k in range(anchor_box_per_grid):
                output[i][j][k][0] = 0.0

    (x, y, w, h) = box.get_yolo_coords()

    # Set the yolo coords for the anchor box of the object.
    output[idi][idj][idk][0] = 1.0
    output[idi][idj][idk][1] = x
    output[idi][idj][idk][2] = y
    output[idi][idj][idk][3] = w
    output[idi][idj][idk][4] = h

    # Set the classes of the object.
    for i in range(5, 5 + class_count):
        output[idi][idj][idk][i] = 0.0

    output[idi][idj][idk][5 + current_class] = 1.0

    return output


def dict_to_csv(d, delimiter='\t', line_feed='\r\n'):
    """
    Converts a dictionary to a csv string.
    :param d: The input dictionary.
    :param delimiter: The delimiter of the csv string.
    :param line_feed: The line feed used in the csv.
    :return: The csv string.
    """
    output = ""

    for (key, item) in d.items():
        output += key
        output += delimiter.join([str(x) for x in item])
        output += line_feed

    return output


def label_images(source_path, target_path, anchor_box_config, image_extension='.jpg', verbose=True):
    """
    Traverses trough the source_path, labeling all images found.
    The images have to be cleaned with black background.
    :param source_path: The path containing the images, traversed recursively.
    :param target_path: The path to save the labels into.
    :param image_extension: The extension of the images.
    :param verbose: Writes to the consoles which file is being converted.
    :return: -
    """
    # Create target path if not exists, truncate otherwise
    fu.create_truncated_folder(target_path)

    labels = {}

    # Loop over files in the source_path folder
    for image_file in fu.list_files(source_path, image_extension):

        if verbose:
            print("Labeling " + image_file)

        # Open image file
        image = cv2.imread(image_file)

        # Image text
        img_text = image_file.split(os.path.sep)[-1]

        # Retrieve YOLO output for the input image
        labels[img_text] = np.ndarray.flatten(get_yolo_output_from_image(image, anchor_box_config, cfg.class_count, 1))

    file_path = os.path.join(target_path, 'labels.json')

    fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_TEXT)

    os.write(fd, dict_to_csv(labels).encode())

    os.close(fd)


