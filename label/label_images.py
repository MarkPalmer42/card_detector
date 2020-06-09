
import os
from utilities import folder_utils as fu
import cv2
import numpy as np
from yolo import box as ab
from config import config as cfg
from cleaning import clean_data as cd


def get_boundary_box_from_image(image):
    """
    Calculates the boundary box for the input image.
    :param image: The input image.
    :return: The boundary box of the object in the image.
    """
    # Gets the mask used for cleaning the data.
    mask = cd.get_mask_for_bluebox_image(image)

    # Determines which columns and rows don't have object in them
    # (columns and rows that are completely masked out).
    vertical_mask = np.any(mask, axis=0)
    horizontal_mask = np.any(mask, axis=1)

    # Determines the boundaries of the object in the image from
    # the left side as well as the right side.
    left_boundaries = np.argmax(mask, axis=1)
    right_boundaries = np.maximum(image.shape[1] - np.argmax(np.flip(mask, 1), axis=1), 0.0)

    # Determines the boundaries of the object in the image from
    # the upper and lower side of the image.
    upper_boundaries = np.argmax(mask, axis=0)
    lower_boundaries = np.maximum(image.shape[0] - np.argmax(np.flip(mask, 0), axis=0), 0.0)

    # Creating filter numpy arrays for maximum and minimum search later.
    max_filter_x = np.ones(mask.shape[0]) * image.shape[1]
    min_filter_x = np.ones(mask.shape[0]) * -1

    max_filter_y = np.ones(mask.shape[1]) * image.shape[0]
    min_filter_y = np.ones(mask.shape[1]) * -1

    # Mask the unneeded data from the boundary arrays with irrelevant data
    # for the min / max search to be ignored. (The original boundary arrays
    # have zeros in the columns and rows that do not have object in them,
    # these are being ignored with this method).
    max_filter_x[horizontal_mask] = left_boundaries[horizontal_mask]
    min_filter_x[horizontal_mask] = right_boundaries[horizontal_mask]

    max_filter_y[vertical_mask] = upper_boundaries[vertical_mask]
    min_filter_y[vertical_mask] = lower_boundaries[vertical_mask]

    # Find the upper left and lower right corners of the object in the array.
    x1 = np.min(max_filter_x)
    y1 = np.min(max_filter_y)

    x2 = np.max(min_filter_x)
    y2 = np.max(min_filter_y)

    # Translate coordinates into YOLO format.
    w = (x2 - x1) / image.shape[1]
    h = (y2 - y1) / image.shape[0]

    midcord_x = x1 / image.shape[1] + w / 2.0
    midcord_y = y1 / image.shape[0] + h / 2.0

    return ab.box(midcord_x, midcord_y, w, h)


def get_yolo_output_from_image(image, anchor_box_conf, class_count, current_class):
    """
    Determines the yolo output of a numpy image.
    :param image: The numpy array of the image.
    :param anchor_box_conf: The anchor box configuration
    :param class_count: The number of classes.
    :param current_class: The class represented in the image.
    :return:
    """
    box = get_boundary_box_from_image(image)

    display_image(image, box)

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


def display_image(image, boundary_box, text=None, display_color=(0, 0, 255)):
    """
    Displays the image.
    :param image: The image to be displayed.
    :param boundary_box: The boundary box to display on the image.
    :param text: The text to display on the boundary box.
    :param display_color: The color of the bounding box displayed in BGR format.
    :return: -
    """
    (x1, y1, x2, y2) = boundary_box.get_iou_coords()

    x1 = int(x1 * image.shape[1])
    x2 = int(x2 * image.shape[1])
    y1 = int(y1 * image.shape[0])
    y2 = int(y2 * image.shape[0])

    image = cv2.rectangle(image, (x1, y1), (x2, y2), display_color, 1)

    if not text is None:
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 1)

    cv2.imshow('image', image)
    cv2.waitKey(0)


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


def get_boundary_box(source_path, image_extension='.jpg', verbose=True):

    # Create target path if not exists, truncate otherwise
    #fu.create_truncated_folder(target_path)


    # Loop over files in the source_path folder
    for image_file in fu.list_files(source_path, image_extension):

        if verbose:
            print("Labeling " + image_file)

        # Open image file
        image = cv2.imread(image_file)

        # Image text
        image_text = image_file.split(os.path.sep)[-1]

        # Get boundary box of image.
        boundary_box = get_boundary_box_from_image(image)

        display_image(image, boundary_box)



def label_images(source_path, target_path, anchor_box_config, image_extension='.jpg', verbose=True):
    """
    Traverses trough the source_path, label all images found.
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


