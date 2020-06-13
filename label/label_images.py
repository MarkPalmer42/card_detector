
import cv2
import numpy as np
from yolo import box as ab


def get_boundary_box_from_image(image, mask):
    """
    Calculates the boundary box for the input image.
    :param image: The input image.
    :param mask: The mask associated with the image.
    :return: The boundary box of the object in the image.
    """
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
    max_filter_x = np.ones((mask.shape[0], mask.shape[2])) * image.shape[1]
    min_filter_x = np.ones((mask.shape[0], mask.shape[2])) * -1

    max_filter_y = np.ones((mask.shape[1], mask.shape[2])) * image.shape[0]
    min_filter_y = np.ones((mask.shape[1], mask.shape[2])) * -1

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


def get_yolo_output_from_image(boundary_box, anchor_box_conf, class_count, current_class):
    """
    Determines the yolo output of a numpy image.
    :param boundary_box: The boundary box of the image.
    :param anchor_box_conf: The anchor box configuration
    :param class_count: The number of classes.
    :param current_class: The class represented in the image.
    :return:
    """

    # Determine the matching anchor box.
    (idi, idj, idk) = anchor_box_conf.get_matching_anchor_box_indices(boundary_box)

    anchor_box_per_grid = anchor_box_conf.get_anchor_boxes_per_grid_cell()
    (grid_v, grid_h) = anchor_box_conf.get_grid_dimensions()

    # Create an array of Nones with the correct dimensions.
    output = np.empty((grid_v, grid_h, anchor_box_per_grid, 5 + class_count), dtype=object)

    # Set the object probability to zero.
    for i in range(grid_v):
        for j in range(grid_h):
            for k in range(anchor_box_per_grid):
                output[i][j][k][0] = 0.0

    (x, y, w, h) = boundary_box.get_yolo_coords()

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


def get_labeled_image(image, boundary_box, text=None, display_color=(0, 0, 255)):
    """
    Generates a labeled image.
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

    return image


def label_yolo_image_batch(batch, labels, masks, abc_config):
    """
    Labels a batch of images.
    :param batch: The batch of images.
    :param labels: Labels for the batch of images containing one value per example in range (0, class_count).
    :param masks: The masks associated with the batch of images.
    :param abc_config: Anchor box configuration of the YOLO algorithm.
    :return: A batch of images with displayed labels as well as the YOLO labels for the batch.
    """
    class_count = int(np.max(labels) + 1)

    yolo_labels = []
    labeled_batch = []

    for i in range(len(batch)):

        boundary_box = get_boundary_box_from_image(batch[i], masks[i])

        yolo_output = get_yolo_output_from_image(boundary_box, abc_config, class_count, int(labels[i]))

        labeled_batch.append(get_labeled_image(batch[i], boundary_box))

        yolo_labels.append(yolo_output)

    return np.array(labeled_batch), np.array(yolo_labels)
