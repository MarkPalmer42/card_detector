
import tensorflow as tf
import numpy as np


def transform_coordinates(midpoints, lengths):
    """
    Transforms coordinates from yolo format to upper-left and lower-right corne
    points to make the IOU calculation easier.
    :param midpoints: The midpoints, the format is [batch_size, 1, 1, 2].
    :param lengths: The lengths (widths and heights), the format is [batch_size, 1, 1, 2].
    :return: The transformed coordinates.
    """
    half_length = tf.divide(lengths, 2.0)

    point1 = tf.subtract(midpoints, half_length)
    point2 = tf.add(midpoints, half_length)

    return point1, point2


def calculate_iou(ground_truth, prediction):
    """
    Calculates the loss for the bounding box prediction.
    Only pass inputs that pass the detection treshold.
    :param ground_truth: The ground truth, the format is [batch_size, grid_w, grid_h, anchor_boxes, (classes + 5)].
    :param prediction: The predictions, the format is [batch_size, grid_w, grid_h, anchor_boxes, (classes + 5)].
    :return: The loss associated with the input.
    """
    ground_truth_midpoint = tf.slice(ground_truth, [0, 0, 1], [-1, 1, 2])
    ground_truth_length = tf.slice(ground_truth, [0, 0, 3], [-1, 1, 2])

    prediction_midpoint = tf.slice(prediction, [0, 0, 1], [-1, 1, 2])
    prediction_lengths = tf.slice(prediction, [0, 0, 3], [-1, 1, 2])

    ground_truth_point1, ground_truth_point2 = transform_coordinates(ground_truth_midpoint, ground_truth_length)
    prediction_point1, prediction_point2 = transform_coordinates(prediction_midpoint, prediction_lengths)

    point1_max = tf.maximum(prediction_point1, ground_truth_point1)
    point2_min = tf.minimum(prediction_point2, ground_truth_point2)

    inter_length = tf.subtract(point2_min, point1_max)

    zero_tensor = tf.zeros(tf.shape(inter_length))

    intersection_area = tf.reduce_prod(tf.maximum(inter_length, zero_tensor), axis=-1)

    ground_truth_area = tf.reduce_prod(ground_truth_length, axis=-1)
    pred_area = tf.reduce_prod(prediction_lengths, axis=-1)

    union_area = tf.subtract(tf.add(ground_truth_area, pred_area), intersection_area)

    return tf.divide(intersection_area, union_area)


def calculate_class_difference(ground_truth, prediction):
    """
    Calculates the loss for the class probabilities.
    Only pass inputs that pass the detection treshold.
    :param ground_truth: The ground truth, the format is [batch_size, grid_w, grid_h, anchor_boxes, (classes + 5)].
    :param prediction: The predictions, the format is [batch_size, grid_w, grid_h, anchor_boxes, (classes + 5)].
    :return: The loss associated with the input.
    """
    class_count = 1

    ground_truth_class = tf.slice(ground_truth, [0, 0, 5], [-1, 1, class_count])

    prediction_class = tf.slice(prediction, [0, 0, 5], [-1, 1, class_count])

    difference = tf.subtract(ground_truth_class, prediction_class)

    square = tf.square(difference)

    return tf.divide(tf.reduce_sum(square, axis=-1), 2.0)


def yolo_loss_function(y_ground_truth, y_prediction):
    """
    Loss function of the yolo model.
    :param y_ground_truth: The ground truth, the format is [batch_size, grid_w * grid_h * anchor_boxes * (classes + 5)].
    :param y_prediction: The prediction, the format is [batch_size, grid_w * grid_h * anchor_boxes * (classes + 5)].
    :return: Returns the loss associated with the inputs.
    """
    y_ground_truth = tf.reshape(y_ground_truth, [-1, 8, 8, 6, 6])
    y_prediction = tf.reshape(y_prediction, [-1, 8, 8, 6, 6])

    y_diff = tf.subtract(y_ground_truth, y_prediction)

    y_diff_squared = tf.square(y_diff)

    mask = np.zeros(y_prediction.shape[-1], dtype=float)
    mask[0] = 1.0
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    masked = tf.multiply(y_diff_squared, mask)

    detection_loss = tf.reduce_sum(masked)

    ge_mat = tf.slice(y_ground_truth, [0, 0, 0, 0, 0], [-1, 8, 8, 1, 1])

    ge_mat = tf.math.greater_equal(ge_mat, 0.5)

    ge_mat = tf.squeeze(ge_mat, [3, 4])

    masked_pred = tf.boolean_mask(y_prediction, ge_mat)

    masked_ground_truth = tf.boolean_mask(y_ground_truth, ge_mat)

    iou = calculate_iou(masked_ground_truth, masked_pred)

    iou_loss = tf.reduce_sum(tf.square(tf.subtract(tf.ones(tf.shape(iou)), iou)))

    class_diff = calculate_class_difference(masked_ground_truth, masked_pred)

    class_loss = tf.reduce_sum(class_diff)

    detection_loss = tf.multiply(detection_loss, 3.0)
    class_loss = tf.multiply(class_loss, 2.0)

    return tf.add(tf.add(detection_loss, class_loss), iou_loss)

