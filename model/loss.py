
import tensorflow as tf
import config.config as cfg


def yolo_loss_fn(y_ground_truth, y_prediction):
    """
    Loss function of the yolo model.
    :param y_ground_truth: The ground truth, the format is [batch_size, grid_w * grid_h * anchor_boxes * (classes + 5)].
    :param y_prediction: The prediction, the format is [batch_size, grid_w * grid_h * anchor_boxes * (classes + 5)].
    :return: Returns the loss associated with the inputs.
    """

    grid_w = cfg.yolo_grids[0]
    grid_h = cfg.yolo_grids[1]
    abc = cfg.anchor_box_count
    cls = cfg.class_count
    lambda_coord = 15
    lambda_noobj = 0.1
    detection_threhold = 0.5

    y_ground_truth = tf.reshape(y_ground_truth, [-1, grid_w, grid_h, abc, cls + 5])
    y_prediction = tf.reshape(y_prediction, [-1, grid_w, grid_h, abc, cls + 5])

    sliced_ground_truth = tf.slice(y_ground_truth, [0, 0, 0, 0, 0], [-1, grid_w, grid_h, 1, 1])

    ge_mat = tf.math.greater_equal(sliced_ground_truth, detection_threhold)
    neg_mat = tf.math.less(sliced_ground_truth, detection_threhold)

    ge_mat = tf.squeeze(ge_mat, [3, 4])
    neg_mat = tf.squeeze(neg_mat, [3, 4])

    masked_pred = tf.boolean_mask(y_prediction, ge_mat)
    masked_ground_truth = tf.boolean_mask(y_ground_truth, ge_mat)

    negated_masked_pred = tf.boolean_mask(y_prediction, neg_mat)
    negated_masked_ground_truth = tf.boolean_mask(y_ground_truth, neg_mat)

    negated_gt_obj = tf.slice(negated_masked_ground_truth, [0, 0, 0], [-1, 1, 1])
    negated_pred_obj = tf.slice(negated_masked_pred, [0, 0, 0], [-1, 1, 1])

    gt_obj = tf.slice(masked_ground_truth, [0, 0, 0], [-1, 1, 1])
    pred_obj = tf.slice(masked_pred, [0, 0, 0], [-1, 1, 1])

    gt_xy = tf.slice(masked_ground_truth, [0, 0, 1], [-1, 1, 2])
    pred_xy = tf.slice(masked_pred, [0, 0, 1], [-1, 1, 2])

    gt_wh = tf.slice(masked_ground_truth, [0, 0, 3], [-1, 1, 2])
    pred_wh = tf.slice(masked_pred, [0, 0, 3], [-1, 1, 2])

    gt_class = tf.slice(masked_ground_truth, [0, 0, 5], [-1, 1, 1])
    pred_class = tf.slice(masked_pred, [0, 0, 5], [-1, 1, 1])

    loss = tf.multiply(tf.reduce_sum(tf.keras.losses.MSE(gt_xy, pred_xy)), lambda_coord)

    loss = tf.add(loss, tf.multiply(tf.reduce_sum(tf.keras.losses.MSE(gt_wh, pred_wh)), lambda_coord))

    loss = tf.add(loss, tf.reduce_sum(tf.keras.losses.binary_crossentropy(gt_obj, pred_obj)))

    loss = tf.add(loss, tf.reduce_sum(tf.keras.losses.binary_crossentropy(gt_class, pred_class)))

    nobj_loss = tf.multiply(tf.reduce_sum(tf.keras.losses.MSE(negated_gt_obj, negated_pred_obj)), lambda_noobj)

    return tf.add(loss, nobj_loss)
