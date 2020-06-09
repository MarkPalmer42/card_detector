
import numpy as np


class box:
    """
        This class represents an anchor box or a boundary box.
    """

    def __init__(self, x, y, w, h):
        """
        Initializes the box object with the values used for the YOLO algorithm.
        :param x: Float, represents the center point of the box relative to the image width.
        :param y: Float, represents the center point of the box relative to the image height.
        :param w: Float, represents the width of the box relative to the image width.
        :param h: Float, represents the height of the box relative to the image height.
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def calculate_iou(self, box):
        """
        Calculates the intersect over union of two box objects.
        :param box: The other box object.
        :return: Float. The intersect over union ratio.
        """
        (box1_x1, box1_y1, box1_x2, box1_y2) = self.get_iou_coords()
        (box2_x1, box2_y1, box2_x2, box2_y2) = box.get_iou_coords()

        # Finding the top left and bottom right corners of the intersection.
        xi1 = np.maximum(box1_x1, box2_x1)
        yi1 = np.maximum(box1_y1, box2_y1)
        xi2 = np.minimum(box1_x2, box2_x2)
        yi2 = np.minimum(box1_y2, box2_y2)

        # Calculating the width and height of the intersection.
        inter_width = xi2 - xi1
        inter_height = yi2 - yi1

        # Calculating the area of the intersection.
        inter_area = np.maximum(0, inter_width) * np.maximum(0, inter_height)

        # Calculating the area of the two boxes.
        self_area = self.w * self.h
        box_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

        # Calculating the area of the union.
        union_area = self_area + box_area - inter_area

        return inter_area / union_area

    def get_yolo_coords(self):
        """
        Returns the coordinates of the box as they are used in the YOLO algorithm.
        :return: A tuple of four floats: the coordinates of the center point, the width and the height.
        """
        return self.x, self.y, self.w, self.h

    def get_iou_coords(self):
        """
        Returns the coordinates of the top left and bottom right corner of the box.
        :return: A tuple of four floats: the coordinates of the top left and bottom right corner.
        """
        return self.x - self.w / 2, self.y - self.h / 2, self.x + self.w / 2, self.y + self.h / 2


