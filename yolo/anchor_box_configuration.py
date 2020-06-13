from yolo import box as ab
import numpy as np


class anchor_box_config:
    """
    The class representing the configuration of the anchor boxes used for the YOLO algorithm.
    There are fixed anchor boxes for each grid cell that are calculated based on the configuration given.
    """

    def __init__(self, img_d, grids, box_h, aspect_ratios):
        """
        Initializes the anchor box configuration.
        The anchor boxes are calculated based on the list of heights and
        the list of aspect ratios given. There will be a total of
        len(box_h) * len(aspect_ratios) anchor boxes for each grid cell.
        :param img_d: Dimensions of the image, tuple of width and height.
        :param grids: Tuple of ints, the vertical and horizontal YOLO grid cells.
        :param box_h: List of anchor box heights used for each YOLO grid cells.
                      This is a list of floats representing the anchor box
                      height relative to the input image height.
        :param aspect_ratios: List of tuples containing floats that represent
                      the aspect ratios of the anchor boxes.
        """
        self.grids_v = grids[0]
        self.grids_h = grids[1]

        img_w = img_d[0]
        img_h = img_d[1]

        self.anchor_boxes_per_grid = len(box_h) * len(aspect_ratios)

        # Creating an empty array for the anchor box objects.
        self.anchor_boxes = np.empty((self.grids_v, self.grids_h, self.anchor_boxes_per_grid), dtype=object)

        image_aspect_ratio = img_h / img_w

        # Traverse trough the vertical and horizontal grid cells.
        for i in range(grids[0]):
            for j in range(grids[1]):

                # Center of the current grid cell.
                grid_x = (j + 0.5) / grids[1]
                grid_y = (i + 0.5) / grids[0]

                # Index of current anchor box.
                k = 0

                # Traverse trough the anchor box heights and aspect ratios.
                for height in box_h:
                    for a_ratio in aspect_ratios:

                        # Calculate the aspect ratio of the anchor box
                        ratio = a_ratio[0] / a_ratio[1]

                        # Calculate the weight based on the height,
                        # aspect ratio of the input image and anchor box.
                        width = height * ratio * image_aspect_ratio

                        self.anchor_boxes[i][j][k] = ab.box(grid_x, grid_y, width, height)
                        k = k + 1

    def get_acnhor_boxes(self):
        """
        Retrieves the calculated anchor box objects.
        :return: The list of anchor box objects.
        """
        return self.anchor_boxes

    def get_grid_dimensions(self):
        """
        Gets the dimensions of the grid.
        :return: The dimensions of the grid.
        """
        return self.grids_v, self.grids_h

    def get_anchor_boxes_per_grid_cell(self):
        """
        Get the number of anchor boxes of each grid cell.
        :return: The number of anchor boxes per grid cell.
        """
        return self.anchor_boxes.shape[2]

    def get_matching_anchor_box_indices(self, box):
        """
        Finds the anchor box in the configuration that matches the given box the most.
        :param box: The boundary box to find an anchor box for.
        :return: Tuple of three ints: the grid cells indices and the anchor box index,
                The anchor box object itself is also returned.
        """
        maxval = -1
        idi, idj, idk = -1, -1, -1

        for i in range(self.grids_v):
            for j in range(self.grids_h):
                for k in range(self.anchor_boxes_per_grid):

                    iou = self.anchor_boxes[i][j][k].calculate_iou(box)

                    if iou > maxval:
                        maxval = iou
                        idi, idj, idk = i, j, k

        return idi, idj, idk
