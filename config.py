
"""
    Width of the input image.
"""
input_width = 640

"""
    Height of the input image.
"""
input_height = 480

"""
    Number of color channels in the input image.
"""
color_channels = 3

"""
    Color used for blue box technique
"""
bluebox_color = [0, 0, 1]

"""
    The threshold for cleaning the original dataset.
"""
bluebox_threshold = 0.22

"""
    Number of classes to detect.
"""
class_count = 55

"""
    Dictionary of classes. Optionally it can be specified, but not necessary.
"""
classes = {}

"""
    Number of vertical and horizontal grids for the YOLO object detection algorithm.
"""
yolo_grids = (8, 8)

"""
    List of the anchor box heights. A list of non-negative floating point numbers.
    Expresses the height of the anchor boxes compare to the height of the input image.
    Example: 0.6 means that the anchor box will take up 60% of the image's height.
"""
anchor_box_heights = [0.95, 0.83, 0.71, 0.59, 0.47, 0.35]

""""
    List of tuples of floating point numbers. These values express the aspect ratio of
    the anchor boxes. The anchor boxes' widths will be calculated based on these values
    as well as the aspect ratio of the input image.
"""
anchor_box_aspect_ratios = [(29.0, 44.0)]

