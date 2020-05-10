
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
yolo_grids = (32, 32)

"""
    List of the anchor box heights. A list of non-negative floating point numbers.
    Expresses the height of the anchor boxes compare to the height of the input image.
    Example: 0.6 means that the anchor box will take up 60% of the image's height.
"""
anchor_box_heights = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.45, 0.35]

""""
    List of tuples of floating point numbers. These values express the aspect ratio of
    the anchor boxes. The anchor boxes' widths will be calculated based on these values
    as well as the aspect ratio of the input image.
"""
anchor_box_aspect_ratios = [(5.8, 8.8), (5.8, 8.4), (5.8, 8.1), (5.8, 7.8), (5.8, 7.4), (5.8, 7.1), (5.8, 6.8)]

