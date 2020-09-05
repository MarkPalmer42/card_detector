
"""
    Rotations to be applied to the data set during augmentation.
"""
rotations = [75, 45, 15]

"""
    Whether or not the image should be flipped horizontally during transformation.
"""
flip_horizontally = True

"""
    Whether or not the image should be flipped vertically during transformation.
"""
flip_vertically = True

"""
    Ranges for randomized background colors.
"""
bg_color_ranges = [(128, 255), (128, 255), (128, 255)]

"""
    Number of different background colors an example should be augmented with.
"""
background_augmentation_count = 1
