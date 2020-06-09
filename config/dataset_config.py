
"""
    Folder of the data set.
"""
dataset_folder = 'dataset'

"""
    Folder of the original data set (the input video files) in the 'dataset_folder' folder.
"""
original_folder = 'original_dataset'

"""
    Folder of the converted data set (the images saved from the videos) in the 'dataset_folder' folder.
"""
converted_folder = 'converted_dataset'

"""
    Folder of the cleaned data set (the blue box color removed) in the 'dataset_folder' folder.
"""
cleaned_folder = 'cleaned_dataset'

"""
    Folder of the masks used to clean the data set in the 'dataset_folder' folder.
"""
mask_folder = 'mask_dataset'

"""
    Folder of the transformed (flipped and rotated) data set in the 'dataset_folder' folder.
"""
transformed_folder = 'transformed_dataset'

"""
    True if all steps during the preparation should be saved, false otherwise.
"""
save_all_data = True

"""
    The frequency of skipping images from the input video files. 1 does not skip, 2 skips each 2nd,
    3 skips each 3rd, etc...
"""
skip_images = 1
