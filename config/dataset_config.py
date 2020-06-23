
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
    Folder of the transformed (flipped and rotated) mask data set in the 'dataset_folder' folder.
"""
transformed_mask_folder = 'transformed_mask_dataset'

"""
    Folder of the labeled data set in the 'dataset_folder' folder.
"""
labeled_folder = 'labeled_dataset'

"""
    Folder of the augmented data set in the 'dataset_folder' folder.
"""
augmented_folder = 'augmented_dataset'

"""
    Folder of the training data set in the 'dataset_folder' folder.
"""
training_folder = 'augmented_dataset'

"""
    True if all steps during the preparation should be saved, false otherwise.
"""
save_all_data = True

"""
    The frequency of keeping images from the input video files. 1 does not skip, 2 keeps each 2nd,
    3 keeps each 3rd, etc...
"""
keep_images = 40

"""
    True if the process status of the data set preparation should be written on the console, false otherwise.
"""
verbose = False
