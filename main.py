
import os
import clean_data as cd
import convert_video as cv

original_dataset = os.path.join('dataset', 'original_dataset')
converted_dataset = os.path.join('dataset', 'converted_dataset')
cleaned_dataset = os.path.join('dataset', 'cleaned_dataset')

# Convert videos to frame images
cv.convert_video_to_frames(original_dataset, converted_dataset)

# Clean images
cd.clean_image(converted_dataset, cleaned_dataset, 4.5)
