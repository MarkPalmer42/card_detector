
import os
import convert_video as cv


original_dataset = os.path.join('dataset', 'original_dataset')
converted_dataset = os.path.join('dataset', 'converted_dataset')

cv.convert_video_to_frames(original_dataset, converted_dataset, '.mov')
