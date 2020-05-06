
import cv2
import os
import truncate_folder


def convert_video_to_frames(video_path, target_path, video_extension, create_dir=True, verbose=True):
    """
    Converts all the videos from the video_path to jpeg frames and stores them in the target_path.
    The target path will be truncated before the conversion.
    :param video_path: The path of the input video files.
    :param target_path: The path to store the generated jpeg images. Will be truncated.
    :param video_extension: The extension of the videos to convert.
    :param create_dir: If set to true, a folder will be created in target_path for each video.
    :param verbose: Writes to the consoles which file is being converted.
    :return: -
    """

    # Create target path if not exists, truncate otherwise
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    else:
        truncate_folder.truncate_folder_contents(target_path)

    # Loop over files in the video_path folder
    for v in os.listdir(video_path):

        # Check if the extension matches
        if v.lower().endswith(video_extension):

            if verbose:
                print("Converting " + v)

            folder_path = target_path

            # Remove extension from file name
            file_name = v[0: len(v) - len(video_extension)]

            # Create directory for the video file if necessary
            if create_dir:
                folder_path = os.path.join(target_path, file_name)
                os.mkdir(folder_path)

            video_file = os.path.join(video_path, v)

            # Open video file
            vidcap = cv2.VideoCapture(video_file)
            success, image = vidcap.read()
            count = 0

            # Loop over the frames of the video
            while success:
                filename = file_name + "_frame%d.jpg" % count
                cv2.imwrite(os.path.join(folder_path, filename), image)
                success, image = vidcap.read()
                count += 1




