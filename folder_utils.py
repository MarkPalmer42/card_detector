
import os
import shutil


def list_files(path, extension):
    """
    Lists all the files with the given extension from the path recursively.
    :param path: The path to search.
    :param extension: The file extension to filter.
    :return: List of files with their paths combined.
    """
    file_list = []

    # Loop over files in the video_path folder
    for v in os.listdir(path):
        # Check if item is directory
        file_path = os.path.join(path, v)
        if os.path.isdir(file_path):
            file_list = file_list + list_files(os.path.join(path, v), extension)
        # Check if the extension matches
        elif v.lower().endswith(extension):
            file_list.append(file_path)

    return file_list


def create_truncated_folder(path):
    """
    Deletes all the contents of the given folder.
    If the folder doesn't exists, creates it.
    :param path: The path of the folder to be truncated or created.
    :return: -
    """
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                # Check if item is file or link
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s, %s' % (file_path, e))
