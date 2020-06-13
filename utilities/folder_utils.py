
import os
import shutil


def list_video_files_with_class(path, extension, cls=-1):
    """
    Lists all the files with the given extension from the path recursively with their classes.
    Each folder represents one class, video files in the root folder have their own classes.
    :param path: The path to search.
    :param extension: The file extension to filter.
    :param cls: The current class to be applied.
    :return: List of files with their paths combined.
    """
    file_list = []

    if cls == -1:
        current_class = 0
        use_input_class = False
    else:
        current_class = cls
        use_input_class = True

    # Loop over files in the path folder
    for v in os.listdir(path):

        # Check if item is directory
        file_path = os.path.join(path, v)

        if os.path.isdir(file_path):
            file_list = file_list + list_video_files_with_class(file_path, extension, current_class)

        # Check if the extension matches
        elif v.lower().endswith(extension):
            file_list.append((file_path, current_class))

        if not use_input_class:
            current_class = current_class + 1

    return file_list


def list_files(path, extension):
    """
    Lists all the files with the given extension from the path recursively.
    :param path: The path to search.
    :param extension: The file extension to filter.
    :return: List of files with their paths combined.
    """
    file_list = []

    # Loop over files in the path folder
    for v in os.listdir(path):

        # Check if item is directory
        file_path = os.path.join(path, v)

        if os.path.isdir(file_path):
            file_list = file_list + list_files(file_path, extension)

        # Check if the extension matches
        elif v.lower().endswith(extension):
            file_list.append(file_path)

    return file_list


def has_file_with_extension(path, extension):
    """
    Decides if the given directory path contains (recursively) at least one file of the given extension.
    :param path: The directory path to be checked.
    :param extension: The extension to look for.
    :return: True if the directory contains at least one file of the extension, false otherwise.
    """
    for v in os.listdir(path):
        file_path = os.path.join(path, v)

        if os.path.isdir(file_path) and has_file_with_extension(file_path, extension):
            return True

        elif v.lower().endswith(extension):
            return True

    return False


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
