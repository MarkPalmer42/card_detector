
import os
import shutil


def truncate_folder_contents(path):
    """
    Deletes all the contents of the given folder.
    :param path: The path of the folder to be truncated.
    :return: -
    """

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