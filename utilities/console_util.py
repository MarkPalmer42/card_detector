

def print_status(message, current_status, length, verbose):
    """
    Print the current status of the preparation of the data set.
    :param message: The message.
    :param current_status: Current counter of the operation (from 0 to length -1).
    :param length: The length of the operation
    :param verbose: True if printing ot console is allowed.
    :return: -
    """
    if verbose:
        percentage = int((float(current_status + 1) / length) * 100)

        previous_percentage = int((float(current_status) / length) * 100)

        if not percentage == previous_percentage:
            print(message + ' ' + str(percentage) + '%')
