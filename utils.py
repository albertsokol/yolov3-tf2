import numpy as np
from configparser import ConfigParser


def normalise_anchors(anchors, cnn_input_size):
    """
    Normalise anchors from their raw pixel values to proportions of the CNN input size (the image dimensions).
    Takes in the list of anchor [width, height] lists and returns normalised anchors as a list of tuples.

    Parameters:
    anchors: list: the list of anchors to normalise and convert to tuples
    cnn_input_size: int: the input size of the network; images will be resized to this size

    :return: anchors as a list of (width, height) tuples
    """
    anchor_pairs = []

    for anchor in anchors:
        anchor = [i / cnn_input_size for i in anchor]
        anchor = list(anchor)
        anchor_pairs.append(anchor)

    return anchor_pairs


def create_class_dict(df):
    """
    Finds all the classes and creates one-hot vector encodings for them by finding unique class names in the dataframe.

    Parameters:
    df: pandas dataframe: the dataframe of training examples, where classes are under the column named 'class'

    :return: dictionary with class names as keys and one-hot vector encodings as values
    """
    class_dict = {}
    classes = df['class'].unique()
    print(classes)

    for i, class_ in enumerate(classes):
        class_dict[class_] = np.zeros([len(classes), 1])
        class_dict[class_][i] = 1

    return class_dict


def write_config_file(class_dict, cnn_input_size, anchors_list):
    """
    Create a config.ini file to save the model hyperparameters. This is used to ensure consistency across the trained
    network and the predict.py file.

    Parameters:
    class_dict: dictionary: all the classes; class ordering is important to get consistent predictions
    cnn_input_size: int: the input size of the network; images will be resized to this size
    anchors_list: list: the list of anchors as normalised (width, height) tuples
    """
    scale = {0: 'small_', 1: 'medium_', 2: 'large_'}
    config = ConfigParser()
    config.read('config.ini')

    # Add class keys in the correct order
    if not config.has_section('classes'):
        config.add_section('classes')
    for i, key in enumerate(class_dict.keys()):
        config.set('classes', str(i), key)

    # Add the cnn input size
    if not config.has_section('input size'):
        config.add_section('input size')
    config.set('input size', 'input size', str(cnn_input_size))

    # Add the normalised anchors in order with their width and height values stored
    if not config.has_section('anchors'):
        config.add_section('anchors')
    for i, anchors in enumerate(anchors_list):
        for j, anchor in enumerate(anchors):
            config.set('anchors', scale[i] + str(j), str(anchor[0]) + ', ' + str(anchor[1]))

    # Write the file
    with open('config.ini', 'w') as f:
        config.write(f)


def read_config_file():
    """ Read the class names, cnn input size and create the anchors list for use in the prediction model. """
    config = ConfigParser()
    config.read('config.ini')

    # Read the classes into a list, used for labelling the predicted bboxes
    classes = []
    for key, value in config.items('classes'):
        classes.append(value)

    # Get the input size of the network
    cnn_input_size = config.getint('input size', 'input size')

    # Read the anchors and re-create the list containing them at correct scales
    anchors = [[], [], []]
    for key, value in config.items('anchors'):
        anchor = tuple(float(x) for x in value.split(sep=', '))
        if key.startswith('small'):
            anchors[0].append(anchor)
        if key.startswith('medium'):
            anchors[1].append(anchor)
        if key.startswith('large'):
            anchors[2].append(anchor)

    return classes, cnn_input_size, anchors
