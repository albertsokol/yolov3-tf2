import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers

import utils
from callbacks import AP50Checkpoint, LearningRateFinder
from generator import YoloGenerator
from losses import wrapper_loss
from model import create_model, create_pretrain_model

if __name__ == '__main__':
    """
    Train the yolo v3 model using this file. Just edit the stuff in the section below. 
    Instructions for data prep are outlined in the readme file.
    A model pretrained on ImageNet is available.  
    """
    # A bit naughty but this error doesn't actually affect the training process
    np.seterr(over='ignore')
    np.set_printoptions(threshold=sys.maxsize)

    # ===========================================================================
    # =================== ADJUST THESE PARAMETERS AS REQUIRED ===================

    # Define where to saved the trained model
    save_path = '/path/to/save/model'
    imgnet_pretrain_path = '/path/to/pretrained/darknet/model'

    # Define the paths to the training and validation image folders, and the csv files containing bbox data
    TRAIN_CSV = '/path/to/training_data.csv'
    TRAIN_PATH = '/path/to/training_image_folder/'
    VAL_CSV = '/path/to/validation_data.csv'
    VAL_PATH = '/path/to/validation_image_folder/'

    # Input and anchor parameters: anchors should be given as raw pixel sizes - these are based on the original paper
    # Anchors are divided up into small, medium and large scales below
    CNN_INPUT_SIZE = 416
    raw_anchors = [[7., 9.], [11., 21.], [23., 16.],
                   [21., 43.], [43., 31.], [41., 83.],
                   [81., 63.], [109., 138.], [261., 228.]]

    # Training parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 60

    # Choose mode as 'lrf' or 'train'; 'lrf' will draw graph for choosing learning rate. 'train' will train the model,
    # and will save to save_path for best validation loss, and save_path + _ap50 for best ap50 model. More details
    # in callbacks.py. If using 'lrf' mode aim for around 1000 training steps for best results
    mode = 'train'

    # If training, set the learning rate here
    lr = 1e-5

    # ===========================================================================
    # ===========================================================================

    assert mode == 'lrf' or mode == 'train', f'mode \'{mode}\' does not exist, please use \'lrf\' or \'train\''

    # Set up the dataframes that contain the bbox data, create a dictionary with class names as keys and one-hot
    # encoding vectors as values
    train_df = pd.read_csv(TRAIN_CSV, index_col='filename')
    val_df = pd.read_csv(VAL_CSV, index_col='filename')
    class_dict = utils.create_class_dict(train_df)
    print(f'# training images: {len(train_df.index.unique())}\n# validation images: {len(val_df.index.unique())}')

    # Normalise the given raw anchors above and organise them by scale: small, medium and large
    ANCHORS = utils.normalise_anchors(raw_anchors, CNN_INPUT_SIZE)

    # The feature map sizes; 3 feature maps are used in yolo v3 for detecting objects at small, medium and large scales
    FMAP_SIZES = [CNN_INPUT_SIZE // (2 ** 3), CNN_INPUT_SIZE // (2 ** 4), CNN_INPUT_SIZE // (2 ** 5)]

    # Set up final model training parameters  and write a config.ini file with the hyper-parameters we've specified
    TRAIN_STEPS = len(train_df.index.unique()) // BATCH_SIZE
    TEST_STEPS = len(val_df.index.unique()) // BATCH_SIZE
    LABEL_LENGTH = 5 + len(class_dict)
    utils.write_config_file(class_dict, CNN_INPUT_SIZE, ANCHORS)

    gen_options = {'batch_size': BATCH_SIZE,
                   'cnn_input_size': CNN_INPUT_SIZE,
                   'fmap_sizes': FMAP_SIZES,
                   'anchors': ANCHORS,
                   'class_dict': class_dict}

    # Set up generators for training - augmentation on the training data gives a better result
    train_generator = YoloGenerator(train_df, TRAIN_PATH, shuffle=True, aug=True, **gen_options)
    val_generator = YoloGenerator(val_df, VAL_PATH, shuffle=False, **gen_options)

    # Set up the correct callbacks for the given mode
    if mode == 'lrf':
        # Moves logarithmically through all learning rates and plots loss vs learning rate once training complete
        lrf = LearningRateFinder(TRAIN_STEPS * NUM_EPOCHS)
        cbs = [lrf]

    else:
        # Measures the mean AP50 at the end of each epoch and saves model with best performance so far
        # To save on computation, AP50 calculations only start once performance is quite good
        ap50cp = AP50Checkpoint(val_generator, BATCH_SIZE, TEST_STEPS, LABEL_LENGTH, FMAP_SIZES, ANCHORS, save_path)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path + 'cp', monitor='val_loss', save_best_only=True,
                                                        verbose=1)
        cbs = [checkpoint, ap50cp]

    # Load the yolo v3 loss function, create models for the DarkNet-53 backbone and yolo v3, load ImageNet weights
    # If you get an error in loading weights, try change path to /path/to/model_name/variables/variables
    loss_fn = wrapper_loss(LABEL_LENGTH, BATCH_SIZE)
    imgnet_model = create_pretrain_model(224)
    imgnet_model.load_weights(imgnet_pretrain_path)
    model = create_model(CNN_INPUT_SIZE, 3, LABEL_LENGTH)

    tf.keras.utils.plot_model(model, to_file='model.png')

    # Because the input sizes and output heads are different, easiest way to transfer weights is loading them layer
    # by layer over the DarkNet-53 backbone
    print('Loading following layer weights from ImageNet pretrained backbone to yolo v3 backbone...')
    for i in range(1, len(imgnet_model.layers) - 2):
        wts = imgnet_model.layers[i].get_weights()
        model.layers[i].set_weights(wts)
        # model.layers[i].trainable = False  # optionally turn off training for ImgNet pretrained part by uncommenting
        print(f'imgnet: {imgnet_model.layers[i].name} --------> yolo v3: {model.layers[i].name}')

    model.summary()

    # Set the optimizer and compile the model
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss_fn, metrics=None)

    # Begin training
    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=TRAIN_STEPS,
                        verbose=1,
                        callbacks=cbs)

    # In case AP50 checkpoint failed, save the final model now as a backup
    model.save(f'{save_path}_epoch_{NUM_EPOCHS}_backup')

    # Plot callback graphs if used
    if mode == 'lrf':
        lrf.plot()

    # Plot losses
    loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']

    fig, axs = plt.subplots(1)
    axs.plot(range(1, 1 + len(loss_history)), loss_history, 'r-')
    axs.plot(range(1, 1 + len(val_loss_history)), val_loss_history, 'b-')
    axs.set(xlabel='epochs', ylabel='loss')

    plt.show()
