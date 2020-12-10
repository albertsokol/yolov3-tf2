import os

import matplotlib.pyplot as plt
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from callbacks import LearningRateFinder
from model import create_pretrain_model


if __name__ == '__main__':
    """
    This file was used for pretraining the DarkNet-53 backbone on ImageNet. 
    """

    # ===========================================================================
    # =================== ADJUST THESE PARAMETERS AS REQUIRED ===================

    # Define where to saved the trained model
    save_path = '/path/to/save/model'

    # Define the paths to the training and validation image folders
    TRAIN_PATH = '/path/to/training_image_folder/'
    VAL_PATH = '/path/to/validation_image_folder/'

    # Input parameters
    CNN_INPUT_SIZE = 224

    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 50

    # Choose mode as 'lrf' or 'train'; 'lrf' will draw graph for choosing learning rate. 'train' will train the model,
    # and will save to save_path for best validation loss, and save_path + _ap50 for best ap50 model. More details
    # in callbacks.py. If using 'lrf' mode aim for around 1000 training steps for best results
    mode = 'train'

    # If training, set the learning rate here
    lr = 1e-5

    # ===========================================================================
    # ===========================================================================

    assert mode == 'lrf' or mode == 'train', f'mode \'{mode}\' does not exist, please use \'lrf\' or \'train\''

    # Set up the remaining training parameters
    n_train = sum([len(files) for r, d, files in os.walk(TRAIN_PATH)])
    n_val = sum([len(files) for r, d, files in os.walk(VAL_PATH)])
    print(f'# training images: {n_train} \n# validation images: {n_val}')
    TRAIN_STEPS = n_train // BATCH_SIZE
    TEST_STEPS = n_val // BATCH_SIZE

    # Set up image augmentation
    train_aug = ImageDataGenerator(rotation_range=12.,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   brightness_range=(0.7, 1.3),
                                   shear_range=6.,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rescale=1./255)
    val_aug = ImageDataGenerator(rescale=1./255)

    gen_options = {'target_size': (CNN_INPUT_SIZE, CNN_INPUT_SIZE), 'batch_size': BATCH_SIZE, 'shuffle': True}

    # Set up generators - ImageNet data is in folders hence why flow from directory is used
    train_gen = train_aug.flow_from_directory(TRAIN_PATH, **gen_options)
    val_gen = val_aug.flow_from_directory(VAL_PATH, **gen_options)

    # Set up the correct callbacks for the given mode
    if mode == 'lrf':
        # Moves logarithmically through all learning rates and plots loss vs learning rate once training complete
        lrf = LearningRateFinder(TRAIN_STEPS * NUM_EPOCHS)
        cbs = [lrf]
    else:
        checkpoint = callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
        cbs = [checkpoint]

    # Build the model and compile
    model = create_pretrain_model(CNN_INPUT_SIZE)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Begin training
    history = model.fit(train_gen,
                        validation_data=val_gen,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=TRAIN_STEPS,
                        validation_steps=TEST_STEPS,
                        verbose=1,
                        callbacks=cbs)

    # In case checkpoint failed, save the final model now as a backup
    model.save(f'{save_path}_epoch_{NUM_EPOCHS}_backup')

    # Plot callback graphs if used
    if mode == 'lrf':
        lrf.plot()

    # Plot the losses and accuracies seen during training
    loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']
    acc_history = history.history['accuracy']
    val_acc_history = history.history['val_accuracy']

    fig, axs = plt.subplots(2)
    fig.set_size_inches(8, 12)

    axs[0].plot(range(1, 1 + len(loss_history)), loss_history, 'r-', label='train loss')
    axs[0].plot(range(1, 1 + len(val_loss_history)), val_loss_history, 'b-', label='val loss')
    axs[0].set(xlabel='epochs', ylabel='loss')
    axs[0].legend(loc="upper right")

    axs[1].plot(range(1, 1 + len(acc_history)), acc_history, 'g-', label='train accuracy')
    axs[1].plot(range(1, 1 + len(val_acc_history)), val_acc_history, 'm-', label='val accuracy')
    axs[1].set(xlabel='epochs', ylabel='classification accuracy')
    axs[1].legend(loc="upper right")

    plt.show()
