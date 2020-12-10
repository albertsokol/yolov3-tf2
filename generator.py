import imageio
import numpy as np
import pandas as pd
import tensorflow.keras
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from gen_utils import create_gt_tensor


class YoloGenerator(tensorflow.keras.utils.Sequence):
    """
    Generator that returns a batch of images and their ground truth tensors in the yolo v3 format.

    ...
    Methods
    _______
    The generator has no public methods and interacts directly with Keras to automatically return images and
    ground truth tensors. Note the required methods __len__ and __getitem__:
    __len__: the same thing as TRAIN_STEPS or TEST_STEPS; this is how many times the generator needs to return data
             in an epoch
    __getitem__: returns the images as a numpy array and the ground truths as a dictionary of numpy arrays across
                 the 3 different scales: small, medium and large. This method is expected by Keras

    ...
    Attributes
    ----------
    dataframe:
        the pandas dataframe object containing the image and bounding box data
    image_path: str:
        path to the folder containing images
    class_dict: dictionary:
        with class names as keys and one-hot vector encodings as values
        
    batch_size: int:
        number of training/validation examples in a batch
    cnn_input_size: int:
        the input size of the network; images will be resized to this size
    fmap_sizes: list:
        organised as small, medium, large scale pixel dimensions of the feature maps
    anchors: list:
        list of small, medium and large anchors as 3 lists of normalised (width, height) tuples
    shuffle: bool:
        if True, data will be randomly shuffled at the end of each epoch
        
    contrast: tuple:
        (low, high) amounts to augment contrast; range 0. - 1. decreases, 1.+ increases
    brightness: tuple:
        (low, high) amounts of augment brightness; range < 0 decreases, > 0 increases
    horizontal_flip: float:
        probability of flipping an image horizontally; 0.5 is 50% chance
    translate: tuple:
        pixel amounts to translate image in x and y directions
    zoom: tuple:
        (out, in) proportions to zoom in on the image; range 0. - 1. zooms out, 1.+ zooms in
    aug: bool:
        if True, data augmentation will be applied to images and bounding boxes (applied simultaneously)
    """

    def __init__(self,
                 dataframe,
                 image_path,
                 class_dict=None,

                 batch_size=8,
                 cnn_input_size=None,
                 fmap_sizes=None,
                 anchors=None,
                 shuffle=True,

                 contrast=(0.75, 1.25),
                 brightness=(-30, 30),
                 horizontal_flip=0.5,
                 translate=(-15, 15),
                 zoom=(0.8, 1.2),
                 aug=False):
        """
        Constructs an instance of the YoloGenerator class.

        ...
        Parameters
        ----------
        dataframe:
            the pandas dataframe object containing the image and bounding box data
        image_path: str:
            path to the folder containing images
        class_dict: dictionary:
            with class names as keys and one-hot vector encodings as values
            
        batch_size: int:
            number of training/validation examples in a batch
        cnn_input_size: int:
            the input size of the network; images will be resized to this size
        fmap_sizes: list:
            organised as small, medium, large scale pixel dimensions of the feature maps
        anchors: list:
            list of small, medium and large anchors as 3 lists of normalised (width, height) tuples
        shuffle: bool:
            if True, data will be randomly shuffled at the end of each epoch
            
        contrast: tuple:
            (low, high) amounts to augment contrast; range 0. - 1. decreases, 1.+ increases
        brightness: tuple:
            (low, high) amounts of augment brightness; range < 0 decreases, > 0 increases
        horizontal_flip: float:
            probability of flipping an image horizontally; 0.5 is 50% chance
        translate: tuple:
            pixel amounts to translate image in x and y directions
        zoom: tuple:
            (out, in) proportions to zoom in on the image; range 0. - 1. zooms out, 1.+ zooms in
        aug: bool:
            if True, data augmentation will be applied to images and bounding boxes (applied simultaneously)
        """

        # Data parameters
        self.df = dataframe
        self.image_path = image_path
        self.image_filenames = self.df.index.unique().to_list()
        self.index = np.arange(len(self.image_filenames))
        self.class_dict = class_dict

        # Model parameters
        self.batch_size = batch_size
        self.cnn_input_size = cnn_input_size
        self.fmap_sizes = fmap_sizes
        self.anchors = anchors
        self.shuffle = shuffle

        # Augmentation parameters
        self.contrast = contrast
        self.brightness = brightness
        self.horizontal_flip = horizontal_flip
        self.translate = translate
        self.zoom = zoom
        self.aug = aug
        self.seq = self.__create_seq()

        # Shuffle the data before starting if shuffling has been turned on
        self.on_epoch_end()

    def on_epoch_end(self):
        # Shuffle order of data at end of each epoch
        if self.shuffle:
            np.random.shuffle(self.index)

    def __len__(self):
        # Number of gradient descent steps that will be taken
        return len(self.image_filenames) // self.batch_size

    def __getitem__(self, index):
        # Create a list of batch_size numerical indices
        indices = self.index[index * self.batch_size:(index + 1) * self.batch_size]

        # Get the filenames that correspond to those indices
        batch = [self.image_filenames[k] for k in indices]

        # Generate the data
        x, y_small, y_medium, y_large = self.__get_data(batch)

        return x, {'small_out': y_small, 'medium_out': y_medium, 'large_out': y_large}

    def __get_data(self, batch):
        # Returns the images and ground truth tensors as numpy arrays
        # Initialise tensor objects with correct shape and -1. placeholder values
        x = np.empty([self.batch_size, self.cnn_input_size, self.cnn_input_size, 3], dtype=np.float32)
        y_small = -np.ones([self.batch_size,
                            self.fmap_sizes[0],
                            self.fmap_sizes[0],
                            len(self.anchors[0]) * (5 + len(self.class_dict))], dtype=np.float32)
        y_medium = -np.ones([self.batch_size,
                            self.fmap_sizes[1],
                            self.fmap_sizes[1],
                            len(self.anchors[1]) * (5 + len(self.class_dict))], dtype=np.float32)
        y_large = -np.ones([self.batch_size,
                            self.fmap_sizes[2],
                            self.fmap_sizes[2],
                            len(self.anchors[2]) * (5 + len(self.class_dict))], dtype=np.float32)

        # Use image ID to fetch data and save as i'th element in batch of arrays
        for i, img_id in enumerate(batch):

            img, bbs = self.__process(img_id)
            x[i, ...] = img
            y_small[i, ...], y_medium[i, ...], y_large[i, ...] = create_gt_tensor(bbs,
                                                                                  self.class_dict,
                                                                                  self.fmap_sizes,
                                                                                  self.anchors,
                                                                                  y_small[i, ...],
                                                                                  y_medium[i, ...],
                                                                                  y_large[i, ...])

        return x, y_small, y_medium, y_large

    def __read_img(self, img_id):
        # Open image and convert to RGB - otherwise some of the objects will be in grayscale which can cause errors
        img = imageio.imread(self.image_path + img_id, pilmode="RGB")

        return img

    def __process(self, img_id):
        out_bbs = []

        # Load image and bounding boxes
        img = self.__read_img(img_id)
        bbs = self.__img_id_to_bboxes(img_id)

        # ia.imshow(bbs.draw_on_image(img)) # Uncomment to preview before processing

        # Perform augmentation, or just resize and pad if augmentation turned off
        img_aug, bbs_aug = self.seq(image=img, bounding_boxes=bbs)

        # Augmentation may result in bboxes with centroids that are outside the image; remove these
        if self.aug:
            bbs_aug.remove_out_of_image_fraction_(fraction=0.45)

        # ia.imshow(bbs_aug.draw_on_image(img_aug)) # Uncomment to preview after processing

        # Convert bboxes from the top-left bottom-right format to YOLO xywh format, proportional to the image
        for bb in bbs_aug:
            x = [z / self.cnn_input_size for z in [bb.center_x, bb.center_y, bb.width, bb.height]]
            out_bbs.append(x + [bb.label])

        # Normalise the pixel values to the 0 to 1 range
        img_np = np.array(img_aug) / 255.

        return img_np, out_bbs

    def __create_seq(self):
        # The imgaug augmentation sequence object is constructed here

        if self.aug:
            # If augmentation is turned on then resizing and padding will be applied, followed by augmentation
            seq = iaa.Sequential([

                # Make the size of the image correct; note we maintain the aspect ratio when resizing 
                iaa.Resize({'longer-side': self.cnn_input_size, 'shorter-side': 'keep-aspect-ratio'}),
                iaa.PadToFixedSize(self.cnn_input_size, self.cnn_input_size, position='center'),

                # Perform augmentation
                iaa.Fliplr(self.horizontal_flip),
                iaa.Affine(translate_px={'x': self.translate, 'y': self.translate},
                           scale={'x': self.zoom, 'y': self.zoom}),
                iaa.GammaContrast(self.contrast),
                iaa.Add(self.brightness)

            ])

        else:
            # If augmentation is turned off, the images and their bboxes still need to be resized and padded
            seq = iaa.Sequential([

                # Make the size of the image correct; note we maintain the aspect ratio when resizing 
                iaa.Resize({'longer-side': self.cnn_input_size, 'shorter-side': 'keep-aspect-ratio'}),
                iaa.PadToFixedSize(self.cnn_input_size, self.cnn_input_size, position='center')

            ])

        return seq

    def __img_id_to_bboxes(self, img_id):
        bbs = []

        # Load the bounding boxes for this image from the dataframe
        img_df = self.df.loc[img_id]
        if type(img_df) is not pd.DataFrame:
            # If there is just 1 bbox a Series object will be created, to prevent errors we will convert to a dataframe
            img_df = img_df.to_frame().T

        # Get each bounding box as an imgaug BoundingBox object
        for index, row in img_df.iterrows():
            bbs.append(BoundingBox(x1=row['x1'], y1=row['y1'], x2=row['x2'], y2=row['y2'], label=row['class']))

        # Place all those BoundingBox objects into an imgaug BoundingBoxesOnImage object for rescaling etc.
        bbs_on_img = BoundingBoxesOnImage(bbs, shape=(img_df.iloc[0]['height'], img_df.iloc[0]['width']))

        return bbs_on_img
