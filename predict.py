import os

import cv2
import imageio
import imgaug as ia
import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from tensorflow.keras import models
from tqdm import tqdm

import utils
from losses import wrapper_loss


def sigmoid(z):
    return 1 / (1 + np.exp(-z) + 1e-7)


class YoloPredictor:
    """
    Class for using a trained yolo v3 model in predicting bounding boxes.

    Loads a saved, trained model in the tensorflow SavedModel format and uses it for predicting bounding boxes on
    images. Can predict on single images, or on directories and optionally save the outputs.

    ...
    Public methods
    --------------
    predict(fname=None):
        predict bounding boxes for a single image and display the result. If a string fname (file name) if given,
        that file will be predicted on. Otherwise, a random image from the folder is selected for prediction
        fname should not contain the image_path, just 'image.jpg' is fine

    predict_folder(folder_paths):
        predict bounding boxes for every image in a list of folders and save them to a new folder with the same root.
        This will save images with hard-coded bboxes on them. This is useful for creating videos from frames, for
        example

    ...
    Attributes
    ----------
    image_path: str:
        the path to the folder containing image files to predict on; automatically updates if using predict_folder
    model_path: str:
        the path to the trained yolo v3 model
    detect_threshold: float:
        the threshold at which to label a predicted objectness value as a predicted object; default 0.5
    nms_threshold: float:
        bounding boxes that overlap more than this threshold will be deleted, iterating in order of objectness

    """

    def __init__(self,
                 image_path,
                 model_path,
                 detect_threshold=0.5,
                 nms_threshold=0.3):
        """
        Constructor for YoloPredictor, used for predicting bounding boxes on a trained yolo v3 model.

        ...
        Parameters
        ----------
        image_path: str:
            the path to the folder containing image files to predict on; automatically updates if using predict_folder
        model_path: str:
            the path to the trained yolo v3 model
        detect_threshold: float:
            the threshold at which to label a predicted objectness value as a predicted object; default 0.5
        nms_threshold: float:
            bounding boxes that overlap more than this threshold will be deleted, iterating in order of objectness
        """

        # Set up paths and get image names from directory
        self.image_path = image_path
        self.image_fnames = sorted(os.listdir(self.image_path))

        # Get information about model from the config file and set up important hyperparameters
        self.classes, self.cnn_input_size, ta = utils.read_config_file()
        self.colors = self.__create_colors()
        self.anchors = [ta[2], ta[1], ta[0]]  # need to reverse anchor order here

        self.fmap_sizes = [self.cnn_input_size // (2 ** 5),
                           self.cnn_input_size // (2 ** 4),
                           self.cnn_input_size // (2 ** 3)]

        self.label_length = 5 + len(self.classes)
        self.detect_threshold = detect_threshold
        self.nms_threshold = nms_threshold

        self.rng = np.random.default_rng()

        # Create the imgaug Sequential objects that will deal with resizing and padding the images and bboxes
        self.seq = self.__create_seq()
        self.seq_inv = None

        # Load model
        self.model = self.__load_model(model_path)

    def __create_colors(self):
        """ Uniformly distribute hue values in HSV color space for each label and then convert to BGR tuples. """
        colors = []

        for i in range(len(self.classes)):
            hsv = np.array((int(i * (360. / (len(self.classes))) / 2.), 220, 255), ndmin=3, dtype=np.uint8)
            color = tuple([v.item() for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0, :]])
            colors.append(color)

        return colors

    def predict(self, fname=None, display=True):
        """ Runs prediction on an image; if fname is None, a random image will be chosen. """

        # Load an image and save the original dimensions for later, updates the resize + pad inversion Sequence
        np_img, original_img_dims = self.__load_image(fname)

        # Predict bboxes for the selected image; note that model outputs tensors in order [large, medium, small]
        outs = self.model.predict(np_img)

        # Convert the output tensor given by yolo into a list of bounding boxes
        pred_bboxes = self.__yolo_output_to_bboxes(outs)

        # If there were no bboxes predicted, just return the original image without going any further
        if not pred_bboxes:
            return None, np_img.shape

        # Carry out non-max suppression on the bboxes
        nms_bboxes = self.__nms(pred_bboxes)

        # Convert the final non-max suppressed bboxes into an imgaug BoundingBoxesOnImage object to undo resizing, pad
        img_fin, bbs_fin = self.__nms_to_bboxes_on_img(nms_bboxes, np_img)

        if not display:
            return bbs_fin, img_fin.shape

        for i, bb in enumerate(bbs_fin.bounding_boxes):
            img_fin = bb.draw_box_on_image(img_fin, size=2)
            img_fin = bb.draw_label_on_image(img_fin, size=0, size_text=20, height=20)

        ia.imshow(img_fin)

    def predict_folder(self, folder_paths):
        """
        Predict bounding boxes for all images in a list of folders and save the full images to new folders,
        ../name_yolo/ with the same root as the current folder.

        Parameters
        ----------
        folder_paths: list[string]:
            the paths to each folder to predict. A single folder is fine, but must be in a list. Paths must end with /
        """
        for folder_path in folder_paths:
            if not os.path.isdir(folder_path[:-1] + '_yolo/'):
                os.mkdir(folder_path[:-1] + '_yolo/')
            self.image_path = folder_path

            for fname in tqdm(sorted(os.listdir(folder_path))):
                image = cv2.imread(f'{folder_path}/{fname}')
                bbs, img_dims = self.predict(fname, display=False)
                p = img_dims[0] / image.shape[0]  # the proportion to resize by

                if bbs:
                    for bb in bbs:
                        x1_ = int(bb.x1 / p)
                        y1_ = int(bb.y1 / p)
                        x2_ = int(bb.x2 / p)
                        y2_ = int(bb.y2 / p)
                        label = bb.label
                        color = self.colors[self.classes.index(label)]
                        image = cv2.rectangle(image, (x1_, y1_), (x2_, y2_), color, 2)
                        cv2.putText(image, label, (x1_ + 5, y1_ + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.imwrite(f'{folder_path[:-1] + "_yolo"}/{fname}', image)

    def __load_model(self, model_path):
        """ Loads the model into memory from the provided path, preparing it for use in predicting. """
        print('Loading model - this may take a couple of minutes ... ')
        loss_fn = wrapper_loss(label_length=self.label_length, batch_size=1)
        model = models.load_model(model_path, custom_objects={"tf": tf, 'yolo_v3_loss': loss_fn})

        return model

    def __load_image(self, fname=None):
        """ Loads an image and creates the inversion imgaug Sequential object. """
        # If no fname is provided, pick a random image from the self.image_fnames list
        if fname is None:
            fname = self.image_fnames[self.rng.choice(len(self.image_fnames))]

        # Open the image and convert to RGB in case of grayscale images
        img = imageio.imread(self.image_path + fname, pilmode="RGB")

        # Save the image dimensions in (w, h) order for consistency
        original_img_dims = (img.shape[1], img.shape[0])

        # Create the inversion Sequence that will undo the resizing and padding for the image and bboxes simultaneously
        self.seq_inv = self.__create_seq_inv(original_img_dims)

        # Resize and normalize the image for use in the model, expand dimensions to account for batch size dim = 1
        img_resized = self.seq(image=img)
        img_resized = np.array(img_resized, dtype=np.float32) / 255.
        img_resized = np.expand_dims(img_resized, axis=0)

        return img_resized, original_img_dims

    def __yolo_output_to_bboxes(self, outs):
        """ Converts the tensors outputted by the yolo v3 prediction model into usable bounding boxes. """
        pred_bboxes = []

        # Iterate across the 3 scales
        for scale, pred_tensor in enumerate(outs):

            # Find boxes where the objectness confidence is above the threshold (default=0.5)
            idxs = np.argwhere(pred_tensor[..., ::self.label_length] >= self.detect_threshold)

            for idx in idxs:
                # Return the predicted box vectors and convert them from transformation vectors to bboxes
                v = (pred_tensor[0, idx[1], idx[2], idx[3] * self.label_length: (idx[3] + 1) * self.label_length])
                pred_bboxes.append(self.__t2box(idx, scale, v))

        return pred_bboxes

    def __t2box(self, idx, scale, bbox):
        """ Converts from yolo v3 output tensor transformation vectors into corners format bounding boxes. """
        # We need to know the grid cell location and anchor to convert predicted transformations into bboxes
        grid_x = idx[1]
        grid_y = idx[2]
        anchor_index = idx[3]

        # Formulae for converting yolo output to full size centroids format
        bx = ((sigmoid(bbox[1]) + grid_x) / self.fmap_sizes[scale]) * self.cnn_input_size
        by = ((sigmoid(bbox[2]) + grid_y) / self.fmap_sizes[scale]) * self.cnn_input_size
        bw = (np.exp(bbox[3]) * self.anchors[scale][anchor_index][0]) * self.cnn_input_size
        bh = (np.exp(bbox[4]) * self.anchors[scale][anchor_index][1]) * self.cnn_input_size

        # Convert from centroids into corners format used in IOU and iaa API
        x1 = bx - (bw / 2)
        y1 = by - (bh / 2)
        x2 = bx + (bw / 2)
        y2 = by + (bh / 2)

        # Get one-hot encodings of the predicted classes for the bbox
        temp_bbox = bbox[5:].tolist()
        class_bbox = [0] * len(self.classes)
        class_bbox[temp_bbox.index(max(temp_bbox))] = 1

        return [bbox[0], x1, y1, x2, y2] + class_bbox

    def __nms(self, pred_bboxes):
        """ Applies non-max suppression: removes boxes which overlap more than the threshold in order of confidence. """
        nms_bboxes = []

        # Create a numpy array of bboxes
        np_bboxes = np.array(pred_bboxes)

        # Sort rows in this array by ascending predicted objectness (confidence)
        sorted_boxes = np_bboxes[np.argsort(np_bboxes[:, 0])]

        while len(sorted_boxes) > 0:
            # Get the highest remaining confidence bbox index, get that bbox, fix dimensions
            last = len(sorted_boxes) - 1
            current_box = np.expand_dims(sorted_boxes[last], axis=-1)

            # Once we have extracted the box, remove it from the array
            sorted_boxes = np.delete(sorted_boxes, last, 0)
            nms_bboxes.append(current_box)

            # Find co-ordinates of intersection rectangles with current best box
            ix1 = np.maximum(sorted_boxes[:, 1], current_box[1])
            iy1 = np.maximum(sorted_boxes[:, 2], current_box[2])
            ix2 = np.minimum(sorted_boxes[:, 3], current_box[3])
            iy2 = np.minimum(sorted_boxes[:, 4], current_box[4])

            # Calculate areas and IOUs
            intersection_areas = np.expand_dims((ix2 - ix1) * (iy2 - iy1), axis=-1)

            all_areas = np.expand_dims((sorted_boxes[:, 3] - sorted_boxes[:, 1]) *
                                       (sorted_boxes[:, 4] - sorted_boxes[:, 2]), axis=-1)

            c_area = (current_box[3] - current_box[1]) * (current_box[4] - current_box[2])

            ious = intersection_areas / (all_areas + c_area - intersection_areas)

            # Delete box from sorted list if it overlaps more than the threshold
            sorted_boxes = np.delete(sorted_boxes, np.where(ious > self.nms_threshold)[0], axis=0)

        return nms_bboxes

    def __nms_to_bboxes_on_img(self, nms_bboxes, np_img):
        # Get the non max suppressed bboxes as a pure numpy array
        np_bboxes = np.array(nms_bboxes)[:, :, 0]

        labels = np.empty([np_bboxes.shape[0]], dtype=str)

        # Iterate through the classes in order and get labels from class vectors
        # In the event a box has more than one label, they will be separated by ', '
        for i, _class in enumerate(self.classes, start=5):
            labels = np.where(np_bboxes[:, i] == 1., np.core.defchararray.add(labels, _class), labels)

        # Get the [x1, y1, x2, y2, label] bboxes as BoundingBox objects
        bb = []

        for box, label in zip(np_bboxes, labels):
            # Truncate any exploding values; this only really applies to poorly trained models
            vals = [max(min(x, 1e6), -1e6) for x in (box[1], box[2], box[3], box[4])]
            bb.append(BoundingBox(x1=vals[0], y1=vals[1], x2=vals[2], y2=vals[3], label=label))

        # Convert to an iaa BoundingBoxesOnImage object
        bbs = BoundingBoxesOnImage(bb, shape=(self.cnn_input_size, self.cnn_input_size))

        # Undo the original resize and padding on image and bboxes simultaneously
        img_fin, bbs_fin = self.seq_inv(image=np_img[0], bounding_boxes=bbs)

        return img_fin, bbs_fin

    def __create_seq(self):
        """
        Resizes the image to the input size required by the model, maintaining aspect ratio and padding blank space.
        """
        seq = iaa.Sequential([

            # Letterbox resizing and padding
            iaa.Resize({'longer-side': self.cnn_input_size, 'shorter-side': 'keep-aspect-ratio'}),
            iaa.PadToFixedSize(self.cnn_input_size, self.cnn_input_size, position='center')

        ])

        return seq

    def __create_seq_inv(self, img_dims):
        """ Inverts the resizing and padding to convert the image and bounding boxes back to original dimensions. """
        # Determine the aspect ratio and whether image is portrait or landscape
        if img_dims[0] >= img_dims[1]:
            aspect = [1, img_dims[1] / img_dims[0]]
        else:
            aspect = [img_dims[0] / img_dims[1], 1]

        # crop by half the padded amount on top and bottom
        h_crop = int((self.cnn_input_size - self.cnn_input_size * aspect[1]) / 2.)
        w_crop = int((self.cnn_input_size - self.cnn_input_size * aspect[0]) / 2.)

        if h_crop > w_crop:
            seq_inv = iaa.Sequential([
                iaa.Crop(px=(h_crop, 0, h_crop, 0), keep_size=False),
            ])
        else:
            seq_inv = iaa.Sequential([
                iaa.Crop(px=(0, w_crop, 0, w_crop), keep_size=False),
            ])

        return seq_inv


if __name__ == '__main__':
    """ 
    Run this file to create a YoloPredictor object that can predict and render bounding boxes.
    """

    yp = YoloPredictor(image_path='/path/to/image/folder/', model_path='/path/to/trained/model')

    print('yolov3 predictor initialised. \n'
          '\nUse yp.predict() to predict on a random image, or yp.predict(fname) to predict on a specific file. '
          '\nyp.predict() is useful for checking single images and debugging. \n'
          '\nUse yp.predict_folder(folder_paths) to predict on every image in every folder in the list, and save the '
          'predicted images to a new directory with the same root as each folder.'
          '\nyp.predict_folder() is useful for rendering an entire video\'s frames.')

    # folder_paths = ['/home/y4tsu/Videos/bhutan/', '/home/y4tsu/Videos/hanoi/']

    # yp.predict_folder(folder_paths)
    # yp.predict()

    # To run in Python shell:
    # exec(open('predict.py').read())
