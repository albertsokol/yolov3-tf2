import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

from gen_utils import get_iou


def sigmoid(z):
    return 1 / (1 + np.exp(-z) + 1e-7)


class AP50Checkpoint(Callback):
    """
    Computes the mean AP50 across all classes, using interpolation and calculating the area under the curve at
    all unique recall points. Saves model at highest AP50 calculated. Subclasses the keras Callback object.
    The criteria for TP and FP are based on COCO criteria; a true positive (TP) predicted bbox has the same class
    predictions as the ground truth and an IOU > 0.5, otherwise it's a false positive (FP) prediction.

    ...
    Methods
    -------
    This class has no public methods, it is accessed directly by tf/keras during training.

    ...
    Attributes
    ----------
    val_generator:
        YoloGenerator: instance of YoloGenerator that is returning validation images and labels
    batch_size: int:
        number of training examples in a batch
    test_steps: int:
        number of that need to be taken to step through all the validation data
    label_length: int:
        length of a gt vector: objectness label + 4x positional labels + a label for each class
    fmap_sizes: list:
        organised as small, medium, large scale pixel dimensions of the feature maps
    anchors: list:
        list of small, medium and large anchors as 3 lists of normalised (width, height) tuples
    save_path: str:
        path to save the model with the current best AP50 to
    per_scale_threshold: int:
        the maximum number of allowed bbox predictions for a given scale in an image
        for example, if 300, then if >300 bboxes predicted for the small scale of an image, AP50 will not be
        measured to save computation under the assumption that a model that predicts 300 bboxes for an image
        is still not performing well. This also prevents saving the model until it's performing quite well
    """

    def __init__(self,
                 val_generator,
                 batch_size,
                 test_steps,
                 label_length,
                 fmap_sizes,
                 anchors,
                 save_path,
                 per_scale_threshold=1000):
        """
        Constructor for the AP50 Checkpoint callback class that measures the mean AP50 across all classes, using
        interpolation and calculating the area under the curve at all unique recall points. Saves model at highest AP50
        calculated.

        ...
        Parameters
        ----------
        val_generator:
            YoloGenerator: instance of YoloGenerator that is returning validation images and labels
        batch_size: int:
            number of training examples in a batch
        test_steps: int:
            number of that need to be taken to step through all the validation data
        label_length: int:
            length of a gt vector: objectness label + 4x positional labels + a label for each class
        fmap_sizes: list:
            organised as small, medium, large scale pixel dimensions of the feature maps
        anchors: list:
            list of small, medium and large anchors as 3 lists of normalised (width, height) tuples
        save_path: str:
            path to save the model with the current best AP50 to
        per_scale_threshold: int:
            the maximum number of allowed bbox predictions for a given scale in an image
            for example, if 300, then if >300 bboxes predicted for the small scale of an image, AP50 will not be
            measured to save computation under the assumption that a model that predicts 300 bboxes for an image
            is still not performing well. This also prevents saving the model until it's performing quite well
        """
        super().__init__()
        # Set up the callback
        self.val_generator = val_generator
        self.batch_size = batch_size
        self.test_steps = test_steps
        self.label_length = label_length
        self.per_scale_threshold = per_scale_threshold
        self.save_path = save_path

        # Set up important values for calculating AP50
        self.gt_box_total = 0.
        self.eval_array = np.empty([0, 2])
        self.ap50 = 0.
        self.best_ap50 = 0.

        # The anchors and fmap sizes are reversed so flip them around
        self.fmap_sizes = fmap_sizes[::-1]
        self.anchors = anchors[::-1]
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        """ Calculate the AP50 using predict at the end of each epoch.

        Slow, but overall saves time by avoiding using Metrics API (which calculates on every batch), and allows use of
        lists, numpy, etc. as we are not working with symbolic Tensor objects. I may try to do this with tf backend
        using on_test_batch_end in the future but it's reaaaaaaaal involved.
        """

        print('\n === Calculating AP50... === ')
        # Re-initialise variables for calculating AP50
        self.eval_array = np.empty([0, 2])
        self.ap50 = 0.
        self.gt_box_total = 0.
        precisions = []
        recalls = []
        start = time.time()

        for i in tqdm(range(self.test_steps)):
            # Get a batch of data
            x, y_dict = self.val_generator.__getitem__(i)

            # Get lists of labels and predictions in order [large, medium, small]
            y = [y_dict['large_out'], y_dict['medium_out'], y_dict['small_out']]
            out = self.model.predict(x)

            # This method returns True if > self.per_scale_threshold bboxes are predicted and breaks out of method
            thresh_hit = self.__get_eval_arrays(y, out)
            if thresh_hit:
                return

        # Sort items in eval array by descending tO (confidence) for constructing precision/recall values
        self.eval_array = self.eval_array[np.argsort(self.eval_array[:, 0])][::-1]

        print(f'predicted {self.eval_array.shape[0] / self.gt_box_total} boxes per gt box')
        print(f'found {100 * np.sum(self.eval_array[:, 1]) / self.gt_box_total:.2f}% of gt bboxes')
        # print('gt boxes total:', self.gt_box_total)

        # Initialise totals for true positives and false positives
        tp_total = 0.
        fp_total = 0.

        # Create the lists containing precision and recall values
        for i in range(self.eval_array.shape[0]):
            tp_total += self.eval_array[i, 1]
            fp_total += (self.eval_array[i, 1] == 0.).astype(float)

            # Precision = TP so far / TP + FP so far
            precisions.append(tp_total / (tp_total + fp_total))

            # Recall = TP so far / all GT boxes found in the dataset
            recalls.append(tp_total / self.gt_box_total)

        # Generally the lists don't use too much memory but uncomment if you wanna check this
        # import sys.getsizeof
        # print('Memory usage of numpy eval_array', self.eval_array.nbytes / 1e6, 'MB')
        # print('Memory usage of precision + recall lists:', (getsizeof(precisions) + getsizeof(recalls)) / 1e6, 'MB')

        # Interpolate the precision values by iterating over precisions backwards
        j = len(precisions) - 2
        while j >= 0:
            if precisions[j + 1] > precisions[j]:
                precisions[j] = precisions[j + 1]
            j -= 1

        # Calculate the area under the interpolated curve at every unique recall point (more like COCO than VOC)
        # Just splits the interpolated curve into rectangles, calculates their area and adds all those areas together
        highest_recall = 0.
        for i in range(len(precisions)):
            recall_width = recalls[i] - highest_recall
            if recalls[i] > highest_recall:
                highest_recall = recalls[i]
            self.ap50 += precisions[i] * recall_width

        print(f' === AP50: {self.ap50 * 100:.4f} === \n')
        print('Time taken to calculate AP50:', time.time() - start, 'seconds')

        # If the current AP50 is the best encountered so far, thethe model will be saved to save_path
        if self.ap50 > self.best_ap50:
            self.best_ap50 = self.ap50
            print(f'New best AP50! Saving model to save_path: {self.save_path}ap50')
            # Save the model to the specified path in the default tensorflow SavedModel format
            self.model.save(self.save_path + 'ap50')

        return

    def __t2box(self, idx, scale, bbox):
        """ Convert bbox positional vectors from transformation values into normalised bboxes. """
        # We need to know the grid cell location and anchor to convert predicted transformations into bboxes
        grid_x = idx[0]
        grid_y = idx[1]
        anchor_index = idx[2]

        # These are the formulae for converting yolo output to normalised bbox format
        bx = (sigmoid(bbox[1]) + grid_x) / self.fmap_sizes[scale]
        by = (sigmoid(bbox[2]) + grid_y) / self.fmap_sizes[scale]
        bw = np.exp(bbox[3]) * self.anchors[scale][anchor_index][0]
        bh = np.exp(bbox[4]) * self.anchors[scale][anchor_index][1]

        # Get one-hot encodings of the predicted classes for the bbox
        class_bbox = (bbox[5:] >= 0.5).astype(int).tolist()

        return [bbox[0], bx, by, bw, bh] + class_bbox

    def __get_eval_arrays(self, y, out):
        """
        Gets all the predicted bboxes and whether they are TP or FP. Also gets total gt bboxes, used for finding recall.
        """
        thresh_hit = False

        # Iterate over scales first
        for j, (gt_batch, pred_batch) in enumerate(zip(y, out)):

            # Then, iterate over images
            for gt, pred in zip(gt_batch, pred_batch):

                # Number of predicted bboxes for the scale
                num_pred_bbox = np.sum(pred[..., ::self.label_length] >= 0.5)

                # While model is training, likely to return huge amounts of boxes due to random initialization which can
                # be very slow to work with. Realistically, a well trained model will not predict more than 100 images
                # for a single scale eg. small. The per scale box threshold can be changed if this is too low.
                if num_pred_bbox > self.per_scale_threshold:
                    print(f'\nPredicted >{self.per_scale_threshold} boxes for scale {j}; AP50 likely very low, '
                          f'therefore not calculated to maintain performance.')
                    thresh_hit = True
                    return thresh_hit

                # Get number of ground truth bboxes for the scale and add these to the cumulative sum
                num_gt_bbox = np.sum(gt[..., ::self.label_length] == 1.)
                self.gt_box_total += num_gt_bbox
                gt_boxes = []
                pred_boxes = []

                # First check if there are any ground truth bboxes; if yes then convert them to full size for IOU calc
                # If no gt bboxes exist, then all predictions will be false positives for this scale
                if num_gt_bbox == 0:
                    gt_boxes = None

                else:
                    gt_idx = np.argwhere(gt[..., ::self.label_length] == 1.)
                    for idx in gt_idx:
                        v = (gt[idx[0], idx[1], idx[2] * self.label_length: (idx[2] + 1) * self.label_length])
                        gt_boxes.append(self.__t2box(idx, j, v))

                # If no bboxes have been predicted, then we don't need to do any more for this image
                if num_pred_bbox == 0:
                    continue

                # Find the locations of predicted bboxes (objectness predictions are greater than the threshold of 0.5)
                pred_idx = np.argwhere(pred[..., ::self.label_length] >= 0.5)
                for idx in pred_idx:
                    # Slice out the prediction vectors for these locations
                    v = (pred[idx[0], idx[1], idx[2] * self.label_length: (idx[2] + 1) * self.label_length])
                    # Convert the transformation vectors into normalised bbox values and store
                    pred_boxes.append(self.__t2box(idx, j, v))

                for pred_box in pred_boxes:

                    # If there are gt bboxes then there may be some TP, otherwise they are all FP
                    if gt_boxes is not None:

                        for gt_box in gt_boxes:
                            # First, check if class predicted correctly
                            classes = gt_box[5:] == pred_box[5:]

                            if classes:
                                # If yes, then calculate IOU and append as a TP if above threshold, 0.5 by default
                                iou = get_iou(gt_box, pred_box)

                                if iou >= 0.5:
                                    self.eval_array = np.append(self.eval_array, [[pred_box[0], 1]], axis=0)
                                    # Remove the bbox from the list and continue evaluating predicted bboxes
                                    gt_boxes.remove(gt_box)

                                    continue

                    # If bbox does not (match class and IOU > 0.5) then predicted bbox is a FP (labelled as 0)
                    self.eval_array = np.append(self.eval_array, [[pred_box[0], 0]], axis=0)

        return thresh_hit


class LearningRateFinder(Callback):
    """ Callback function that graphs learning rate against loss to help find optimal learning rate.

    Callback function that inherits from Keras Callback module. Tries a range of learning rates from min_lr to max_lr
    along a logarithmic scale, split up into total_steps steps. Saves learning rate and loss after each batch update in
    the history attribute. The plot can then be used to find the best learning rates to use for training.

    ...
    Methods
    -------
    plot(): plot learning rate against loss. Use this graph to find optimal lr for cyclical learning rate: the part
            where the loss juuust starts to decrease consistently is the min_lr. The part where the loss stops
            decreasing smoothly and flattens out or gets volatile is the max_lr. Note you might need to zoom in a fair
            bit on the graph if the loss explodes at the end

    ...
    Attributes
    ----------
    total_steps: int:
        the total number of batch updates to perform. This is number of epochs * training steps per epoch
        the more total_steps, the more stable the output graph will be, around 1000 steps is usually a good amount
    min_lr: float:
        base learning rate to start from
    max_lr: float:
        maximum possible learning rate to test

    ...
    Public attributes
    -----------------
    history: the history dictionary contains keys 'lr' for learning rate values and 'loss' for losses
    """

    def __init__(self, total_steps, min_lr=1e-9, max_lr=1.):
        """
        Constructor for callback function that graphs learning rate against loss to help find optimal learning rate.

        ...
        Parameters
        ----------
        total_steps: int:
            the total number of batch updates to perform. This is number of epochs * training steps per epoch
            the more total_steps, the more stable the output graph will be, around 1000 steps is usually a good amount
        min_lr: float:
            base learning rate to start from
        max_lr: float:
            maximum possible learning rate to test

        ...
        Public attributes
        -----------------
        history: the history dictionary contains keys 'lr' for learning rate values and 'loss' for losses
        """
        super(LearningRateFinder, self).__init__()
        self._history = {'lr': [], 'loss': []}
        self.__min_lr = min_lr
        self.__max_lr = max_lr
        self.__total_steps = total_steps
        self.__batch_number = 0.
        self.__k = np.log(max_lr / min_lr) / self.__total_steps

    @property
    def history(self):
        """ Getter for the history dictionary containing keys 'lr' for learning rates and 'loss' for losses. """
        return self._history

    def __lr(self):
        """ Returns learning rate on an exponentially increasing scale from min_lr to max_lr. """
        return self.__min_lr * np.exp(self.__k * self.__batch_number)

    def on_train_begin(self, logs=None):
        """ Set learning rate to base, minimum value when training begins. """
        K.set_value(self.model.optimizer.lr, self.__min_lr)

    def on_batch_end(self, batch, logs=None):
        """ Update learning rate, append loss and learning rate to history. """
        self.__batch_number += 1

        K.set_value(self.model.optimizer.lr, self.__lr())
        
        self.history['lr'].append(K.get_value(self.model.optimizer.lr))
        current_loss = logs.get('loss')
        self.history['loss'].append(current_loss)

    def plot(self):
        """ Plots learning rate against loss.

        Recommended to pick the point where the loss just starts to drop as the lower bound for cyclical learning rate,
        pick point where loss stops decreasing or becomes ragged as upper bound for cyclical learning rate.
        Alternatively, pick the minimum loss as static learning rate, or starting rate for a decreasing regimen.
        """
        lr_history = self.history['lr']
        smooth_loss_history = self.__smooth_losses()

        _fig, _axs = plt.subplots(2)
        _axs[0].plot(lr_history, smooth_loss_history, 'r-')
        _axs[0].set(xlabel='learning rate', ylabel='loss')
        _axs[0].set_xscale('log')

        _axs[1].plot(range(len(lr_history)), lr_history, 'b-')
        _axs[1].set(xlabel='iterations', ylabel='learning rate')
        plt.show()

    def __smooth_losses(self, beta=0.96):
        """ Returns an exponential moving average of the losses for smoother plotting.

        By default, beta is set to 0.96 which will result in moving average of around 25 data points. Decrease beta
        to reduce smoothing. Also applies bias correction to improve plotting at start of data.
        """
        avg_loss = 0
        coarse_loss = self.history['loss']
        smooth_loss = []

        for i in range(len(coarse_loss)):
            avg_loss = (beta * avg_loss) + ((1 - beta) * coarse_loss[i])
            corr_avg_loss = avg_loss / (1 - (beta ** (i + 1)))
            smooth_loss.append(corr_avg_loss)

        return smooth_loss
