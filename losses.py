import tensorflow.keras.backend as K


def clip(z):
    """ Clip all values in a tensor to prevent divide by 0 errors. """
    z = K.clip(z, 1e-7, 1)
    return z


def get_pos_loss(y_true, y_pred, obj_mask, label_length):
    """ Calculates the position loss; the loss from x, y, w and h transformation predictions. """
    tx_true = y_true[..., 1::label_length]
    tx_pred = y_pred[..., 1::label_length]

    ty_true = y_true[..., 2::label_length]
    ty_pred = y_pred[..., 2::label_length]

    tw_true = y_true[..., 3::label_length]
    tw_pred = y_pred[..., 3::label_length]

    th_true = y_true[..., 4::label_length]
    th_pred = y_pred[..., 4::label_length]

    # Mean squared error, only applicable to anchors which have an object assigned hence use obj_mask
    tx_se = ((tx_pred - tx_true) * obj_mask) ** 2
    ty_se = ((ty_pred - ty_true) * obj_mask) ** 2
    tw_se = ((tw_pred - tw_true) * obj_mask) ** 2
    th_se = ((th_pred - th_true) * obj_mask) ** 2

    # Sum the values to get final xy_loss for a given example
    tx_sum = K.sum(tx_se)
    ty_sum = K.sum(ty_se)
    tw_sum = K.sum(tw_se)
    th_sum = K.sum(th_se)
    pos_loss = tx_sum + ty_sum + tw_sum + th_sum

    return pos_loss / 16


def get_obj_loss(y_true, y_pred, obj_mask, label_length):
    """ Calculates the objectness loss; the loss from predicting the presence of an object for an anchor. """
    to_true = y_true[..., ::label_length]
    to_pred = y_pred[..., ::label_length]

    # Binary cross entropy loss for the objectness values that have objectness to* = 1
    to_bce = (to_true * (-K.log(clip(to_pred))) + (1 - to_true) * (-K.log(clip(1 - to_pred)))) * obj_mask
    obj_loss = K.sum(to_bce)

    return obj_loss


def get_no_obj_loss(y_true, y_pred, obj_mask, ignore_mask, label_length):
    """ Calculates the non-objectness loss; the loss from predicting the absence of an object for an anchor. """
    to_true = y_true[..., ::label_length]
    to_pred = y_pred[..., ::label_length]

    # Want to prevent only predicting 0, therefore calculate non_object loss: this does not apply to to* = 1 or -1
    tno_bce = (to_true * (-K.log(clip(to_pred))) + (1 - to_true) * (-K.log(clip(1 - to_pred))))
    tno_bce *= K.cast((obj_mask == 0.), float) * ignore_mask

    no_obj_loss = K.sum(tno_bce)

    return no_obj_loss


def get_class_loss(y_true, y_pred, obj_mask, label_length):
    """ Calculates the loss for class predictions. """
    class_loss = 0.

    for i in range(5, label_length):
        # Iterate through each of the classes and perform binary cross-entropy loss for anchors with to* = 1
        class_true = y_true[..., i::label_length]
        class_pred = y_pred[..., i::label_length]
        class_bce = (class_true * (-K.log(clip(class_pred))) + (1 - class_true) *
                     (-K.log(clip(1 - class_pred)))) * obj_mask
        class_loss += K.sum(class_bce)

    return class_loss


def wrapper_loss(label_length, batch_size):
    """
    Wrapper for the yolo v3 loss function - this allows us to pass label length and batch size as extra parameters to
    the loss function.

    Parameters:
    label_length: int: objectness label + 4x positional labels + a label for each class
    batch_size: int: number of training examples in a batch

    :return: returns the yolo v3 loss value calculated by the nested loss function
    """
    def yolo_v3_loss(y_true, y_pred):
        """
        Calculates the average loss of a prediction tensor. See the original yolo v3 paper for details.

        Parameters:
        y_true: tf tensor: the ground truth label tensor for a given batch, given automatically by Keras
        y_pred: tf tensor: the prediction tensor for a given batch, given automatically by Keras

        :return: yolo v3 loss normalised over the batch size
        """

        # Weighting for positional loss and non-objectness loss; helps to balance the loss function for a good result
        lambda_coord = 5.
        lambda_no_obj = 0.2
        print('WE OUCHEAAAAaaaa')  # print essential information; lil WEEZY !

        # Get the value of all the ground truth objectness labels for each anchor at each scale
        gt_obj = y_true[..., ::label_length]

        # The objectness mask is 0. where there's no object, and 1. where there is an object
        obj_mask = K.clip(gt_obj, 0., 1.)

        # The ignore mask helps prevent punishing good but not best predictions, and is 1. where gt objectness is -1.
        ignore_mask = K.cast((gt_obj != -1.), float)

        # Calculate the individual components of the loss function
        pos_loss = get_pos_loss(y_true, y_pred, obj_mask, label_length)
        obj_loss = get_obj_loss(y_true, y_pred, obj_mask, label_length)
        no_obj_loss = get_no_obj_loss(y_true, y_pred, obj_mask, ignore_mask, label_length)
        class_loss = get_class_loss(y_true, y_pred, obj_mask, label_length)

        # Find the total loss by summing and weighting the appropriate components
        loss = (lambda_coord * pos_loss) + obj_loss + (lambda_no_obj * no_obj_loss) + class_loss

        # Return the final loss by averaging over the batch
        return loss / batch_size

    return yolo_v3_loss
