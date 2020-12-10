import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import models


def create_mask(mask_size, anchors_per_det, label_length):
    # Create masks to apply sigmoid activation to objectness and class predictions only
    mask = np.ones([mask_size, mask_size, anchors_per_det * label_length])
    mask[..., 1::label_length] = 0
    mask[..., 2::label_length] = 0
    mask[..., 3::label_length] = 0
    mask[..., 4::label_length] = 0

    mask = mask.astype(bool)

    return mask


def mask_wrapper(mask):

    def _activations_by_mask(x):
        # Apply masks to the outputs of the model
        z = tf.where(mask, K.sigmoid(x), x)

        return z

    return _activations_by_mask


def down_conv_res_block(m, current_sf):
    """ Apply down convolutions as a residual block as per yolo v3 paper. """
    z = layers.Conv2D(current_sf, (1, 1), padding='same')(m)
    z = layers.LeakyReLU()(z)
    z = layers.BatchNormalization()(z)

    z = layers.Conv2D(current_sf * 2, (3, 3), padding='same')(z)
    z = layers.LeakyReLU()(z)
    z = layers.BatchNormalization()(z)

    z = layers.Add()([m, z])

    return z


def darknet53_backbone(model_input):
    """ Create the DarkNet-53 backbone which will supply the feature maps to feed to the neck and head of the model. """
    # Initial block; sf = number of starting filters
    sf = 32
    conv0 = layers.Conv2D(sf, (3, 3), padding='same')(model_input)
    conv0 = layers.LeakyReLU()(conv0)
    conv0 = layers.BatchNormalization()(conv0)
    conv0 = layers.Conv2D(sf * 2, (3, 3), strides=(2, 2), padding='same')(conv0)
    conv0 = layers.LeakyReLU()(conv0)
    conv0 = layers.BatchNormalization()(conv0)

    # Convolutional and residual blocks
    conv1 = down_conv_res_block(conv0, sf)
    conv1 = layers.Conv2D(sf * 4, (3, 3), strides=(2, 2), padding='same')(conv1)
    conv1 = layers.LeakyReLU()(conv1)
    conv1 = layers.BatchNormalization()(conv1)

    conv2 = down_conv_res_block(conv1, sf * 2)
    conv2 = down_conv_res_block(conv2, sf * 2)
    conv2 = layers.Conv2D(sf * 8, (3, 3), strides=(2, 2), padding='same')(conv2)
    conv2 = layers.LeakyReLU()(conv2)
    conv2 = layers.BatchNormalization()(conv2)

    conv3 = down_conv_res_block(conv2, sf * 4)
    conv3 = down_conv_res_block(conv3, sf * 4)
    conv3 = down_conv_res_block(conv3, sf * 4)
    conv3 = down_conv_res_block(conv3, sf * 4)
    conv3 = down_conv_res_block(conv3, sf * 4)
    conv3 = down_conv_res_block(conv3, sf * 4)
    conv3 = down_conv_res_block(conv3, sf * 4)

    # The first output here; this is the small scale feature map
    conv3_out = down_conv_res_block(conv3, sf * 4)

    conv3 = layers.Conv2D(sf * 16, (3, 3), strides=(2, 2), padding='same')(conv3_out)
    conv3 = layers.LeakyReLU()(conv3)
    conv3 = layers.BatchNormalization()(conv3)

    conv4 = down_conv_res_block(conv3, sf * 8)
    conv4 = down_conv_res_block(conv4, sf * 8)
    conv4 = down_conv_res_block(conv4, sf * 8)
    conv4 = down_conv_res_block(conv4, sf * 8)
    conv4 = down_conv_res_block(conv4, sf * 8)
    conv4 = down_conv_res_block(conv4, sf * 8)
    conv4 = down_conv_res_block(conv4, sf * 8)

    # The second output; this is the medium scale feature map
    conv4_out = down_conv_res_block(conv4, sf * 8)

    conv4 = layers.Conv2D(sf * 32, (3, 3), strides=(2, 2), padding='same')(conv4_out)
    conv4 = layers.LeakyReLU()(conv4)
    conv4 = layers.BatchNormalization()(conv4)

    conv5 = down_conv_res_block(conv4, sf * 16)
    conv5 = down_conv_res_block(conv5, sf * 16)
    conv5 = down_conv_res_block(conv5, sf * 16)

    # The third and final output; this is the large scale feature map
    conv5_out = down_conv_res_block(conv5, sf * 16)

    return conv5_out, conv4_out, conv3_out


def imgnet_neck_head(large_from_backbone):
    """ Creates the neck and head to attach to DarkNet-53 backbone for training on ImageNet. """
    m = layers.GlobalAveragePooling2D()(large_from_backbone)
    out = layers.Dense(1000, activation='softmax')(m)

    return out


def neck_conv_block(m):
    m = layers.Conv2D(512, (1, 1), padding='same')(m)
    m = layers.LeakyReLU()(m)
    m = layers.BatchNormalization()(m)

    m = layers.Conv2D(1024, (3, 3), padding='same')(m)
    m = layers.LeakyReLU()(m)
    m = layers.BatchNormalization()(m)

    m = layers.Conv2D(512, (1, 1), padding='same')(m)
    m = layers.LeakyReLU()(m)
    m = layers.BatchNormalization()(m)

    m = layers.Conv2D(1024, (3, 3), padding='same')(m)
    m = layers.LeakyReLU()(m)
    m = layers.BatchNormalization()(m)

    return m


def yolo_v3_neck_and_head(large_from_backbone,
                          medium_from_backbone,
                          small_from_backbone,
                          anchors_per_det,
                          label_length):
    """
    Neck and head: takes small, medium and large fmap inputs from DarkNet-53 backbone, applies convolutions and
    generates the final prediction tensors. Uses upsampling layers as per the original paper.
    """
    # Get the output for the large detector, and create a path to upsample
    large = neck_conv_block(large_from_backbone)

    large = layers.Conv2D(512, (1, 1), padding='same')(large)
    large = layers.LeakyReLU()(large)
    large_up = layers.BatchNormalization()(large)

    large = layers.Conv2D(1024, (3, 3), padding='same')(large_up)
    large = layers.LeakyReLU()(large)
    large = layers.BatchNormalization()(large)

    # Upsample and concatenate the large detector output with the medium detector output
    large_up = layers.UpSampling2D()(large_up)
    medium = layers.Concatenate()([large_up, medium_from_backbone])
    medium = neck_conv_block(medium)

    medium = layers.Conv2D(512, (1, 1), padding='same')(medium)
    medium = layers.LeakyReLU()(medium)
    medium_up = layers.BatchNormalization()(medium)

    medium = layers.Conv2D(1024, (3, 3), padding='same')(medium_up)
    medium = layers.LeakyReLU()(medium)
    medium = layers.BatchNormalization()(medium)

    # Upsample and concatenate the medium detector output with the small detector output
    medium_up = layers.UpSampling2D()(medium_up)
    small = layers.Concatenate()([medium_up, small_from_backbone])
    small = neck_conv_block(small)

    small = layers.Conv2D(512, (1, 1), padding='same')(small)
    small = layers.LeakyReLU()(small)
    small = layers.BatchNormalization()(small)

    small = layers.Conv2D(1024, (3, 3), padding='same')(small)
    small = layers.LeakyReLU()(small)
    small = layers.BatchNormalization()(small)

    # Finally, output the prediction tensors
    large_out = layers.Conv2D(anchors_per_det * label_length, (1, 1), padding='same')(large)
    medium_out = layers.Conv2D(anchors_per_det * label_length, (1, 1), padding='same')(medium)
    small_out = layers.Conv2D(anchors_per_det * label_length, (1, 1), padding='same')(small)

    # Create masks to apply sigmoid activation to objectness and class predictions only
    # (don't want to clip positional transforms to the 0 - 1; there is no activation needed on these)
    large_mask = create_mask(K.int_shape(large_out)[1], anchors_per_det, label_length)
    medium_mask = create_mask(K.int_shape(medium_out)[1], anchors_per_det, label_length)
    small_mask = create_mask(K.int_shape(small_out)[1], anchors_per_det, label_length)

    # Apply the sigmoid activation mask
    large_out_masked = layers.Lambda(mask_wrapper(large_mask), name='large_out')(large_out)
    medium_out_masked = layers.Lambda(mask_wrapper(medium_mask), name='medium_out')(medium_out)
    small_out_masked = layers.Lambda(mask_wrapper(small_mask), name='small_out')(small_out)

    return large_out_masked, medium_out_masked, small_out_masked


def create_model(cnn_input_size, anchors_per_det, label_length):
    """ Creates the full yolo v3 network with DarkNet-53 backbone and my modified Conv2D neck and head, returns model.

    Parameters:
    cnn_input_size: int: the input size of the network; images will be resized to this size
    anchors_per_det: int: the number of anchors at each fmap scale; this is basically just gonna be 3
    label_length: int: length of a gt vector: objectness label + 4x positional labels + a label for each class
    """

    # Set up the input layer
    model_input = layers.Input((cnn_input_size, cnn_input_size, 3))

    # Build the model backbone and return the feature maps for all 3 scales
    bbone_large, bbone_medium, bbone_small = darknet53_backbone(model_input)

    # Process the feature maps through the neck and head to get final outputs on all 3 scales
    large_out, medium_out, small_out = yolo_v3_neck_and_head(bbone_large,
                                                             bbone_medium,
                                                             bbone_small,
                                                             anchors_per_det,
                                                             label_length)

    # Create the model
    model = models.Model(inputs=model_input, outputs=[large_out, medium_out, small_out])

    return model


def create_pretrain_model(cnn_input_size):
    """ Creates the DarkNet-53 backbone that will be trained on ImageNet, returns model.

    Parameters:
    cnn_input_size: int: the input size of the network; images will be resized to this size
    """

    # Set the input layer: my model was pretrained on (224, 224) images in RGB
    model_input = layers.Input((cnn_input_size, cnn_input_size, 3))

    # Only the large output from the backbone is needed; discard the others
    bbone_large, *_ = darknet53_backbone(model_input)

    # Use the basic ImageNet head; just a global pooling and 1000 neuron Dense softmax layer
    out = imgnet_neck_head(bbone_large)

    # Create the model
    model = models.Model(inputs=model_input, outputs=out)

    return model
