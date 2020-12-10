import numpy as np


def centroids2corners(bbox):
    """ Convert from centroids, width, height format to upper-left and lower-right corners format, used in IOU. """
    x1 = bbox[0] - (bbox[2] / 2)
    y1 = bbox[1] - (bbox[3] / 2)
    x2 = bbox[0] + (bbox[2] / 2)
    y2 = bbox[1] + (bbox[3] / 2)
    corners = [x1, y1, x2, y2]
    return corners


def get_iou(a_bbox, b_bbox):
    """ Convert bboxes to correct format for IOU calculation. Formatting returns bboxes as [x1, y1, x2, y2]. """
    a = centroids2corners(a_bbox)
    b = centroids2corners(b_bbox)

    # Find co-ordinates of intersection rectangle, return 0 if no overlap
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # If there is no overlap, just return IOU as 0 to avoid negative values
    if x2 < x1 or y2 < y1:
        return 0.

    # Calculate areas of bboxes and area of intersection
    intersection_area = (x2 - x1) * (y2 - y1)
    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])

    # Calculate IOU
    iou = intersection_area / float(a_area + b_area - intersection_area)

    return iou


def get_best_anchor(raw_bbox, grid_coordinates, fmap_sizes, anchors_list, ignore_threshold=0.5):
    """
    Returns the index of the best anchor for a given bounding box, as well as the anchor's co-ordinates.
    Anchors that have an IOU > 0.5 but are not the best will need to be ignored later in the non-objectness loss
    to avoid punishing the model for making decent predictions as per the original yolo v3 paper. Therefore, save a
    value of 1 for anchors that meet the threshold.
    """
    ious = []
    temp_anchors = []

    for i in range(len(anchors_list)):
        # Note that anchors are defined using their width and height; this is the information in anchor_dims
        # The x and y co-ordinates of anchors correspond to the midpoint of a grid cell
        anchor_x = (grid_coordinates[i][0] + 0.5) / fmap_sizes[i]
        anchor_y = (grid_coordinates[i][1] + 0.5) / fmap_sizes[i]

        for anchor_dims in anchors_list[i]:
            anchor_box = [anchor_x, anchor_y, anchor_dims[0], anchor_dims[1]]  # xywh

            # Get IOU for given anchor
            iou = get_iou(raw_bbox, anchor_box)
            temp_anchors.append(anchor_box)
            ious.append(iou)

    # Find the ious that are above the threshold (0.5 in original paper)
    meet_ignore_threshold = np.array([x >= ignore_threshold for x in ious]).astype(int)

    # Save the anchor box with the highest IOU for this bbox
    best_anchor_idx = int(np.argmax(ious))
    best_anchor_box = temp_anchors[best_anchor_idx]

    return best_anchor_idx, best_anchor_box, meet_ignore_threshold


def get_tx(bbox, grid_x, fmap_size):
    # Return the value that maps the ground truth x value into the transformation value
    tx = bbox[0] * fmap_size - grid_x
    return np.log((tx / (1 - tx)) + 1e-7)


def get_ty(bbox, grid_y, fmap_size):
    # Return the value that maps the ground truth y value into the transformation value
    ty = bbox[1] * fmap_size - grid_y
    return np.log((ty / (1 - ty)) + 1e-7)


def get_tw(bbox, anchor):
    # Return the value that maps the ground truth width value into the transformation value
    return np.log((bbox[2] / anchor[2]) + 1e-7)


def get_th(bbox, anchor):
    # Return the value that maps the ground truth height value into the transformation value
    return np.log((bbox[3] / anchor[3]) + 1e-7)


def create_gt_tensor(bbs, class_dict, fmap_sizes, anchors_list, y_small, y_medium, y_large):
    """
    Accessed by the YoloGenerator class only. Used to convert augmented images and BoundingBoxOnImage objects into the
    tensor formats required by tf.
    """
    label_length = 5 + len(class_dict)

    # Initialise the ground truth objectness values to 0; step along the tensors to do this
    y_small[..., ::label_length] = 0.
    y_medium[..., ::label_length] = 0.
    y_large[..., ::label_length] = 0.

    # Iterate through the bounding boxes for the given image; bboxes are currently in [x, y, w, h, label] format
    for bb in bbs:
        raw_bbox = [bb[0], bb[1], bb[2], bb[3]]

        grid_coordinates = []

        # Find the co-ordinates of the grid cell containing the centroid in each of the fmap_sizes
        for fmap_size in fmap_sizes:
            grid_x = int(bb[0] * fmap_size)
            grid_y = int(bb[1] * fmap_size)
            grid_coordinates.append((grid_x, grid_y))

        # Get the most similar anchor box to the gt bbox by measuring IOU between all anchors with bbox
        best_anchor_idx, best_anchor_box, ignore_anchors = get_best_anchor(raw_bbox,
                                                                           grid_coordinates,
                                                                           fmap_sizes,
                                                                           anchors_list)

        # scale refers to whether the gt bbox is classed as small, medium or large (0, 1 or 2)
        scale = int(best_anchor_idx // 3.)
        # anchor index in the scale refers to whether it is the 1st, 2nd or 3rd anchor in that scale which is best
        anchor_idx_in_scale = best_anchor_idx % 3

        # Find the ground truth transformations to turn the anchor box into the ground truth bbox
        tx = get_tx(raw_bbox, grid_coordinates[scale][0], fmap_sizes[scale])
        ty = get_ty(raw_bbox, grid_coordinates[scale][1], fmap_sizes[scale])
        tw = get_tw(raw_bbox, best_anchor_box)
        th = get_th(raw_bbox, best_anchor_box)

        # The one hot encoded vector that corresponds to the class of the current bbox
        # Currently supports one ground truth class per box only
        class_vector = class_dict[bb[4]]

        # The ground truth 1-dimensional vector of length label_length that contains all the bbox information for
        # inserting into the full ground truth tensor
        # Important: ground truth vectors will be in order: [to, tx, ty, tw, th, classes]
        gt_vector = np.concatenate([np.ravel(np.array([1., tx, ty, tw, th])), np.ravel(class_vector)])

        # Update the ground truth tensor for this example
        for i, gt_tensor in enumerate((y_small, y_medium, y_large)):

            # Update the correct ground truth tensor with the gt_vector values
            if scale == i:
                gt_tensor[grid_coordinates[scale][0],
                          grid_coordinates[scale][1],
                          (anchor_idx_in_scale * label_length):((anchor_idx_in_scale + 1) * label_length)] = gt_vector

            # Set the anchors which had IOU >= 0.5 but were not the best anchors to have objectness -1.
            # This is important to be able to set up the ignore mask later used in the loss function
            for j in range(3):
                # Anchors that should be ignored will have had a value of 1 assigned to them by get_best_anchor function
                if ignore_anchors[i * 3 + j] == 1:
                    # Check that the anchor is not already being used by a gt bbox, or not already ignored
                    if gt_tensor[grid_coordinates[i][0], grid_coordinates[i][1], label_length * j] == 0.:
                        # Update the ground truth objectness value and set it to -1. as a placeholder
                        gt_tensor[grid_coordinates[i][0], grid_coordinates[i][1], label_length * j] = -1.

    return y_small, y_medium, y_large
