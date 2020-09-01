import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches, lines

STANDARD_COLORS = [
    'Red', 'RoyalBlue', 'SaddleBrown', 'RosyBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Red', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'YellowGreen', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Cornsilk', 'Crimson', 'Cyan',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Lavender', 'LavenderBlush', 'MediumAquaMarine', 'MediumOrchid',
    'AliceBlue', 'LawnGreen', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'Magenta', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey'
]


def convert_to_square(bbox):
    """Convert bbox to square
    Parameters:
    ----------
    bbox: numpy array , shape n x 4
        input bbox
    Returns:
    -------
    square bbox
    """
    # square_bbox = bbox.copy()

    w = bbox[..., 2] - bbox[..., 0] + 1
    h = bbox[..., 3] - bbox[..., 1] + 1
    max_side = tf.maximum(h, w)

    square_bbox_x1 = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox_y1 = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox_x2 = square_bbox_x1 + max_side - 1
    square_bbox_y2 = square_bbox_y1 + max_side - 1
    return tf.stack([square_bbox_x1, square_bbox_y1, square_bbox_x2, square_bbox_y2], axis=-1)


def pad(bboxes, img_shape):
    """
        pad the the bboxes, alse restrict the size of it
    Parameters:
    ----------
        bboxes: numpy array, n x 5
            input bboxes
        w: float number
            width of the input image
        h: float number
            height of the input image
    Returns :
    ------
        dy, dx : numpy array, n x 1
            start point of the bbox in target image
        edy, edx : numpy array, n x 1
            end point of the bbox in target image
        y, x : numpy array, n x 1
            start point of the bbox in original image
        ex, ex : numpy array, n x 1
            end point of the bbox in original image
        tmph, tmpw: numpy array, n x 1
            height and width of the bbox
    """
    w, h = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[0], tf.float32)

    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = tf.shape(bboxes)[0]

    dx, dy = tf.zeros((num_box,)), tf.zeros((num_box,))
    edx, edy = tmpw - 1, tmph - 1

    x, y, ex, ey = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]

    edx = tf.where(ex > w - 1, tmpw + w - 2 - ex, edx)
    ex = tf.where(ex > w - 1, w - 1, ex)

    edy = tf.where(ey > h - 1, tmph + h - 2 - ey, edy)
    ey = tf.where(ey > h - 1, h - 1, ey)

    dx = tf.where(x < 0, 0. - x, dx)
    x = tf.where(x < 0, 0., x)

    dy = tf.where(y < 0, 0. - y, dy)
    y = tf.where(y < 0, 0., y)

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [tf.cast(item, tf.int32) for item in return_list]
    return return_list


def calibrate_box(bbox, reg):
    """
        calibrate bboxes
    Parameters:
    ----------
        bbox: numpy array, shape n x 5
            input bboxes
        reg:  numpy array, shape n x 4
            bboxes adjustment
    Returns:
    -------
        bboxes after refinement
    """
    w = bbox[..., 2] - bbox[..., 0] + 1
    w = tf.expand_dims(w, 1)
    h = bbox[..., 3] - bbox[..., 1] + 1
    h = tf.expand_dims(h, 1)
    reg_m = tf.concat([w, h, w, h], axis=-1)
    aug = reg_m * reg
    bbox_c = bbox + aug
    return bbox_c


def compute_iou(box, boxes, box_area, boxes_area):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])

    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / (union + 1e-10)
    return iou


def cal_accuracy(cls_prob, label):
    '''
    :param cls_prob:
    :param label:
    :return:calculate classification accuracy for pos and neg examples only
    '''
    # get the index of maximum value along axis one from cls_prob
    # 0 for negative 1 for positive
    pred = tf.argmax(cls_prob, axis=1)
    label_int = tf.cast(label, tf.int64)
    # return the index of pos and neg examples
    cond = tf.where(tf.greater_equal(label_int, 0))
    picked = tf.squeeze(cond)
    # gather the label of pos and neg examples
    label_picked = tf.gather(label_int, picked)
    pred_picked = tf.gather(pred, picked)
    # calculate the mean value of a vector contains 1 and 0, 1 for correct classification, 0 for incorrect
    # ACC = (TP+FP)/total population
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked, pred_picked), tf.float32))
    return accuracy_op


def processed_image(img, scale):
    '''
    rescale/resize the image according to the scale
    :param img: image
    :param scale:
    :return: resized image
    '''
    img_shape = tf.shape(img)
    img_shape = tf.cast(img_shape, tf.float32)
    new_shape = tf.cast(img_shape[:2] * scale, tf.int32)
    img_resized = tf.image.resize(img, new_shape)
    img_resized = (img_resized - 127.5) / 128
    return img_resized


def generate_bbox(cls_map, reg, stride, cellsize, scale, threshold):
    # index of class_prob larger than threshold
    t_index = tf.where(cls_map > threshold)
    if tf.size(t_index) == 0:
        return tf.cast([], tf.float32)
    # offset
    reg = tf.gather_nd(reg, t_index)
    score = tf.gather_nd(cls_map, t_index)[..., tf.newaxis]
    t_index = tf.cast(t_index, tf.float32)
    bbox = tf.stack([stride * t_index[..., 1],
                     stride * t_index[..., 0],
                     stride * t_index[..., 1] + cellsize,
                     stride * t_index[..., 0] + cellsize], axis=-1)
    bbox = bbox / scale
    boundingbox = tf.concat([bbox, score, reg], axis=1)
    return boundingbox


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5, mode="Union"):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    if max_boxes is None:
        max_boxes = -1
    return keep[:max_boxes]


def clip_bbox(bbox, image_size):
    # bbox format is x_min, y_min, x_max, y_max
    xy_min = tf.maximum(1., bbox[..., :2])
    xy_max = tf.minimum(tf.cast([image_size[1], image_size[0]], tf.float32) - 1.,
                        bbox[..., 2:])
    return tf.concat([xy_min, xy_max], axis=-1)


def display_instances(image, boxes, class_names=None, class_ids=None,
                      landmarks=None, scores=None, figsize=(12, 12), ax=None,
                      score_threshold=0.5, captions=None):
    # Number of instances
    if boxes is None or not boxes.shape[0]:
        print("\n*** No instances to display *** \n")
        return
    if class_ids is None:
        class_ids = np.zeros([boxes.shape[0]], dtype=int)
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = STANDARD_COLORS
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(boxes.shape[0]):
        score = scores[i] if scores is not None else 1.
        if score_threshold > score:
            continue
        color = colors[class_ids[i]]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        x1, y1, x2, y2 = boxes[i]
        h = y2 - y1
        w = x2 - x1
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)
        if not captions:
            class_id = class_ids[i]
            label = class_names[class_id] if class_names is not None else ''
            caption = "{} {:.3f}".format(label, score) if score < 1. else label
        else:
            caption = captions[i]
        ax.text(x1, y1 - 10, caption,
                color=color, size=12, backgroundcolor="none")
        if landmarks is not None:
            for landmark in landmarks[i]:
                c = patches.Circle(landmark, radius=(h / 25 + w / 25) / 2, edgecolor=(1, 0, 0), facecolor='none')
                ax.add_patch(c)
    ax.imshow(masked_image.astype(np.uint8), aspect='equal')
