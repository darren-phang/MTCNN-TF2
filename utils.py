import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches, lines

STANDARD_COLORS = [
    'RoyalBlue', 'SaddleBrown', 'Red', 'RosyBrown', 'Green', 'SandyBrown',
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
    height, width, channels = img.shape
    new_height = int(height * scale)  # resized new height
    new_width = int(width * scale)  # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
    img_resized = (img_resized - 127.5) / 128
    return np.array(img_resized, np.float32)


def generate_bbox(cls_map, reg, stride, cellsize, scale, threshold):
    # index of class_prob larger than threshold
    t_index = np.where(cls_map > threshold)

    # find nothing
    if t_index[0].size == 0:
        return np.array([])
    # offset
    dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

    reg = np.array([dx1, dy1, dx2, dy2])
    score = cls_map[t_index[0], t_index[1]]
    boundingbox = np.vstack([
        np.round((stride * t_index[1]) / scale),
        np.round((stride * t_index[0]) / scale),
        np.round((stride * t_index[1] + cellsize) / scale),
        np.round((stride * t_index[0] + cellsize) / scale),
        score,
        reg])

    return boundingbox.T


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
    # bbox format is y_min, x_min, y_max, x_max
    bbox[..., :2] = np.maximum(1, bbox[..., :2])
    bbox[..., 2:] = np.minimum(np.array([image_size[1], image_size[0]]) - 1, bbox[..., 2:])
    return bbox


def display_instances(image, boxes, class_names=None, class_ids=None,
                      scores=None, figsize=(12, 12), ax=None,
                      score_threshold=0.5, captions=None):
    if class_ids is None:
        class_ids = np.zeros([boxes.shape[0]], dtype=int)
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = STANDARD_COLORS
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        score = scores[i] if scores is not None else 1.
        if score_threshold > score:
            continue
        color = colors[class_ids[i]]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        x1, y1, x2, y2 = boxes[i]
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
    ax.imshow(masked_image.astype(np.uint8), aspect='equal')
