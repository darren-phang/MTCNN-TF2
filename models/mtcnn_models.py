import tensorflow.keras.layers as layers
from utils import *
import time
import cv2


def xy2yx(bbox):
    x1, y1, x2, y2 = tf.unstack(bbox, 4, axis=-1)
    bbox = tf.stack([y1, x1, y2, x2], axis=-1)
    return bbox


class PNet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = layers.Conv2D(10, 3, 1, activation=layers.PReLU(shared_axes=[1, 2]),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.pool1 = layers.MaxPool2D([2, 2], 2, padding='SAME')
        self.conv2 = layers.Conv2D(16, 3, 1, activation=layers.PReLU(shared_axes=[1, 2]),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.conv3 = layers.Conv2D(32, 3, 1, activation=layers.PReLU(shared_axes=[1, 2]),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005))

        # batch*H*W*2
        self.cls = layers.Conv2D(2, 1, 1, activation=layers.Softmax())
        # batch*H*W*4
        self.box = layers.Conv2D(4, 1, 1, activation=None)
        self.landmark = layers.Conv2D(10, 1, 1, activation=None)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        cls_prob = self.cls(x)
        bbox_pred = self.box(x)
        landmark_pred = self.landmark(x)
        if training:
            cls_prob = tf.squeeze(cls_prob, [1, 2])
            bbox_pred = tf.squeeze(bbox_pred, [1, 2])
            landmark_pred = tf.squeeze(landmark_pred, [1, 2])
        return cls_prob, bbox_pred, landmark_pred

    @tf.function
    def detect(self, img, net_size=12, min_ob_size=20,
               scale_factor=0.79, thresh=0.6):

        def cond(img_shape, all_boxes, im_resized, current_scale):
            return tf.minimum(img_shape[0], img_shape[1]) > net_size

        def body(img_shape, all_boxes, img_resized, current_scale):
            cls_cls_map, reg, _ = self.call(img_resized[np.newaxis, ...], training=False)
            boxes = generate_bbox(cls_cls_map[0, :, :, 1], reg[0],
                                  2, net_size, current_scale, thresh)

            current_scale *= scale_factor
            img_resized = processed_image(img, current_scale)
            img_shape = tf.shape(img_resized)
            if tf.size(boxes) == 0:
                return img_shape, all_boxes, img_resized, current_scale
            selected_indices = tf.image.non_max_suppression(boxes=xy2yx(boxes[:, :4]),
                                                            scores=boxes[:, 4],
                                                            max_output_size=1000,
                                                            iou_threshold=0.5)
            boxes = tf.gather(boxes, selected_indices)
            all_boxes = tf.concat([all_boxes, boxes], axis=0)
            return img_shape, all_boxes, img_resized, current_scale

        img_shape = tf.shape(img)
        current_scale = float(net_size) / min_ob_size  # find initial scale
        img_resized = processed_image(img, current_scale)
        all_boxes = tf.zeros([0, 9], tf.float32)
        im_shape, all_boxes, img_resized, current_scale = \
            tf.while_loop(cond, body, [tf.shape(img_resized), all_boxes, img_resized, current_scale],
                          shape_invariants=[tf.TensorShape([3]), tf.TensorShape([None, 9]),
                                            tf.TensorShape([None, None, 3]), tf.TensorShape([1])])

        if tf.size(all_boxes) == 0:
            return tf.cast([], tf.float32), tf.cast([], tf.float32), tf.cast([], tf.float32)

        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1

        # refine the boxes
        x1, x2 = all_boxes[:, 0] + all_boxes[:, 5] * bbw, all_boxes[:, 2] + all_boxes[:, 7] * bbw
        y1, y2 = all_boxes[:, 1] + all_boxes[:, 6] * bbh, all_boxes[:, 3] + all_boxes[:, 8] * bbh

        boxes = tf.concat([
            tf.minimum(x1, x2)[..., tf.newaxis],
            tf.minimum(y1, y2)[..., tf.newaxis],
            tf.maximum(x1, x2)[..., tf.newaxis],
            tf.maximum(y1, y2)[..., tf.newaxis]], axis=-1)

        boxes = clip_bbox(boxes, img_shape[:2])
        scores = all_boxes[:, 4]
        selected_indices = tf.image.non_max_suppression(boxes=xy2yx(boxes),
                                                        scores=scores,
                                                        max_output_size=1000,
                                                        iou_threshold=0.7)
        boxes = tf.gather(boxes, selected_indices)
        scores = tf.gather(scores, selected_indices)
        return boxes, scores, tf.cast([], tf.float32)


class RNet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = layers.Conv2D(28, 3, 1, activation=layers.PReLU(shared_axes=[1, 2]),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.pool1 = layers.MaxPool2D(3, 2, padding='SAME')
        self.conv2 = layers.Conv2D(48, 3, 1, activation=layers.PReLU(shared_axes=[1, 2]),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.pool2 = layers.MaxPool2D(3, 2)
        self.conv3 = layers.Conv2D(64, 2, 1, activation=layers.PReLU(shared_axes=[1, 2]),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.fc1 = layers.Dense(128, activation='relu')
        self.cls = layers.Dense(2, activation=tf.nn.softmax)
        self.box = layers.Dense(4, activation=None)
        self.landmark = layers.Dense(10, activation=None)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = layers.Flatten()(x)
        x = self.fc1(x)
        cls_pred = self.cls(x)
        box_pred = self.box(x)
        landmark_pred = self.landmark(x)
        return cls_pred, box_pred, landmark_pred

    @tf.function
    def detect(self, img, bbox_pnet, net_size=24, batch_size=256,
               thresh=0.6, iou_threshold=0.7, max_detect=100):
        img_shape = tf.shape(img)
        dets = convert_to_square(bbox_pnet)
        dets = tf.round(dets)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, img_shape[:2])
        num_boxes = tf.shape(dets)[0]
        cropped_ims = tf.zeros((0, net_size, net_size, 3), tf.float32)

        if tf.size(bbox_pnet) == 0:
            return tf.cast([], tf.float32), tf.cast([], tf.float32), tf.cast([], tf.float32)

        def cond(i, cropped_ims):
            return i < num_boxes

        def body(i, cropped_ims):
            tmp_img = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            tmp_img = tf.pad(tmp_img, [[dy[i], tmph[i] - edy[i] - 1],
                                       [dx[i], tmpw[i] - edx[i] - 1],
                                       [0, 0]])
            tmp_img = tf.cast(tmp_img, tf.float32)
            tmp_img = (tf.image.resize(tmp_img, (net_size, net_size)) - 127.5) / 128
            cropped_ims = tf.concat([cropped_ims, tmp_img[tf.newaxis, ...]], axis=0)
            i = i + 1
            return i, cropped_ims

        i, cropped_ims = tf.while_loop(cond, body, [0, cropped_ims],
                                       [tf.TensorShape(None), tf.TensorShape([None, net_size, net_size, 3])])
        scores = tf.zeros((0,), tf.float32)
        reg = tf.zeros((0, 4), tf.float32)
        data = tf.data.Dataset.from_tensor_slices(cropped_ims).batch(batch_size)
        for batch_in in data:
            _scores, _reg, _ = self.call(batch_in, training=False)
            scores = tf.concat([scores, _scores[..., 1]], axis=0)
            reg = tf.concat([reg, _reg], axis=0)
        boxes = calibrate_box(dets, reg)
        selected_indices = tf.image.non_max_suppression(boxes=xy2yx(boxes),
                                                        scores=scores,
                                                        max_output_size=max_detect,
                                                        iou_threshold=iou_threshold,
                                                        score_threshold=thresh)
        boxes, scores = tf.gather(boxes, selected_indices), tf.gather(scores, selected_indices)
        return boxes, scores, tf.cast([], tf.float32)


class ONet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = layers.Conv2D(32, 3, 1, activation=layers.PReLU(shared_axes=[1, 2]),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.pool1 = layers.MaxPool2D(3, 2, padding='SAME')
        self.conv2 = layers.Conv2D(64, 3, 1, activation=layers.PReLU(shared_axes=[1, 2]),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.pool2 = layers.MaxPool2D(3, 2)
        self.conv3 = layers.Conv2D(64, 3, 1, activation=layers.PReLU(shared_axes=[1, 2]),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.pool3 = layers.MaxPool2D(2, 2, padding='SAME')
        self.conv4 = layers.Conv2D(128, 2, 1, activation=layers.PReLU(shared_axes=[1, 2]),
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.fc1 = layers.Dense(256, activation='relu')
        self.cls = layers.Dense(2, activation=tf.nn.softmax)
        self.box = layers.Dense(4, activation=None)
        self.landmark = layers.Dense(10, activation=None)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = layers.Flatten()(x)
        x = self.fc1(x)
        cls_pred = self.cls(x)
        box_pred = self.box(x)
        landmark_pred = self.landmark(x)
        return cls_pred, box_pred, landmark_pred

    @tf.function
    def detect(self, img, bbox_rnet, net_size=48, batch_size=128,
               thresh=0.7, iou_threshold=0.5, max_detect=100):
        img_shape = tf.shape(img)
        dets = convert_to_square(bbox_rnet)
        dets = tf.round(dets)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, img_shape[:2])
        num_boxes = tf.shape(dets)[0]
        cropped_ims = tf.zeros((0, net_size, net_size, 3), tf.float32)

        if tf.size(bbox_rnet) == 0:
            return tf.cast([], tf.float32), tf.cast([], tf.float32), tf.cast([], tf.float32)

        def cond(i, cropped_ims):
            return i < num_boxes

        def body(i, cropped_ims):
            tmp_img = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            tmp_img = tf.pad(tmp_img, [[dy[i], tmph[i] - edy[i] - 1],
                                       [dx[i], tmpw[i] - edx[i] - 1],
                                       [0, 0]])
            tmp_img = tf.cast(tmp_img, tf.float32)
            tmp_img = (tf.image.resize(tmp_img, (net_size, net_size)) - 127.5) / 128
            cropped_ims = tf.concat([cropped_ims, tmp_img[tf.newaxis, ...]], axis=0)
            i = i + 1
            return i, cropped_ims

        i, cropped_ims = tf.while_loop(cond, body, [0, cropped_ims],
                                       [tf.TensorShape(None), tf.TensorShape([None, net_size, net_size, 3])])

        scores = tf.zeros((0,), tf.float32)
        reg = tf.zeros((0, 4), tf.float32)
        data = tf.data.Dataset.from_tensor_slices(cropped_ims).batch(batch_size)
        for batch_in in data:
            _scores, _reg, _ = self.call(batch_in, training=False)
            scores = tf.concat([scores, _scores[..., 1]], axis=0)
            reg = tf.concat([reg, _reg], axis=0)
        boxes = calibrate_box(dets, reg)
        selected_indices, selected_scores = tf.image.non_max_suppression_with_scores(boxes=xy2yx(boxes),
                                                                                     scores=scores,
                                                                                     max_output_size=max_detect,
                                                                                     iou_threshold=iou_threshold,
                                                                                     score_threshold=thresh)
        boxes, scores = tf.gather(boxes, selected_indices), tf.gather(scores, selected_indices)
        return boxes, scores, tf.cast([], tf.float32)
        # w = boxes[:, 2] - boxes[:, 0] + 1
        # h = boxes[:, 3] - boxes[:, 1] + 1
        # landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        # landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        # return boxes, scores, np.reshape(landmark, [landmark.shape[0], -1, 2])