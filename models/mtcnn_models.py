import tensorflow.keras.layers as layers
from utils import *
import time
import cv2


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

    def detect(self, img, net_size=12, min_ob_size=20,
               scale_factor=0.79, thresh=0.6, padding=None):
        h, w, c = img.shape
        current_scale = float(net_size) / min_ob_size  # find initial scale
        im_resized = processed_image(img, current_scale)
        current_height, current_width, _ = im_resized.shape
        all_boxes = list()
        while min(current_height, current_width) > net_size:
            start_time = time.time()
            cls_cls_map, reg, _ = self.call(tf.expand_dims(im_resized, axis=0), training=False)
            print(time.time() - start_time)
            boxes = generate_bbox(cls_cls_map.numpy()[0, :, :, 1], reg.numpy()[0],
                                  2, net_size, current_scale, thresh)

            current_scale *= scale_factor
            im_resized = processed_image(img, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue
            # get the index from non-maximum s
            keep = py_nms(boxes[:, :4], boxes[:, 4], None, 0.5)
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None, None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = py_nms(all_boxes[:, :4], all_boxes[:, 4], None, 0.7)
        all_boxes = all_boxes[keep]

        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1

        # refine the boxes
        x1, x2 = all_boxes[:, 0] + all_boxes[:, 5] * bbw, all_boxes[:, 2] + all_boxes[:, 7] * bbw
        y1, y2 = all_boxes[:, 1] + all_boxes[:, 6] * bbh, all_boxes[:, 3] + all_boxes[:, 8] * bbh
        boxes = np.vstack([
            np.minimum(x1, x2),
            np.minimum(y1, y2),
            np.maximum(x1, x2),
            np.maximum(y1, y2),
            all_boxes[:, 4]])
        boxes = boxes.T
        if padding is not None:
            if boxes.shape[0] < padding:
                boxes = np.pad(boxes, [[0, padding - boxes.shape[0]], [0, 0]])
            else:
                boxes = boxes[:padding]

        rois = boxes[..., :4]
        rois = clip_bbox(rois, (h, w))
        return rois, boxes[..., 4], None


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

    def detect(self, img, bbox_pnet, net_size=24, batch_size=256,
               score_thresh=0.6, iou_threshold=0.6, max_detect=100):
        h, w, c = img.shape
        dets = convert_to_square(bbox_pnet)
        dets = np.round(dets)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, net_size, net_size, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp_img = np.zeros((tmph[i], tmpw[i], 3))
            tmp_img[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp_img, (net_size, net_size)) - 127.5) / 128

        cls_scores, reg = [], []
        for i in range(int(np.ceil(num_boxes / batch_size))):
            data_in = cropped_ims[i * batch_size:(i + 1) * batch_size]
            _cls_scores, _reg, _ = self.call(data_in, training=False)
            _cls_scores = _cls_scores[:, 1]
            cls_scores.append(_cls_scores)
            reg.append(_reg)
        if len(cls_scores) == 0:
            return None, None, None
        cls_scores = np.concatenate(cls_scores, axis=0)
        reg = np.concatenate(reg, axis=0)
        keep_inds = np.where(cls_scores > score_thresh)[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            scores = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None, None
        keep = py_nms(boxes, scores, max_detect, iou_threshold)
        boxes, scores, reg = boxes[keep], scores[keep], reg[keep]
        boxes = calibrate_box(boxes, reg)
        return boxes, scores, None


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

    def detect(self, img, bbox_rnet, net_size=48, batch_size=128, thresh=0.6, iou_threshold=0.6, max_detect=100):
        h, w, c = img.shape
        dets = convert_to_square(bbox_rnet)
        dets = np.round(dets)
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, net_size, net_size, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp_img = np.zeros((tmph[i], tmpw[i], 3))
            tmp_img[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp_img, (net_size, net_size)) - 127.5) / 128

        cls_scores, reg, landmark = [], [], []
        for i in range(int(np.ceil(num_boxes / batch_size))):
            data_in = cropped_ims[i * batch_size:(i + 1) * batch_size]
            _cls_scores, _reg, _landmark = self.call(data_in, training=False)
            _cls_scores = _cls_scores[:, 1]
            cls_scores.append(_cls_scores)
            reg.append(_reg)
            landmark.append(_landmark)
        if len(cls_scores) == 0:
            print("b")
            return None, None, None
        cls_scores = np.concatenate(cls_scores, axis=0)
        reg = np.concatenate(reg, axis=0)
        landmark = np.concatenate(landmark, axis=0)

        keep_inds = np.where(cls_scores > thresh)[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            scores = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            print("a")
            return None, None, None
        boxes = calibrate_box(boxes, reg)
        keep = py_nms(boxes, scores, max_detect, iou_threshold, mode="Minimum")
        boxes, scores, landmark = boxes[keep], scores[keep], landmark[keep]
        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1
        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        return boxes, scores, np.reshape(landmark, [landmark.shape[0], -1, 2])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    ckpt = tf.train.Checkpoint(pnet=pnet, rnet=rnet, onet=onet)
    ckpt.restore("/media/cdut9c403/新加卷/darren/logs/MTCNN/Pnet/ckpt-18")
    # ckpt.restore("../data/pnet/ckpt-18")
    # ckpt.restore("../data/rnet/ckpt-14")
    # ckpt.restore("../data/onet/ckpt-16")
    image_dir = "/media/cdut9c403/新加卷/darren/wider_face/WIDER_val/images/28--Sports_Fan/28_Sports_Fan_Sports_Fan_28_487.jpg"
    image = cv2.imread(image_dir)
    roi, score, _ = pnet.detect(image, thresh=0.7)
    # roi, score, _ = rnet.detect(image, roi)
    # roi, score, landmarks = onet.detect(image, roi)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_instances(image, roi, landmarks=None)
    plt.show()
