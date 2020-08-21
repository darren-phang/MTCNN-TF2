import tensorflow as tf
import tensorflow.keras.layers as layers


class PNet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        kwargs_conv = {"activation": layers.ReLU(),
                       "kernel_regularizer": tf.keras.regularizers.l2(0.0005)}
        self.conv1 = layers.Conv2D(10, 3, 1, **kwargs_conv)
        self.pool1 = layers.MaxPool2D([2, 2], 2, padding='SAME')
        self.conv2 = layers.Conv2D(16, 3, 1, **kwargs_conv)
        self.conv3 = layers.Conv2D(32, 3, 1, **kwargs_conv)

        # batch*H*W*2
        self.cls = layers.Conv2D(2, 1, 1, activation=layers.Softmax())
        # batch*H*W*4
        self.box = layers.Conv2D(4, 1, 1, activation=None)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        cls = self.cls(x)
        box = self.box(x)
        cls_prob = tf.squeeze(cls, [1, 2], name='cls_prob')
        bbox_pred = tf.squeeze(box, [1, 2], name='bbox_pred')

        return cls_prob, bbox_pred

    def detect(self, img, net_size=48, min_ob_size=48,
               scale_factor=0.79, thresh=0.6, padding=None):
        h, w, c = img.shape
        current_scale = float(net_size) / min_ob_size  # find initial scale
        im_resized = processed_image(img, current_scale)
        current_height, current_width, _ = im_resized.shape
        # fcn
        all_boxes = list()
        while min(current_height, current_width) > net_size:
            cls_cls_map, reg = self.call(tf.expand_dims(im_resized, axis=0), training=False)
            boxes = generate_bbox(cls_cls_map.numpy()[0, :, :, 1], reg.numpy()[0],
                                  8, net_size, current_scale, thresh)
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
            return None, None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = py_nms(all_boxes[:, :4], all_boxes[:, 4], None, 0.7)
        all_boxes = all_boxes[keep]

        bbh = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbw = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # refine the boxes
        y1, y2 = all_boxes[:, 0] + all_boxes[:, 5] * bbw, all_boxes[:, 2] + all_boxes[:, 7] * bbw
        x1, x2 = all_boxes[:, 1] + all_boxes[:, 6] * bbh, all_boxes[:, 3] + all_boxes[:, 8] * bbh
        boxes_y = np.vstack([
            np.minimum(y1, y2),
            np.minimum(x1, x2),
            np.maximum(y1, y2),
            np.maximum(x1, x2),
            all_boxes[:, 4]])
        boxes_y = boxes_y.T
        if padding is not None:
            if boxes_y.shape[0] < padding:
                boxes_y = np.pad(boxes_y, [[0, padding - boxes_y.shape[0]], [0, 0]])
            else:
                boxes_y = boxes_y[:padding]

        rois = boxes_y[..., :4]
        rois = clip_bbox(rois, (h, w)).numpy()
        rois = rois / np.array([h, w, h, w])

        return rois, boxes_y[..., 4]
