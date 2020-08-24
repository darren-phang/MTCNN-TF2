import tensorflow as tf
import tensorflow.keras.layers as layers
from utils import *
import time
import cv2


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

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        cls_prob = self.cls(x)
        bbox_pred = self.box(x)

        return cls_prob, bbox_pred

    def detect(self, img, net_size=12, min_ob_size=20,
               scale_factor=0.79, thresh=0.6, padding=None):
        h, w, c = img.shape
        current_scale = float(net_size) / min_ob_size  # find initial scale
        im_resized = processed_image(img, current_scale)
        current_height, current_width, _ = im_resized.shape
        all_boxes = list()
        while min(current_height, current_width) > net_size:
            cls_cls_map, reg = self.call(tf.expand_dims(im_resized, axis=0), training=False)

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
            return None, None

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
        rois = rois / np.array([w, h, w, h])

        return rois, boxes[..., 4]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ckpt = tf.train.load_checkpoint('../data/PNet_landmark/PNet-18')
    print(ckpt.get_variable_to_shape_map())
    pnet = PNet()
    pnet(np.zeros([1, 12, 12, 3]))
    pnet.conv1.kernel.assign(ckpt.get_tensor("conv1/weights"))
    pnet.conv1.bias.assign(ckpt.get_tensor("conv1/biases"))
    pnet.conv2.kernel.assign(ckpt.get_tensor("conv2/weights"))
    pnet.conv2.bias.assign(ckpt.get_tensor("conv2/biases"))
    pnet.conv3.kernel.assign(ckpt.get_tensor("conv3/weights"))
    pnet.conv3.bias.assign(ckpt.get_tensor("conv3/biases"))

    pnet.cls.kernel.assign(ckpt.get_tensor("conv4_1/weights"))
    pnet.cls.bias.assign(ckpt.get_tensor("conv4_1/biases"))
    pnet.box.kernel.assign(ckpt.get_tensor("conv4_2/weights"))
    pnet.box.bias.assign(ckpt.get_tensor("conv4_2/biases"))
    # ckpt = tf.train.Checkpoint(pnet=pnet)
    # ckpt.restore("/media/cdut9c403/新加卷/darren/logs/MTCNN/Pnet/ckpt-7")
    #
    image_dir = "/media/darren/新加卷/Datasets/WiderFace/WIDER_val/images/1--Handshaking/1_Handshaking_Handshaking_1_94.jpg"
    image = cv2.imread(image_dir)
    roi, score = pnet.detect(image, thresh=0.6)

    h, w, c = image.shape
    roi *= np.array([w, h, w, h])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_instances(image, roi)
    plt.show()
