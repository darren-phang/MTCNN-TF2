# @Author : Darren Pang
# @Time : 2020/8/31 9:45 上午
# @File : servce.py

import tensorflow as tf
from models import PNet, RNet, ONet


class MTCNNServing(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        ckpt = tf.train.Checkpoint(pnet=self.pnet, rnet=self.rnet, onet=self.onet)
        ckpt.restore("ckpt/origin/pnet/ckpt-18")
        ckpt.restore("ckpt/origin/rnet/ckpt-14")
        ckpt.restore("ckpt/origin/onet/ckpt-16")

    @tf.function(input_signature=[tf.TensorSpec([None, ], tf.string, name='base64'),
                                  tf.TensorSpec([None, ], tf.int32, name='height'),
                                  tf.TensorSpec([None, ], tf.int32, name='width')])
    def call(self, base64, height, width):
        image = tf.io.decode_image(base64[0], channels=3)
        red, green, blue = tf.split(image, num_or_size_splits=3, axis=-1)
        image = tf.concat(values=[blue, green, red], axis=-1)
        image = tf.reshape(image, [height[0], width[0], 3])
        image = tf.cast(image, tf.float32)
        roi, score, _ = self.pnet.detect(image, thresh=0.6)
        roi = tf.reshape(roi, [-1, 4])
        roi, score, _ = self.rnet.detect(image, roi, thresh=0.6)
        roi = tf.reshape(roi, [-1, 4])
        roi, score, landmarks = self.onet.detect(image, roi, thresh=0.6)
        return roi, score


model = MTCNNServing()
image = tf.keras.Input(shape=[], dtype=tf.string, name='image')
height = tf.keras.Input(shape=[], dtype=tf.int32, name='height')
width = tf.keras.Input(shape=[], dtype=tf.int32, name='width')
roi, score = model(image, height, width)
roi = tf.reshape(roi, [-1, 4], name='bbox')
score = tf.reshape(score, [-1], name='score')
mtcnn = tf.keras.Model(inputs=[image, height, width],
                       outputs=[roi, score])
mtcnn.save('mtcnn/0', save_format='tf')
