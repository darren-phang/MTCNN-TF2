# @Author : Darren Pang
# @Time : 2020/9/1 2:54 下午
# @File : test_mdoel.py
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from models.mtcnn_models import PNet, RNet, ONet
from utils import display_instances

if __name__ == '__main__':
    image_dir = "/Users/darrenpang/Documents/datasets/LFW/train/lfw_5590/Alex_Barros_0001.jpg"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for i in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[i], True)
            tf.config.set_soft_device_placement(True)
    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    ckpt = tf.train.Checkpoint(pnet=pnet, rnet=rnet, onet=onet)
    ckpt.restore("ckpt/origin/pnet/ckpt-18")
    ckpt.restore("ckpt/origin/rnet/ckpt-14")
    ckpt.restore("ckpt/origin/onet/ckpt-16")
    image = cv2.imread(image_dir)
    roi, score, _ = pnet.detect(image, thresh=0.6)
    roi, score, _ = rnet.detect(image, roi, thresh=0.6)
    roi, score, landmarks = onet.detect(image, roi, thresh=0.6)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    display_instances(image, roi, landmarks=None)
    plt.show()
