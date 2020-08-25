import tensorflow as tf
import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import compute_iou
import json
import random
from utils import processed_image

TRAIN_IMAGE_ROOT = "WIDER_train/images"
VAL_IMAGE_ROOT = "WIDER_val/images"
TRAIN_ANNOTATION = "wider_face_split/wider_face_train_bbx_gt.txt"
VAL_ANNOTATION = "wider_face_split/wider_face_val_bbx_gt.txt"
INFO = ['x1', 'y1', 'w', 'h', 'blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose']


class WiderFace:
    def __init__(self, dataset_root, subset):
        if subset == 'train':
            self.annotation = os.path.join(dataset_root, TRAIN_ANNOTATION)
            self.image_root = os.path.join(dataset_root, TRAIN_IMAGE_ROOT)
        elif subset == 'val':
            self.annotation = os.path.join(dataset_root, VAL_ANNOTATION)
            self.image_root = os.path.join(dataset_root, VAL_IMAGE_ROOT)
        self.image_info = []
        self.read_annotation()
        self.image_num = len(self.image_info)

    def read_annotation(self):
        with open(self.annotation, "r") as file:
            all_lines = file.readlines()
            idx = 0
            while idx < len(all_lines):
                image_name = all_lines[idx].strip()
                face_num = int(all_lines[idx + 1].strip())
                if face_num == 0:
                    idx = idx + 3
                    continue
                _image_info = {'image_path': os.path.join(self.image_root, image_name),
                               'num': face_num, 'bbox': [], 'bbox_info': []}
                # read the face bbox
                for i in range(1, face_num + 1):
                    _bbox = [int(_) for _ in all_lines[idx + i + 1].strip().split(" ")]
                    # if _bbox[2] <= 0 or _bbox[3] <= 0:
                    #     print(image_name, _bbox)
                    _image_info['bbox'].append([_bbox[0], _bbox[1], _bbox[0] + _bbox[2], _bbox[1] + _bbox[3]])
                    _image_info['bbox_info'].append(_bbox[4:])
                idx = idx + 2 + face_num
                self.image_info.append(_image_info)

    def generator_one_ann(self, image_info, output_size, n_ind, n_pos, n_par):
        annotation = []
        image = cv2.imread(image_info['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # x, y, x, y
        bboxes = np.array(image_info["bbox"]).astype(np.float32)
        bboxes_wh = bboxes[..., 2:] - bboxes[..., :2]
        bboxes_area = bboxes_wh[..., 0] * bboxes_wh[..., 1]

        height, width, channel = image.shape
        # 1. crop 50 negative example from each image, iou < 0.3
        neg_num = 0
        while neg_num < 50:
            size = np.random.randint(output_size, min(width, height) / 2)
            nx, ny = np.random.randint(0, width - size), np.random.randint(0, height - size)
            crop_bbox = [nx, ny, nx + size, ny + size]
            crop_area = size * size
            iou = compute_iou(crop_bbox, bboxes, crop_area, bboxes_area)
            cropped_img = image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
            resize_img = cv2.resize(cropped_img, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

            if np.max(iou) < 0.3:
                image_name = "negative/%s.jpg" % n_ind
                annotation.append({
                    'image_name': image_name,
                    'image': resize_img,
                    'label': 0,
                    'offset': [0, 0, 0, 0]
                })
                neg_num += 1
                n_ind += 1
        for bbox_id, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            box_w, box_h = bboxes_wh[bbox_id] + 1
            if max(width, height) < 20 or x1 < 0 or y1 < 0:
                continue

            for i in range(5):
                size = np.random.randint(output_size, min(height, width) / 2)
                delta_x = np.random.randint(max(-size, -x1), box_w)
                delta_y = np.random.randint(max(-size, -y1), box_h)
                nx1, ny1 = int(max(0, x1 + delta_x)), int(max(0, y1 + delta_y))
                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_bbox = np.array([nx1, ny1, nx1 + size, ny1 + size])
                crop_area = size * size
                iou = compute_iou(crop_bbox, bboxes, crop_area, bboxes_area)

                cropped_img = image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
                resize_img = cv2.resize(cropped_img, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
                if np.max(iou) < 0.3:
                    image_name = "negative/%s.jpg" % n_ind
                    annotation.append({
                        'image_name': image_name,
                        'image': resize_img,
                        'label': 0,
                        'offset': [0, 0, 0, 0]
                    })
                    n_ind += 1
            # 2.2 crop pos and part images
            for i in range(15):
                size = np.random.randint(int(min(box_w, box_h) * 0.8), np.ceil(1.25 * max(box_w, box_h)))

                if box_h < 5 or box_w < 5:
                    continue

                delta_x = np.random.randint(-box_w * 0.2, box_w * 0.2)
                delta_y = np.random.randint(-box_h * 0.2, box_h * 0.2)

                nx1 = int(max(x1 + box_w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + box_h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size
                if nx2 > width or ny2 > height:
                    continue
                crop_bbox = np.array([nx1, ny1, nx2, ny2])
                crop_area = output_size * output_size
                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)
                iou = compute_iou(crop_bbox, bbox[np.newaxis, ...],
                                  crop_area, bboxes_area[bbox_id][np.newaxis, ...])[0]
                cropped_img = image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
                resize_img = cv2.resize(cropped_img, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
                if iou >= 0.65:
                    image_name = "positive/%s.jpg" % n_pos
                    annotation.append({
                        'image_name': image_name,
                        'image': resize_img,
                        'label': 1,
                        'offset': [offset_x1, offset_y1, offset_x2, offset_y2]
                    })
                    n_pos += 1
                elif iou >= 0.4:
                    image_name = "part/%s.jpg" % n_par
                    annotation.append({
                        'image_name': image_name,
                        'image': resize_img,
                        'label': -1,
                        'offset': [offset_x1, offset_y1, offset_x2, offset_y2]
                    })
                    n_par += 1
        return annotation, n_ind, n_pos, n_par

    def pnet_generator_image(self, output_dir, output_size=12):
        annotation = []
        root_dir = os.path.join(output_dir, "PNet")
        os.makedirs(root_dir, exist_ok=True)
        os.makedirs(os.path.join(root_dir, "positive"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, "part"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, "negative"), exist_ok=True)
        n_ind = 0
        n_pos = 0
        n_par = 0
        information = tqdm(range(self.image_num), desc='generating data: ')
        for image_info in self.image_info:
            ann, n_ind, n_pos, n_par = self.generator_one_ann(image_info, output_size, n_ind, n_pos, n_par)
            for _ann in ann:
                img = _ann.pop('image')
                cv2.imwrite(os.path.join(root_dir, _ann["image_name"]), img)
            annotation += ann
            information.set_description("positive:%d, part:%d, negative:%d" % (n_pos, n_par, n_ind))
            information.update(1)
        with open(os.path.join(root_dir, "annotation.json"), "w") as file:
            json.dump(annotation, file, indent=1)

    def pnet_generator_tfrecord(self, output_dir, output_size=12):
        root_dir = os.path.join(output_dir, "PNet_TFRCORD")
        os.makedirs(root_dir, exist_ok=True)
        self.writer = tf.io.TFRecordWriter(os.path.join(root_dir, "data.record"))
        n_ind = 0
        n_pos = 0
        n_par = 0
        information = tqdm(range(self.image_num), desc='generating data: ')
        for image_info in self.image_info:
            ann, n_ind, n_pos, n_par = self.generator_one_ann(image_info, output_size, n_ind, n_pos, n_par)
            for _ann in ann:
                self.add_sample(_ann["image"], _ann["image_name"], _ann["label"], _ann["offset"])
            information.set_description("positive:%d, part:%d, negative:%d" % (n_pos, n_par, n_ind))
            information.update(1)
        self.writer.close()

    def add_sample(self, image, image_name, label, offset):
        exam = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_name.encode()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                    'offset': tf.train.Feature(float_list=tf.train.FloatList(value=offset)),
                    'shape': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[image.shape[0], image.shape[1], image.shape[2]])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_png(image).numpy()]))
                }
            )
        )
        self.writer.write(exam.SerializeToString())


class MTCNNGenerator:
    def __init__(self, dataset_root, sub_model):
        self.model_dataset_path = os.path.join(dataset_root, sub_model)
        self.annotation = os.path.join(self.model_dataset_path, "annotation.json")
        self.feature_description = {
            'name': tf.io.FixedLenFeature([1], tf.string),
            'label': tf.io.FixedLenFeature([1], tf.int64),
            'offset': tf.io.FixedLenFeature([4], tf.float32),
            'shape': tf.io.FixedLenFeature([3], tf.int64),
            'data': tf.io.FixedLenFeature([1], tf.string)
        }
        if os.path.exists(self.annotation):
            self.image_info = json.load(open(self.annotation, "r"))
            random.shuffle(self.image_info)
            self.image_number = len(self.image_info)
        else:
            record_list = []
            for name in os.listdir(self.model_dataset_path):
                if name.split(".")[-1] == "record":
                    record_list.append(os.path.join(self.model_dataset_path, name))
            if len(record_list) == 0:
                raise ValueError("no data to onload")
            self.tfrecord = record_list[0]

    @staticmethod
    @tf.function
    def random_flip_images(image, offset, label):
        # negative image have no offset annotation
        if label != 1: return image, offset, label
        if tf.random.uniform([], 1) > 0.5:
            image = tf.image.flip_left_right(image)
            # the y offset is not changed,
            # offset = tf.unstack(offset, axis=-1)
            # offset = tf.stack([offset[2], offset[1], offset[0], offset[3]], axis=-1)
            # offset = offset * tf.constant([-1, 1, -1, 1], dtype=tf.float32)
        return image, offset, label

    @staticmethod
    def image_color_distort(image, bbox, label):
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        return image, bbox, label

    def generator_image(self):
        for image_info in self.image_info:
            image = cv2.imread(os.path.join(self.model_dataset_path, image_info["image_name"]))
            image = (image - 127.5) / 128
            offset = np.array(image_info["offset"])
            label = image_info["label"]
            image, offset, label = self.random_flip_images(image, offset, label)
            yield image, offset, label

    def _parse_function(self, exam_proto):
        data = tf.io.parse_single_example(exam_proto, self.feature_description)
        label = tf.cast(data["label"][0], tf.float32)
        offset = data["offset"]
        shape = data["shape"]
        image = tf.io.decode_image(data['data'][0], 3, tf.float32)
        image = tf.reshape(image, shape)
        image, offset, label = self.image_color_distort(image, offset, label)
        # image = (image * 255 - 127.5) / 128
        image, offset, label = self.random_flip_images(image, offset, label)
        return image, offset, label

    def get_generator(self, batch_size=32):
        if self.tfrecord is not None:
            dataset = tf.data.TFRecordDataset(self.tfrecord, num_parallel_reads=8) \
                .map(self._parse_function) \
                .shuffle(batch_size * 5) \
                .batch(batch_size) \
                .repeat(-1) \
                .prefetch(tf.data.experimental.AUTOTUNE)
        else:
            dataset = tf.data.Dataset.from_generator(self.generator_image, (tf.float32, tf.float32, tf.float32)) \
                .map(self.image_color_distort) \
                .shuffle(batch_size * 5) \
                .batch(batch_size) \
                .repeat(-1) \
                .prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import utils

    # generate the dataset
    # wider_face_dataset = '/media/cdut9c403/新加卷/darren/wider_face'
    # dataset = WiderFace(wider_face_dataset, "train")
    # dataset.pnet_generator_tfrecord(wider_face_dataset, 12)

    # for image_info in dataset.image_info:
    #     image = cv2.imread(image_info['image_path'])
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     bboxes = np.array(image_info["bbox"]).astype(np.float32)
    #
    #     utils.display_instances(image, bboxes)
    #     plt.show()

    # read the tfrecord
    dataset_path = '/media/cdut9c403/新加卷/darren/wider_face'
    generator = MTCNNGenerator(dataset_path, "PNet_TFRCORD")
    for image, offset, label in generator.get_generator(1):
        image = image.numpy()[0] * 255
        if label[0] == 1:
            plt.imshow(image.astype(np.uint8))
            plt.show()
