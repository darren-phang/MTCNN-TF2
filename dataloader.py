import tensorflow as tf
import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import compute_iou
import json
import random

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

    def pnet_generator(self, output_dir, output_size=12):
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
            image = cv2.imread(image_info['image_path'])
            # x, y, x, y
            bboxes = np.array(image_info["bbox"]).astype(np.float32)
            bboxes_wh = bboxes[..., [2, 3]] - bboxes[..., [0, 1]]
            bboxes_area = bboxes_wh[..., 0] * bboxes_wh[..., 1]

            h, w, c = image.shape
            # 1. crop 50 negative example from each image, iou < 0.3
            neg_num = 0
            while neg_num < 50:
                size = np.random.randint(output_size, min(h, w) / 2)
                nx, ny = np.random.randint(0, w - size), np.random.randint(0, h - size)
                crop_bbox = [nx, ny, nx + size, ny + size]
                crop_area = size * size
                iou = compute_iou(crop_bbox, bboxes, crop_area, bboxes_area)

                cropped_img = image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2], :]
                resize_img = cv2.resize(cropped_img, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

                if np.max(iou) < 0.3:
                    image_name = "negative/%s.jpg" % n_ind
                    cv2.imwrite(os.path.join(root_dir, image_name), resize_img)
                    annotation.append({
                        'image_name': image_name,
                        'label': 0,
                        'offset': [0, 0, 0, 0]
                    })
                    neg_num += 1
                    n_ind += 1
            for bbox_id, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                box_w, box_h = bboxes_wh[bbox_id] + 1
                if max(w, h) < 20 or x1 < 0 or y1 < 0:
                    continue

                for i in range(5):
                    size = np.random.randint(output_size, min(h, w) / 2)
                    delta_x = np.random.randint(max(-size, -x1), box_w)
                    delta_y = np.random.randint(max(-size, -y1), box_h)
                    nx1, ny1 = int(max(0, x1 + delta_x)), int(max(0, y1 + delta_y))
                    if nx1 + size > w or ny1 + size > h:
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
                            'label': 0,
                            'offset': [0, 0, 0, 0]
                        })
                        cv2.imwrite(os.path.join(root_dir, image_name), resize_img)
                        n_ind += 1
                # 2.2 crop pos and part images
                for i in range(20):
                    size = np.random.randint(int(min(box_w, box_h) * 0.8), np.ceil(1.25 * max(box_w, box_h)))

                    if box_h < 5 or box_w < 5:
                        continue

                    delta_x = np.random.randint(-box_w * 0.2, box_w * 0.2)
                    delta_y = np.random.randint(-box_h * 0.2, box_h * 0.2)

                    nx1 = int(max(x1 + box_w / 2 + delta_x - size / 2, 0))
                    ny1 = int(max(y1 + box_h / 2 + delta_y - size / 2, 0))
                    nx2 = nx1 + size
                    ny2 = ny1 + size
                    if nx2 > w or ny2 > h:
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
                        cv2.imwrite(os.path.join(root_dir, image_name), resize_img)
                        annotation.append({
                            'image_name': image_name,
                            'label': 1,
                            'offset': [offset_x1, offset_y1, offset_x2, offset_y2]
                        })
                        n_pos += 1
                    elif iou >= 0.4:
                        image_name = "part/%s.jpg" % n_par
                        cv2.imwrite(os.path.join(root_dir, image_name), resize_img)
                        annotation.append({
                            'image_name': image_name,
                            'label': -1,
                            'offset': [offset_x1, offset_y1, offset_x2, offset_y2]
                        })
                        n_par += 1
            information.set_description("positive:%d, part:%d, negative:%d" % (n_pos, n_par, n_ind))
            information.update(1)
        with open(os.path.join(root_dir, "annotation.json"), "w") as file:
            json.dump(annotation, file, indent=1)


class MTCNNGenerator:
    def __init__(self, dataset_root, sub_model):
        self.model_dataset_path = os.path.join(dataset_root, sub_model)
        self.annotation = os.path.join(self.model_dataset_path, "annotation.json")
        self.image_info = json.load(open(self.annotation, "r"))
        random.shuffle(self.image_info)
        self.image_number = len(self.image_info)

    @staticmethod
    def random_flip_images(image, offset, label):
        if label != 1: return image, offset, label
        if np.random.random() > 0.5:
            image = image[:, ::-1]
            # the y offset is not changed,
            offset = offset[[2, 1, 0, 3]] * np.array([-1, 1, -1, 1])
        return image, offset, label

    @staticmethod
    def image_color_distort(image, bbox, label):
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        return image, bbox, label

    def generator(self):
        for image_info in self.image_info:
            image = cv2.imread(os.path.join(self.model_dataset_path, image_info["image_name"]))
            offset = np.array(image_info["offset"])
            label = image_info["label"]
            image, offset, label = self.random_flip_images(image, offset, label)
            yield image, offset, label

    def get_generator(self, batch_size=32):
        dataset = tf.data.Dataset.from_generator(self.generator, (tf.float32, tf.float32, tf.float32)) \
            .map(self.image_color_distort) \
            .shuffle(batch_size * 5) \
            .batch(batch_size) \
            .repeat(-1) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


if __name__ == '__main__':
    wider_face_dataset = '/media/cdut9c403/新加卷/darren/wider_face'
    dataset = WiderFace(wider_face_dataset, "train")
    dataset.pnet_generator(wider_face_dataset)