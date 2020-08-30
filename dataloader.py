import tensorflow as tf
import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import compute_iou
import json
import random
from utils import processed_image, convert_to_square


@tf.function
def random_flip_images(image, offset, label):
    # negative image have no offset annotation
    if label != 1: return image, offset, label
    if tf.random.uniform([], 1) > 0.5:
        image = tf.image.flip_left_right(image)
        # the y offset is not changed,
        offset = tf.unstack(offset, axis=-1)
        offset = tf.stack([offset[2], offset[1], offset[0], offset[3]], axis=-1)
        offset = offset * tf.constant([-1, 1, -1, 1], dtype=tf.float32)
    return image, offset, label


@tf.function
def random_flip_landmark(image, landmark):
    if tf.random.uniform([], 1) > 0.5:
        image = tf.image.flip_left_right(image)
        landmark = tf.reshape(landmark, [-1, 2])
        landmark = landmark * tf.cast([[-1, 1]], tf.float32)
        landmark = landmark + tf.cast([[1, 0]], tf.float32)
        landmark = tf.unstack(landmark, axis=0)
        landmark = tf.stack([landmark[1], landmark[0], landmark[2], landmark[4], landmark[3]], axis=0)
        # landmark[[0, 1]] = landmark[[1, 0]]  # left eye<->right eye
        # landmark[[3, 4]] = landmark[[4, 3]]  # left mouth<->right mouth
        landmark = tf.reshape(landmark, [-1])

    return image, landmark


def image_color_distort(image):
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    return image


class WiderFace:
    def __init__(self, dataset_root, subset):
        # WIDER FACE
        TRAIN_IMAGE_ROOT = "WIDER_train/images"
        VAL_IMAGE_ROOT = "WIDER_val/images"
        TRAIN_ANNOTATION = "wider_face_split/wider_face_train_bbx_gt.txt"
        VAL_ANNOTATION = "wider_face_split/wider_face_val_bbx_gt.txt"
        INFO = ['x1', 'y1', 'w', 'h', 'blur', 'expression', 'illumination', 'invalid', 'occlusion', 'pose']

        if subset == 'train':
            self.annotation = os.path.join(dataset_root, TRAIN_ANNOTATION)
            self.image_root = os.path.join(dataset_root, TRAIN_IMAGE_ROOT)
        elif subset == 'val':
            self.annotation = os.path.join(dataset_root, VAL_ANNOTATION)
            self.image_root = os.path.join(dataset_root, VAL_IMAGE_ROOT)
        self.image_info = []
        self.read_annotation()
        self.image_num = len(self.image_info)

    def read_annotation(self, invalid=False, blur=1):
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
                    _bbox = [float(_) for _ in all_lines[idx + i + 1].strip().split(" ")]
                    if not invalid and _bbox[7]:
                        print("invalid annotation")
                        continue
                    if blur < _bbox[4]:
                        print("blur annotation")
                        continue
                    _image_info['bbox'].append([_bbox[0], _bbox[1], _bbox[0] + _bbox[2], _bbox[1] + _bbox[3]])
                    _image_info['bbox_info'].append(_bbox[4:])
                idx = idx + 2 + face_num
                if len(_image_info['bbox']) != 0:
                    _image_info["num"] = len(_image_info['bbox'])
                    self.image_info.append(_image_info)

    def generate_sample_example(self, image_info, output_size, n_ind, n_pos, n_par):
        annotation = []
        image = cv2.imread(image_info['image_path'])
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
                if nx2 > width or ny2 > height:
                    continue
                crop_bbox = np.array([nx1, ny1, nx2, ny2])
                crop_area = size * size
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

    def generate_hard_example(self, image_info, output_size, n_ind, n_pos, n_par):
        annotation = []
        image = cv2.imread(image_info['image_path'])
        # the preprocessing can be changed by yourself
        pre_bbox, pre_scores, _ = self.detect_model[0].detect(image, thresh=0.5)
        if self.detect_model.__len__() != 1:
            pre_bbox, pre_scores, _ = self.detect_model[1].detect(image, pre_bbox, thresh=0.3)
        if pre_bbox is None:
            return annotation, n_ind, n_pos, n_par
        # pre_bbox = np.concatenate([pre_bbox, pre_scores], axis=-1)
        gts = np.array(image_info["bbox"], dtype=np.float32).reshape(-1, 4)
        gts_areas = gts[..., [2, 3]] - gts[..., [0, 1]]
        gts_areas = gts_areas[..., 0] * gts_areas[..., 1]
        # change to square
        dets = convert_to_square(pre_bbox)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1
            box = np.array([x_left, y_top, x_right, y_bottom])

            # ignore box beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > image.shape[1] - 1 or y_bottom > image.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            iou = compute_iou(box, gts, width * height, gts_areas)

            crop_img = image[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resize_img = cv2.resize(crop_img, (output_size, output_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write new_tools
            # Iou with all gts must below 0.3
            if np.max(iou) < 0.3 and neg_num < 60:
                # save the examples
                image_name = "negative/%s.jpg" % n_ind
                annotation.append({
                    'image_name': image_name,
                    'image': resize_img,
                    'label': 0,
                    'offset': [0, 0, 0, 0]
                })
                neg_num += 1
                n_ind += 1
            else:
                # find gt_box with the highest iou
                idx = np.argmax(iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg new_tools
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)
                # save positive and part-face images and write labels
                if np.max(iou) >= 0.65:
                    image_name = "positive/%s.jpg" % n_pos
                    annotation.append({
                        'image_name': image_name,
                        'image': resize_img,
                        'label': 1,
                        'offset': [offset_x1, offset_y1, offset_x2, offset_y2]
                    })
                    n_pos += 1
                elif np.max(iou) >= 0.4:
                    image_name = "part/%s.jpg" % n_par
                    annotation.append({
                        'image_name': image_name,
                        'image': resize_img,
                        'label': -1,
                        'offset': [offset_x1, offset_y1, offset_x2, offset_y2]
                    })
                    n_par += 1
        return annotation, n_ind, n_pos, n_par

    def pnet_generate_image(self, output_dir, output_size=12):
        annotation = []
        root_dir = os.path.join(output_dir, "PNet")
        os.makedirs(root_dir, exist_ok=True)
        os.makedirs(os.path.join(root_dir, "positive"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, "part"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, "negative"), exist_ok=True)
        n_ind = 0
        n_pos = 0
        n_par = 0
        information = tqdm(range(self.image_num), desc='generating data_origin: ')
        for image_info in self.image_info:
            ann, n_ind, n_pos, n_par = self.generate_sample_example(image_info, output_size, n_ind, n_pos, n_par)
            for _ann in ann:
                img = _ann.pop('image')
                cv2.imwrite(os.path.join(root_dir, _ann["image_name"]), img)
            annotation += ann
            information.set_description("positive:%d, part:%d, negative:%d" % (n_pos, n_par, n_ind))
            information.update(1)
        with open(os.path.join(root_dir, "annotation.json"), "w") as file:
            json.dump(annotation, file, indent=1)

    def generate_tfrecord(self, output_dir, stage_num, output_size=12, detect_model=None, record_name="data_origin"):
        root_name = "%sNet_TFRCORD" % (["P", "R", "O"][stage_num])
        self.detect_model = detect_model
        if stage_num > 1 and self.detect_model is None:
            raise ValueError("the detect model is None")
        generator_fun = self.generate_sample_example if stage_num == 0 else self.generate_hard_example
        root_dir = os.path.join(output_dir, root_name)
        tfrecord_name = os.path.join(root_dir, "%s.record" % record_name)
        os.makedirs(root_dir, exist_ok=True)
        n_ind = 0
        n_pos = 0
        n_par = 0
        label = {-1: 'par', 1: 'pos', 0: 'neg'}
        annotation = {'pos': [], 'neg': [], 'par': []}
        information = tqdm(range(self.image_num), desc='generating data_origin: ')
        for image_info in self.image_info:
            ann, n_ind, n_pos, n_par = generator_fun(image_info, output_size, n_ind, n_pos, n_par)
            for _ann in ann:
                annotation[label[_ann["label"]]].append(_ann)
            information.set_description("positive:%d, part:%d, negative:%d" % (n_pos, n_par, n_ind))
            information.update(1)
        information.close()
        self.balance_dataset(tfrecord_name, annotation, basenum=n_pos)

    def add_sample(self, image, image_name, label, offset):
        exam = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_name.encode()])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
                    'offset': tf.train.Feature(float_list=tf.train.FloatList(value=offset)),
                    'shape': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[image.shape[0], image.shape[1], image.shape[2]])),
                    'data_origin': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tf.io.encode_png(image).numpy()]))
                }
            )
        )
        self.writer.write(exam.SerializeToString())

    def get_sample(self, exam_proto):
        feature_description = {
            'name': tf.io.FixedLenFeature([1], tf.string),
            'label': tf.io.FixedLenFeature([1], tf.int64),
            'offset': tf.io.FixedLenFeature([4], tf.float32),
            'shape': tf.io.FixedLenFeature([3], tf.int64),
            'data_origin': tf.io.FixedLenFeature([1], tf.string)
        }
        data = tf.io.parse_single_example(exam_proto, feature_description)
        return data['name'], data['label'], data["offset"], data["shape"], data["data_origin"]

    def balance_dataset(self, tfrecord_name, annotation, basenum=100000, ratio_n_p_pr=(1, 1, 1)):
        each_label_num = {'neg': ratio_n_p_pr[0] * basenum,
                          'pos': ratio_n_p_pr[1] * basenum,
                          'par': ratio_n_p_pr[2] * basenum}
        all_num = 0
        for label in annotation:
            choose_num = min(len(annotation[label]), each_label_num[label])
            annotation[label] = np.random.choice(annotation[label], size=choose_num, replace=False)
            all_num += len(annotation[label])
        print(len(annotation['neg']), len(annotation['pos']), len(annotation['par']))
        self.writer = tf.io.TFRecordWriter(tfrecord_name)
        annotation = annotation['neg'].tolist() + annotation['pos'].tolist() + annotation['par'].tolist()
        random.shuffle(annotation)
        process = tqdm(range(all_num))
        for _ann in annotation:
            self.add_sample(_ann["image"], _ann["image_name"], _ann["label"], _ann["offset"])
            process.update(1)
        self.writer.close()


class LFW:
    def __init__(self, dataset_root):
        self.TRAIN_IMAGE_ANNOTATION = "train/trainImageList.txt"
        self.TEST_IMAGE_ANNOTATION = "train/testImageList.txt"
        self.IMAGE_ROOT = "train"
        self.dataset_root = dataset_root
        self.image_info = []
        self.read_annotation()
        self.read_annotation(read_test=True)
        self.image_num = len(self.image_info)

    def read_annotation(self, read_test=False):
        annotation = self.TEST_IMAGE_ANNOTATION if read_test else self.TRAIN_IMAGE_ANNOTATION
        annotation = os.path.join(self.dataset_root, annotation)
        with open(annotation, "r") as file:
            for line in file.readlines():
                info = line.strip().split(" ")
                _bbox = np.array([int(_) for _ in info[1: 5]])[[0, 2, 1, 3]]
                self.image_info.append({
                    "image_path": os.path.join(self.dataset_root, self.IMAGE_ROOT, info[0].replace("\\", "/")),
                    "bbox": _bbox.tolist(),
                    "landmark": [float(_) for _ in info[5:]],
                })

    def generate_tfrecord(self, output_dir, image_size, record_name="landmark"):
        os.makedirs(output_dir, exist_ok=True)
        tfrecord_name = os.path.join(output_dir, "%s.record" % record_name)
        self.writer = tf.io.TFRecordWriter(tfrecord_name)
        information = tqdm(range(self.image_num), desc='generating data_origin: ')
        for idx, image_info in enumerate(self.image_info):
            information.set_description("processing image: %d" % idx)
            # crop and resize the image
            image = cv2.imread(image_info["image_path"])
            _bbox = image_info["bbox"]
            image = image[_bbox[1]: _bbox[3], _bbox[0]: _bbox[2], :]
            image = cv2.resize(image, (image_size, image_size))
            # compute the landmark offset
            _landmark = np.array(image_info["landmark"]).reshape([-1, 2])
            _landmark = _landmark - np.array([_bbox[0], _bbox[1]])
            _landmark = _landmark / image_size
            _landmark = _landmark.flatten()
            exam = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'name': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[image_info["image_path"].encode()])),
                        'landmark': tf.train.Feature(float_list=tf.train.FloatList(value=_landmark.tolist())),
                        'shape': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[image.shape[0], image.shape[1], image.shape[2]])),
                        'data_origin': tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[tf.io.encode_png(image).numpy()]))
                    }
                )
            )
            self.writer.write(exam.SerializeToString())
            information.update(1)
        information.close()


class MTCNNGenerator:
    def __init__(self, dataset_root, record_name='data_origin'):
        self.tfrecord = os.path.join(dataset_root, "%s.record" % record_name)
        self.set_feature_description()

    def set_feature_description(self):
        raise NotImplementedError

    def _parse_function(self, exam_proto):
        raise NotImplementedError

    def get_generator(self, batch_size=32):
        dataset = tf.data.TFRecordDataset(self.tfrecord, num_parallel_reads=8) \
            .map(self._parse_function) \
            .shuffle(batch_size * 5) \
            .batch(batch_size) \
            .repeat(-1) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class BboxGenerator(MTCNNGenerator):
    def set_feature_description(self):
        self.feature_description = {
            'name': tf.io.FixedLenFeature([1], tf.string),
            'label': tf.io.FixedLenFeature([1], tf.int64),
            'offset': tf.io.FixedLenFeature([4], tf.float32),
            'shape': tf.io.FixedLenFeature([3], tf.int64),
            'data_origin': tf.io.FixedLenFeature([1], tf.string)
        }

    def _parse_function(self, exam_proto):
        data = tf.io.parse_single_example(exam_proto, self.feature_description)
        label = tf.cast(data["label"][0], tf.float32)
        offset = data["offset"]
        shape = data["shape"]
        image = tf.io.decode_png(data['data_origin'][0])
        image = tf.reshape(image, shape)
        image = image_color_distort(image)
        image, offset, label = random_flip_images(image, offset, label)
        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 128
        return image, offset, label


class LandmarkGenerator(MTCNNGenerator):
    def set_feature_description(self):
        self.feature_description = {
            'name': tf.io.FixedLenFeature([1], tf.string),
            'landmark': tf.io.FixedLenFeature([10], tf.float32),
            'shape': tf.io.FixedLenFeature([3], tf.int64),
            'data_origin': tf.io.FixedLenFeature([1], tf.string)
        }

    def _parse_function(self, exam_proto):
        data = tf.io.parse_single_example(exam_proto, self.feature_description)
        landmark = data["landmark"]
        shape = data["shape"]
        image = tf.io.decode_png(data['data_origin'][0])
        image = tf.reshape(image, shape)
        image = image_color_distort(image)
        image, landmark = random_flip_landmark(image, landmark)
        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 128
        return image, landmark


if __name__ == '__main__':
    import utils
    from models.mtcnn_models import PNet, RNet

    wider_face_dataset = ''
    dataset = WiderFace(wider_face_dataset, "train")
    # generate the dataset
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    pnet = PNet()
    rnet = RNet()
    ckpt = tf.train.Checkpoint(pnet=pnet, rnet=rnet)
    # 在生成Rnet训练集时需要加载Pnet的参数
    ckpt.restore("/media/cdut9c403/新加卷/darren/logs/MTCNN/Pnet/ckpt-18")
    # 在生成Onet训练集时需要加载Pnet和Rnet的参数
    ckpt.restore("/media/cdut9c403/新加卷/darren/logs/MTCNN/Rnet/ckpt-18")

    dataset.generate_tfrecord(wider_face_dataset,
                              stage_num=0,  # pnet=0, rnet=1, onet=2
                              output_size=12,  # pnet=12, rnet=24, onet=48
                              detect_model=None,  # pnet=None, rnet=[pnet], onet=[pnet, rnet]
                              record_name='train')

    # 我只在Onet的训练中加入了landmark的信息，并且是在全部训练完成后，单独训练landmark头
    lfw_face_dataset = '/Users/darrenpang/Documents/datasets/LFW'
    lfw = LFW(lfw_face_dataset)
    lfw.generate_tfrecord('/Users/darrenpang/Documents/datasets/LFW/train', 48)
