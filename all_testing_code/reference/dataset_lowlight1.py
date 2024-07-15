#! /usr/bin/env python
# coding=utf-8


import os
import cv2
import random
import numpy as np
#! 有问题
import tensorflow as tf
import utils as utils
from config_lowlight import cfg



class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset_type):
        self.annot_path  = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size  = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug    = cfg.TRAIN.DATA_AUG   if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    # wxl：没有用到数据集类型，默认数据格式是逐行读入的，并且每行是一个单元的标注信息
    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        print('###################the total image:', len(annotations))
        return annotations

    def __iter__(self):
        return self

    def __next__(self):

        with tf.device('/cpu:0'):
            self.train_input_size = random.choice(self.train_input_sizes)
            self.train_output_sizes = self.train_input_size // self.strides

            # wxl：构建一批的输入图片的空数据结构
            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))
            # wxl：构建一批的输出数据的空数据结构
            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes))
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes))
            # wxl：构建一批真值框的空数据结构
            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):

        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        image = np.array(cv2.imread(image_path))
        # print('*****************read image***************************')
        # wxl：取出标注信息中的bbox信息，字符串——>浮点数——>整数，
        bboxes = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in line[1:]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area


    # wxl：遍历真值框，并返回结果
    # wxl：这个函数的主要作用：
    # wxl：每个真值框需要经过不同尺度的栅格化，但是在栅格化的过程中很有可能伴随着，这个框和这个栅格的尺度本身并不适配，（因为预测框是在栅格的基础上偏移来的）
    # wxl：因此应该将真值框划分到与之尺度适配的一个或多个栅格中。
    # wxl：在本脚本中真值框到栅格框的划分依据是：如果真值框与所属的栅格框的iou大于0.3，则认为该两者适配，可以将真值框保留到该栅格尺度下
    # wxl：如果真值框与所有尺度的栅格框的交幷比都小于0.3，则把该真值框划分到交幷比最接近的一个栅格尺度下。
    def preprocess_true_boxes(self, bboxes):

        # wxl：注意label的尺度是：[3,output_size,output_size,每个尺度下的每个栅格上的框的数目，xywh con + prob]
        # wxl：同时每个的output_size对应着每个尺度下的栅格数目，是会随着栅格尺度的变化而改变的。
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        # wxl：bboxes_xywh存储着栅格化的框的信息，形状是[3,150,4]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        # wxl：这是一个计数的矩阵，用于判断在遍历所有的真值框时，在栅格化后，每个尺度下的真值框数目小于150。
        # wxl：且后面的策略是，大于150的部分，索引对150取模，直接覆盖掉原本的值。
        # wxl：这里是合理的，因为一张图片中有150个目标已经是非常amazing的事情了。
        bbox_count = np.zeros((3,))

        # wxl：遍历所有的框，这里是一张图片对应的标注文件中所有的框
        for bbox in bboxes:
            # wxl：首先取出所有框的坐标和类别信息
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            # wxl：对于所有的框在标注信息中是没有类别置信度信息的，但是回归时是需要的，因此需要依据类别信息制备类别置信度信息，并且类别置信度信息应该是独热的。
            # wxl：类别信息在标注文件中会通过类别index给出。
            # wxl：将输入的以数字大小代表的类别编码，转换成独热类别编码。
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            # wxl：均值分布，让分布的尺度等于类别数，分布的值等于类别数的倒数
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            # wxl：保持值的和是1不变，同时让每个不是类别的非独热位置上，有小值。
            # smooth_onehot是经过平滑的独热编码。
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            # wxl：将（x1,y1,x2,y2）转换成（xc，yc，w，h）
            # wxl：注意这里是将框的尺寸从像素坐标系归一化到栅格坐标系，由于栅格大小有多种，所以归一化的到栅格坐标系的结果也有多种
            # wxl：bbox_xywh这个变量十分重要，代表着在像素空间下，框的（xc，yc，w，h）
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # wxl：这里是将框从像素空间，化到了栅格空间
            # wxl：bbox_xywh_scaled这个变量也很重要，代表着在栅格空间下所有框的（xc，yc，w，h）
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            # wxl：下面要逐尺度得到每个框对应的栅格框。
            exist_positive = False
            for i in range(3):
                # wxl：这里得到的是一个栅格对应的锚框数目。
                # wxl：anchors_xywh代表所有的栅格框。
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                # wxl：计算得到框的中心点所在的栅格的中心点。
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                # wxl：计算得到栅格的宽高。
                anchors_xywh[:, 2:4] = self.anchors[i]

                # wxl：这里计算交幷比的函数，是在栅格坐标系中，真值框与真值框所在栅格做交幷比。
                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)

                # wxl：留下真值框与栅格交幷比大于0.3的框的结果
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):

                    # wxl：取出交幷比大于0.3部分的栅格左上角。
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    # wxl：通过交幷比的筛选，将真值框和类别概率附到label中。
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    # wxl：进行取模运算，保证每个尺度上的边界框不会超过最大值。
                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    # wxl：将bbox_xywh满足最大数目要求的前面个数个，放入bboxes_xywh中
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    # wxl：在最外层的循环中会在bboxes中进行遍历，因此每当有一个新的框这里都会加1，大于最大数之后，会将前面覆盖。
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                # wxl：这里返回的是IOU最大的框的索引，框的索引对应的框是按如下方法进行分布的
                # wxl：在标注文件中的第一个像素框，其在每个栅格空间都对应着一个栅格框。
                # wxl：先排列完一个栅格尺度下的所有框，再排列下一个栅格尺度。
                # wxl：因此下面就可以通过简单的方法，在已知框的总的索引的前提下，推断出框所属的栅格尺度，和所属当前栅格尺度下的索引。
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                # wxl：除法再取整，会舍去小数点后的数字，索引是从0开始，因此可以得到所属栅格尺度的索引。
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                # wxl：取模再取整，得到的是在某一个栅格尺度下的数目索引。
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                # wxl：取出栅格空间中的 与栅格交幷比最高的框的中心点，对应的栅格的左上角坐标
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                # wxl：将最好的框的坐标按索引放置在label的对应位置。
                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                # wxl：这里是将最终的结构放置在bboxes中。
                # wxl：需要保证每次这样得到的框的数目也不会大于每个尺度下所能拥有的最多的框的数目
                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        # wxl：最终将框的真值信息进行返回、
        # wxl：返回的label项，是单张图片按照预测结果的数据结构返回的，所以其形状如下[尺度数目，一张图片上的框的数目（小于最大的框的数目），4（xywh）+1（conf）+ 类别数]
        label_sbbox, label_mbbox, label_lbbox = label
        # wxl：返回的bbox项仅仅是按照xxywh进行返回。
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs




