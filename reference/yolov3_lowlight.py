#! /usr/bin/env python
# coding=utf-8


import numpy as np
import tensorflow as tf
import core.utils as utils
import darkdet.common as common
import core.backbone as backbone
from core.config_lowlight import cfg


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""
    def __init__(self, input_data, trainable, input_data_clean):

        self.trainable        = trainable
        self.classes          = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class        = len(self.classes)
        self.strides          = np.array(cfg.YOLO.STRIDES)
        self.anchors          = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh  = cfg.YOLO.IOU_LOSS_THRESH
        self.upsample_method  = cfg.YOLO.UPSAMPLE_METHOD
        self.isp_flag = cfg.YOLO.ISP_FLAG


        try:
            self.conv_lbbox, self.conv_mbbox, self.conv_sbbox, self.recovery_loss= self.__build_nework(input_data, self.isp_flag, input_data_clean)
        except:
            raise NotImplementedError("Can not build up yolov3 network!")

        with tf.variable_scope('pred_sbbox'):
            self.pred_sbbox = self.decode(self.conv_sbbox, self.anchors[0], self.strides[0])

        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[1], self.strides[1])

        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[2], self.strides[2])

    def __build_nework(self, input_data, isp_flag, input_data_clean):

        filtered_image_batch = input_data
        self.filter_params = input_data
        filter_imgs_series = []

        if isp_flag:
            with tf.variable_scope('extract_parameters_2'):
                input_data = tf.image.resize_images(input_data, [256, 256], method=tf.image.ResizeMethod.BILINEAR)
                filter_features = common.extract_parameters_2(input_data, cfg, self.trainable)

            # filter_features = tf.random_normal([1, 10], 0.5, 0.1)

            filters = cfg.filters
            filters = [x(input_data, cfg) for x in filters]
            filter_parameters = []
            for j, filter in enumerate(filters):
                with tf.variable_scope('filter_%d' % j):
                    print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.',
                          filter.get_short_name())
                    print('      filter_features:', filter_features.shape)

                    filtered_image_batch, filter_parameter = filter.apply(
                        filtered_image_batch, filter_features)
                    filter_parameters.append(filter_parameter)
                    filter_imgs_series.append(filtered_image_batch)


                    print('      output:', filtered_image_batch.shape)
            self.filter_params = filter_parameters
        self.image_isped = filtered_image_batch
        self.filter_imgs_series = filter_imgs_series

        recovery_loss = tf.reduce_sum(tf.pow(filtered_image_batch - input_data_clean, 2.0))#/(2.0 * batch_size)

        input_data = filtered_image_batch
        route_1, route_2, input_data = backbone.darknet53(input_data, self.trainable)

        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv52')
        input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv53')
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv54')
        input_data = common.convolutional(input_data, (3, 3,  512, 1024), self.trainable, 'conv55')
        input_data = common.convolutional(input_data, (1, 1, 1024,  512), self.trainable, 'conv56')

        conv_lobj_branch = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, name='conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_lbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1,  512,  256), self.trainable, 'conv57')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route_2], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 768, 256), self.trainable, 'conv58')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv59')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv60')
        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv61')
        input_data = common.convolutional(input_data, (1, 1, 512, 256), self.trainable, 'conv62')

        conv_mobj_branch = common.convolutional(input_data, (3, 3, 256, 512),  self.trainable, name='conv_mobj_branch' )
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_mbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv63')
        input_data = common.upsample(input_data, name='upsample1', method=self.upsample_method)

        with tf.variable_scope('route_2'):
            input_data = tf.concat([input_data, route_1], axis=-1)

        input_data = common.convolutional(input_data, (1, 1, 384, 128), self.trainable, 'conv64')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv65')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv66')
        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv67')
        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv68')

        conv_sobj_branch = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, name='conv_sobj_branch')
        conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3*(self.num_class + 5)),
                                          trainable=self.trainable, name='conv_sbbox', activate=False, bn=False)

        return conv_lbbox, conv_mbbox, conv_sbbox, recovery_loss

    def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape       = tf.shape(conv_output)
        batch_size       = conv_shape[0]
        output_size      = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5: ]

        ### 总结，对于xy_grid，一个批次有多少个，最外层就重复多少次
        ###                 ，第二层和第三层是方形栅格的个数，对应方形栅格的坐标，也对应着，最内层的数字（最内层数字也是坐标（列， 行） or（x，y））
        ###                 ，第四层是一个scale下一个栅格对应多少个框，有多少个框就需要将坐标重复多少次  

        # 例如output_size = 3
        # y= [[0, 0, 0],
        #     [1, 1, 1],
        #     [2, 2, 2]]
        # x= [[0, 1, 2],
        #     [0, 1, 2],
        #     [0, 1, 2]]
        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        # xy_grid = [[[0, 0], [1, 0], [2, 0]],
        #            [[0, 1], [1, 1], [2, 1]],
        #            [[0, 2], [1, 2], [2, 2]]]

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)

        # 假设 batch_size 为 2，anchor_per_scale 为 3，则 xy_grid 变为：
        # xy_grid = [[[[[0, 0], [0, 0], [0, 0]], [[1, 0], [1, 0], [1, 0]], [[2, 0], [2, 0], [2, 0]]],
        #             [[[0, 1], [0, 1], [0, 1]], [[1, 1], [1, 1], [1, 1]], [[2, 1], [2, 1], [2, 1]]],
        #             [[[0, 2], [0, 2], [0, 2]], [[1, 2], [1, 2], [1, 2]], [[2, 2], [2, 2], [2, 2]]]],
        
        #            [[[[0, 0], [0, 0], [0, 0]], [[1, 0], [1, 0], [1, 0]], [[2, 0], [2, 0], [2, 0]]],
        #             [[[0, 1], [0, 1], [0, 1]], [[1, 1], [1, 1], [1, 1]], [[2, 1], [2, 1], [2, 1]]],
        #             [[[0, 2], [0, 2], [0, 2]], [[1, 2], [1, 2], [1, 2]], [[2, 2], [2, 2], [2, 2]]]]]  # 形状为 [2, 3, 3, 3, 2]
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # x，y都是偏移量，因此加上栅格坐标，就是在栅格空间的预测坐标，乘以栅格尺缩，就是像素空间的坐标。
        # 关于sigmoid函数 & exp函数映射， 需要解释训练时怎么学，推理时就怎么算，所以任意合理的映射关系都可以。
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        # 这里anchor代表栅格的大小，因此tf.exp(conv_raw_dwdh)代表预测框相对于栅格缩放的倍数，conv_raw_dwdh代表预测框相对于栅格缩放倍数的对数
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        # 关于置信度 & 类别概率，二者经过sigmoid函数映射也是相同的道理。
        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        # 所以 raw 到 最终的结果 的关系 就是 ： 适当的非线性激活函数映射  +  栅格空间 ——> 像素空间映射
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

        # 将（cx,cy,w,h） 转换成 （x1,y1,x2,y2）
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        # 将可能存在的左下，右上点，强制转换成左上，右下点
        # 即（x_min,y_min,x_max,y_max）
        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        # 计算两个框各自的面积
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        # 取出两个框的最大内正接矩形框的坐标
        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        # 如果最大内正接矩形不存在，则置零
        inter_section = tf.maximum(right_down - left_up, 0.0)

        # 计算最大内正接矩形的面积，作为交集面积
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # 计算并集面积 & 交幷比
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        # 计算最小正外接矩形的坐标
        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])

        # 确保最小正外接矩形是大于0的
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # 计算最小正外接矩形的面积
        enclose_area = enclose[..., 0] * enclose[..., 1]

        # 后一部分是空白占大矩形的比例
        # 整体可以理解为 iou + 并集/整体 -1
        # 好，解决一些iou不能表征的误差
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape  = tf.shape(conv)
        batch_size  = conv_shape[0]
        output_size = conv_shape[1]
        input_size  = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh     = pred[:, :, :, :, 0:4]
        pred_conf     = pred[:, :, :, :, 4:5]

        label_xywh    = label[:, :, :, :, 0:4]
        respond_bbox  = label[:, :, :, :, 4:5]
        label_prob    = label[:, :, :, :, 5:]
        
        # wxl：先计算giou，再依据框的尺寸计算，giou的权重因子，最后为有框之处计算giou_loss
        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

        # wxl：同时满足：那处没框 & 那处计算的最大iou也小于阈值， 认为是背景
        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < self.iou_loss_thresh, tf.float32 )

        # wxl：将有无真值框的01矩阵 & 预测框置信度的概率矩阵，求差方矩阵，作为权重因子。
        conf_focal = self.focal(respond_bbox, pred_conf)
        # wxl：
        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

        return giou_loss, conf_loss, prob_loss



    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        with tf.name_scope('smaller_box_loss'):
            loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                         anchors = self.anchors[0], stride = self.strides[0])

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors = self.anchors[1], stride = self.strides[1])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors = self.anchors[2], stride = self.strides[2])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]
        with tf.name_scope('recovery_loss'):
            recovery_loss = self.recovery_loss

        return giou_loss, conf_loss, prob_loss, recovery_loss