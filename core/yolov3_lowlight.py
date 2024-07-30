import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import core.utils as utils
import core.common as common
from core.config_lowlight  import cfg
from core.backbone import Darknet53
from einops import rearrange

class YOLOV3(nn.Module):
    """Implement PyTorch yolov3 here"""
    def __init__(self):
        super(YOLOV3, self).__init__()

        self.classes            = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_class          = len(self.classes)
        self.strides            = np.array(cfg.YOLO.STRIDES)
        self.anchors            = utils.get_anchors(cfg.YOLO.ANCHORS)
        self.upsample_method    = cfg.YOLO.UPSAMPLE_METHOD
        self.isp_flag           = cfg.YOLO.ISP_FLAG
        self.anchor_per_scale   = cfg.YOLO.ANCHOR_PER_SCALE
        self.iou_loss_thresh    = cfg.YOLO.IOU_LOSS_THRESH

        self.conv_lbbox         = None
        self.conv_mbbox         = None
        self.conv_sbbox         = None
        self.recovery_loss      = None
        self.pred_sbbox         = None
        self.pred_mbbox         = None
        self.pred_lbbox         = None

        self.darknet53 = Darknet53()

        self.conv_lbbox_model = nn.Sequential(
            common.ConvBlock(1024, 512, kernel_size=1, downsample=False, bn=True, activate=True),
            common.ConvBlock(512, 1024, kernel_size=3, downsample=False, bn=True, activate=True),
            common.ConvBlock(1024, 512, kernel_size=1, downsample=False, bn=True, activate=True),
            common.ConvBlock(512, 1024, kernel_size=3, downsample=False, bn=True, activate=True),
            common.ConvBlock(1024, 512, kernel_size=1, downsample=False, bn=True, activate=True)
        )

        self.output_lbbox = nn.Sequential(
            common.ConvBlock(512, 1024, kernel_size=3, downsample=False, bn=True, activate=True),
            common.ConvBlock(1024, 3*(self.num_class + 5), kernel_size=1, downsample=False, bn=False, activate=False)
        )

        self.conv_mbbox_model = nn.Sequential(
            common.ConvBlock(768, 256, kernel_size=1, downsample=False, bn=True, activate=True),
            common.ConvBlock(256, 512, kernel_size=3, downsample=False, bn=True, activate=True),
            common.ConvBlock(512, 256, kernel_size=1, downsample=False, bn=True, activate=True),
            common.ConvBlock(256, 512, kernel_size=3, downsample=False, bn=True, activate=True),
            common.ConvBlock(512, 256, kernel_size=1, downsample=False, bn=True, activate=True)
        )

        self.output_mbbox = nn.Sequential(
            common.ConvBlock(256, 512, kernel_size=3, downsample=False, bn=True, activate=True),
            common.ConvBlock(512, 3*(self.num_class + 5), kernel_size=1, downsample=False, bn=False, activate=False)
        )

        self.conv_sbbox_model = nn.Sequential(
        common.ConvBlock(384, 128, kernel_size=1, downsample=False, bn=True, activate=True),
        common.ConvBlock(128, 256, kernel_size=3, downsample=False, bn=True, activate=True),
        common.ConvBlock(256, 128, kernel_size=1, downsample=False, bn=True, activate=True),
        common.ConvBlock(128, 256, kernel_size=3, downsample=False, bn=True, activate=True),
        common.ConvBlock(256, 128, kernel_size=1, downsample=False, bn=True, activate=True),
        )

        self.output_sbbox = nn.Sequential(
            common.ConvBlock(128, 256, kernel_size=3, downsample=False, bn=True, activate=True),
            common.ConvBlock(256, 3*(self.num_class + 5), kernel_size=1, downsample=False, bn=False, activate=False)
        )

        self.conv57 = common.ConvBlock(512, 256, kernel_size=1, downsample=False, bn=True, activate=True)
        self.conv63 = common.ConvBlock(256, 128, kernel_size=1, downsample=False, bn=True, activate=True)

        self.extractor = common.ExtractParameters2(cfg)

    def forward(self, input_data, input_data_clean):

        filtered_image_batch    = input_data
        self.filter_params      = input_data
        filter_imgs_series      = []

        if self.isp_flag:
            input_data = F.interpolate(input_data, size=(256, 256), mode='bilinear')
            filter_features = self.extractor(input_data)

            filters = cfg.filters
            filters = [x(input_data, cfg) for x in filters]
            filter_parameters = []
            for j, filter in enumerate(filters):
                print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.', filter.get_short_name())
                print('      filter_features:', filter_features.shape)
                filtered_image_batch, filter_parameter = filter.apply(filtered_image_batch, filter_features)
                filter_parameters.append(filter_parameter)
                filter_imgs_series.append(filtered_image_batch)
                print('      output:', filtered_image_batch.shape)

        # wxl：一下三者，包括：滤波器参数、所有滤波器处理后的图像、滤波器处理的中间系列图像
            self.filter_params = filter_parameters
        self.image_isped = filtered_image_batch
        self.filter_imgs_series = filter_imgs_series

        self.recovery_loss = torch.sum((filtered_image_batch - input_data_clean) ** 2)

        route_1, route_2, input_data = self.darknet53(input_data)

        input_data = self.conv_lbbox_model(input_data)
        self.conv_lbbox = self.output_lbbox(input_data)

        input_data = self.conv57(input_data)
        input_data = F.interpolate(input_data, scale_factor=2, mode=self.upsample_method)
        input_data = torch.cat([input_data, route_2], dim=1)

        input_data = self.conv_mbbox_model(input_data)
        self.conv_mbbox = self.output_mbbox(input_data)

        input_data = self.conv63(input_data)
        input_data = F.interpolate(input_data, scale_factor=2, mode=self.upsample_method)
        input_data = torch.cat([input_data, route_1], dim=1)

        input_data = self.conv_sbbox_model(input_data)
        self.conv_sbbox = self.output_sbbox(input_data)

        self.pred_sbbox = self.decoder(self.conv_sbbox, self.anchors[0], self.strides[0])
        self.pred_mbbox = self.decoder(self.conv_mbbox, self.anchors[1], self.strides[1])
        self.pred_lbbox = self.decoder(self.conv_lbbox, self.anchors[2], self.strides[2])


    def decoder(self, conv_output, anchors, stride):
        batch_size, _, output_size, _ = conv_output.shape

        conv_output = rearrange(conv_output, 'b (anchor classes) h w -> b h w anchor classes', classes=5 + self.num_class, anchor=self.anchor_per_scale)
        # conv_output = conv_output.reshape(batch_size, output_size, output_size, self.anchor_per_scale, 5 + self.num_class)

        conv_raw_dxdy = conv_output[..., 0:2]
        conv_raw_dwdh = conv_output[..., 2:4]
        conv_raw_conf = conv_output[..., 4:5]
        conv_raw_prob = conv_output[..., 5:]

        y = torch.arange(output_size, dtype=torch.int).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(output_size, dtype=torch.int).unsqueeze(0).repeat(output_size, 1)

        xy_grid = torch.stack([x, y], dim=-1)
        xy_grid = xy_grid.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1 ,self.anchor_per_scale ,1).float().to("cuda:0")

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * torch.from_numpy(anchors).to("cuda:0").unsqueeze(0).unsqueeze(1).unsqueeze(2)) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)

        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)

        return torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)


    def compute_loss(self, label_sbbox, label_mbbox, label_lbbox, true_sbbox, true_mbbox, true_lbbox):

        loss_sbbox = self.loss_layer(self.conv_sbbox, self.pred_sbbox, label_sbbox, true_sbbox,
                                     anchors=self.anchors[0], stride=self.strides[0])
        loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                     anchors=self.anchors[1], stride=self.strides[1])
        loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                     anchors=self.anchors[2], stride=self.strides[2])

        giou_loss = loss_sbbox[0] + loss_mbbox[0] + loss_lbbox[0]
        conf_loss = loss_sbbox[1] + loss_mbbox[1] + loss_lbbox[1]
        prob_loss = loss_sbbox[2] + loss_mbbox[2] + loss_lbbox[2]

        recovery_loss = self.recovery_loss

        return giou_loss, conf_loss, prob_loss, recovery_loss
        
    def focal(self, target, actual, alpha=1, gamma=2):

        focal_loss = alpha * torch.pow(torch.abs(target - actual), gamma)

        return focal_loss
    
    def bbox_giou(self, boxes1, boxes2):
        boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
        boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

        boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                            torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
        boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                            torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
        right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = torch.clamp(right_down - left_up, min=0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area

        enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])

        enclose = torch.clamp(enclose_right_down - enclose_left_up, min=0.0)
        enclose_area = enclose[..., 0] * enclose[..., 1]
        giou = iou - (enclose_area - union_area) / enclose_area

        return giou
    
    def bbox_iou(self, boxes1, boxes2):
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
        boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)
        left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
        right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = torch.clamp(right_down - left_up, min=0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = inter_area / union_area
        return iou
    
    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):
        conv_shape  = conv.shape
        batch_size  = conv_shape[0]
        output_size = conv_shape[2]
        input_size  = stride * output_size
        conv = rearrange(conv, 'b (anchor classes) h w -> b h w anchor classes', classes=5 + self.num_class, anchor=self.anchor_per_scale)
        # conv = conv.view(batch_size, output_size, output_size, self.anchor_per_scale, 5 + self.num_class)
        conv_raw_conf = conv[..., 4:5]
        conv_raw_prob = conv[..., 5:]

        pred_xywh     = pred[..., 0:4]
        pred_conf     = pred[..., 4:5]

        label_xywh    = label[..., 0:4]
        respond_bbox  = label[..., 4:5]
        label_prob    = label[..., 5:]

        giou = self.bbox_giou(pred_xywh, label_xywh).unsqueeze(-1)
        input_size = float(input_size)
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = self.bbox_iou(pred_xywh.unsqueeze(-2), bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        max_iou = torch.max(iou, dim=-1, keepdim=True)[0]
        respond_bgd = (1.0 - respond_bbox) * (max_iou < self.iou_loss_thresh).float()

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
            respond_bbox * F.binary_cross_entropy_with_logits(conv_raw_conf, respond_bbox, reduction='none')
            +
            respond_bgd * F.binary_cross_entropy_with_logits(conv_raw_conf, respond_bbox, reduction='none')
        )

        prob_loss = respond_bbox * F.binary_cross_entropy_with_logits(conv_raw_prob, label_prob, reduction='none')

        giou_loss = torch.mean(torch.sum(giou_loss, dim=[1, 2, 3, 4]))
        conf_loss = torch.mean(torch.sum(conf_loss, dim=[1, 2, 3, 4]))
        prob_loss = torch.mean(torch.sum(prob_loss, dim=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss