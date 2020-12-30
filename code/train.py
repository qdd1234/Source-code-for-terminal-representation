#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : zhoukunyang
#   Created date: 2020-12-08
#   
#
# ================================================================
import datetime
import json
from collections import deque
import paddle.fluid as fluid
import paddle.fluid.layers as P
import sys
import time
import shutil
import math
import copy
import random
import threading
import numpy as np
import os

from config import *
from model.head import YOLOv3Head
from model.resnet import Resnet50Vd
from model.yolov3 import YOLOv3
from model.yolov4 import YOLOv4
from tools.cocotools import get_classes, catid2clsid, clsid2catid
from model.decode_np import Decode
from tools.cocotools import eval
from tools.data_process import data_clean, get_samples
from tools.transform import *
from pycocotools.coco import COCO

import logging

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
# logging.basicConfig(level=logging.INFO, format=FORMAT)
# logger = logging.getLogger(__name__)

def init_log_config():
    """
    初始化日志相关配置
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train.log')
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

logger = init_log_config()

def bbox_ciou(boxes1, boxes2):
    '''
    计算ciou = iou - p2/c2 - av
    :param boxes1: (8, 13, 13, 3, 4)   pred_xywh
    :param boxes2: (8, 13, 13, 3, 4)   label_xywh
    :return:
    '''

    # 变成左上角坐标、右下角坐标
    boxes1_x0y0x1y1 = P.concat([boxes1[:, :, :, :, :2] - boxes1[:, :, :, :, 2:] * 0.5,
                                boxes1[:, :, :, :, :2] + boxes1[:, :, :, :, 2:] * 0.5], axis=-1)
    boxes2_x0y0x1y1 = P.concat([boxes2[:, :, :, :, :2] - boxes2[:, :, :, :, 2:] * 0.5,
                                boxes2[:, :, :, :, :2] + boxes2[:, :, :, :, 2:] * 0.5], axis=-1)
    '''
    逐个位置比较boxes1_x0y0x1y1[..., :2]和boxes1_x0y0x1y1[..., 2:]，即逐个位置比较[x0, y0]和[x1, y1]，小的留下。
    比如留下了[x0, y0]
    这一步是为了避免一开始w h 是负数，导致x0y0成了右下角坐标，x1y1成了左上角坐标。
    '''
    boxes1_x0y0x1y1 = P.concat([P.elementwise_min(boxes1_x0y0x1y1[:, :, :, :, :2], boxes1_x0y0x1y1[:, :, :, :, 2:]),
                                P.elementwise_max(boxes1_x0y0x1y1[:, :, :, :, :2], boxes1_x0y0x1y1[:, :, :, :, 2:])],
                               axis=-1)
    boxes2_x0y0x1y1 = P.concat([P.elementwise_min(boxes2_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, 2:]),
                                P.elementwise_max(boxes2_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, 2:])],
                               axis=-1)

    # 两个矩形的面积
    boxes1_area = (boxes1_x0y0x1y1[:, :, :, :, 2] - boxes1_x0y0x1y1[:, :, :, :, 0]) * (
                boxes1_x0y0x1y1[:, :, :, :, 3] - boxes1_x0y0x1y1[:, :, :, :, 1])
    boxes2_area = (boxes2_x0y0x1y1[:, :, :, :, 2] - boxes2_x0y0x1y1[:, :, :, :, 0]) * (
                boxes2_x0y0x1y1[:, :, :, :, 3] - boxes2_x0y0x1y1[:, :, :, :, 1])

    # 相交矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    left_up = P.elementwise_max(boxes1_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, :2])
    right_down = P.elementwise_min(boxes1_x0y0x1y1[:, :, :, :, 2:], boxes2_x0y0x1y1[:, :, :, :, 2:])

    # 相交矩形的面积inter_area。iou
    inter_section = P.relu(right_down - left_up)
    inter_area = inter_section[:, :, :, :, 0] * inter_section[:, :, :, :, 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / (union_area + 1e-9)

    # 包围矩形的左上角坐标、右下角坐标，shape 都是 (8, 13, 13, 3, 2)
    enclose_left_up = P.elementwise_min(boxes1_x0y0x1y1[:, :, :, :, :2], boxes2_x0y0x1y1[:, :, :, :, :2])
    enclose_right_down = P.elementwise_max(boxes1_x0y0x1y1[:, :, :, :, 2:], boxes2_x0y0x1y1[:, :, :, :, 2:])

    # 包围矩形的对角线的平方
    enclose_wh = enclose_right_down - enclose_left_up
    enclose_c2 = P.pow(enclose_wh[:, :, :, :, 0], 2) + P.pow(enclose_wh[:, :, :, :, 1], 2)

    # 两矩形中心点距离的平方
    p2 = P.pow(boxes1[:, :, :, :, 0] - boxes2[:, :, :, :, 0], 2) + P.pow(boxes1[:, :, :, :, 1] - boxes2[:, :, :, :, 1],
                                                                         2)

    # 增加av。
    atan1 = P.atan(boxes1[:, :, :, :, 2] / (boxes1[:, :, :, :, 3] + 1e-9))
    atan2 = P.atan(boxes2[:, :, :, :, 2] / (boxes2[:, :, :, :, 3] + 1e-9))
    v = 4.0 * P.pow(atan1 - atan2, 2) / (math.pi ** 2)
    a = v / (1 - iou + v)

    ciou = iou - 1.0 * p2 / enclose_c2 - 1.0 * a * v
    return ciou


def bbox_iou(boxes1, boxes2):
    '''
    预测框          boxes1 (?, grid_h, grid_w, 3,   1, 4)，神经网络的输出(tx, ty, tw, th)经过了后处理求得的(bx, by, bw, bh)
    图片中所有的gt  boxes2 (?,      1,      1, 1, 70, 4)
    '''
    boxes1_area = boxes1[:, :, :, :, :, 2] * boxes1[:, :, :, :, :, 3]  # 所有格子的3个预测框的面积
    boxes2_area = boxes2[:, :, :, :, :, 2] * boxes2[:, :, :, :, :, 3]  # 所有ground truth的面积

    # (x, y, w, h)变成(x0, y0, x1, y1)
    boxes1 = P.concat([boxes1[:, :, :, :, :, :2] - boxes1[:, :, :, :, :, 2:] * 0.5,
                       boxes1[:, :, :, :, :, :2] + boxes1[:, :, :, :, :, 2:] * 0.5], axis=-1)
    boxes2 = P.concat([boxes2[:, :, :, :, :, :2] - boxes2[:, :, :, :, :, 2:] * 0.5,
                       boxes2[:, :, :, :, :, :2] + boxes2[:, :, :, :, :, 2:] * 0.5], axis=-1)

    # 所有格子的3个预测框 分别 和  70个ground truth  计算iou。 所以left_up和right_down的shape = (?, grid_h, grid_w, 3, 70, 2)
    expand_boxes1 = P.expand(boxes1, [1, 1, 1, 1, P.shape(boxes2)[4], 1])  # 不同于pytorch和tf，boxes1和boxes2都要扩展为相同shape
    expand_boxes2 = P.expand(boxes2, [1, P.shape(boxes1)[1], P.shape(boxes1)[2], P.shape(boxes1)[3], 1,
                                      1])  # 不同于pytorch和tf，boxes1和boxes2都要扩展为相同shape
    left_up = P.elementwise_max(expand_boxes1[:, :, :, :, :, :2], expand_boxes2[:, :, :, :, :, :2])  # 相交矩形的左上角坐标
    right_down = P.elementwise_min(expand_boxes1[:, :, :, :, :, 2:], expand_boxes2[:, :, :, :, :, 2:])  # 相交矩形的右下角坐标

    inter_section = P.relu(right_down - left_up)  # 相交矩形的w和h，是负数时取0  (?, grid_h, grid_w, 3, 70, 2)
    inter_area = inter_section[:, :, :, :, :, 0] * inter_section[:, :, :, :, :, 1]  # 相交矩形的面积   (?, grid_h, grid_w, 3, 70)
    expand_boxes1_area = P.expand(boxes1_area, [1, 1, 1, 1, P.shape(boxes2)[4]])
    expand_boxes2_area = P.expand(boxes2_area, [1, P.shape(expand_boxes1_area)[1], P.shape(expand_boxes1_area)[2],
                                                P.shape(expand_boxes1_area)[3], 1])
    union_area = expand_boxes1_area + expand_boxes2_area - inter_area  # union_area                (?, grid_h, grid_w, 3, 70)
    iou = 1.0 * inter_area / union_area  # iou                       (?, grid_h, grid_w, 3, 70)

    return iou


def loss_layer(conv, pred, label, bboxes, stride, num_class, iou_loss_thresh):
    conv_shape = P.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = stride * output_size

    pred_xywh = pred[:, :, :, :, 0:4]
    pred_conf = pred[:, :, :, :, 4:5]
    pred_prob = pred[:, :, :, :, 5:]

    label_xywh = label[:, :, :, :, 0:4]
    respond_bbox = label[:, :, :, :, 4:5]
    label_prob = label[:, :, :, :, 5:]

    ciou = P.reshape(bbox_ciou(pred_xywh, label_xywh),
                     (batch_size, output_size, output_size, 3, 1))  # (8, 13, 13, 3, 1)
    input_size = P.cast(input_size, dtype='float32')

    # 每个预测框xxxiou_loss的权重 = 2 - (ground truth的面积/图片面积)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    ciou_loss = respond_bbox * bbox_loss_scale * (1 - ciou)  # 1. respond_bbox作为mask，有物体才计算xxxiou_loss

    # 2. respond_bbox作为mask，有物体才计算类别loss
    prob_pos_loss = label_prob * (0 - P.log(pred_prob + 1e-9))  # 二值交叉熵，tf中也是加了极小的常数防止nan
    prob_neg_loss = (1 - label_prob) * (0 - P.log(1 - pred_prob + 1e-9))  # 二值交叉熵，tf中也是加了极小的常数防止nan
    prob_mask = P.expand(respond_bbox, [1, 1, 1, 1, num_class])
    prob_loss = prob_mask * (prob_pos_loss + prob_neg_loss)

    # 3. xxxiou_loss和类别loss比较简单。重要的是conf_loss，是一个二值交叉熵损失
    # 分两步：第一步是确定 grid_h * grid_w * 3 个预测框 哪些作为反例；第二步是计算二值交叉熵损失。
    expand_pred_xywh = P.reshape(pred_xywh, (batch_size, output_size, output_size, 3, 1, 4))  # 扩展为(?, grid_h, grid_w, 3,   1, 4)
    expand_bboxes = P.reshape(bboxes, (batch_size, 1, 1, 1, P.shape(bboxes)[1], 4))  # 扩展为(?,      1,      1, 1, 70, 4)
    iou = bbox_iou(expand_pred_xywh, expand_bboxes)  # 所有格子的3个预测框 分别 和  70个ground truth  计算iou。   (?, grid_h, grid_w, 3, 70)
    max_iou, max_iou_indices = P.topk(iou, k=1)  # 与70个ground truth的iou中，保留最大那个iou。  (?, grid_h, grid_w, 3, 1)

    # respond_bgd代表  这个分支输出的 grid_h * grid_w * 3 个预测框是否是 反例（背景）
    # label有物体，respond_bgd是0。 没物体的话：如果和某个gt(共70个)的iou超过iou_loss_thresh，respond_bgd是0；如果和所有gt(最多70个)的iou都小于iou_loss_thresh，respond_bgd是1。
    # respond_bgd是0代表有物体，不是反例（或者是忽略框）；  权重respond_bgd是1代表没有物体，是反例。
    # 有趣的是，模型训练时由于不断更新，对于同一张图片，两次预测的 grid_h * grid_w * 3 个预测框（对于这个分支输出）  是不同的。用的是这些预测框来与gt计算iou来确定哪些预测框是反例。
    # 而不是用固定大小（不固定位置）的先验框。
    respond_bgd = (1.0 - respond_bbox) * P.cast(max_iou < iou_loss_thresh, 'float32')

    # 二值交叉熵损失
    pos_loss = respond_bbox * (0 - P.log(pred_conf + 1e-9))
    neg_loss = respond_bgd  * (0 - P.log(1 - pred_conf + 1e-9))

    conf_loss = pos_loss + neg_loss
    # 回顾respond_bgd，某个预测框和某个gt的iou超过iou_loss_thresh，不被当作是反例。在参与“预测的置信位 和 真实置信位 的 二值交叉熵”时，这个框也可能不是正例(label里没标这个框是1的话)。这个框有可能不参与置信度loss的计算。
    # 这种框一般是gt框附近的框，或者是gt框所在格子的另外两个框。它既不是正例也不是反例不参与置信度loss的计算。（论文里称之为ignore）

    ciou_loss = P.reduce_sum(ciou_loss) / batch_size
    conf_loss = P.reduce_sum(conf_loss) / batch_size
    prob_loss = P.reduce_sum(prob_loss) / batch_size

    return ciou_loss, conf_loss, prob_loss


def decode(conv_output, anchors, stride, num_class):
    conv_shape = P.shape(conv_output)
    batch_size = conv_shape[0]
    n_grid = conv_shape[1]
    anchor_per_scale = len(anchors)

    conv_output = P.reshape(conv_output, (batch_size, n_grid, n_grid, anchor_per_scale, 5 + num_class))

    conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
    conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
    conv_raw_conf = conv_output[:, :, :, :, 4:5]
    conv_raw_prob = conv_output[:, :, :, :, 5:]

    rows = P.range(0, n_grid, 1, 'float32')
    cols = P.range(0, n_grid, 1, 'float32')
    rows = P.expand(P.reshape(rows, (1, -1, 1)), [n_grid, 1, 1])
    cols = P.expand(P.reshape(cols, (-1, 1, 1)), [1, n_grid, 1])
    offset = P.concat([rows, cols], axis=-1)
    offset = P.reshape(offset, (1, n_grid, n_grid, 1, 2))
    offset = P.expand(offset, [batch_size, 1, 1, anchor_per_scale, 1])

    pred_xy = (P.sigmoid(conv_raw_dxdy) + offset) * stride
    anchor_t = fluid.layers.assign(np.copy(anchors).astype(np.float32))
    pred_wh = (P.exp(conv_raw_dwdh) * anchor_t)
    pred_xywh = P.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = P.sigmoid(conv_raw_conf)
    pred_prob = P.sigmoid(conv_raw_prob)

    return P.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def yolo_loss(args, num_classes, iou_loss_thresh, anchors):
    conv_lbbox = args[0]  # (?, ?, ?, 3*(num_classes+5))
    conv_mbbox = args[1]  # (?, ?, ?, 3*(num_classes+5))
    conv_sbbox = args[2]  # (?, ?, ?, 3*(num_classes+5))
    label_sbbox = args[3]  # (?, ?, ?, 3, num_classes+5)
    label_mbbox = args[4]  # (?, ?, ?, 3, num_classes+5)
    label_lbbox = args[5]  # (?, ?, ?, 3, num_classes+5)
    true_bboxes = args[6]  # (?, 70, 4)
    pred_sbbox = decode(conv_sbbox, anchors[0], 8, num_classes)
    pred_mbbox = decode(conv_mbbox, anchors[1], 16, num_classes)
    pred_lbbox = decode(conv_lbbox, anchors[2], 32, num_classes)
    sbbox_ciou_loss, sbbox_conf_loss, sbbox_prob_loss = loss_layer(conv_sbbox, pred_sbbox, label_sbbox, true_bboxes, 8,
                                                                   num_classes, iou_loss_thresh)
    mbbox_ciou_loss, mbbox_conf_loss, mbbox_prob_loss = loss_layer(conv_mbbox, pred_mbbox, label_mbbox, true_bboxes, 16,
                                                                   num_classes, iou_loss_thresh)
    lbbox_ciou_loss, lbbox_conf_loss, lbbox_prob_loss = loss_layer(conv_lbbox, pred_lbbox, label_lbbox, true_bboxes, 32,
                                                                   num_classes, iou_loss_thresh)

    ciou_loss = sbbox_ciou_loss + mbbox_ciou_loss + lbbox_ciou_loss
    conf_loss = sbbox_conf_loss + mbbox_conf_loss + lbbox_conf_loss
    prob_loss = sbbox_prob_loss + mbbox_prob_loss + lbbox_prob_loss
    return [ciou_loss, conf_loss, prob_loss]


def multi_thread_op(i, samples, decodeImage, context, train_dataset, with_mixup, mixupImage,
                    photometricDistort, randomFlipImage, normalizeBox, padBox, bboxXYXY2XYWH):
    samples[i] = decodeImage(samples[i], context, train_dataset)
    if with_mixup:
        samples[i] = mixupImage(samples[i], context)
    samples[i] = photometricDistort(samples[i], context)
    # samples[i] = randomCrop(samples[i], context)
    samples[i] = randomFlipImage(samples[i], context)
    samples[i] = normalizeBox(samples[i], context)
    samples[i] = padBox(samples[i], context)
    samples[i] = bboxXYXY2XYWH(samples[i], context)

def clear_model(save_dir):
    path_dir = os.listdir(save_dir)
    it_ids = []
    for name in path_dir:
        sss = name.split('.')
        if sss[0] == '':
            continue
        if sss[0] == 'best_model':   # 不会删除最优模型
            it_id = 9999999999
        else:
            it_id = int(sss[0])
        it_ids.append(it_id)
    if len(it_ids) >= 11 * 3:
        it_id = min(it_ids)
        pdopt_path = '%s/%d.pdopt' % (save_dir, it_id)
        pdmodel_path = '%s/%d.pdmodel' % (save_dir, it_id)
        pdparams_path = '%s/%d.pdparams' % (save_dir, it_id)
        if os.path.exists(pdopt_path):
            os.remove(pdopt_path)
        if os.path.exists(pdmodel_path):
            os.remove(pdmodel_path)
        if os.path.exists(pdparams_path):
            os.remove(pdparams_path)

if __name__ == '__main__':
    use_gpu = False
    use_gpu = True

    # 选择配置
    cfg = YOLOv4_Config_1()
    # cfg = YOLOv3_Config_1()


    algorithm = cfg.algorithm

    class_names = get_classes(cfg.classes_path)
    num_classes = len(class_names)
    _anchors = copy.deepcopy(cfg.anchors)
    num_anchors = len(cfg.anchor_masks[0])  # 每个输出层有几个先验框
    _anchors = np.array(_anchors)
    _anchors = np.reshape(_anchors, (-1, num_anchors, 2))
    _anchors = _anchors.astype(np.float32)

    # 步id，无需设置，会自动读。
    iter_id = 0

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            # 多尺度训练
            inputs = P.data(name='input_1', shape=[-1,3,-1,-1], append_batch_size=True, dtype='float32')
            if algorithm == 'YOLOv4':
                output_l, output_m, output_s = YOLOv4(inputs, num_classes, num_anchors, is_test=False, trainable=True)
            elif algorithm == 'YOLOv3':
                backbone = Resnet50Vd()
                head = YOLOv3Head()
                yolov3 = YOLOv3(backbone, head)
                output_l, output_m, output_s = yolov3(inputs)

            # 建立损失函数
            label_sbbox = P.data(name='input_2', shape=[-1, -1, -1, 3, (num_classes + 5)], append_batch_size=False, dtype='float32')
            label_mbbox = P.data(name='input_3', shape=[-1, -1, -1, 3, (num_classes + 5)], append_batch_size=False, dtype='float32')
            label_lbbox = P.data(name='input_4', shape=[-1, -1, -1, 3, (num_classes + 5)], append_batch_size=False, dtype='float32')
            true_bboxes = P.data(name='input_5', shape=[cfg.num_max_boxes, 4], dtype='float32')
            args = [output_l, output_m, output_s, label_sbbox, label_mbbox, label_lbbox, true_bboxes]
            ciou_loss, conf_loss, prob_loss = yolo_loss(args, num_classes, cfg.iou_loss_thresh, _anchors)
            loss = ciou_loss + conf_loss + prob_loss
            
            base_lr = cfg.lr
            lr = fluid.layers.cosine_decay( learning_rate = base_lr, step_each_epoch=10000, epochs=650000)
            optimizer = fluid.optimizer.Adam(learning_rate=lr)
            optimizer.minimize(loss)


    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            # 多尺度训练
            inputs = P.data(name='input_1', shape=[-1,3,-1,-1], append_batch_size=False, dtype='float32')
            if algorithm == 'YOLOv4':
                output_l, output_m, output_s = YOLOv4(inputs, num_classes, num_anchors, is_test=False, trainable=True)
            elif algorithm == 'YOLOv3':
                backbone = Resnet50Vd()
                head = YOLOv3Head(keep_prob=1.0)   # 一定要设置keep_prob=1.0, 为了得到一致的推理结果
                yolov3 = YOLOv3(backbone, head)
                output_l, output_m, output_s = yolov3(inputs)
            eval_fetch_list = [output_l, output_m, output_s]
    eval_prog = eval_prog.clone(for_test=True)

    # 参数随机初始化
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    compiled_eval_prog = fluid.compiler.CompiledProgram(eval_prog)
    _decode = Decode(algorithm, cfg.anchors, cfg.conf_thresh, cfg.nms_thresh, cfg.input_shape, exe, compiled_eval_prog, class_names)

    if cfg.pattern == 1:
        fluid.load(train_prog, cfg.model_path, executor=exe)
        strs = cfg.model_path.split('weights/')
        if len(strs) == 2:
            iter_id = int(strs[1])

    # 种类id
    _catid2clsid = copy.deepcopy(catid2clsid)
    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:  # 如果不是COCO数据集，而是自定义数据集
        _catid2clsid = {}
        _clsid2catid = {}
        for k in range(num_classes):
            _catid2clsid[k] = k
            _clsid2catid[k] = k
    # 训练集
    train_dataset = COCO(cfg.train_path)
    train_img_ids = train_dataset.getImgIds()
    train_records = data_clean(train_dataset, train_img_ids, _catid2clsid, cfg.train_pre_path)
    num_train = len(train_records)
    train_indexes = [i for i in range(num_train)]
    # 验证集
    with open(cfg.val_path, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            dataset = json.loads(line)
            val_images = dataset['images']

    batch_size = cfg.batch_size
    print(batch_size)
    with_mixup = cfg.with_mixup
    context = cfg.context
    # 预处理
    # sample_transforms
    decodeImage = DecodeImage(with_mixup=with_mixup)   # 对图片解码。最开始的一步。
    mixupImage = MixupImage()                   # mixup增强
    photometricDistort = PhotometricDistort()   # 颜色扭曲
    randomCrop = RandomCrop()                   # 随机裁剪
    randomFlipImage = RandomFlipImage()         # 随机翻转
    normalizeBox = NormalizeBox()               # 将物体的左上角坐标、右下角坐标中的横坐标/图片宽、纵坐标/图片高 以归一化坐标。
    padBox = PadBox(cfg.num_max_boxes)          # 如果gt_bboxes的数量少于num_max_boxes，那么填充坐标是0的bboxes以凑够num_max_boxes。
    bboxXYXY2XYWH = BboxXYXY2XYWH()             # sample['gt_bbox']被改写为cx_cy_w_h格式。
    # batch_transforms
    randomShape = RandomShape()                 # 多尺度训练。随机选一个尺度。也随机选一种插值方式。
    normalizeImage = NormalizeImage(algorithm, is_scale=True, is_channel_first=False)   # 图片归一化。直接除以255。
    gt2YoloTarget = Gt2YoloTarget(cfg.anchors,
                                  cfg.anchor_masks,
                                  cfg.downsample_ratios,
                                  num_classes)  # 填写target0、target1、target2张量。

    # 保存模型的目录
    if not os.path.exists('./weights'): os.mkdir('./weights')

    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()

    # 一轮的步数。丢弃最后几个样本。
    train_steps = num_train // batch_size
    best_ap_list = [0.0, 0]  # [map, iter]
    while True:  # 无限个epoch
        # 每个epoch之前洗乱
        np.random.shuffle(train_indexes)
        for step in range(train_steps):
            iter_id += 1

            # 估计剩余时间
            start_time = end_time
            end_time = time.time()
            time_stat.append(end_time - start_time)
            time_cost = np.mean(time_stat)
            eta_sec = (cfg.max_iters - iter_id) * time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))

            # ==================== train ====================
            samples = get_samples(train_records, train_indexes, step, batch_size, with_mixup)
            # sample_transforms用多线程
            threads = []
            for i in range(batch_size):
                t = threading.Thread(target=multi_thread_op,
                                     args=(i, samples, decodeImage, context, train_dataset, with_mixup, mixupImage,
                                           photometricDistort, randomFlipImage, normalizeBox, padBox,
                                           bboxXYXY2XYWH))
                threads.append(t)
                t.start()
            # 等待所有线程任务结束。
            for t in threads:
                t.join()

            # batch_transforms
            samples = randomShape(samples, context)
            samples = normalizeImage(samples, context)
            batch_image, batch_label, batch_gt_bbox = gt2YoloTarget(samples, context)

            # 一些变换
            batch_image = batch_image.transpose(0, 3, 1, 2)
            batch_image = batch_image.astype(np.float32)

            
            batch_label[2] = batch_label[2].astype(np.float32)
            batch_label[1] = batch_label[1].astype(np.float32)
            batch_label[0] = batch_label[0].astype(np.float32)

            batch_gt_bbox = batch_gt_bbox.astype(np.float32)

            losses = exe.run(train_prog, feed={"input_1": batch_image, "input_2": batch_label[2],
                                                  "input_3": batch_label[1], "input_4": batch_label[0],
                                                  "input_5": batch_gt_bbox, },
                             fetch_list=[loss, ciou_loss, conf_loss, prob_loss])

            # ==================== log ====================
            if iter_id % 20 == 0:
                strs = 'Train iter: {}, all_loss: {:.6f}, ciou_loss: {:.6f}, conf_loss: {:.6f}, prob_loss: {:.6f}, eta: {}'.format(
                    iter_id, losses[0][0], losses[1][0], losses[2][0], losses[3][0], eta)
                logger.info(strs)

            # ==================== save ====================
            if iter_id % cfg.save_iter == 0:
                save_path = './weights/%d' % iter_id
                fluid.save(train_prog, save_path)
                logger.info('Save model to {}'.format(save_path))
                clear_model('weights')

            # ==================== eval ====================
            if iter_id % cfg.eval_iter == 0:
                box_ap = eval(_decode, eval_fetch_list, val_images, cfg.val_pre_path, cfg.val_path, cfg.eval_batch_size,
                              _clsid2catid, cfg.draw_image)
                logger.info("box ap: %.3f" % (box_ap[0],))

                # 以box_ap作为标准
                ap = box_ap
                if ap[0] > best_ap_list[0]:
                    best_ap_list[0] = ap[0]
                    best_ap_list[1] = iter_id
                    save_path = './weights/best_model'
                    fluid.save(train_prog, save_path)
                    logger.info('Save model to {}'.format(save_path))
                    clear_model('weights')
                logger.info("Best test ap: {}, in iter: {}".format(
                    best_ap_list[0], best_ap_list[1]))

            # ==================== exit ====================
            if iter_id == cfg.max_iters:
                logger.info('Done.')
                exit(0)

