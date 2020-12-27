#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : zhoukunyang
#   Created date: 2020-12-08

#
# ================================================================
from collections import deque
import datetime
import cv2
import os
import copy
import json
import time
import numpy as np
from config import *
from model.head import YOLOv3Head
from model.resnet import Resnet50Vd
from model.yolov3 import YOLOv3
from tools.cocotools import eval
import paddle.fluid as fluid
import paddle.fluid.layers as P
from tools.cocotools import get_classes, clsid2catid
from model.yolov4 import YOLOv4
from model.decode_np import Decode

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)



# 6G的卡，训练时如果要预测，则设置use_gpu = False，否则显存不足。
use_gpu = False
use_gpu = True


if __name__ == '__main__':
    # 选择配置
    cfg = YOLOv4_Config_1()
    # cfg = YOLOv3_Config_1()


    algorithm = cfg.algorithm
    classes_path = cfg.classes_path

    # 读取的模型
    model_path = cfg.infer_model_path

    # input_shape越大，精度会上升，但速度会下降。
    input_shape = cfg.input_shape

    # 推理时的分数阈值和nms_iou阈值
    conf_thresh = cfg.conf_thresh
    nms_thresh = cfg.nms_thresh

    # 是否给图片画框。
    draw_image = cfg.draw_image

    # 验证时的批大小
    eval_batch_size = cfg.eval_batch_size

    # 验证集图片的相对路径
    # eval_pre_path = '../COCO/val2017/'
    # anno_file = '../COCO/annotations/instances_val2017.json'
    eval_pre_path = cfg.val_pre_path
    anno_file = cfg.val_path
    with open(anno_file, 'r', encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            dataset = json.loads(line)
            images = dataset['images']

    anchors = cfg.anchors
    num_anchors = len(cfg.anchor_masks[0])
    all_classes = get_classes(classes_path)
    num_classes = len(all_classes)


    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            # 多尺度训练
            inputs = P.data(name='input_1', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
            if algorithm == 'YOLOv4':
                output_l, output_m, output_s = YOLOv4(inputs, num_classes, num_anchors, is_test=False, trainable=True)
            elif algorithm == 'YOLOv3':
                backbone = Resnet50Vd()
                head = YOLOv3Head(keep_prob=1.0)   # 一定要设置keep_prob=1.0, 为了得到一致的推理结果
                yolov3 = YOLOv3(backbone, head)
                output_l, output_m, output_s = yolov3(inputs)
            eval_fetch_list = [output_l, output_m, output_s]
    eval_prog = eval_prog.clone(for_test=True)
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    fluid.load(eval_prog, model_path, executor=exe)
    _decode = Decode(algorithm, anchors, conf_thresh, nms_thresh, input_shape, exe, eval_prog, all_classes)


    _clsid2catid = copy.deepcopy(clsid2catid)
    if num_classes != 80:   # 如果不是COCO数据集，而是自定义数据集
        _clsid2catid = {}
        for k in range(num_classes):
            _clsid2catid[k] = k
    box_ap = eval(_decode, eval_fetch_list, images, eval_pre_path, anno_file, eval_batch_size, _clsid2catid, draw_image)

