#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : zhoukunyang
#   Created date: 2020-12-08
# 
#
# ================================================================
from config import YOLOv4_Config_1, YOLOv3_Config_1
from model.head import YOLOv3Head
from model.resnet import Resnet50Vd
from model.yolov3 import YOLOv3
from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from model.decode_np import Decode
import json
import os
import paddle.fluid as fluid
import paddle.fluid.layers as P
from tools.cocotools import test_dev

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


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

    # test集图片的相对路径
    test_pre_path = '../data/data7122/test2017/'
    anno_file = '../data/data7122/annotations/image_info_test-dev2017.json'
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

    test_dev(_decode, eval_fetch_list, images, test_pre_path, eval_batch_size, draw_image)

