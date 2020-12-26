#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-10 10:20:27
#   Description : paddlepaddle_yolov4
#
# ================================================================
from collections import deque
import datetime
import cv2
import os
import time
import numpy as np
from collections import OrderedDict
import paddle.fluid as fluid
import paddle.fluid.layers as P
from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from tools.visualize import get_colors, draw

import logging


FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def process_image(img, input_shape):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale_x = float(input_shape[1]) / w
    scale_y = float(input_shape[0]) / h
    img = cv2.resize(img, None, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC)
    pimage = img.astype(np.float32) / 255.
    pimage = np.expand_dims(pimage, axis=0)
    return pimage

use_gpu = False
use_gpu = True

if __name__ == '__main__':
    # classes_path = 'data/voc_classes.txt'
    classes_path = 'data/coco_classes.txt'
    # model_path可以是'yolov4'、'./weights/1000'这些。
    model_path = 'yolov4'
    model_path = './weights/66000'

    # input_shape越大，精度会上升，但速度会下降。
    # input_shape = (320, 320)
    input_shape = (416, 416)
    # input_shape = (608, 608)

    # 验证时的分数阈值和nms_iou阈值
    conf_thresh = 0.05
    nms_thresh = 0.45
    keep_top_k = 100
    nms_top_k = 100

    # 是否给图片画框。不画可以提速。读图片、后处理还可以继续优化。
    draw_image = True
    # draw_image = False

    # 是否用fluid.layers.yolo_box()来对预测框解码。
    use_yolo_box = True

    # 初始卷积核个数
    initial_filters = 32
    anchors = np.array([
        [[12, 16], [19, 36], [40, 28]],
        [[36, 75], [76, 55], [72, 146]],
        [[142, 110], [192, 243], [459, 401]]
    ])
    # 一些预处理
    anchors = anchors.astype(np.float32)
    num_anchors = len(anchors[0])  # 每个输出层有几个先验框

    all_classes = get_classes(classes_path)
    num_classes = len(all_classes)


    startup_prog = fluid.Program()
    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs = P.data(name='image', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')

            resize_shape = P.data(name='resize_shape', shape=[-1, 2], append_batch_size=False, dtype='int32')
            origin_shape = P.data(name='origin_shape', shape=[-1, 2], append_batch_size=False, dtype='int32')
            param = {}
            param['resize_shape'] = resize_shape
            param['origin_shape'] = origin_shape
            param['anchors'] = anchors
            param['conf_thresh'] = conf_thresh
            param['nms_thresh'] = nms_thresh
            param['keep_top_k'] = keep_top_k
            param['nms_top_k'] = nms_top_k
            param['use_yolo_box'] = use_yolo_box

            boxes, scores, classes = YOLOv4(inputs, num_classes, num_anchors, is_test=False, trainable=True, postprocess='fastnms', param=param)

            eval_fetch_list = [boxes, scores, classes]
    eval_prog = eval_prog.clone(for_test=True)
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)

    fluid.load(eval_prog, model_path, executor=exe)

    if not os.path.exists('images/res/'): os.mkdir('images/res/')

    # 获取颜色
    colors = get_colors(num_classes)


    path_dir = os.listdir('images/test')
    # warm up
    if use_gpu:
        for k, filename in enumerate(path_dir):
            image = cv2.imread('images/test/' + filename)
            ori_h, ori_w, _ = image.shape
            pimage = process_image(np.copy(image), input_shape)
            pimage = pimage.transpose(0, 3, 1, 2)
            _, _, ih, iw = pimage.shape
            _resize_shape = np.array([[iw, ih]], np.float32)
            _origin_shape = np.array([[ori_w, ori_h]], np.float32)

            outs = exe.run(eval_prog, feed={"image": pimage, "resize_shape": _resize_shape, "origin_shape": _origin_shape, }, fetch_list=eval_fetch_list)
            if k == 10:
                break


    time_stat = deque(maxlen=20)
    start_time = time.time()
    end_time = time.time()
    num_imgs = len(path_dir)
    start = time.time()
    for k, filename in enumerate(path_dir):
        image = cv2.imread('images/test/' + filename)
        ori_h, ori_w, _ = image.shape
        pimage = process_image(np.copy(image), input_shape)
        pimage = pimage.transpose(0, 3, 1, 2)
        _, _, ih, iw = pimage.shape
        _resize_shape = np.array([[iw, ih]], np.float32)
        _origin_shape = np.array([[ori_w, ori_h]], np.float32)

        outs = exe.run(eval_prog, feed={"image": pimage, "resize_shape": _resize_shape, "origin_shape": _origin_shape, }, fetch_list=eval_fetch_list)
        boxes, scores, classes = outs[0][0], outs[1][0], outs[2][0]

        # 后处理那里，一定不会返回空。若没有物体，scores[0]会是负数，由此来判断有没有物体。
        if scores[0] > 0:
            if draw_image:
                draw(image, boxes, scores, classes, all_classes, colors)

        # 估计剩余时间
        start_time = end_time
        end_time = time.time()
        time_stat.append(end_time - start_time)
        time_cost = np.mean(time_stat)
        eta_sec = (num_imgs - k) * time_cost
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        logger.info('Infer iter {}, num_imgs={}, eta={}.'.format(k, num_imgs, eta))
        if draw_image:
            cv2.imwrite('images/res/' + filename, image)
            logger.info("Detection bbox results save in images/res/{}".format(filename))
    cost = time.time() - start
    logger.info('total time: {0:.6f}s'.format(cost))
    logger.info('Speed: %.6fs per image,  %.1f FPS.'%((cost / num_imgs), (num_imgs / cost)))


