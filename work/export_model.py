#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-10 10:20:27
#   Description : paddlepaddle_yolov4
#
# ================================================================
import os
import tempfile
import copy
import shutil
from collections import OrderedDict
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as P

from model.head import YOLOv3Head
from model.resnet import Resnet50Vd
from model.yolov3 import YOLOv3
from tools.cocotools import get_classes
from model.yolov4 import YOLOv4
from config import *

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)




def _strip_postfix(path):
    path, ext = os.path.splitext(path)
    assert ext in ['', '.pdparams', '.pdopt', '.pdmodel'], \
            "Unknown postfix {} from weights".format(ext)
    return path

def _load_state(path):
    if os.path.exists(path + '.pdopt'):
        # XXX another hack to ignore the optimizer state
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, os.path.basename(os.path.normpath(path)))
        shutil.copy(path + '.pdparams', dst + '.pdparams')
        state = fluid.io.load_program_state(dst)
        shutil.rmtree(tmp)
    else:
        state = fluid.io.load_program_state(path)
    return state

def load_params(exe, prog, path, ignore_params=[]):
    """
    Load model from the given path.
    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): load weight to which Program object.
        path (string): URL string or loca model path.
        ignore_params (list): ignore variable to load when finetuning.
            It can be specified by finetune_exclude_pretrained_params
            and the usage can refer to docs/advanced_tutorials/TRANSFER_LEARNING.md
    """

    path = _strip_postfix(path)
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    logger.debug('Loading parameters from {}...'.format(path))
    state = _load_state(path)
    fluid.io.set_program_state(prog, state)

def prune_feed_vars(feeded_var_names, target_vars, prog):
    """
    Filter out feed variables which are not in program,
    pruned feed variables are only used in post processing
    on model output, which are not used in program, such
    as im_id to identify image order, im_shape to clip bbox
    in image.
    """
    exist_var_names = []
    prog = prog.clone()
    prog = prog._prune(targets=target_vars)
    global_block = prog.global_block()
    for name in feeded_var_names:
        try:
            v = global_block.var(name)
            exist_var_names.append(str(v.name))
        except Exception:
            logger.info('save_inference_model pruned unused feed '
                        'variables {}'.format(name))
            pass
    return exist_var_names

def save_infer_model(save_dir, exe, feed_vars, test_fetches, infer_prog):
    feed_var_names = [var.name for var in feed_vars.values()]
    fetch_list = sorted(test_fetches.items(), key=lambda i: i[0])
    target_vars = [var[1] for var in fetch_list]
    feed_var_names = prune_feed_vars(feed_var_names, target_vars, infer_prog)
    logger.info("Export inference model to {}, input: {}, output: "
                "{}...".format(save_dir, feed_var_names,
                               [str(var.name) for var in target_vars]))
    fluid.io.save_inference_model(
        save_dir,
        feeded_var_names=feed_var_names,
        target_vars=target_vars,
        executor=exe,
        main_program=infer_prog,
        params_filename="__params__")


def dump_infer_config(save_dir, cfg):
    if os.path.exists('%s/infer_cfg.yml' % save_dir): os.remove('%s/infer_cfg.yml' % save_dir)
    content = ''
    with open('tools/template_cfg.yml', 'r', encoding='utf-8') as f:
        for line in f:
            for key in cfg:
                key2 = '${%s}' % key
                if key2 in line:
                    if key == 'class_names':
                        line = ''
                        for cname in cfg[key]:
                            line += '- %s\n' % cname
                    else:
                        line = line.replace(key2, str(cfg[key]))
                    break
            content += line
    with open('%s/infer_cfg.yml' % save_dir, 'w', encoding='utf-8') as f:
        f.write(content)
        f.close()


if __name__ == '__main__':
    # 推理模型保存目录
    save_dir = 'inference_model'

    # 导出时用fastnms还是不后处理
    # postprocess = 'fastnms'
    postprocess = 'multiclass_nms'
    # postprocess = 'numpy_nms'

    # need 3 for YOLO arch
    min_subgraph_size = 3

    # 是否使用Padddle Executor进行推理。
    use_python_inference = False

    # 使用GPU时，默认为fluid, 可选（fluid/trt_fp32/trt_fp16）
    mode = 'fluid'

    # 对模型输出的预测框再进行一次分数过滤的阈值。设置为0.0表示不再进行分数过滤。
    # 与conf_thresh不同，需要修改这个值的话直接编辑导出的inference_model/infer_cfg.yml配置文件，不需要重新导出模型。
    # 总之，inference_model/infer_cfg.yml里的配置可以手动修改，不需要重新导出模型。
    draw_threshold = 0.0

    # 选择配置
    cfg = YOLOv4_Config_1()
    # cfg = YOLOv3_Config_1()


    # =============== 以下不用设置 ===============
    algorithm = cfg.algorithm
    classes_path = cfg.classes_path

    # 读取的模型
    model_path = cfg.infer_model_path

    # input_shape越大，精度会上升，但速度会下降。
    input_shape = cfg.infer_input_shape

    # 推理时的分数阈值和nms_iou阈值。注意，这些值会写死进模型，如需修改请重新导出模型。
    conf_thresh = cfg.infer_conf_thresh
    nms_thresh = cfg.infer_nms_thresh
    keep_top_k = cfg.infer_keep_top_k
    nms_top_k = cfg.infer_nms_top_k


    # 初始卷积核个数
    initial_filters = 32
    # 先验框
    _anchors = copy.deepcopy(cfg.anchors)
    num_anchors = len(cfg.anchor_masks[0])  # 每个输出层有几个先验框
    _anchors = np.array(_anchors)
    _anchors = np.reshape(_anchors, (-1, num_anchors, 2))
    _anchors = _anchors.astype(np.float32)
    num_anchors = len(_anchors[0])  # 每个输出层有几个先验框

    all_classes = get_classes(classes_path)
    num_classes = len(all_classes)


    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs = P.data(name='image', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')

            if postprocess == 'fastnms' or postprocess == 'multiclass_nms':
                resize_shape = P.data(name='resize_shape', shape=[-1, 2], append_batch_size=False, dtype='int32')
                origin_shape = P.data(name='origin_shape', shape=[-1, 2], append_batch_size=False, dtype='int32')
                param = {}
                param['resize_shape'] = resize_shape
                param['origin_shape'] = origin_shape
                param['anchors'] = _anchors
                param['conf_thresh'] = conf_thresh
                param['nms_thresh'] = nms_thresh
                param['keep_top_k'] = keep_top_k
                param['nms_top_k'] = nms_top_k
                param['num_classes'] = num_classes
                param['num_anchors'] = num_anchors
                # 输入字典
                feed_vars = [('image', inputs), ('resize_shape', resize_shape), ('origin_shape', origin_shape)]
                feed_vars = OrderedDict(feed_vars)
            if postprocess == 'numpy_nms':
                param = None
                # 输入字典
                feed_vars = [('image', inputs), ]
                feed_vars = OrderedDict(feed_vars)

            if algorithm == 'YOLOv4':
                if postprocess == 'fastnms':
                    boxes, scores, classes = YOLOv4(inputs, num_classes, num_anchors, is_test=False, trainable=True, export=True, postprocess=postprocess, param=param)
                    test_fetches = {'boxes': boxes, 'scores': scores, 'classes': classes, }
                if postprocess == 'multiclass_nms':
                    pred = YOLOv4(inputs, num_classes, num_anchors, is_test=False, trainable=True, export=True, postprocess=postprocess, param=param)
                    test_fetches = {'pred': pred, }
            elif algorithm == 'YOLOv3':
                backbone = Resnet50Vd()
                head = YOLOv3Head(keep_prob=1.0)   # 一定要设置keep_prob=1.0, 为了得到一致的推理结果
                yolov3 = YOLOv3(backbone, head)
                if postprocess == 'fastnms':
                    boxes, scores, classes = yolov3(inputs, export=True, postprocess=postprocess, param=param)
                    test_fetches = {'boxes': boxes, 'scores': scores, 'classes': classes, }
                if postprocess == 'multiclass_nms':
                    pred = yolov3(inputs, export=True, postprocess=postprocess, param=param)
                    test_fetches = {'pred': pred, }
    infer_prog = infer_prog.clone(for_test=True)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(startup_prog)


    logger.info("postprocess: %s" % postprocess)
    load_params(exe, infer_prog, model_path)

    save_infer_model(save_dir, exe, feed_vars, test_fetches, infer_prog)

    # 导出配置文件
    cfg = {}
    input_shape_h = input_shape[0]
    input_shape_w = input_shape[1]
    cfg['arch'] = algorithm
    cfg['min_subgraph_size'] = min_subgraph_size
    cfg['use_python_inference'] = use_python_inference
    cfg['mode'] = mode
    cfg['postprocess'] = postprocess
    cfg['draw_threshold'] = draw_threshold
    cfg['input_shape_h'] = input_shape_h
    cfg['input_shape_w'] = input_shape_w
    cfg['class_names'] = all_classes
    if algorithm == 'YOLOv4':
        cfg['is_scale'] = True
        cfg['mean0'] = 0.0
        cfg['mean1'] = 0.0
        cfg['mean2'] = 0.0
        cfg['std0'] = 1.0
        cfg['std1'] = 1.0
        cfg['std2'] = 1.0
    elif algorithm == 'YOLOv3':
        cfg['is_scale'] = False
        cfg['mean0'] = 0.485
        cfg['mean1'] = 0.456
        cfg['mean2'] = 0.406
        cfg['std0'] = 0.229
        cfg['std1'] = 0.224
        cfg['std2'] = 0.225
    dump_infer_config(save_dir, cfg)
    logger.info("Done.")


