#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : zhoukunyang
#   Created date: 2020-12-08
#   Description : 复制权重
#
# ================================================================
import os
import torch
from model.yolov4 import YOLOv4
import paddle.fluid as fluid


def load_weights(path):
    """ Loads weights from a compressed save file. """
    # state_dict = torch.load(path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    return state_dict

state_dict = load_weights('yolov4.pt')
print('============================================================')


def copy1(idx, place):
    keyword1 = 'conv%d.weight' % idx
    keyword2 = 'bn%d.weight' % idx
    keyword3 = 'bn%d.bias' % idx
    keyword4 = 'bn%d.running_mean' % idx
    keyword5 = 'bn%d.running_var' % idx
    for key in state_dict:
        value = state_dict[key].numpy()
        if keyword1 in key:
            w = value
        elif keyword2 in key:
            y = value
        elif keyword3 in key:
            b = value
        elif keyword4 in key:
            m = value
        elif keyword5 in key:
            v = value
    tensor = fluid.global_scope().find_var('conv%.3d.conv.weights' % idx).get_tensor()
    tensor2 = fluid.global_scope().find_var('conv%.3d.bn.scale' % idx).get_tensor()
    tensor3 = fluid.global_scope().find_var('conv%.3d.bn.offset' % idx).get_tensor()
    tensor4 = fluid.global_scope().find_var('conv%.3d.bn.mean' % idx).get_tensor()
    tensor5 = fluid.global_scope().find_var('conv%.3d.bn.var' % idx).get_tensor()
    tensor.set(w, place)
    tensor2.set(y, place)
    tensor3.set(b, place)
    tensor4.set(m, place)
    tensor5.set(v, place)

def copy2(idx, place):
    keyword1 = 'conv%d.weight' % idx
    keyword2 = 'conv%d.bias' % idx
    for key in state_dict:
        value = state_dict[key].numpy()
        if keyword1 in key:
            w = value
        elif keyword2 in key:
            b = value
    tensor = fluid.global_scope().find_var('conv%.3d.conv.weights' % idx).get_tensor()
    tensor2 = fluid.global_scope().find_var('conv%.3d.conv.bias' % idx).get_tensor()
    tensor.set(w, place)
    tensor2.set(b, place)


num_classes = 80
num_anchors = 3
use_gpu = False

inputs = fluid.layers.data(name='input_1', shape=[-1, 3, -1, -1], append_batch_size=False, dtype='float32')
pred_outs = YOLOv4(inputs, num_classes, num_anchors)

# Create an executor using CPU as an example
gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
place = fluid.CUDAPlace(gpu_id) if use_gpu else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


print('\nCopying...')
for i in range(1, 94, 1):
    copy1(i, place)
for i in range(95, 102, 1):
    copy1(i, place)
for i in range(103, 110, 1):
    copy1(i, place)

copy2(94, place)
copy2(102, place)
copy2(110, place)

fluid.io.save_persistables(exe, 'yolov4', fluid.default_startup_program())
print('\nDone.')


