#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : zhoukunyang
#   Created date: 2020-12-08
#   
#
# ================================================================
import paddle.fluid as fluid

from model.yolov4 import yolo_decode


class YOLOv3(object):
    def __init__(self, backbone, head):
        super(YOLOv3, self).__init__()
        self.backbone = backbone
        self.head = head

    def __call__(self, x, export=False, postprocess=None, param=None):
        body_feats = self.backbone(x)
        output_l, output_m, output_s = self.head(body_feats)
        if export:
            return yolo_decode(output_l, output_m, output_s, postprocess, param)

        # 相当于numpy的transpose()，交换下标
        output_l = fluid.layers.transpose(output_l, perm=[0, 2, 3, 1], name='output_l')
        output_m = fluid.layers.transpose(output_m, perm=[0, 2, 3, 1], name='output_m')
        output_s = fluid.layers.transpose(output_s, perm=[0, 2, 3, 1], name='output_s')
        return output_l, output_m, output_s




