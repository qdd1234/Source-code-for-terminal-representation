#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : miemie2013
#   Created date: 2020-06-10 10:20:27
#   Description : paddlepaddle_yolov4
#
# ================================================================
import paddle.fluid as fluid

from model.yolov4 import yolo_decode


class YOLACT(object):
    def __init__(self, backbone, head):
        super(YOLACT, self).__init__()
        self.backbone = backbone
        self.head = head

    def __call__(self, x, export=False, postprocess=None, param=None):
        body_feats = self.backbone(x)
        outputs, mcf_outputs, proto_out, segm = self.head(body_feats)
        # if export:
        #     return yolo_decode(output_l, output_m, output_s, postprocess, param)

        output_l = outputs[0]
        output_m = outputs[1]
        output_s = outputs[2]
        mcf_l = mcf_outputs[0]
        mcf_m = mcf_outputs[1]
        mcf_s = mcf_outputs[2]
        # 相当于numpy的transpose()，交换下标
        output_l = fluid.layers.transpose(output_l, perm=[0, 2, 3, 1], name='output_l')
        output_m = fluid.layers.transpose(output_m, perm=[0, 2, 3, 1], name='output_m')
        output_s = fluid.layers.transpose(output_s, perm=[0, 2, 3, 1], name='output_s')
        mcf_l = fluid.layers.transpose(mcf_l, perm=[0, 2, 3, 1], name='mcf_l')
        mcf_m = fluid.layers.transpose(mcf_m, perm=[0, 2, 3, 1], name='mcf_m')
        mcf_s = fluid.layers.transpose(mcf_s, perm=[0, 2, 3, 1], name='mcf_s')
        return output_l, output_m, output_s, mcf_l, mcf_m, mcf_s, proto_out, segm




