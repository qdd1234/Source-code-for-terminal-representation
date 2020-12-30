#! /usr/bin/env python
# coding=utf-8
# ================================================================
#
#   Author      : zhoukunyang
#   Created date: 2020-12-08
#   Description : 配置文件。
#
# ================================================================


class YOLOv4_Config_1(object):
    """
    YOLOv4默认配置
    """
    def __init__(self):
        self.algorithm = 'YOLOv4'

        # 自定义数据集
        # self.train_path = 'annotation_json/voc2012_train.json'
        # self.val_path = 'annotation_json/voc2012_val.json'
        # self.classes_path = 'data/voc_classes.txt'
        # self.train_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'   # 训练集图片相对路径
        # self.val_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'     # 验证集图片相对路径

        # COCO数据集
        self.train_path = '../data/annotations/instances_train2017.json'
        # self.train_path = '../data/data7122/annotations/instances_val2017.json'
        self.val_path = '../data/annotations/instances_val2017.json'
        self.classes_path = 'data/coco_classes.txt'
        self.train_pre_path = '../data/train2017/'  # 训练集图片相对路径
        # self.train_pre_path = '../data/data7122/val2017/'      # 验证集图片相对路径
        self.val_pre_path = '../data/val2017/'      # 验证集图片相对路径

        # 训练时若预测框与所有的gt小于阈值self.iou_loss_thresh时视为反例
        self.iou_loss_thresh = 0.7

        # 模式。 0-从头训练，1-读取之前的模型继续训练（model_path可以是'yolov4.h5'、'./weights/step00001000.h5'这些。）
        self.pattern = 0
        self.lr = 0.0001
        self.batch_size = 32
        # 如果self.pattern = 1，需要指定self.model_path表示从哪个模型读取权重继续训练。
        self.model_path = 'weights1/best_model'
        # self.model_path = './weights/1000'

        # ========= 一些设置 =========
        # 每隔几步保存一次模型
        self.save_iter = 1000
        # 每隔几步计算一次eval集的mAP
        self.eval_iter = 5000
        # 训练多少步
        self.max_iters = 650000


        # 验证
        # self.input_shape越大，精度会上升，但速度会下降。
        # self.input_shape = (320, 320)
        self.input_shape = (416, 416)
        #self.input_shape = (608, 608)
        # 验证时的分数阈值和nms_iou阈值
        self.conf_thresh = 0.001
        self.nms_thresh = 0.45
        # 是否画出验证集图片
        self.draw_image = False
        # 验证时的批大小
        self.eval_batch_size = 16


        # ============= 训练时预处理相关 =============
        self.with_mixup = False
        self.context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
        # PadBox
        self.num_max_boxes = 70
        # Gt2YoloTarget
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anchors = [[12, 16], [19, 36], [40, 28],
                        [36, 75], [76, 55], [72, 146],
                        [142, 110], [192, 243], [459, 401]]
        self.downsample_ratios = [32, 16, 8]


        # ============= 推理、导出时相关 =============
        # 读取的模型
        self.infer_model_path = 'yolov4'
        # self.infer_model_path = './weights/66000'

        # infer_input_shape越大，精度会上升，但速度会下降。
        # self.infer_input_shape = (320, 320)
        self.infer_input_shape = (416, 416)
        # self.infer_input_shape = (608, 608)

        # 推理时的分数阈值和nms_iou阈值
        self.infer_conf_thresh = 0.05
        self.infer_nms_thresh = 0.45
        self.infer_keep_top_k = 100
        self.infer_nms_top_k = 100

        # 是否给图片画框。
        self.infer_draw_image = True
        # self.infer_draw_image = False




class YOLOv3_Config_1(object):
    """
    YOLOv3默认配置
    """
    def __init__(self):
        self.algorithm = 'YOLOv3'

        # 自定义数据集
        # self.train_path = 'annotation_json/voc2012_train.json'
        # self.val_path = 'annotation_json/voc2012_val.json'
        # self.classes_path = 'data/voc_classes.txt'
        # self.train_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'   # 训练集图片相对路径
        # self.val_pre_path = '../VOCdevkit/VOC2012/JPEGImages/'     # 验证集图片相对路径

        # COCO数据集
        self.train_path = '../data/annotations/instances_train2017.json'
        # self.train_path = '../data/data7122/annotations/instances_val2017.json'
        self.val_path = '../data/annotations/instances_val2017.json'
        self.classes_path = 'data/coco_classes.txt'
        self.train_pre_path = '../data/train2017/'  # 训练集图片相对路径
        # self.train_pre_path = '../data/data7122/val2017/'      # 验证集图片相对路径
        self.val_pre_path = '../data/val2017/'      # 验证集图片相对路径

        # 训练时若预测框与所有的gt小于阈值self.iou_loss_thresh时视为反例
        self.iou_loss_thresh = 0.7

        # 模式。 0-从头训练，1-读取之前的模型继续训练（model_path可以是'yolov4.h5'、'./weights/step00001000.h5'这些。）
        self.pattern = 0
        self.lr = 0.0001
        self.batch_size = 64
        # 如果self.pattern = 1，需要指定self.model_path表示从哪个模型读取权重继续训练。
        self.model_path = 'yolov3_r50vd_dcn_obj365_dropblock_iouloss'
        # self.model_path = './weights/1000'

        # ========= 一些设置 =========
        # 每隔几步保存一次模型
        self.save_iter = 1000
        # 每隔几步计算一次eval集的mAP
        self.eval_iter = 1000
        # 训练多少步
        self.max_iters = 60000


        # 验证
        # self.input_shape越大，精度会上升，但速度会下降。
        # self.input_shape = (320, 320)
        self.input_shape = (416, 416)
        # self.input_shape = (608, 608)
        # 验证时的分数阈值和nms_iou阈值
        self.conf_thresh = 0.001
        self.nms_thresh = 0.45
        # 是否画出验证集图片
        self.draw_image = False
        # 验证时的批大小
        self.eval_batch_size = 4


        # ============= 训练时预处理相关 =============
        self.with_mixup = False
        self.context = {'fields': ['image', 'gt_bbox', 'gt_class', 'gt_score']}
        # PadBox
        self.num_max_boxes = 70
        # Gt2YoloTarget
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anchors = [[10, 13], [16, 30], [33, 23],
                        [30, 61], [62, 45], [59, 119],
                        [116, 90], [156, 198], [373, 326]]
        self.downsample_ratios = [32, 16, 8]


        # ============= 推理、导出时相关 =============
        # 读取的模型
        self.infer_model_path = 'yolov3_r50vd_dcn_obj365_dropblock_iouloss'
        # self.infer_model_path = './weights/1000'

        # infer_input_shape越大，精度会上升，但速度会下降。
        # self.infer_input_shape = (320, 320)
        self.infer_input_shape = (416, 416)
        # self.infer_input_shape = (608, 608)

        # 推理时的分数阈值和nms_iou阈值
        self.infer_conf_thresh = 0.05
        self.infer_nms_thresh = 0.45
        self.infer_keep_top_k = 100
        self.infer_nms_top_k = 100

        # 是否给图片画框。
        self.infer_draw_image = True
        # self.infer_draw_image = False


class PostprocessNumpyNMSConfig(object):
    """
    deploy_infer.py后处理配置
    """
    def __init__(self):
        self.anchors = [[12, 16], [19, 36], [40, 28],
                        [36, 75], [76, 55], [72, 146],
                        [142, 110], [192, 243], [459, 401]]
        self.conf_thresh = 0.05
        self.nms_thresh = 0.45
        self.keep_top_k = 100
        self.nms_top_k = 100
        self.nms_top_k = 100


class TrainConfig_2(object):
    """
    其它配置
    """
    def __init__(self):
        pass




