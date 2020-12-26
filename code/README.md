[English](README_en.md) | 简体中文

# YOLOv4-Tiny PLUS源码

## 代码位置
模型的源代码在work/model/yolov4内

## 推荐
本项目已经开源到AIStudio中，可直接跑：
https://aistudio.baidu.com/aistudio/projectdetail/570310

## 训练
开始训练前安装pycocotools依赖、解压COCO2017数据集,文件夹的路径需要修改成自己的：
```
cd ~
pip install pycocotools
cd data
cd data7122
unzip ann*.zip
unzip val*.zip
unzip tes*.zip
unzip image_info*.zip
unzip train*.zip
cd ~/w*
```

运行train.py进行训练:
```
rm -f train.txt
nohup python train.py>> train.txt 2>&1 &
```
通过修改config.py代码来进行更换数据集、更改超参数以及训练参数。
训练时默认每5000步计算一次验证集的mAP。或者运行eval.py评估指定模型的mAP。该mAP是val集的结果。



## 训练自定义数据集
自带的voc2012数据集是一个很好的例子。

将自己数据集的txt注解文件放到annotation目录下，txt注解文件的格式如下：
```
xxx.jpg 18.19,6.32,424.13,421.83,20 323.86,2.65,640.0,421.94,20
xxx.jpg 48,240,195,371,11 8,12,352,498,14
# 图片名 物体1左上角x坐标,物体1左上角y坐标,物体1右下角x坐标,物体1右下角y坐标,物体1类别id 物体2左上角x坐标,物体2左上角y坐标,物体2右下角x坐标,物体2右下角y坐标,物体2类别id ...
```
运行1_txt2json.py会在annotation_json目录下生成两个coco注解风格的json注解文件，这是train.py支持的注解文件格式。
在config.py里修改train_path、val_path、classes_path、train_pre_path、val_pre_path这5个变量（自带的voc2012数据集直接解除注释就ok了）就可以开始训练自己的数据集了。
如果需要跑demo.py、eval.py，与数据集有关的变量也需要修改一下，应该很容易看懂。

## 评估
训练时默认每5000步计算一次验证集的mAP。或者运行eval.py评估指定模型的mAP。该mAP是val集的结果。

## test-dev
运行test_dev.py。
运行完之后，进入results目录，把bbox_detections.json压缩成bbox_detections.zip，提交到
https://competitions.codalab.org/competitions/20794#participate


## 预测
运行demo.py。

## 导出
```
python export_model.py
```
关于导出的参数请看export_model.py中的注释。导出后的模型默认存放在inference_model目录下，带有一个配置文件infer_cfg.yml。

用导出后的模型预测图片：
```
python deploy_infer.py --model_dir inference_model --image_dir images/test/
```

用导出后的模型预测视频：
```
python deploy_infer.py --model_dir inference_model --video_file 视频路径
```

用导出后的模型播放视频：（按esc键停止播放）
```
python deploy_infer.py --model_dir inference_model --play_video 视频路径
```



