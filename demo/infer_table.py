
from mmdet.apis import init_detector, inference_detector, show_result
import matplotlib.pyplot as plt
import os
import sys
import mmcv
import cv2 as cv
import numpy as np
sys.path.append(os.path.dirname(__file__))
from utils import merge_bbox, nms_cpu, merge_bbox_overlap


# config_file = '../configs/table_ocr/table_faster_rcnn_r50_fpn_1x_base.py'
config_file = '../configs/table_ocr/faster_rcnn_r50_highIoU.py'

# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../work_dirs/table_faster_rcnn_r50_fpn_1x_base/epoch_12.pth'
checkpoint_file = '../work_dirs/table_faster_rcnn_r50_fpn_1x_base_high_IoU/epoch_12.pth'
# output_path = '../work_dirs/table_faster_rcnn_r50_fpn_1x_base/test.jpg'
output_path = '../work_dirs/table_faster_rcnn_r50_fpn_1x_base_high_IoU/test.jpg'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# test a single image
img = '/data/home/zjw/dataset/daxin/daxin_image/0.jpg'
result = inference_detector(model, img)

# print('result shape {}'.format(len(result)))
# print('result cell shape {}'.format(len(result[0])))
if isinstance(result, tuple):
    bbox_result, segm_result = result
else:
    bbox_result, segm_result = result, None

print('result shape {}'.format(len(bbox_result)))
print('result cell shape {}'.format(bbox_result[0].shape))
print(bbox_result)
bboxes = np.vstack(result)

print('bboxes type {}'.format(type(bboxes)))
print('bboxes shape {}'.format(bboxes.shape))
print('bboxes len {}'.format(len(bboxes)))

if len(bboxes) == 0 or len(bboxes) == 1:
    merge_bboxes = bboxes
else:
    # merge high iou bbox to cover entire table
    merge_bboxes = merge_bbox(bboxes, 0.60)
    if len(merge_bboxes) > 1:
        print('second nms')
        merge_bboxes = nms_cpu(merge_bboxes, 0.2)

    print('merged boxes shape: {}'.format(merge_bboxes.shape))

print(merge_bboxes)

# show the results
show_result(img, [merge_bboxes], [model.CLASSES], score_thr=0.01, show=False, out_file=output_path)

