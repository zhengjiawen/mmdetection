
from mmdet.apis import init_detector, inference_detector, show_result
import matplotlib.pyplot as plt

import mmcv
import cv2 as cv
import os
import time
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils import merge_bbox, nms_cpu, merge_bbox_overlap


config_file = '../configs/table_ocr/table_faster_rcnn_r50_fpn_1x_base.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '../work_dirs/table_faster_rcnn_r50_fpn_1x_base/epoch_12.pth'

output_path = '../work_dirs/table_faster_rcnn_r50_fpn_1x_base/show_det/'

img_base_path = '/data/home/zjw/dataset/daxin/daxin_image/'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# test a single image
for i in range(1044):
    img_path = os.path.join(img_base_path, str(i)+'.jpg')
    start = time.time()
    result = inference_detector(model, img_path)
    print('single img cost time {:.2f} s '.format(time.time() - start))

    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(result)

    if len(bboxes) == 0 or len(bboxes) == 1:
        merge_bboxes = bboxes
    else:
        # merge high iou bbox to cover entire table
        merge_bboxes = merge_bbox_overlap(bboxes, 0.60)
        if len(merge_bboxes) > 1:
            merge_bboxes = nms_cpu(merge_bboxes, 0.2)

        print('merged boxes shape: {}'.format(merge_bboxes.shape))

    output_img_path = os.path.join(output_path, str(i)+'.jpg')
    # show the results
    show_result(img_path, [merge_bboxes], [model.CLASSES], score_thr=0.2, show=False, out_file=output_img_path)

