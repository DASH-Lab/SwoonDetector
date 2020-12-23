import argparse
import sys
sys.path.insert(0, './yolov5_test')
import os

import shutil
import time
from pathlib import Path

import cv2
import math
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from yolov5_test.utils.general import xywh2xyxy, pre_WBF
print(torch.__version__)
print(torch.cuda.is_available())
import torch.backends.cudnn as cudnn
from numpy import random

from yolov5_test.models.experimental import attempt_load
from yolov5_test.utils.datasets import LoadStreams, LoadImages
from yolov5_test.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from yolov5_test.utils.torch_utils import select_device, load_classifier, time_synchronized
from ensemble_boxes import *
def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = boxes[:, 0].clamp(0, img_shape[1])  # x1
    boxes[:, 1] = boxes[:, 1].clamp(0, img_shape[0])
    boxes[:, 2] = boxes[:, 2].clamp(0, img_shape[1])
    boxes[:, 3] = boxes[:, 3].clamp(0, img_shape[0])
    return boxes

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

class HumanDetector:
    def __init__(self, weight_path):
        self.confidence_thresh = 0.01
        self.iou_thresh = 0.55
        self.device = select_device('cuda')
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(weight_path, map_location=self.device)
        # self.imgsz = check_img_size(1280, s=self.model.stride.max())
        if self.half:
            self.model.half()

    def predict(self, image, iou, conf):
        h0, w0 = image.shape[:2]
        r = 1280 / max(h0, w0)  # resize image to img_size
        interp = cv2.INTER_AREA
        img = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=interp)
        (h, w) = img.shape[:2]
        img, ratio, pad = letterbox(img, new_shape=(1280, 1280))
        shapes = (h0, w0), ((h / h0, w / w0), pad)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, False)[0]
        output = non_max_suppression(pred, conf_thres=conf, iou_thres=iou, merge=False)
        try:
            nms_result = output[0]
            coords = nms_result
            coords = clip_coords(coords, (1080, 1920))
            box = coords[:, :4].clone()
            confidence_score = coords[:, 4].clone()
            coords = scale_coords(img.shape[1:], box, shapes[0], shapes[1])
            coords = coords.cpu().detach().numpy()

            result_coord = [np.round(coord).astype(np.int).tolist() for coord in coords]
            confidence_score = [score.cpu().detach().numpy() for score in confidence_score]
        except:
            return {'img': [], 'label': [], 'score': []}

        return {'img': [image[x[1]:x[3], x[0]:x[2], :] for x in result_coord],
                'label': result_coord
            , 'score': confidence_score}



if __name__ == "__main__":
    a = Human_detector('./weight/best_.pt')
    cv = cv2.imread('/media/data2/AGC/IITP_Track01_Sample/01/GH010171 484.jpg')
    import matplotlib.pyplot as plt
    aa = a.predict(cv)['img']
    print(aa)
    for i in aa:
        plt.imshow(i)
        plt.show()

