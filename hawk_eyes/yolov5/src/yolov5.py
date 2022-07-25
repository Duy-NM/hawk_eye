from time import time
from click import argument
import cv2
from cv2 import waitKey
import torch, os, time
import numpy as np
import onnxruntime as ort

from .models.common import DetectMultiBackend
from .utils.augmentations import letterbox
from .utils.general import (LOGGER, check_img_size,scale_coords, cv2, non_max_suppression)


class Yolov5():
    def __init__(self, pt_model_path, yaml_path, imgsz=(640,640), augment=False,conf_thres=0.5, iou_thres=0.45, classes=None, agnostic_nms=False, max_det=100) -> None:
        self.imgsz = imgsz
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = DetectMultiBackend(pt_model_path, device=self.device, dnn=False, data=yaml_path, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size

    def inference(self, img):
        h,w = img.shape[0:2]
        gain = max(w/self.imgsz[0], h/self.imgsz[1])
        img = letterbox(img, self.imgsz, stride=self.stride, auto=self.pt)[0]
        im = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None] 
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = self.model(im, augment=self.augment)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        pred = pred[0].cpu().detach().numpy()
        for i, pre in enumerate(pred):
            pred[i][:4] = np.array(pre[:4]*gain)

        # pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], img.shape).round()

        return pred


class Yolov5():
    def __init__(self, model_path, imw=640, imh=640, conf_thres=0.5, iou_thres=0.45, max_det=100) -> None:
        self.imw = imw
        self.imh = imh
        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.max_det=max_det
        self.model = ort.InferenceSession(model_path)
        self.input_name = self.model.get_inputs()[0].name

    def inference(self, image):
        h,w = image.shape[0:2]
        rw = w/self.imw
        rh = h/self.imh
        img = cv2.resize(image,(self.imw,self.imh))
        img = img.transpose((2, 0, 1))[::-1]
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        pred = self.model.run(None, {self.input_name: img.astype(np.float32)})
        pred = torch.FloatTensor(pred)

        pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=None, agnostic=False, max_det=self.max_det)
        pred = pred[0].cpu().detach().numpy()

        for i, pre in enumerate(pred):
            pred[i][:4] = np.array(pre[:4]*[rw,rh,rw,rh])

        return pred
