import cv2
import numpy as np
from .general import non_max_suppression
import torch
import onnxruntime as ort

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

def pth2onnx():
    pass