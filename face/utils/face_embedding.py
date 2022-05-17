import cv2
import numpy as np

from utils import face_align
import onnx
import onnxruntime
# import tensorflow as tf
import cv2
from sklearn.preprocessing import normalize


class ArcFaceONNX:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        self.taskname = 'recognition'
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            #print(nid, node.name)
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        if find_sub and find_mul:
            #mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        #print('input mean and std:', self.input_mean, self.input_std)
        if self.session is None:
            self.session = onnxruntime.InferenceSession(self.model_file, None)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names)==1
        self.output_shape = outputs[0].shape
    
    def get(self, img, kps):
        # print(onnxruntime.get_device())
        # cv2.imshow('d', img)
        aimg = face_align.norm_crop(img, landmark=kps)
        # cv2.imshow('aaasd', aimg)
        # cv2.waitKey(1)
        cv2.imwrite('aaaa.jpg', aimg)
        embedding = self.get_feat(aimg).flatten()
        return embedding

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size
        
        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

if __name__ == '__main__':
    from face_detection import RetinaFace
    retinaface_ = RetinaFace(model_file='/home/vti/.insightface/models/buffalo_l/det_10g.onnx')
    arcface = ArcFaceONNX(model_file='/home/vti/.insightface/models/buffalo_l/w600k_r50.onnx')

    img = cv2.imread('/media/vti/DATA/Work/VTI_Project/FaceRecognition/face_recognition/data_vti_register_0930v4/anh.nguyenthi/2.jpg')
    bboxes, kpss = retinaface_.detect(img)
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        out = arcface.get(img, kps)
        bbox = bbox.astype(int)
        cv2.imshow('aaa', img[bbox[1]:bbox[3], bbox[0]:bbox[2], :])
        cv2.waitKey(0)
        print(out)
