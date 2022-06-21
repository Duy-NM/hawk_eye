import cv2
import numpy as np

from . import face_align
import onnx
import onnxruntime
# import tensorflow as tf
import cv2
from sklearn.preprocessing import normalize
import os, gdown

model_dir = os.path.join(os.path.expanduser('~'), '.hawkeye/model')
class ArcFace:
    def __init__(self, model_name='', model_file=None, session=None):

        '''
        model_name: choice in ['arcface_s', 'arcface_m', 'arcface_l']
                    default is arcface_m
                    arcface_s -> SCRFD-500MF
                    arcface_m -> SCRFD-2.5GF
                    arcface_l -> SCRFD-10GF
        '''
        if model_file is None:
            assert model_name in ['', 'arcface_s', 'arcface_m', 'arcface_l'], 'model_name is wrong'
            if model_name == 'arcface_s':
                self.model_file = model_dir + '/s_w600k_mbf.onnx'
                if os.path.exists(model_dir + '/s_w600k_mbf.onnx') == False:
                    gdown.download('https://drive.google.com/u/0/uc?id=1Dvxphqyc84urh001xFPUWxmhVvrI7IbS&export=download', model_dir + '/s_w600k_mbf.onnx', quiet=False)
            if model_name == 'arcface_m' or  model_name == '':
                self.model_file = model_dir + '/glint360_r50.onnx'
                if os.path.exists(model_dir + '/glint360_r50.onnx') == False:
                    gdown.download('https://drive.google.com/u/0/uc?id=1obFVt88uEuoOYkOV2G6oAoQdBwSJGgy2&export=download', model_dir + '/glint360_r50.onnx', quiet=False)
            if model_name == 'arcface_l':
                self.model_file = model_dir + '/glint360_r100.onnx'
                if os.path.exists(model_dir + '/glint360_r100.onnx') == False:
                    gdown.download('https://drive.google.com/u/0/uc?id=1eHNQK7QbHFNEHqRUOlQNC0ju5E-ctjHX&export=download', model_dir + '/glint360_r100.onnx', quiet=False)
        else:
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
        # cv2.imwrite('aaaa.jpg', aimg)
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


