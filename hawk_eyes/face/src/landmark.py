from __future__ import division
import numpy as np
import cv2
import onnx
import onnxruntime
from . import face_align
from . import reference_world as world
from .aux_functions import get_line, get_points_on_chin, get_angle, convert_106p_to_86p
import os, gdown

face3Dmodel = world.ref3DModel()
model_dir = os.path.join(os.path.expanduser('~'), '.hawkeye/model')

class Landmark:
    def __init__(self, model_name='', model_file=None, session=None):
        if model_file is None:
            assert model_name in ['', '2d', '3d'], 'model_name is wrong'
            if model_name == '2d' or  model_name == '':
                self.model_file = model_dir + '/2d106det.onnx'
                if os.path.exists(self.model_file) == False:
                    gdown.download('https://drive.google.com/u/0/uc?id=18ngkuvwYATi1Y0SDA9ni8fh9SKrQFp0q&export=download', self.model_file, quiet=False)
            if model_name == '3d':
                self.model_file = model_dir + '/1k3d68.onnx'
                if os.path.exists(self.model_file) == False:
                    gdown.download('https://drive.google.com/u/0/uc?id=1nRcEJrfWjLPWCDurhH3_JkM40hD_FJ8D&export=download', self.model_file, quiet=False)
        else:
            self.model_file = model_file

        self.session = session
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
            if nid<3 and node.name=='bn_data':
                find_sub = True
                find_mul = True
        if find_sub and find_mul:
            #mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 128.0
        self.input_mean = input_mean
        self.input_std = input_std
        #print('input mean and std:', model_file, self.input_mean, self.input_std)
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
        output_shape = outputs[0].shape
        #print('init output_shape:', output_shape)
        if output_shape[1]==3309:
            self.lmk_dim = 3
            self.lmk_num = 68
        else:
            self.lmk_dim = 2
            self.lmk_num = output_shape[1]//self.lmk_dim
        self.taskname = 'landmark_%dd_%d'%(self.lmk_dim, self.lmk_num)

    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])

    def get(self, img, bbox):
        # bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0]  / (max(w, h)*1.5)
        #print('param:', img.shape, bbox, center, self.input_size, _scale, rotate)
        aimg, M = face_align.transform(img, center, self.input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])
        #assert input_size==self.input_size
        blob = cv2.dnn.blobFromImage(aimg, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        pred = self.session.run(self.output_names, {self.input_name : blob})[0][0]
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num*-1:,:]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.input_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (self.input_size[0] // 2)

        IM = cv2.invertAffineTransform(M)
        pred = face_align.trans_points(pred, IM)
        # face[self.taskname] = pred
        return pred
    
    def get_face_angle(self, image, landmark, draw=True):
        if draw:
            for la in landmark:
                cv2.circle(image, la.astype(int), 1, (155,155,155), 1)
        refImgPts = np.array([landmark[86], landmark[0], landmark[35], landmark[93], landmark[52], landmark[61]], dtype=np.float64)
        # print(refImgPts)
        height, width, channel = image.shape
        focalLength = width
        cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))
        mdists = np.zeros((4, 1), dtype=np.float64)
        # calculate rotation and translation vector using solvePnP
        success, rotationVector, translationVector = cv2.solvePnP(face3Dmodel, refImgPts, cameraMatrix, mdists)
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)
        p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
        p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))

        # print(angles[1])
        return angles[1] #, p1,p2

    def get_face_angle2(self, image, landmark):
        dlib_face_landmark = convert_106p_to_86p(landmark)
        perp_line, _, _, _, _ = get_line(dlib_face_landmark, image, type="perp_line")
        # face_e = points1[0]
        nose_mid_line, _, _, _, _ = get_line(dlib_face_landmark, image, type="nose_long")

        angle = get_angle(perp_line, nose_mid_line)

        return angle
