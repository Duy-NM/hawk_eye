import cv2
import numpy as np

from face.src.face_detection import RetinaFace
from face.src.face_embedding import ArcFace
from face.src.landmark import Landmark
from tracking.src.byte_tracker import BYTETracker

retina = RetinaFace()
arc = ArcFace()
lan = Landmark()
bt = BYTETracker()

img = cv2.imread('/media/vti/SSD/VTI/FaceMask/face_recognize_v2/5.png')
boxes, kpss = retina.detect(img)

for b,k in zip(boxes, kpss):
    emb = arc.get(img, k)
    print(emb)
    l = lan.get(img, b)
    print(l)


while True:
    frame = img
    boxes,_ = retina.detect(frame)
    boxes, ids = bt.predict(frame, boxes)




