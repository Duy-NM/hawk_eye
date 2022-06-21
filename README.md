# HawkEye
A library for Computer Vision, flexible and easy to use. The library uses a lot of models like ArcFace, RetinaFace, FaceLandmark, OCRModels .... These models will be downloaded automatically when used for the first time, please make sure your network connection is not blocked to google drive.

# Install
**HawkEye** is available on [pypi.org](https://pypi.org/project/hawk-eyes/), if you just want to use it for your project, install it using pip.
Requires python>=3.8, torch>=1.8
```
pip install hawk-eyes
```
# Methods
Supported methods:
- [x] [Retina Face](https://arxiv.org/abs/1905.00641)
- [x] [ArcFace](https://arxiv.org/abs/1801.07698)
- [ ] [Face similar <coming soon>]()
- [x] [face landmarks]()
- [x] [face angles]()
- [x] [Tracking]()

# 1. Face Recognition
### 1.1 Face detection
[**Retina Face**](https://arxiv.org/abs/1905.00641): Single-stage Dense Face Localisation in the Wild
```py
import cv2
from hawkeye.face import RetinaFace

retina = RetinaFace(model_name='retina_s')

img = cv2.imread('path/to/image.png')
bboxes, kpss = retina.detect(img)
for box in bboxes:
    box = box[:4].astype(int)
    cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), (0,255,0),thickness=2)
cv2.imshow('image', img)
cv2.waitKey(0)
```

### 1.2 Face extract feature use ArcFace
[**ArcFace**](https://arxiv.org/abs/1801.07698): Additive Angular Margin Loss for Deep Face Recognition
```py
import cv2
from hawkeye.face import RetinaFace
from hawkeye.face import ArcFace

arc = ArcFace(model_name='arcface_s')
retina = RetinaFace(model_name='retina_s')

img = cv2.imread('path/to/image.png')
bboxes, kpss = retina.detect(img)
for box,kps in zip(bboxes, kpss):
    emb = arc.get(img, kps)
    print(emb)
```

### 1.3 Get similar
Coming soon

### 1.4 Get face landmarks, face angles
```py
import cv2
from hawkeye.face import RetinaFace, Landmark

retina = RetinaFace(model_name='retina_s')
landmark = Landmark()

img = cv2.imread('path/to/image.png')
bboxes, kpss = retina.detect(img)
for box,kps in zip(bboxes, kpss):
    land = landmark.get(img, box)
    angle = landmark.get_face_angle(img, land)
    print(angle)
```

# 2. Tracking

```py
import cv2
from hawkeye.face import RetinaFace
from hawkeye.tracking import BYTETracker

bt = BYTETracker()
retina = RetinaFace(model_name='retina_s')

cap = cv2.videoCapture(0)
ret, _ = cap.read()
while ret:
    ret, frame = cap.read()
    boxes,_ = retina.detect(frame)
    boxes, ids = bt.predict(frame, boxes)
    print(ids)
```
