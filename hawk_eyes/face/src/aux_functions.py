from configparser import ConfigParser
import cv2, math, os
from PIL import Image, ImageDraw
from cv2 import mean
from matplotlib.pyplot import axis
from tqdm import tqdm
import random
from imutils import face_utils
import requests
from zipfile import ZipFile
from tqdm import tqdm
import bz2, shutil
import numpy as np



def get_line(face_landmark, image, type="eye", debug=False):
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    left_eye = face_landmark["left_eye"]
    right_eye = face_landmark["right_eye"]
    left_eye_mid = np.mean(np.array(left_eye), axis=0)
    right_eye_mid = np.mean(np.array(right_eye), axis=0)
    eye_line_mid = (left_eye_mid + right_eye_mid) / 2

    if type == "eye":
        left_point = left_eye_mid
        right_point = right_eye_mid
        mid_point = eye_line_mid

    elif type == "nose_mid":
        nose_length = (
            face_landmark["nose_bridge"][-1][1] - face_landmark["nose_bridge"][0][1]
        )
        left_point = [left_eye_mid[0], left_eye_mid[1] + nose_length / 2]
        right_point = [right_eye_mid[0], right_eye_mid[1] + nose_length / 2]
        # mid_point = (
        #     face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
        # ) / 2

        mid_pointY = (
            face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
        ) / 2
        mid_pointX = (
            face_landmark["nose_bridge"][-1][0] + face_landmark["nose_bridge"][0][0]
        ) / 2
        mid_point = (mid_pointX, mid_pointY)

    elif type == "nose_tip":
        nose_length = (
            face_landmark["nose_bridge"][-1][1] - face_landmark["nose_bridge"][0][1]
        )
        left_point = [left_eye_mid[0], left_eye_mid[1] + nose_length]
        right_point = [right_eye_mid[0], right_eye_mid[1] + nose_length]
        mid_point = (
            face_landmark["nose_bridge"][-1][1] + face_landmark["nose_bridge"][0][1]
        ) / 2

    elif type == "bottom_lip":
        bottom_lip = face_landmark["bottom_lip"]
        bottom_lip_mid = np.max(np.array(bottom_lip), axis=0)
        shiftY = bottom_lip_mid[1] - eye_line_mid[1]
        left_point = [left_eye_mid[0], left_eye_mid[1] + shiftY]
        right_point = [right_eye_mid[0], right_eye_mid[1] + shiftY]
        mid_point = bottom_lip_mid

    elif type == "perp_line":
        bottom_lip = face_landmark["bottom_lip"]
        bottom_lip_mid = np.mean(np.array(bottom_lip), axis=0)

        left_point = eye_line_mid
        left_point = face_landmark["nose_bridge"][0]
        right_point = bottom_lip_mid

        mid_point = bottom_lip_mid

    elif type == "nose_long":
        nose_bridge = face_landmark["nose_bridge"]
        left_point = [nose_bridge[0][0], nose_bridge[0][1]]
        right_point = [nose_bridge[-1][0], nose_bridge[-1][1]]

        mid_point = left_point

    # d.line(eye_mid, width=5, fill='red')
    y = [left_point[1], right_point[1]]
    x = [left_point[0], right_point[0]]
    # cv2.imshow('h', image)
    # cv2.waitKey(0)
    eye_line = fit_line(x, y, image)
    d.line(eye_line, width=5, fill="blue")

    # Perpendicular Line
    # (midX, midY) and (midX - y2 + y1, midY + x2 - x1)
    y = [
        (left_point[1] + right_point[1]) / 2,
        (left_point[1] + right_point[1]) / 2 + right_point[0] - left_point[0],
    ]
    x = [
        (left_point[0] + right_point[0]) / 2,
        (left_point[0] + right_point[0]) / 2 - right_point[1] + left_point[1],
    ]
    perp_line = fit_line(x, y, image)
    if debug:
        d.line(perp_line, width=5, fill="red")
        pil_image.show()
    return eye_line, perp_line, left_point, right_point, mid_point


def get_points_on_chin(line, face_landmark, chin_type="chin"):
    chin = face_landmark[chin_type]
    points_on_chin = []
    for i in range(len(chin) - 1):
        chin_first_point = [chin[i][0], chin[i][1]]
        chin_second_point = [chin[i + 1][0], chin[i + 1][1]]

        flag, x, y = line_intersection(line, (chin_first_point, chin_second_point))
        if flag:
            points_on_chin.append((x, y))

    return points_on_chin


def plot_lines(face_line, image, debug=False):
    pil_image = Image.fromarray(image)
    if debug:
        d = ImageDraw.Draw(pil_image)
        d.line(face_line, width=4, fill="white")
        pil_image.show()


def line_intersection(line1, line2):
    # mid = int(len(line1) / 2)
    start = 0
    end = -1
    line1 = ([line1[start][0], line1[start][1]], [line1[end][0], line1[end][1]])

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    x = []
    y = []
    flag = False

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return flag, x, y

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    segment_minX = min(line2[0][0], line2[1][0])
    segment_maxX = max(line2[0][0], line2[1][0])

    segment_minY = min(line2[0][1], line2[1][1])
    segment_maxY = max(line2[0][1], line2[1][1])

    if (
        segment_maxX + 1 >= x >= segment_minX - 1
        and segment_maxY + 1 >= y >= segment_minY - 1
    ):
        flag = True

    return flag, x, y


def fit_line(x, y, image):
    if x[0] == x[1]:
        x[0] += 0.1
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    x_axis = np.linspace(0, image.shape[1], 50)
    y_axis = polynomial(x_axis)
    eye_line = []
    for i in range(len(x_axis)):
        eye_line.append((x_axis[i], y_axis[i]))

    return eye_line


def get_angle(line1, line2):
    delta_y = line1[-1][1] - line1[0][1]
    delta_x = line1[-1][0] - line1[0][0]
    perp_angle = math.degrees(math.atan2(delta_y, delta_x))
    if delta_x < 0:
        perp_angle = perp_angle + 180
    if perp_angle < 0:
        perp_angle += 360
    if perp_angle > 180:
        perp_angle -= 180

    # print("perp", perp_angle)
    delta_y = line2[-1][1] - line2[0][1]
    delta_x = line2[-1][0] - line2[0][0]
    nose_angle = math.degrees(math.atan2(delta_y, delta_x))

    if delta_x < 0:
        nose_angle = nose_angle + 180
    if nose_angle < 0:
        nose_angle += 360
    if nose_angle > 180:
        nose_angle -= 180
    # print("nose", nose_angle)

    angle = nose_angle - perp_angle
    return angle




def draw_landmarks(face_landmarks, image):
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=5, fill="white")
    pil_image.show()



def get_avg_brightness(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(v)


def get_avg_saturation(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    return np.mean(v)


def change_brightness(img, value=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    v = value * v
    v[v > 255] = 255
    v = np.asarray(v, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def change_saturation(img, value=1.0):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    s = value * s
    s[s > 255] = 255
    s = np.asarray(s, dtype=np.uint8)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def check_path(path):
    is_directory = False
    is_file = False
    is_other = False
    if os.path.isdir(path):
        is_directory = True
    elif os.path.isfile(path):
        is_file = True
    else:
        is_other = True

    return is_directory, is_file, is_other



def convert_106p_to_86p(pt106):
    landmark106to68=[1,10,12,14,16,3,5,7,0,23,21,19,32,30,28,26,17,    # 脸颊17点
                 43,48,49,51,50,      # 左眉毛5点
                 102,103,104,105,101, # 右眉毛5点
                 72,73,74,86,78,79,80,85,84, # 鼻子9点
                 35,41,42,39,37,36,   # 左眼睛6点
                 89,95,96,93,91,90,   # 右眼睛6点
                 52,64,63,71,67,68,61,65,66,62,70,69, # 嘴巴20点
                 61,58,59,53,56,55,52,69,57,60,54,65 #bottom_lip
                 ]
    pt86 = {'left_eyebrow':[], 'right_eyebrow':[], 'nose_bridge':[], 'nose_tip':[], 'left_eye':[], 'right_eye':[], 'top_lip':[], 'bottom_lip':[], 'chin':[]}
    for i in range(68):
        if i < 17:
            index = landmark106to68[i]
            pt86['chin'].append(pt106[index])
        elif i<22:
            index = landmark106to68[i]
            pt86['left_eyebrow'].append(pt106[index])
        elif i<27:
            index = landmark106to68[i]
            pt86['right_eyebrow'].append(pt106[index])
        elif i<31:
            index = landmark106to68[i]
            pt86['nose_bridge'].append(pt106[index])
        elif i<36:
            index = landmark106to68[i]
            pt86['nose_tip'].append(pt106[index])
        elif i<42:
            index = landmark106to68[i]
            pt86['left_eye'].append(pt106[index])
        elif i<48:
            index = landmark106to68[i]
            pt86['right_eye'].append(pt106[index])
        elif i<60:
            index = landmark106to68[i]
            pt86['top_lip'].append(pt106[index])
        else:
            index = landmark106to68[i]
            pt86['bottom_lip'].append(pt106[index])

    return pt86


def is_image(path):
    try:
        extensions = path[-4:]
        image_extensions = ["png", "PNG", "jpg", "JPG"]

        if extensions[1:] in image_extensions:
            return True 
        else:
            print("Please input image file. png / jpg")
            return False 
    except: 
        return False 


