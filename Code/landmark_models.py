import cv2
import numpy as np
import dlib
from math import hypot
from scipy.spatial import distance as dist

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, frame, facial_landmarks):

    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio



def get_gaze_ratio(eye_points, frame, facial_landmarks):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ##################### Keep only eye region in gray image. Remove rest. ##################################
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    ###################### Apply thresholding in eye region to separate eyeball (black) from rest eye(white) #################
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y:max_y, min_x:max_x]
    x, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    ################################ Divide eye into - left part and right part #####################################
    height, width = threshold_eye.shape

    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]


    up_side_threshold = threshold_eye[0: int(height/2), 0: width]
    down_side_threshold = threshold_eye[int(height/2): height, 0: width]

    ########################## Check which part (left/right) has more white region #################################
    # If left part has more white -> means eyeball is on right
    # If right part has more white -> means eyeball is on left
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio1 = 0.4
    elif right_side_white == 0:
        gaze_ratio1 = 5
    else:
        gaze_ratio1 = left_side_white / right_side_white

    up_side_white = cv2.countNonZero(up_side_threshold)
    down_side_white = cv2.countNonZero(down_side_threshold)

    if up_side_white == 0:
        gaze_ratio2 = 0.4
    elif down_side_white == 0:
        gaze_ratio2 = 5
    else:
        gaze_ratio2 = up_side_white / down_side_white

    return gaze_ratio1,gaze_ratio2



def get_mouth_ratio(mouth_points, frame, facial_landmarks):

    left = (facial_landmarks.part(mouth_points[0]).x, facial_landmarks.part(mouth_points[0]).y) # L1
    right = (facial_landmarks.part(mouth_points[2]).x, facial_landmarks.part(mouth_points[2]).y) # L5
    top = (facial_landmarks.part(mouth_points[1]).x, facial_landmarks.part(mouth_points[1]).y) # L3
    bottom = (facial_landmarks.part(mouth_points[3]).x, facial_landmarks.part(mouth_points[3]).y) # L7

    dist1 = dist.euclidean(top,bottom)
    dist2 = dist.euclidean(left,right)

    ratio = float(dist1/dist2)

    return ratio