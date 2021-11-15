import numpy as np
import cv2
import joblib
import dlib

clf = joblib.load('models/face_spoofing.pkl')

sample_number = 1
count = 0
measures = np.zeros(sample_number, dtype=np.float)

def calc_hist(img):

    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)



def face_spoof(img, face):
    x = face[0]*4
    y = face[1]*4
    x1 = face[2]*4
    y1 = face[3]*4

    measures[count%sample_number]=0
    
    roi = img[y:y1, x:x1]
    point = (0,0)
    
    img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
    img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

    ycrcb_hist = calc_hist(img_ycrcb)
    luv_hist = calc_hist(img_luv)

    feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
    feature_vector = feature_vector.reshape(1, len(feature_vector))

    prediction = clf.predict_proba(feature_vector)
    prob = prediction[0][1]

    measures[count % sample_number] = prob
    
    #cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
    point = (x, y-5)

    return measures
