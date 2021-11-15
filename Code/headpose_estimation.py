import numpy as np
import os
import sys
import cv2
from tensorflow.keras.models import load_model
from glob import glob
import random
from tensorflow.keras.utils import Sequence
from random import randrange
import json
import math
#import pandas as pd


# # Load Model


def load_hp_model(oModelPath):
    oHpModel = load_model(oModelPath)
    return oHpModel


# # Headpose Display Fuction


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.identity(3)
    R_y = np.identity(3)
    R_z = np.identity(3)
     
    R_x = np.array([[1,0,0],
                    [0,math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,math.sin(theta[0]), math.cos(theta[0])]
                    ])
         
         
                     
    R_y = np.array([[math.cos(theta[1]),0,math.sin(theta[1])],
                    [0,1,0],
                    [-math.sin(theta[1]),0,math.cos(theta[1])]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),-math.sin(theta[2]),0],
                    [math.sin(theta[2]),math.cos(theta[2]),0],
                    [0,0,1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R

#display headpose

#output of function contains headpose vector drawn on top of it
def displayHeadpose(oImgNp, oAnglesNp,oOffset = 0):
    #convert degree to radians
    oAnglesNp = oAnglesNp * np.pi/180.0
    oHeadposeRotationMat = eulerAnglesToRotationMatrix(oAnglesNp)
    oPointsNp = np.array([(100, 0 ,0),
                         (0, 100, 0),
                         (0, 0, 100)])
    oPointsNp = np.dot(oHeadposeRotationMat, oPointsNp)
    oOriginT = (int(oImgNp.shape[1]/2 + oOffset), int(oImgNp.shape[0]/2))
    oLineXT = (int(oOriginT[0] - oPointsNp[0,0]), int(oOriginT[1] - oPointsNp[1,0]))
    oLineYT = (int(oOriginT[0] - oPointsNp[0,1]), int(oOriginT[1] - oPointsNp[1,1]))
    oLineZT = (int(oOriginT[0] - oPointsNp[0,2]), int(oOriginT[1] - oPointsNp[1,2]))
    cv2.line(oImgNp, oOriginT, oLineXT, (0,0,255),3)
    cv2.line(oImgNp, oOriginT, oLineYT, (0,255,0),3)
    cv2.line(oImgNp, oOriginT, oLineZT, (255,0,0),3)
    return oImgNp


# # Headpose Inference


def expand_bbox(oBBox,oImage):
    x, y, width, height = oBBox[0],oBBox[1], oBBox[2] - oBBox[0],oBBox[3] - oBBox[1]
    #enlarged box
    x = int(x - width/2)
    if x < 0:
        x = 0
    
    y = int(y - height/2)
    if y< 0:
        y = 0
    
    width = int(width + width)
    if x+width > oImage.shape[1]:
        width = oImage.shape[1]
    
    height = int(height + height)
    if y+height > oImage.shape[0]:
        height = oImage.shape[0]
        
        
    bbox = [x,y,x+width,y+height]

    return bbox

#input image should be in BGR FORMAT
#input face box should be in [x1,y1,x2,y2] in other words [left, top,right,bottom]
def headpose_inference(oModel,oImage,face):
    
    left = face[0]*4
    top = face[1]*4
    right = face[2]*4
    bottom = face[3]*4

    oBBox = (left, top,right,bottom)
    
    #expand bounding box
    oBboxExpanded = expand_bbox(oBBox,oImage)

    oImage = cv2.cvtColor(oImage, cv2.COLOR_BGR2RGB)
    #crop face region
    crop = oImage[oBboxExpanded[1]:oBboxExpanded[3], oBboxExpanded[0]:oBboxExpanded[2]]
    #print(crop.shape)
    #resize crop
    crop = cv2.resize(crop, (100,100))
    crop = np.reshape(crop,(-1,100,100,3))
    #normalize
    crop = crop/255.0
    #predict headpose 
    oHpAngles = oModel.predict(crop)
    #convert radian to degree
    oHpAngles = oHpAngles[0] * 180/np.pi
    
    return oHpAngles,oBboxExpanded


