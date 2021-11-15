################################################ Import Libraries  ##########################################
import cv2
import sys
import os
import matplotlib
import numpy as np
from collections import Counter
import face_recognition
import dlib
from math import hypot

from object_detection import yoloV3Detect
from landmark_models import *
from face_spoofing import *
from headpose_estimation import *
from face_detection import get_face_detector, find_faces

################################################ Setup  ######################################################

# face recognition
l = os.listdir('student_db')
known_face_encodings = []
known_face_names = []
face_locations = []
face_encodings = []
face_names = []

for image in l:
    obama_image = face_recognition.load_image_file('student_db/'+image)
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    known_face_encodings.append(obama_face_encoding)
    known_face_names.append(image.split('.')[0])


# headpose model
h_model = load_hp_model('models/Headpose_customARC_ZoomShiftNoise.hdf5')

# face detection model
face_model = get_face_detector()

# face landmark model
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Others
video_capture = cv2.VideoCapture(0)
process_this_frame = False
no_of_frames_0 = 0
no_of_frames_1 = 0
no_of_frames_2 = 0
no_of_frames_3 = 0
no_of_frames_4 = 0
no_of_frames_5 = 0
no_of_frames_6 = 0
no_of_frames_7 = 0
font = cv2.FONT_HERSHEY_PLAIN
flag = True

#################################################### ALERT #####################################################
def alert(condition,no_of_frames):
    if(condition):
        no_of_frames=no_of_frames+1
    else:
        no_of_frames=0

    return no_of_frames

#################################################### MAIN #####################################################

while True:
    # frame skipping to save time
    process_this_frame = not process_this_frame 

    # Grab a single frame of video
    ret, frame = video_capture.read()

    frame2 = frame.copy()
    frame3 = frame.copy()
    report = np.zeros((frame3.shape[0],frame3.shape[1], 3), np.uint8)
  
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Functionalities
    if process_this_frame:
        try:
            ##### Object Detection #####
            try:
                fboxes,fclasses=yoloV3Detect(small_frame)
            
                
                to_detect=['person','laptop','cell phone','book','tv']

                temp1,temp2=[],[]

                for i in range(len(fclasses)):
                    if(fclasses[i] in to_detect):
                        temp1.append(fboxes[i])
                        temp2.append(fclasses[i])
             

                # Conter
                count_items = Counter(temp2)
            except Exception as e:
                count_items = {}
                count_items['person'] = 0
                count_items['laptop'] = 0
                count_items['cell phone'] = 0
                count_items['book'] = 0
                count_items['tv'] = 0
                print(e)

            # Multiple Person Buffer
            condition = (count_items['person']!=1)
            no_of_frames_0 = alert(condition,no_of_frames_0)

            y_pos = 20
            alert_pos = (120,190)

            # Display
            cv2.putText(report, "Number of people detected: "+str(count_items['person']), (1, y_pos), font, 1.1, (0, 255, 0), 2)

            # Alert
            if(no_of_frames_0>10):
                cv2.putText(report, "Number of people detected: "+str(count_items['person']), (1, y_pos), font, 1.1, (0, 0, 255), 2)
                cv2.putText(report, "ALERT", alert_pos, font, 4, (0, 0, 255), 2)
            
            # Object Detection Buffer
            condition = (count_items['laptop']>=1 or 
                        count_items['cell phone']>=1 or 
                        count_items['book']>=1 or 
                        count_items['tv']>=1)
         
            no_of_frames_1 = alert(condition,no_of_frames_1)

            # Display
            cv2.putText(report, "Banned objects detected: "+str(condition), (1, y_pos+20), font, 1.1, (0, 255, 0), 2)

            # Alert
            if(no_of_frames_1>10):
                cv2.putText(report, "Banned objects detected: "+str(condition), (1, y_pos+20), font, 1.1, (0, 0, 255), 2)
                cv2.putText(report, "ALERT", alert_pos, font, 4, (0, 0, 255), 2)
                

            if(count_items['person']==1):

                #### face detection using caffe model of OpenCV's DNN module ####
                
                # detect face
                faces = find_faces(small_frame, face_model)
                if len(faces) >0:
                    face = faces[0]
                else:
                    condition = (len(faces) < 1)
                    no_of_frames_7 = alert(condition,no_of_frames_7)
                    y_pos = 60
                    alert_pos = (120,190)

                    # Display
                    cv2.putText(report, "Number of face detected: "+str(len(faces)), (1, y_pos), font, 1.1, (0, 255, 0), 2)
                    # Alert
                    if(no_of_frames_7>10):
                        cv2.putText(report, "Number of face detected: "+str(len(faces)), (1, y_pos), font, 1.1, (0, 0, 255), 2)
                        cv2.putText(report, "ALERT", alert_pos, font, 4, (0, 0, 255), 2)        
                    
                    horizontalAppendedImg = np.hstack((frame3,report))
                    
                    cv2.imshow("Proctoring_Window", horizontalAppendedImg)
                    continue
                
                # Display Face Detection
                (left, top,right,bottom) = face
                cv2.rectangle(frame3, (left*4, top*4), (right*4, bottom*4), (0, 0, 255), 2)
               

                if(flag==True):
                    #### face verification using face_recognition library ####
                    
                    # modifying order
                    face_locations = [[top, right, bottom, left]]
                   
                    # Convert BGR image to RGB image (which  uses)
                    rgb_small_frame = small_frame[:, :, ::-1]

                    # get CNN feature vector
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    # get similarity
                    face_encoding = face_encodings[0]
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    else:
                        name = "Unknown"
                    flag = False
                
                # Buffer
                condition = (name=='Unknown')  
                no_of_frames_2 = alert(condition,no_of_frames_2)

                # Display
                cv2.putText(report, "Face Recognized: "+str(name), (1, y_pos+40), font, 1.1, (0, 255, 0), 2)


                # Alert
                if(no_of_frames_2>10):
                    cv2.putText(report, "Face Recognized: "+str(name), (1, y_pos+40), font, 1.1, (0, 0, 255), 2)
                    cv2.putText(report, "ALERT", alert_pos, font, 4, (0, 0, 255), 2)
                   

                #### mouth movement ####

                # get landmarks
                left = face[0]*4
                top = face[1]*4
                right = face[2]*4
                bottom = face[3]*4
                face_dlib = dlib.rectangle(left, top, right, bottom)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                facial_landmarks = predictor(gray, face_dlib)

                mouth_ratio = get_mouth_ratio([60,62,64,66], frame2,facial_landmarks)
                
                # Buffer
                condition = (mouth_ratio>0.1)
                no_of_frames_4 = alert(condition,no_of_frames_4)

                # Display
                cv2.putText(report, "Mouth Open: "+str(condition), (1, y_pos+80), font, 1.1, (0, 255, 0), 2)

                # Alert
                if(no_of_frames_4>10):
                    cv2.putText(report, "Mouth Open: "+str(condition), (1, y_pos+80), font, 1.1, (0, 0, 255), 2)
                    cv2.putText(report, "ALERT", alert_pos, font, 4, (0, 0, 255), 2)

                #### head pose ####
                oAnglesNp,oBboxExpanded = headpose_inference(h_model, frame2, face)

                # Display (head angle)
                frame3 = displayHeadpose(frame3, oAnglesNp,oOffset = 0)
                
                
                # Buffer
                condition1=(round(oAnglesNp[0],1) not in [0.0,-1.0,-1.1,-1.2,-1.3,-1.4,-1.5,-1.6,-1.7] and 
                            round(oAnglesNp[1],0) not in [0.0,1.0,2.0,3.0,4.0,5.0])
                no_of_frames_5 = alert(condition1,no_of_frames_5)


                # Display
                if(condition1):
                    cv2.putText(report, "Head Pose: Looking away from screen", (1, y_pos+100), font, 1.1, (0, 255, 0), 2)
                else:
                    cv2.putText(report, "Head Pose: Looking at screen", (1, y_pos+100), font, 1.1, (0, 255, 0), 2)

                # Alert
                if(no_of_frames_5>10):
                    cv2.putText(report, "Head Pose: Looking away from screen", (1, y_pos+100), font, 1.1, (0, 0, 255), 2)
                    cv2.putText(report, "ALERT", alert_pos, font, 4, (0, 0, 255), 2)
            

                ##### Blinking (to support down eye tracking) ######

                left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], frame2,facial_landmarks)
                right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], frame2,facial_landmarks)
                blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                
                ##### eye tracker #####

                gaze_ratio1_left_eye, gaze_ratio2_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], frame2,facial_landmarks)

                gaze_ratio1_right_eye, gaze_ratio2_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], frame2,facial_landmarks)

                # Left/Right
                new_frame1 = np.zeros((500, 500, 3), np.uint8)
                gaze_ratio1 = (gaze_ratio1_right_eye + gaze_ratio1_left_eye) / 2
                    
                if gaze_ratio1 <= 0.35:
                    new_frame1[:] = (0, 0, 255) # red, right
                elif 0.35 < gaze_ratio1 < 4:
                    new_frame1[:] = (0, 0, 0) #black, center
                else:
                    new_frame1[:] = (255, 0, 0) # blue, left

                # Up/Down
                new_frame2 = np.zeros((500, 500, 3), np.uint8)


                # Buffer
                condition = (gaze_ratio1 <= 0.35 or gaze_ratio1>=4 or condition1==True)
                no_of_frames_3 = alert(condition,no_of_frames_3)

                # Display
                if(condition):
                    cv2.putText(report, "Eye Tracking: Looking away from screen", (1, y_pos+60), font, 1.1, (0, 255, 0), 2)
                else:
                    cv2.putText(report, "Eye Tracking: Looking at screen", (1, y_pos+60), font, 1.1, (0, 255, 0), 2)

                # Alert
                if(no_of_frames_3>10):
                    cv2.putText(report, "Eye Tracking: Looking away from screen", (1, y_pos+60), font, 1.1, (0, 0, 255), 2)
                    cv2.putText(report, "ALERT", alert_pos, font, 4, (0, 0, 255), 2)
                    

                #### face spoofing ####
                measures = face_spoof(frame2,face)

                # Buffer
                condition = (np.mean(measures) < 0.7)
                no_of_frames_6 = alert(condition,no_of_frames_6)

                # Display
                cv2.putText(report, "Spoof Face detected: "+str(condition), (1, y_pos+120), font, 1.1, (0, 255, 0), 2)
                
                # Alert
                if(no_of_frames_6>10):
                    cv2.putText(report, "Spoof Face detected: "+str(condition), (1, y_pos+120), font, 1.1, (0, 0, 255), 2)
                    cv2.putText(report, "ALERT", alert_pos, font, 4, (0, 0, 255), 2)
            

            else:
                flag = True

   
            horizontalAppendedImg = np.hstack((frame3,report))
            cv2.imshow("Proctoring_Window", horizontalAppendedImg)

        except Exception as e:
            print(e) 
            flag = True
            report = np.zeros((frame3.shape[0],frame3.shape[1], 3), np.uint8)

            #final display frame
            horizontalAppendedImg = np.hstack((frame3,report))
            cv2.imshow("Proctoring_Window", horizontalAppendedImg)
            

    # Display the resulting image
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("closing window...")
        break
    
    
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
