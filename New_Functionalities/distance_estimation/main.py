########################################### Import Libraries #############################################
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

########################################## Define variables ###############################################
cap = cv2.VideoCapture(0)

# find 1 face in frame
detector = FaceMeshDetector(maxFaces=1)

#################################### Finding the depth of face(object) from camera ###################################
while True:
    # capture frame
    success, img = cap.read()

    # return image with face mesh
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        # get first face coordinates
        face = faces[0]

        # get left eye coordinates 
        pointLeft = face[145]

        # get right eye coordinates 
        pointRight = face[374] 

        # w = width of object in camera in pixels (dynamic value. will change with distance)
        # distance in pixels between eyes
        w, _ = detector.findDistance(pointLeft, pointRight)

        # W = object width in cm (constant value. won't change with distance)
        # actual distance between eyes for men = 64 mm & women = 62 mm. So we take avg which is 63 mm.
        W = 6.3

        # focal length of camera (constant value)
        f = 840

        # Finding depth
        d = (W * f) / w

        # display depth on screen
        # face[10] = forehead coordinates
        cvzone.putTextRect(img, f'Depth: {int(d)}cm',
                           (face[10][0] - 100, face[10][1] - 50),
                           scale=2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)