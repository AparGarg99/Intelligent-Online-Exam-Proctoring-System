########################################### Import Libraries #############################################
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

########################################## Define variables ###############################################
cap = cv2.VideoCapture(0)

# find 1 face in frame
detector = FaceMeshDetector(maxFaces=1)

#################################### Finding the Focal Length of camera ###################################
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

        # d = distance of object from camera in cm
        # we have to sit 50 cm away from camera. use inch tape or scale for this.
        d = 50

        # calculate focal length
        f = (w*d)/W
        print(f)


    cv2.imshow("Image", img)
    cv2.waitKey(1)