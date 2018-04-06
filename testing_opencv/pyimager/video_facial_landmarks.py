# import the necessary packages
from imutils.video import VideoStream
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
from graphics import *
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
                help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()

time.sleep(2.0)
blank_image = np.ones((700, 700, 3), np.uint8)

s_img = cv2.imread("panda-face.png", -1)
orig_mask = s_img[:,:,3]
orig_mask_inv = cv2.bitwise_not(orig_mask)
img = s_img[:,:,0:3]
origHeight, origWidth = s_img.shape[:2]

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    blank_image.fill(255)
    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.circle(blank_image, (x, y), 1, (0, 0, 255), -1)
    #alpha = 1
    #cv2.addWeighted(s_img, alpha, blank_image, 1 - alpha,
    #                0, blank_image)
    # show the frame
            #print('shapesize', len(shape))
            x1, x2 = shape[2][0], shape[14][0]
            y1, y2 = shape[20][1], shape[8][1]
            x1 -= 25
            x2 += 25
            y1 -= 30
            y2 += 30
            face_height = y2 - y1
            face_width = x2 - x1
            #print(face_height, face_width)


        image = cv2.resize(img, (face_width, face_height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(orig_mask, (face_width, face_height), interpolation=cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask_inv, (face_width, face_height), interpolation=cv2.INTER_AREA)

        roi = blank_image[y1:y2, x1:x2]
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        roi_fg = cv2.bitwise_and(image, image, mask=mask)
        dst = cv2.add(roi_bg, roi_fg)
        blank_image[y1:y2, x1:x2] = dst



    #cv2.imshow("Frame", frame)
    cv2.imshow("Frame2", blank_image)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()