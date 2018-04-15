from imutils.video import WebcamVideoStream
from imutils import face_utils
import imutils
import dlib
import cv2
import time
import numpy as np
from PIL import ImageEnhance, Image


def enhance(frame):

    # enhance sharpness, color, brightness and contrast of each frame
    frame = Image.fromarray(frame)

    enhancer = ImageEnhance.Sharpness(frame)
    frame = enhancer.enhance(1.2)

    enhancer = ImageEnhance.Color(frame)
    frame = enhancer.enhance(1.2)

    enhancer = ImageEnhance.Brightness(frame)
    frame = enhancer.enhance(2.0)

    enhancer = ImageEnhance.Contrast(frame)
    frame = enhancer.enhance(1.3)

    frame = imutils.resize(np.array(frame), width=600)
    return frame


def draw(im_floodfill, parameters, blank_image, mask):

    # initializing colors for different face parts
    border_thickness = 2
    border_color = (1, 1, 1)
    ear_color = (96, 96, 96)

    # unbinding all coordinates
    (x1, x2, y1, y2, face_height, face_width, degree_of_rotaion, fc_x, fc_y, m_x, m_y, temp_x1, temp_x2, temp_y1,
     temp_y2, left_eye_height, nose1_x, nose1_y, nose2_x, nose2_y, left_eyeball_height,
     right_eyeball_height) = parameters

    # drawing rightear
    cv2.ellipse(im_floodfill, (fc_x + 95, fc_y - 105), (40, 55), 35, 15, -175, ear_color, -4)
    cv2.ellipse(im_floodfill, (fc_x + 95, fc_y - 105), (45, 60), 35, 15, -175, border_color, border_thickness + 2)

    # drawing leftear
    cv2.ellipse(im_floodfill, (fc_x - 96, fc_y - 105), (40, 55), -45, 3, -187, ear_color, -4)
    cv2.ellipse(im_floodfill, (fc_x - 96, fc_y - 105), (45, 60), -45, 3, -187, border_color, border_thickness + 2)

    # drawing upper face
    cv2.ellipse(im_floodfill, (fc_x, fc_y), (135, 150), 0, -155, -25, border_color, border_thickness)
    cv2.ellipse(im_floodfill, (fc_x, fc_y), (150, 110), 0, -145, -230, border_color, border_thickness)
    cv2.ellipse(im_floodfill, (fc_x, fc_y), (150, 110), 0, -35, 50, border_color, border_thickness)

    # drawing jaws
    cv2.ellipse(im_floodfill, (fc_x - 51 - (m_x // 4), fc_y + 95), (39 - int(m_x / 8), 2), 12 + int(m_x // 8), 0,
                180, border_color, border_thickness)
    cv2.ellipse(im_floodfill, (fc_x + 51 + (m_x // 4), fc_y + 95), (39 - int(m_x / 8), 2), -12 - int(m_x // 8), 0,
                180, border_color, border_thickness)

    # Filling white color inside face
    cv2.floodFill(im_floodfill, mask, (fc_x, fc_y), (255, 255, 255))

    # drawing boundaries for both eye
    cv2.ellipse(im_floodfill, (fc_x - 33, fc_y + 15), (30, 20 + left_eye_height), 10, 0, -180, border_color,
                -border_thickness)
    cv2.ellipse(im_floodfill, (fc_x - 33, fc_y + 15), (30, 45), 10, 0, 180, border_color, -border_thickness)
    cv2.ellipse(im_floodfill, (fc_x + 33, fc_y + 15), (30, 45), -10, 0, 180, border_color, -border_thickness)
    cv2.ellipse(im_floodfill, (fc_x + 33, fc_y + 15), (30, 20 + left_eye_height), -10, 0, -180, border_color,
                -border_thickness)

    # drawing eyeballs
    cv2.ellipse(im_floodfill, (fc_x - 33, fc_y + 15), (6, left_eyeball_height), 0, 0, 360, (255, 255, 255),
                -border_thickness)

    cv2.ellipse(im_floodfill, (fc_x + 33, fc_y + 15), (6, right_eyeball_height), 0, 0, 360, (255, 255, 255),
                -border_thickness)

    # drawing lips
    cv2.ellipse(im_floodfill, (fc_x, fc_y + 102), (int(m_x / 1.8), 3 + int(m_y // 5)), 0, 0, -180, border_color,
                border_thickness)
    cv2.ellipse(im_floodfill, (fc_x, fc_y + 102), (int(m_x / 1.8), 12 + int(m_y // 1.5)), 0, 0, 180, border_color,
                border_thickness)
    cv2.ellipse(im_floodfill, (fc_x, fc_y + 102), (int(m_x / 1.8), 3 + int(m_y // 1.5)), 0, 0, 180, border_color,
                -border_thickness)

    # drawing nose
    cv2.ellipse(im_floodfill, (fc_x, fc_y + 58), (12, 3), 0, 20, -160, border_color, -border_thickness)
    cv2.ellipse(im_floodfill, (fc_x - 6, fc_y + 60), (15, 3), 60, 0, 105, border_color, -border_thickness)
    cv2.ellipse(im_floodfill, (fc_x + 8, fc_y + 60), (15, 3), -60, 75, 155, border_color, -border_thickness)

    # drawing boundaries of nose
    cv2.ellipse(im_floodfill, (nose1_x + 20, nose1_y + 16), (30 + int(m_x / 8), 32), -30 + (int(m_x / 2)), 140, 240,
                border_color, border_thickness)
    cv2.ellipse(im_floodfill, (nose2_x - 20, nose2_y + 16), (30 + int(m_x / 8), 32), 30 - (int(m_x / 2)), -60, 40,
                border_color, border_thickness)

    # cropping panda
    im_floodfill = cv2.resize(im_floodfill, (face_width, face_height), interpolation=cv2.INTER_AREA)

    # rotating the frame
    dst = imutils.rotate(im_floodfill, degree_of_rotaion)

    # inserting rotated frame onto the blank one
    blank_image[y1:y2, x1:x2] += dst

    # using padding to remove additional black lines
    temp2 = blank_image[temp_y1:temp_y2, temp_x1:temp_x2]
    blank_image[y1:y2, x1:x2] = np.pad(temp2, ((33, 33), (33, 33), (0, 0)), mode='constant', constant_values=255)

    return blank_image


def calculate(shape, im_floodfill, mask, blank_image):

    # extract coordinates for calculating size of face
    x1, x2 = shape[2][0], shape[14][0]
    y1, y2 = shape[20][1], shape[8][1]

    # appended coordinates required for padding function.
    temp_x1, temp_y1 = x1 - 47, y1 - 47
    temp_x2, temp_y2 = x2 + 47, y2 + 47

    # appended coordinates to scale the face
    x1 -= 80
    x2 += 80
    y1 -= 80
    y2 += 80
    face_height = (y2 - y1)
    face_width = (x2 - x1)

    # calculating the degree of rotation
    arc_of_rotation = -((shape[14][1] - shape[3][1]) / (shape[14][0] - shape[3][0]))
    degree_of_rotaion = (np.degrees(np.arctan(arc_of_rotation)) - 3) / 1.5

    # center coordinates of face
    fc_x = int(abs(face_width / 2) + 200)
    fc_y = int(abs(face_width / 2) + 150)

    # length of mouth
    m_x = abs(shape[54][0] - shape[48][0])
    m_y = abs(shape[66][1] - shape[62][1])

    # height of boundaries of eyes
    left_eye_height = int(shape[36][1] - shape[19][1])
    left_eye_height -= int((face_height / 650) * left_eye_height)

    # height of each eyeball
    left_eyeball_height = int(1000 * ((shape[41][1] - shape[37][1]) / face_height) / 2)
    right_eyeball_height = int(1000 * ((shape[47][1] - shape[43][1]) / face_height) / 2)

    # coordinates for boundaries of nose
    nose1_x = (fc_x - 20 + fc_x - int(m_x / 2)) // 2
    nose1_y = (fc_y + 25 + fc_y + 110) // 2
    nose2_x = (fc_x + 20 + fc_x + int(m_x / 2)) // 2
    nose2_y = (fc_y + 25 + fc_y + 110) // 2

    # binding all coordinated in a tuple
    parameters = (x1, x2, y1, y2, face_height, face_width, degree_of_rotaion, fc_x, fc_y, m_x, m_y, temp_x1, temp_x2,
                  temp_y1, temp_y2, left_eye_height, nose1_x, nose1_y, nose2_x, nose2_y, left_eyeball_height,
                  right_eyeball_height)

    # calling function to draw
    return draw(im_floodfill, parameters, blank_image, mask)


def start():

    # initialize dlib's face detector
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] camera sensor warming up...")
    vs = WebcamVideoStream(src=0).start()

    time.sleep(1.0)

    # finale frame and frame on which panda will be drawn initially
    blank_image = np.ones((650, 650, 3), np.uint8)*255
    drawing = np.ones((650, 650, 3), np.uint8)*255

    # array to store previous landmarks(used in removing distortion)
    prev_shape = []

    # loop over the frames from the video stream
    while True:
        try:
            # reading frame from webcam and resizing it
            frame = vs.read()
            frame = cv2.flip(frame, 1)
            frame = imutils.resize(frame, width=600)

            # Calling function to enhance the frame(for better landmark detection)
            frame = enhance(frame)

            # converting frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect faces in frame
            rects = detector(gray, 0)

            # clear previous frame
            blank_image.fill(255)
            drawing.fill(255)

            # distortion handling (averaging last three frames)
            if rects:
                if len(prev_shape) == 3:
                    del (prev_shape[0])

                # predicting facial landmarks for the face, then converting the (x, y)-coordinates to a NumPy array
                shape = predictor(gray, rects[0])
                shape = imutils.face_utils.shape_to_np(shape)

                prev_shape.append(shape)
            shape = sum(prev_shape) // len(prev_shape)

            # setup to use flood_fill algorithm
            th, im_th = cv2.threshold(drawing, 220, 255, cv2.THRESH_BINARY)
            im_floodfill = im_th.copy()

            # Mask used to flood filling.Notice the size needs to be 2 pixels than the image.(255, 255, 255)
            h, w = im_th.shape[:2]
            mask = np.ones((h + 2, w + 2), np.uint8) * 255

            # Time to unleash the main code. Calling function calculate.
            blank_image = calculate(shape, im_floodfill, mask, blank_image)

            # displaying panda
            cv2.imshow("Animoji", blank_image)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        except:
            pass

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    start()
