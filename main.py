import cv2 as opencv
from gesture_api import *


'''Variables and Parameters'''
HSV_Lower_Threshold = 150
Gaussian_K_Size = 11
Gaussian_Sigma = 0
Morph_Element_Size = 13
Median_K_Size = 3
Capture_Box_Count = 9
Capture_Box_Dimension = 20
Capture_Box_Separation_X = 8
Capture_Box_Separation_Y = 18
Capture_Position_X = 500
Capture_Position_Y = 150

'''Starting Point'''
Capture_Region_X_Begin = 0.5

'''Ending Point'''
Capture_Region_Y_End = 0.8
Finger_Threshold_Lower = 2.0
Finger_Threshold_Upper = 3.8

'''Factor of Width of Full Frame'''
Radius_Threshold = 0.04
First_Iteration = True
Finger_Count_History = [0, 0]


'''Function Declarations'''


def hand_region_capture(inframe, box_x, box_y):

    """For Retrieving The Hand Histogram"""

    hsv = opencv.cvtColor(inframe, opencv.COLOR_BGR2HSV)
    roi = numpy.zeros([Capture_Box_Dimension * Capture_Box_Count, Capture_Box_Dimension, 3], dtype=hsv.dtype)

    for a in range(Capture_Box_Count):
        roi[a * Capture_Box_Dimension: a * Capture_Box_Dimension + Capture_Box_Dimension, 0: Capture_Box_Dimension] =\
            hsv[box_y[a]: box_y[a] + Capture_Box_Dimension, box_x[a]: box_x[a] + Capture_Box_Dimension]

    hand_histogram = opencv.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    opencv.normalize(hand_histogram, hand_histogram, 0, 255, opencv.NORM_MINMAX)

    return hand_histogram


def hand_region_threshold(inframe, hand_histogram):

    """For Retrieving The Threshold And Applying Filters"""

    inframe = opencv.medianBlur(inframe, 3)

    hsv = opencv.cvtColor(inframe, opencv.COLOR_BGR2HSV)
    hsv[0: int(Capture_Region_Y_End * hsv.shape[0]), 0: int(Capture_Region_X_Begin * hsv.shape[1])] = 0
    hsv[int(Capture_Region_Y_End * hsv.shape[0]): hsv.shape[0], 0: hsv.shape[1]] = 0

    back_projection = opencv.calcBackProject([hsv], [0, 1], hand_histogram, [0, 180, 0, 256], 1)
    disc = opencv.getStructuringElement(opencv.MORPH_ELLIPSE, (Morph_Element_Size, Morph_Element_Size))

    opencv.filter2D(back_projection, -1, disc, back_projection)
    back_projection = opencv.GaussianBlur(back_projection, (Gaussian_K_Size, Gaussian_K_Size), Gaussian_Sigma)
    back_projection = opencv.medianBlur(back_projection, Median_K_Size)

    return_value, threshold = opencv.threshold(back_projection, HSV_Lower_Threshold, 255, 0)

    return threshold


def find_hand_contours(contours):

    """For Retrieving The Largest Hand Contour"""

    maximum_area = 0
    largest_contour = -1

    for a in range(len(contours)):
        contour = contours[a]
        area = opencv.contourArea(contour)
        if area > maximum_area:
            maximum_area = area
            largest_contour = a

    if largest_contour == -1:
        return False, 0
    else:
        hand_contour = contours[largest_contour]
        return True, hand_contour


def detect_fingers(inframe, hull, point, radius):

    """For Detecting And Marking Fingers"""

    global First_Iteration
    global Finger_Count_History

    finger = [(hull[0][0][0], hull[0][0][1])]
    i = j = 0

    for a in range(len(hull)):
        distance = numpy.sqrt((hull[-a][0][0] - hull[1 - a][0][0]) ** 2 + (hull[-a][0][1] - hull[1 - a][0][1]) ** 2)
        if distance > 18:
            if j == 0:
                finger = [(hull[-a][0][0], hull[-a][0][1])]
            else:
                finger.append((hull[-a][0][0], hull[-a][0][1]))
            j += 1

    temporary_length = len(finger)
    while i < temporary_length:
        distance = numpy.sqrt((finger[i][0] - point[0]) ** 2 + (finger[i][1] - point[1]) ** 2)
        if distance < Finger_Threshold_Lower * radius or distance > Finger_Threshold_Upper * radius or\
                finger[i][1] > point[1] + radius:
            finger.remove((finger[i][0], finger[i][1]))
            temporary_length -= 1
        else:
            i += 1

    if len(finger) > 5:
        for a in range(1, len(finger) - 4):
            finger.remove((finger[len(finger) - a][0], finger[len(finger) - a][1]))

    palm = [(point[0], point[1]), radius]

    if First_Iteration:
        Finger_Count_History[0] = Finger_Count_History[1] = len(finger)
        First_Iteration = False
    else:
        Finger_Count_History[0] = 0.34 * (Finger_Count_History[0] + Finger_Count_History[1] + len(finger))

    if (Finger_Count_History[0] - int(Finger_Count_History[0])) > 0.8:
        finger_count = int(Finger_Count_History[0]) + 1
    else:
        finger_count = int(Finger_Count_History[0])

    Finger_Count_History[1] = len(finger)

    text = "FINGERS: " + str(finger_count)
    opencv.putText(inframe, text, (int(0.62 * inframe.shape[1]), int(0.88 * inframe.shape[0])),
                   opencv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, 8)

    for k in range(len(finger)):
        opencv.circle(inframe, finger[k], 10, 255, 2)
        opencv.line(inframe, finger[k], (point[0], point[1]), 255, 2)

    return inframe, finger, palm


def mark_hand_center(inframe, contour):

    """For Marking The Hand Center Circle"""

    maximum_distance = 0
    point = (0, 0)

    x, y, w, h = opencv.boundingRect(contour)

    for ind_y in range(int(y + 0.3 * h), int(y + 0.8 * h)):
        for ind_x in range(int(x + 0.3 * w), int(x + 0.6 * w)):
            distance = opencv.pointPolygonTest(contour, (ind_x, ind_y), True)
            if distance > maximum_distance:
                maximum_distance = distance
                point = (ind_x, ind_y)
    if maximum_distance > Radius_Threshold * inframe.shape[1]:
        threshold_score = True
        opencv.circle(inframe, point, int(maximum_distance), (255, 0, 0), 2)
    else:
        threshold_score = False

    return inframe, point, maximum_distance, threshold_score


def detect_gesture(inframe, finger, palm):

    """For Detecting and Displaying The Gestures"""

    frame_gesture.set_palm(palm[0], palm[1])
    frame_gesture.set_finger_position(finger)
    frame_gesture.calculate_angles()

    gesture_found = decide_gesture(frame_gesture, gesture_dictionary)
    gesture_text = "GESTURE: " + str(gesture_found)
    opencv.putText(inframe, gesture_text, (int(0.56 * inframe.shape[1]), int(0.97 * inframe.shape[0])),
                   opencv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, 8)

    return inframe, gesture_found


def remove_background(frame):

    """For Removing The Background From The Original Image"""

    foreground_mask = background_model.apply(frame)
    kernel = numpy.ones((3, 3), numpy.uint8)
    foreground_mask = opencv.erode(foreground_mask, kernel, iterations=1)
    frame = opencv.bitwise_and(frame, frame, mask=foreground_mask)

    return frame


image = opencv.VideoCapture(0)
capture_completed = 0
background_captured = 0
gesture_dictionary = define_gestures()
frame_gesture = Gesture("frame_gesture")

while (1):
    # Capture frame from camera
    return_value, frame = image.read()
    frame = opencv.bilateralFilter(frame, 5, 50, 100)
    # Operations on the frame
    frame = opencv.flip(frame, 1)
    opencv.rectangle(frame, (int(Capture_Region_X_Begin * frame.shape[1]), 0),
                  (frame.shape[1], int(Capture_Region_Y_End * frame.shape[0])), (255, 0, 0), 1)
    frame_original = numpy.copy(frame)

    if (background_captured):
        foreground_frame = remove_background(frame)

    if not (capture_completed and background_captured):
        if not background_captured:
            opencv.putText(frame, "Remove hand from the frame and press 'b' to capture background",
                        (int(0.05 * frame.shape[1]), int(0.97 * frame.shape[0])), opencv.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, 8)
        else:
            opencv.putText(frame, "Place hand inside boxes and press 'c' to capture hand histogram",
                        (int(0.08 * frame.shape[1]), int(0.97 * frame.shape[0])), opencv.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, 8)

        first_iteration = True
        finger_count_history = [0, 0]
        box_position_x = numpy.array([Capture_Position_X, Capture_Position_X + Capture_Box_Dimension + Capture_Box_Separation_X,
                              Capture_Position_X + 2 * Capture_Box_Dimension + 2 * Capture_Box_Separation_X, Capture_Position_X,
                              Capture_Position_X + Capture_Box_Dimension + Capture_Box_Separation_X,
                              Capture_Position_X + 2 * Capture_Box_Dimension + 2 * Capture_Box_Separation_X, Capture_Position_X,
                              Capture_Position_X + Capture_Box_Dimension + Capture_Box_Separation_X,
                              Capture_Position_X + 2 * Capture_Box_Dimension + 2 * Capture_Box_Separation_X], dtype=int)
        box_position_y = numpy.array(
            [Capture_Position_Y, Capture_Position_Y, Capture_Position_Y, Capture_Position_Y + Capture_Box_Dimension + Capture_Box_Separation_Y,
             Capture_Position_Y + Capture_Box_Dimension + Capture_Box_Separation_Y, Capture_Position_Y + Capture_Box_Dimension + Capture_Box_Separation_Y,
             Capture_Position_Y + 2 * Capture_Box_Dimension + 2 * Capture_Box_Separation_Y,
             Capture_Position_Y + 2 * Capture_Box_Dimension + 2 * Capture_Box_Separation_Y,
             Capture_Position_Y + 2 * Capture_Box_Dimension + 2 * Capture_Box_Separation_Y], dtype=int)

        for i in range(Capture_Box_Count):
            opencv.rectangle(frame, (box_position_x[i], box_position_y[i]),
                          (box_position_x[i] + Capture_Box_Dimension, box_position_y[i] + Capture_Box_Dimension), (255, 0, 0), 1)
    else:
        frame = hand_region_threshold(foreground_frame, hand_histogram)
        contour_frame = numpy.copy(frame)
        _, contours, hierarchy = opencv.findContours(contour_frame, opencv.RETR_TREE, opencv.CHAIN_APPROX_SIMPLE)
        found, hand_contour = find_hand_contours(contours)
        if found:
            hand_convex_hull = opencv.convexHull(hand_contour)
            frame, hand_center, hand_radius, hand_size_score = mark_hand_center(frame_original, hand_contour)
            if hand_size_score:
                frame, finger, palm = detect_fingers(frame, hand_convex_hull, hand_center, hand_radius)
                frame, gesture_found = detect_gesture(frame, finger, palm)
        else:
            frame = frame_original

    # Display frame in a window
    opencv.imshow('Gestures Alive!', frame)
    interrupt = opencv.waitKey(10)

    # Quit by pressing 'q'
    if interrupt & 0xFF == ord('q'):
        break
    # Capture hand by pressing 'c'
    elif interrupt & 0xFF == ord('c'):
        if background_captured:
            capture_completed = 1
            hand_histogram = hand_region_capture(frame_original, box_position_x, box_position_y)
    # Capture background by pressing 'b'
    elif interrupt & 0xFF == ord('b'):
        background_model = opencv.createBackgroundSubtractorMOG2()
        background_captured = 1
    # Reset captured hand by pressing 'r'
    elif interrupt & 0xFF == ord('r'):
        capture_completed = 0
        background_captured = 0

# Release camera & end program
image.release()
opencv.destroyAllWindows()
