from functions import *


image = opencv.VideoCapture(0)
capture_completed = 0
background_captured = 0

while (1):
    ''' Capture Frames From Camera '''
    return_value, frame = image.read()
    frame = opencv.bilateralFilter(frame, 5, 50, 100)
    ''' Operations On The Frame '''
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

    ''' Display Frames In A Window '''
    opencv.imshow('Gestures Alive!', frame)
    interrupt = opencv.waitKey(1)

    ''' Quit by pressing 'q' '''
    if interrupt & 0xFF == ord('q'):
        break
    elif interrupt & 0xFF == ord('c'):
        if background_captured:
            capture_completed = 1
            hand_histogram = hand_region_capture(frame_original, box_position_x, box_position_y)
    elif interrupt & 0xFF == ord('b'):
        background_model = opencv.createBackgroundSubtractorMOG2()
        background_captured = 1
    elif interrupt & 0xFF == ord('r'):
        capture_completed = 0
        background_captured = 0

# Release camera & end program
image.release()
opencv.destroyAllWindows()
