import cv2
import numpy


capture = cv2.VideoCapture(0)

while (capture.isOpened()):
    rectangle, image = capture.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    rectangle, primary_threshold = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    primary_threshold, contours, hierarchy = cv2.findContours(primary_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = numpy.zeros(image.shape, numpy.uint8)

    maximum_area = 0

    for i in range(len(contours)):
        count = contours[i]
        area = cv2.contourArea(count)
        if (area > maximum_area):
            maximum_area = area
            contour_index = i

    count = contours[contour_index]
    hull = cv2.convexHull(count)
    moments = cv2.moments(count)

    if moments['m00'] != 0:
        cx = int(moments['m10']/moments['m00'])
        cy = int(moments['m01']/moments['m00'])

    center = (cx, cy)
    cv2.circle(image, center, 5, [0, 0, 255], 2)
    cv2.drawContours(drawing, [count], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing, [hull], 0, (0,  0, 255), 2)

    count = cv2.approxPolyDP(count, 0.01 * cv2.arcLength(count, True), True)
    hull = cv2.convexHull(count, returnPoints = False)

    if (True):
        defects = cv2.convexityDefects(count, hull)
        minimum_distance = 0
        maximum_distance = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(count[s][0])
            end = tuple(count[e][0])
            far = tuple(count[f][0])
            distance = cv2.pointPolygonTest(count, center, True)
            cv2.line(image, start, end, [0, 255, 0], 2)
            cv2.circle(image, far, 5, [0, 0, 255], -1)
        print(i)
        i = 0

    #cv2.imshow('output', drawing)
    #cv2.imshow('input', image)

    k = cv2.waitKey(10)
    if k == 27:
        break