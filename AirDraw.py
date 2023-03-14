import numpy as np
import cv2
from collections import deque


def color_recognition():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width, _ = frame.shape

        cx = int(width / 2)
        cy = int(height / 2)

        pixel_center = hsv_frame[cy, cx]
        hue_value = pixel_center[0]

        color = "Undefined"
        if hue_value:
            color = 'Color Detected'

        pixel_center_bgr = frame[cy, cx]
        b, g, r = int(pixel_center_bgr[0]), int(pixel_center_bgr[1]), int(pixel_center_bgr[2])

        cv2.rectangle(frame, (cx - 220, 10), (cx + 200, 120), (255, 255, 255), -1)
        cv2.putText(frame, color, (cx - 200, 100), 0, 1.6, (b, g, r), 5)
        cv2.circle(frame, (cx, cy), 5, (25, 25, 25), 3)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("s"):
            break

    capture.release()
    cv2.destroyAllWindows()

    return (hue_value)
hue_value=color_recognition()

colorLower=np.array([hue_value-10, 100, 100])
colorUpper=np.array([hue_value+10, 255, 255])

kernel = np.ones((5, 5), np.uint8)

bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

bindex = 0
gindex = 0
rindex = 0
yindex = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

paintWindow = np.zeros((720,1280,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (10,10), (110,60), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (10,80), (110,130),  colors[0], -1)
paintWindow = cv2.rectangle(paintWindow, (10,150), (110,200), colors[1], -1)
paintWindow = cv2.rectangle(paintWindow, (10,220), (110,270), colors[2], -1)
paintWindow = cv2.rectangle(paintWindow, (10,290), (110,340), colors[3], -1)
cv2.putText(paintWindow, "CLEAR ALL", (25, 30), 0, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (25, 100), 0, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (25, 170), 0, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (25, 240), 0, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (25, 310), 0, 0.5, (255,255,255), 2, cv2.LINE_AA)

cv2.namedWindow('Paint Window', cv2.WINDOW_AUTOSIZE)

camera = cv2.VideoCapture(0)

while True:
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    frame = cv2.rectangle(frame, (10,10), (110,60), (122,122,122), -1)
    frame = cv2.rectangle(frame, (10,80), (110,130), colors[0], -1)
    frame = cv2.rectangle(frame, (10,150), (110,200), colors[1], -1)
    frame = cv2.rectangle(frame, (10,220), (110,270), colors[2], -1)
    frame = cv2.rectangle(frame, (10,290), (110,340), colors[3], -1)
    cv2.putText(frame, "CLEAR ALL", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (25, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (25, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (25, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 2, cv2.LINE_AA)

    if not grabbed:
        break

    imageMask = cv2.inRange(hsv, colorLower, colorUpper)
    # mask1 = np.zeros((720, 1280, 3)) + 255
    # masked = cv2.bitwise_and(frame, frame, mask=blueMask)

    # cv2.imshow(masked)
    imageMask = cv2.erode(imageMask, kernel, iterations=2)
    imageMask = cv2.morphologyEx(imageMask, cv2.MORPH_OPEN, kernel)
    imageMask = cv2.dilate(imageMask, kernel, iterations=1)

    (cnts, _) = cv2.findContours(imageMask.copy(), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    center = None
    contour_frame = np.zeros(frame.shape, np.uint8)
    if len(cnts) > 0:
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

        cv2.drawContours(contour_frame, [cnt], 0, (0, 255, 0), 3)
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if center[0] <= 110:
            if 10 <= center[1] <= 60:
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                bindex = 0
                gindex = 0
                rindex = 0
                yindex = 0

                paintWindow[:,112:,:] = 255
            elif 80 <= center[1] <= 130:
                    colorIndex = 0 # Blue
            elif 150 <= center[1] <= 200:
                    colorIndex = 1 # Green
            elif 220 <= center[1] <= 270:
                    colorIndex = 2 # Red
            elif 290 <= center[1] <= 340:
                    colorIndex = 3 # Yellow
        else :
            if colorIndex == 0:
                bpoints[bindex].appendleft(center)
            elif colorIndex == 1:
                gpoints[gindex].appendleft(center)
            elif colorIndex == 2:
                rpoints[rindex].appendleft(center)
            elif colorIndex == 3:
                ypoints[yindex].appendleft(center)
    else:
        bpoints.append(deque(maxlen=512))
        bindex += 1
        gpoints.append(deque(maxlen=512))
        gindex += 1
        rpoints.append(deque(maxlen=512))
        rindex += 1
        ypoints.append(deque(maxlen=512))
        yindex += 1

    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("User Window", frame)
    cv2.imshow("Paint Window", paintWindow)
    # optional code, uncomment below to see contour of the tool detected
   # cv2.imshow("contour Window", contour_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cropped_image = paintWindow[:, 112:]  # Slicing to crop the image

cv2.imwrite('saved_drawing.jpg', cropped_image)

camera.release()
cv2.destroyAllWindows()
