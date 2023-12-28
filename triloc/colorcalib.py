"""Simple script for choosing the color of the ball for future detection.

Opens a window of the video stream with scroll bars for sensitivity. Clicking at
a location saves its color. Changing the HSV sliders changes the target color.
The mask filters out all pixels that are at most delta_HSV away from the target
color.
"""

import cv2 as cv
import numpy as np


def nothing(_):
    pass


def set_target(event, x, y, flags, param):
    global input_window, h_name, s_name, v_name, frame_hsv
    if event == cv.EVENT_LBUTTONUP:
        # need to scale coordinates in case the frame is not the same dimensions
        # as the image in the window
        _, _, w, h = cv.getWindowImageRect(input_window)

        frame_x = frame_hsv.shape[1]
        frame_y = frame_hsv.shape[0]

        x_scaled = round(frame_x * (x / w))
        y_scaled = round(frame_y * (y / h))

        # print(f"{x_scaled=} {y_scaled=} {x=} {y=} {w=} {h=}")
        # print(f"{frame_hsv.shape}")
        # print()

        cv.setTrackbarPos(h_name, input_window, frame_hsv[y_scaled, x_scaled, 0])
        cv.setTrackbarPos(s_name, input_window, frame_hsv[y_scaled, x_scaled, 1])
        cv.setTrackbarPos(v_name, input_window, frame_hsv[y_scaled, x_scaled, 2])


input_window = "Color Calibration Input"
output_window = "Color Calibration Output"

min_value = 0
max_value = 255

# Names and values for the 6 trackbars
h_name = "H"
h_value = 0

s_name = "S"
s_value = 0

v_name = "V"
v_value = 0

del_h_name = "delta_H"
del_h_value = 0
del_h_initial = 31

del_s_name = "delta_S"
del_s_value = 0
del_s_initial = 31

del_v_name = "delta_V"
del_v_value = 0
del_v_initial = 127

cap = cv.VideoCapture(0)
cv.namedWindow(input_window)
cv.namedWindow(output_window)
cv.setMouseCallback(input_window, set_target)

cv.createTrackbar(h_name, input_window, min_value, max_value, nothing)
cv.createTrackbar(s_name, input_window, min_value, max_value, nothing)
cv.createTrackbar(v_name, input_window, min_value, max_value, nothing)
cv.createTrackbar(del_h_name, input_window, min_value, max_value, nothing)
cv.createTrackbar(del_s_name, input_window, min_value, max_value, nothing)
cv.createTrackbar(del_v_name, input_window, min_value, max_value, nothing)

# Set some sane default delta values
cv.setTrackbarPos(del_h_name, input_window, del_h_initial)
cv.setTrackbarPos(del_s_name, input_window, del_s_initial)
cv.setTrackbarPos(del_v_name, input_window, del_v_initial)


if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()

    if not ret:
        print("From not received successfully")
        break

    h = cv.getTrackbarPos(h_name, input_window)
    s = cv.getTrackbarPos(s_name, input_window)
    v = cv.getTrackbarPos(v_name, input_window)
    del_h = cv.getTrackbarPos(del_h_name, input_window)
    del_s = cv.getTrackbarPos(del_s_name, input_window)
    del_v = cv.getTrackbarPos(del_v_name, input_window)

    lower_threshold = (h - del_h, s - del_s, v - del_v)
    higher_threshold = (h + del_h, s + del_s, v + del_v)

    frame = cv.GaussianBlur(frame, (3, 3), 0)
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV_FULL)
    mask = cv.inRange(frame_hsv, lower_threshold, higher_threshold)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=1)

    mask_frame = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    sample_frame = np.zeros(frame.shape, np.uint8)
    sample_frame[:, :] = [h, s, v]

    output_frame = np.hstack((mask_frame, sample_frame))

    cv.imshow(input_window, frame)
    cv.imshow(output_window, output_frame)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
