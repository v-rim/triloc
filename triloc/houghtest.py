import cv2 as cv
import numpy as np

window_name = "Hough Transform Test"

cap = cv.VideoCapture(0)
cv.namedWindow(window_name)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    frame = cv.GaussianBlur(frame, (5, 5), 0)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if not ret:
        print("From not received successfully")
        break

    circles = cv.HoughCircles(
        gray, cv.HOUGH_GRADIENT, 1, 20, param1=125, param2=50, minRadius=0, maxRadius=0
    )

    if circles is not None:
        for x, y, r in circles[0]:
            # print(f"{circles[0] = }")
            cv.circle(frame, (round(x), round(y)), round(r), (0, 0, 255), 2)

    cv.imshow(window_name, frame)

    if cv.waitKey(1) == ord("q"):
        break
