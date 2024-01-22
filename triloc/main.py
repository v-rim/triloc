import cv2 as cv
import numpy as np
from camera import Camera, assign_captures
from detect import BinaryMotionDetector


def single_camera_test():
    camera_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

    hsv = [12, 175, 225]
    deltas = [16, 80, 120]

    cam_1 = Camera("Main Camera", camera_matrix)
    detector = BinaryMotionDetector(hsv, deltas)

    assign_captures([cam_1])
    while True:
        ret, frame = cam_1.get_frame()
        if not ret:
            print("Frame not received successfully")
            break

        ret, y_median, x_median = detector.detect(frame)
        if ret:
            cv.circle(frame, (x_median, y_median), 8, (0, 0, 255), -1)
            ray = cam_1.point_to_ray((x_median, y_median))
            print(f"Ray to ball center: {ray}")

        cv.imshow("Motion detection", frame)
        if cv.waitKey(1) == ord("q"):
            break

    cam_1.release()
    cv.destroyAllWindows()


def double_camera_test():
    pass


if __name__ == "__main__":
    single_camera_test()
