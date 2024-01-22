import cv2 as cv
import numpy as np
from camera import Camera, assign_captures, release_captures
from detect import BinaryMotionDetector
from triangulation import LSLocalizer


def single_camera_test():
    hsv = [12, 175, 225]
    deltas = [16, 80, 120]

    camera_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

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
    """Not tested"""
    cam_1_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    cam_2_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

    hsv = [12, 175, 225]
    deltas = [16, 80, 120]

    # first camera at origin
    T_1 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # second camera rotated pi/2 about Z at (1, 1, 0)
    T_2 = np.array(
        [
            [0, -1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    cam_1 = Camera("Left Camera", cam_1_matrix)
    cam_2 = Camera("Right Camera", cam_2_matrix)

    detector = BinaryMotionDetector(hsv, deltas, 1e10)  # Turn off isolation

    lsl = LSLocalizer([T_1, T_2])

    assign_captures([cam_1, cam_2])
    while True:
        ret_1, frame_1 = cam_1.get_frame()
        ret_2, frame_2 = cam_2.get_frame()

        ret_1, y_median_1, x_median_1 = detector.detect(frame_1)
        ret_2, y_median_2, x_median_2 = detector.detect(frame_2)

        if ret_1:
            cv.circle(frame_1, (x_median_1, y_median_1), 8, (0, 0, 255), -1)
            ray_1 = cam_1.point_to_ray((x_median_1, y_median_1))
            # print(f"Left camera ray to ball center: {ray_1}")

        if ret_2:
            cv.circle(frame_2, (x_median_2, y_median_2), 8, (0, 0, 255), -1)
            ray_2 = cam_2.point_to_ray((x_median_2, y_median_2))
            # print(f"Right camera ray to ball center: {ray_2}")

        if ret_1 and ret_2:
            predicted_point = lsl.predict([ray_1, ray_2], [1, 1])
            print(f"Predicted ball position: {predicted_point}")

        cv.imshow("Left view", frame_1)
        cv.imshow("Right view", frame_2)

        if cv.waitKey(1) == ord("q"):
            break

    release_captures([cam_1, cam_2])
    cv.destroyAllWindows()


if __name__ == "__main__":
    # single_camera_test()
    double_camera_test()
