import cv2 as cv
import numpy as np


if __name__ == "__main__":
    # HSV : 12 175 225
    # Delta : 16 80 120

    h = 12
    s = 175
    v = 225
    del_h = 16
    del_s = 80
    del_v = 120

    lower_threshold = (h - del_h, s - del_s, v - del_v)
    higher_threshold = (h + del_h, s + del_s, v + del_v)

    cap = cv.VideoCapture(0)

    past_frame_queue = []
    past_frame = 0
    frame_delay = 15

    while True:
        ret, frame = cap.read()
        if not ret:
            print("From not received successfully")
            break

        frame = cv.GaussianBlur(frame, (3, 3), 0)
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV_FULL)

        kernel = np.ones((3, 3), np.uint8)

        mask = cv.inRange(frame_hsv, lower_threshold, higher_threshold)
        delay_mask = mask - past_frame
        delay_mask[delay_mask < 128] = 0 # Strange edge case where it can be 1
        delay_mask = cv.erode(delay_mask, kernel, iterations=5)
        delay_mask = cv.dilate(delay_mask, kernel, iterations=1)

        # print(f"{np.nonzero(delay_mask) = }")
        # print(np.nonzero(delay_mask))
        # input()

        # print(f"{np.mean(delay_mask)}")
        x_list, y_list = np.nonzero(delay_mask)

        if x_list is None or len(x_list) == 0:
            x_list = [0]
        if y_list is None or len(y_list) == 0:
            y_list = [0]

        # print(f"{x_list = }")
        # print(f"{y_list = }")
        # print(f"{np.median(x_list) = }")
        # print(f"{np.median(y_list) = }")
        x_median = int(np.median(x_list))
        y_median = int(np.median(y_list))

        mask_frame = cv.cvtColor(delay_mask, cv.COLOR_GRAY2BGR)

        # print(f"{x_median = }")
        # print(f"{y_median = }")

        # mask_frame[x_median, y_median] = [0, 0, 100]
        # print(f"{delay_mask[x_median, y_median] = }")
        # print(f"{np.mean(delay_mask)}")
        if np.mean(delay_mask) > 0.1:
            cv.circle(mask_frame, (y_median, x_median), 8, (0, 0, 255), -1)

        cv.imshow("Motion detection", mask_frame)

        past_frame_queue.append(mask)

        if frame_delay > 1:
            frame_delay -= 1
        else:
            past_frame = past_frame_queue.pop(0)

        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
