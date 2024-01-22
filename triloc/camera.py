import cv2 as cv
import numpy as np


class Camera:
    def __init__(self, name, camera_matrix=None):
        self.name = name
        self.K = camera_matrix

        self.id = -1
        self.cap = None
        self.Ki = np.linalg.inv(camera_matrix)

    def get_name(self):
        return self.name

    def set_id(self, id):
        self.id = id
        return id

    def get_id(self):
        return self.id

    def has_id(self):
        return not self.id == -1

    def set_cap(self, cap):
        self.cap = cap

    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()

    def point_to_ray(self, point):
        """https://stackoverflow.com/a/55083660"""
        # point = np.array(point)
        return self.Ki @ np.array([point[0], point[1], 1.0])


def assign_captures(camera_list):
    print("Press [0-9] to assign the capture to that camera")
    print("Press [s] to skip")
    print("Press [q] to quit")

    print("-")
    for i, cam in enumerate(camera_list):
        print(f"[{i}] {cam.get_name()}")
    print("-")

    capture_index = 0
    cap = cv.VideoCapture(capture_index, cv.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv.imshow(f"Capture [{capture_index}]", frame)

        key = cv.waitKey(1)
        if key == ord("q"):
            print("Ending capture assignment")
            break
        elif key == ord("s"):
            print(f"Skipping capture [{capture_index}]")
            capture_index += 1
            cap = cv.VideoCapture(capture_index, cv.CAP_DSHOW)
            cv.destroyAllWindows()
        elif key in (ord(c) for c in "0123456789"):
            camera_index = int(chr(key))
            print(f"Assigning capture [{capture_index}] to camera [{camera_index}]")
            camera_list[camera_index].set_id(capture_index)
            camera_list[camera_index].set_cap(cap)
            capture_index += 1
            cap = cv.VideoCapture(capture_index, cv.CAP_DSHOW)
            cv.destroyAllWindows()


def release_captures(camera_list):
    for cam in camera_list:
        cam.release()


if __name__ == "__main__":
    camera_matrix = np.array(
        [
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1],
        ]
    ) 

    cam_1 = Camera("Main Camera Name", camera_matrix)
    print(f"Does cam_1 currently have a set id? {cam_1.has_id()}")
    assign_captures([cam_1])
    print(f"Does cam_1 currently have a set id? {cam_1.has_id()}")
    if cam_1.has_id():
        print(f"{cam_1.get_id() = }")
        
    image_point_1 = [0, 0]
    image_point_2 = [320, 240]
    
    print(f"{cam_1.point_to_ray(image_point_1) = }")
    print(f"{cam_1.point_to_ray(image_point_2) = }")

