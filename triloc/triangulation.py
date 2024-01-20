import numpy as np


class LSLocalizer:
    """Takes multiple rays to predict a position using a least squares method.

    Stores the space transforms of each camera in the form of SE(3) matrices to
    unify objects detections from each viewpoint. The convention for coordinates
    is that X is to the right of the camera, Y is in the direction of the
    camera, and Z is up from the camera.

    Rotations are represented as a pair of angles theta and phi rotating a unit
    vector in the Y direction about the Z and X axes respectively. A positive
    theta rotates the vector towards positive X and a positive phi rotates
    the vector towards positive Z.
    """

    def __init__(self, camera_transforms):
        self.camera_transforms = np.array(camera_transforms)

    def predict(self, angle_pair_array, weights):
        ray_directions = self.angles_to_rays(angle_pair_array)
        ray_points = self.camera_transforms[:, :3, 3]
        # print(f"{ray_points = }")
        return self.find_nearest_point(ray_points, ray_directions, weights)

    def angles_to_rays(self, ray_angles):
        transformed_rays = []
        for i in range(len(ray_angles)):
            theta, phi = ray_angles[i]
            transform = self.camera_transforms[i]
            transform_origin = transform[:, 3]

            ray = [
                np.cos(phi) * np.sin(theta),
                np.cos(phi) * np.cos(theta),
                np.sin(phi),
                1,
            ]
            ray = transform @ ray
            ray -= transform_origin
            transformed_rays.append(ray[:3])

        return np.array(transformed_rays)

    def find_nearest_point(self, ray_points, ray_directions, weights=None):
        """Implements the method found at https://stackoverflow.com/a/48201730.

        Finds the point that minimizes the least squared distance to each of the
        rays. Utilizes the magic number 3 in places as it only works in R^3
        (it may work in higher dimension but that proof uses cross products).
        """
        m = np.zeros((3, 3))
        b = np.zeros(3)

        n = len(ray_points)

        if not len(ray_directions) == n or not len(weights) == n:
            print("Size of ray_points and ray_directions do not match")
            return np.zeros(3)

        if weights is None:
            weights = np.ones(n)

        direction_magnitudes = np.linalg.norm(ray_directions, axis=1)
        ray_directions = weights[:, None] * (
            ray_directions / direction_magnitudes[:, None]
        )

        for i in range(n):
            dd = np.dot(ray_directions[i], ray_directions[i])
            da = np.dot(ray_directions[i], ray_points[i])
            for j in range(3):
                for k in range(3):
                    m[j, k] += ray_directions[i][j] * ray_directions[i][k]
                m[j, j] -= dd
                b[j] += ray_directions[i, j] * da - ray_points[i][j] * dd

        return np.linalg.solve(m, b)


if __name__ == "__main__":
    np.set_printoptions(formatter={"float": "{: 0.4f}".format})

    # first camera at origin
    T_cam1 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # second camera rotated pi/2 about Z at (1, 1, 0)
    T_cam2 = np.array(
        [
            [0, -1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    camera_transforms = [T_cam1, T_cam2]

    lsl = LSLocalizer(camera_transforms)

    # test find_nearest_point
    ray_points = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    ray_directions = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
        ]
    )

    weights = np.array([1, 1, 1, 1])

    nearest_point = lsl.find_nearest_point(ray_points, ray_directions, weights)
    print(f"{nearest_point = }")

    # test angles_to_ray
    ray_1_angles = (0, 0)
    ray_2_angles = (np.pi / 4, 0)
    ray_angles = [ray_1_angles, ray_2_angles]

    transformed_rays = lsl.angles_to_rays(ray_angles)
    print(f"transformed_rays = \n{transformed_rays}")

    # test predict
    predicted_point = lsl.predict(ray_angles, np.array([1, 1]))
    print(f"{predicted_point = }")
