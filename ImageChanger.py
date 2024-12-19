import cv2
import numpy as np

class ImageChanger:
    def __init__(self, image):
        """ Tutaj będzie identyfikacja po punktach znacznikowych rogu parkingu i zapis do src_points"""

        self.src_points = [
            [143, 105],
            [1093, 114],
            [1162, 660],
            [85, 654]

        ]

        pass


    def perspective_correction(self, image):
        height, width, channels = image.shape

        dest_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype='float32')

        src_points = np.array(self.src_points, dtype='float32')
        matrix = cv2.getPerspectiveTransform(src_points, dest_points)
        corrected_image = cv2.warpPerspective(image, matrix, (width, height))

        return corrected_image