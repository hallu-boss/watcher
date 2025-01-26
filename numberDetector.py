import numpy as np
import re
import easyocr
import cv2
from skimage.filters import threshold_otsu


class numberDetector:
    def validate_polish_license_plate(self, text):
        if text is None:
            return None
        text = text.strip().replace(" ", "")
        pattern = r"^[A-Z]{2,3}[0-9A-Z]{4,5}$"
        return text if re.match(pattern, text) else None

    def cut_smal_frame(self, frame):
        if frame is None:
            return frame
        return frame[200:500, 200:500]


    def get_2_point(self, points):
        points_x = [points[i][0][0] for i in range(len(points))]
        points_y = [points[i][0][1] for i in range(len(points))]
        left_point = (min(points_x), min(points_y))
        right_point = (max(points_x), max(points_y))
        return left_point, right_point


    def get_image_plate(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # fx, ax = try_all_threshold(gray)
        # plt.show()

        thresh= threshold_otsu(image)
        _, binary_thresh = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        # _, binary_thresh = cv2.threshold(gray, thresh, thresh + 10, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=False)

        ret_approx = []
        for contour in sorted_contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(image, [approx], -1, (0, 0, 255), 2)
            # print(cv2.contourArea(contour))
            if len(approx) == 4 and 500 < cv2.contourArea(contour) < 10_500:  # Kontury z 4 wierzchołkami

                x, y, w, h = cv2.boundingRect(approx)
                # print(w/h)
                # sleep(1)
                aspect_ratio = w / h
                if 1.4 <= aspect_ratio <= 2.2:
                    # cv2.drawContours(image, [approx], -1, (0, 0, 255), 2)
                    # print(cv2.contourArea(contour))
                    ret_approx.append(approx)

        if len(ret_approx) == 1:
            points = self.get_2_point(ret_approx[0])
            x1, y1 = points[0]
            x2, y2 = points[1]
            if x1 < x2 and y1 < y2:
                plate_img = image[y1:y2, x1:x2]
                if plate_img.size > 0:
                    return plate_img
        return None

    def compile_number_plate(self, result_ocr):
        if len(result_ocr) < 1 or len(result_ocr) > 2:
            return None
        if len(result_ocr) == 1:
            ret = re.sub(r'[^A-Za-z0-9]', '', result_ocr[0][1]).upper()
            if ret[0] != 'E':
                ret = ret[1::]
            return ret

        cz1 = re.sub(r'[^A-Za-z0-9]', '', result_ocr[0][1]).upper()
        cz2 = re.sub(r'[^A-Za-z0-9]', '', result_ocr[1][1]).upper()
        ret = (cz2 + cz1 if 2 <= len(cz2) <= 3 else cz1 + cz2)
        # print(f'pierwszy ret {ret}')
        if ret[0] != 'E':
            ret = ret[1::]
        # print(f'drugi ret {ret}')
        return ret

    def recognition_plate(self, image):
        if image is None:
            return None

        plate_image = self.get_image_plate(image)
        if plate_image is None:
            return None
        # cv2.imshow('frame', plate_image)
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)


        scale_percent = 10
        width = int(gray.shape[1] * scale_percent)
        height = int(gray.shape[0] * scale_percent)
        dim = (width, height)
        resized = cv2.resize(gray, dim, interpolation=cv2.INTER_CUBIC)

        blurred = cv2.medianBlur(resized, 23)
        image_enhanced = cv2.equalizeHist(blurred)
        _, thresh = cv2.threshold(~image_enhanced, 190, 255, cv2.THRESH_BINARY)


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        image_cleared = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel).astype(np.uint8)

        reader = easyocr.Reader(['en'], gpu=True)
        result_ocr = reader.readtext(image_cleared)
        # print(result_ocr)

        plate_nr = ""
        if len(result_ocr) == 1:
            plate_nr = result_ocr[0][1].upper()

        elif len(result_ocr) > 1:
            plate_nr = self.compile_number_plate(result_ocr)

        return self.validate_polish_license_plate(plate_nr)


    def number_plate(self, nr_wideo):
        default_dir = "recordings/wjazd/"
        cap = cv2.VideoCapture(default_dir + str(nr_wideo) + ".mod")
        if not cap.isOpened():
            print("Nie można otworzyć pliku wideo.")
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
        plate_all = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            # cv2.imshow('frame', frame)
            # cv2.waitKey(2)
            frame = self.cut_smal_frame(frame)
            plate_number = self.recognition_plate(frame)
            if plate_number:
                plate_all.append(plate_number)
        cap.release()

        return list(set(plate_all))


if __name__ == "__main__":
    nd = numberDetector()
    wynik = []
    for i in range(1, 7):
        wynik.append(nd.number_plate(nr_wideo=i))

    print(wynik)


