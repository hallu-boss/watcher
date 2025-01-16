import cv2
import numpy as np
import re
import easyocr

def validate_polish_license_plate(text):
    if text is None:
        return None
    text = text.strip().replace(" ", "")
    pattern = r"^[A-Z]{2,3}[0-9A-Z]{4,5}$"
    return text if re.match(pattern, text) else None

def cut_smal_frame(frame):
    if frame is None:
        return frame
    return frame[200:500, 200:550]


def image_sharpering(image):
    kernel_sharpen = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(image, -1, kernel_sharpen)


def get_2_point(points):
    points_x = [points[i][0][0] for i in range(len(points))]
    points_y = [points[i][0][1] for i in range(len(points))]
    left_point = (min(points_x), min(points_y))
    right_point = (max(points_x), max(points_y))
    return left_point, right_point

import cv2

def get_image_plate(image, thresh=175):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_thresh = cv2.threshold(gray, thresh, thresh + 10, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ret_approx = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and 2_000 < cv2.contourArea(contour) < 7_500:  # Kontury z 4 wierzchołkami
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if 1.7 <= aspect_ratio <= 2.2:
                print(aspect_ratio, w, h)
                ret_approx.append(approx)

    if len(ret_approx) == 1:
        points = get_2_point(ret_approx[0])
        x1, y1 = points[0]
        x2, y2 = points[1]
        if x1 < x2 and y1 < y2:
            plate_img = image[y1:y2, x1:x2]
            if plate_img.size > 0:
                return plate_img
    return None

def compile_number_plate(result_ocr):
    if len(result_ocr) != 2:
        return None

    cz1 = re.sub(r'[^A-Z0-9]', '', result_ocr[0][1]).upper()
    cz2 = re.sub(r'[^A-Z0-9]', '', result_ocr[1][1]).upper()
    return cz2 + cz1 if 2 <= len(cz2) <= 3 else cz1 + cz2

def recognition_plate(image):
    if image is None:
        return None

    plate_image = get_image_plate(image)
    if plate_image is None or plate_image.size == 0:  # Check if plate_image is valid
        return None

    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    image_enhanced = cv2.equalizeHist(gray)

    scale_percent = 10
    width = int(image_enhanced.shape[1] * scale_percent)
    height = int(image_enhanced.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(image_enhanced, dim, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('resized', resized)
    cv2.waitKey(100)
    blurred = ~cv2.medianBlur(resized, 11)
    _, thresh = cv2.threshold(blurred, 190, 235, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    image_cleared = cv2.morphologyEx(~thresh, cv2.MORPH_ERODE, kernel).astype(np.uint8)

    reader = easyocr.Reader(['pl', 'en'], gpu=True)
    result_ocr = reader.readtext(image_cleared)

    plate_nr = compile_number_plate(result_ocr)
    return validate_polish_license_plate(plate_nr)

if __name__ == '__main__':
    default_dir = "recordings/wjazd/"
    cap = cv2.VideoCapture(default_dir + "1.mod")

    if not cap.isOpened():
        print("Nie można otworzyć pliku wideo.")
        exit()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 45)
    plate_all = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        plate_number = recognition_plate(frame)
        if plate_number:
            plate_all.append(plate_number)
            print(f"Rozpoznana tablica: {plate_number}")
        else:
            print("Nie rozpoznano tablicy.")
    cap.release()
cv2.destroyAllWindows()




