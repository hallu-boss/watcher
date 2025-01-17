


import cv2
import numpy as np
import re
import easyocr
import os

def validate_polish_license_plate(text):
    if text is None:
        return None
    text = text.strip().replace(" ", "")
    pattern = r"^[A-Z]{2,3}[0-9A-Z]{4,5}$"

    if re.match(pattern, text):
        return text
    else:
        return None


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

def get_image_plate(image, tresh = 175):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, tresh, tresh + 10, cv2.THRESH_BINARY)

    # cv2.imshow('gray', gray)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ret_approx = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4 and cv2.contourArea(contour) > 2_000 and cv2.contourArea(contour) < 6_500 :
            # print(f'wierzchołki: {approx}  ')
            ret_approx.append(approx)
            # cv2.drawContours(image, [approx], 0, (0, 255, 0), 5)

    # print(f'Len end approx: {len(ret_approx)}')
    # cv2.imshow('Detected Rectangle', image)

    if len(ret_approx) == 1:
        points = get_2_point(ret_approx[0])
        plate_img = image[points[0][1]:points[1][1], points[0][0]:points[1][0]]
        # cv2.imshow('plate_img', plate_img)

        return plate_img
    else:
        return None


def complie_number_plate(result_ocr):
    if len(result_ocr) != 2:
        return None

    cz1 = re.sub(r'[^A-Z0-9]', '', result_ocr[0][1]).upper()
    cz2 = re.sub(r'[^A-Z0-9]', '', result_ocr[1][1]).upper()

    if len(cz2) >=2 and len(cz2) <=3:
        plate_nr = cz2 + cz1
    else:
        plate_nr = cz1 + cz2

    return plate_nr
def recognition_plate(image):
    if image is None:
        return None

    cv2.imshow("image", image)
    plate_image = get_image_plate(image)
    cv2.imshow("plate image", plate_image)
    if plate_image is None:
        return None

    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    image_enhanced = cv2.equalizeHist(gray)

    scale_percent = 10  # Increase by 10
    width = int(image_enhanced.shape[1] * scale_percent)
    height = int(image_enhanced.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(image_enhanced, dim, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('plate', resized)

    # Usuń szumy
    blurred = ~cv2.medianBlur(resized, 11)
    _, thresh = cv2.threshold(blurred, 190, 235, cv2.THRESH_BINARY_INV )
    cv2.imshow('thresh', thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    image_cleared = cv2.morphologyEx(~thresh, cv2.MORPH_ERODE, kernel)
    image_cleared = image_cleared.astype(np.uint8)


    image_cleared = ~cv2.bitwise_not(image_cleared)
    cv2.imshow('Obraz do odczytu', image_cleared )


    reader = easyocr.Reader(['pl', 'en'], gpu=True)
    result_ocr = reader.readtext(image_cleared.astype(np.uint8))

    plate_nr = complie_number_plate(result_ocr)
    print(f'Smieci: {plate_nr}')

    ret = validate_polish_license_plate(plate_nr)

    return ret



if __name__ == '__main__':
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


    default_dir = "recordings/wjazd/"
    cap = cv2.VideoCapture(default_dir + "1.mod")

    if not cap.isOpened():
        print("Nie można otworzyć pliku wideo.")
        exit()

    #cap.set(cv2.CAP_PROP_POS_FRAMES, 2 * cap.get(cv2.CAP_PROP_FRAME_COUNT)) # odtwarzanie od sekundy
    cap.set(cv2.CAP_PROP_POS_FRAMES, 45) # odtwarzanie od sekundy

    plate_all = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        plate_number = recognition_plate(frame)
        plate_all.append(plate_number)
        print(f"Rozpoznana tablica: {plate_number}")
        cv2.waitKey(100)
    cv2.destroyAllWindows()




