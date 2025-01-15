import cv2
import numpy as np

class BoundingBox:
    def __init__(self, tag, bounding_rect):
        self.bounding_rect = bounding_rect
        self.tag = tag

    def isSuccesor(self, bounding_rect):
        tolerance = 45
        values = self.bounding_rect
        values2 = bounding_rect
        res = [abs(values[i] - values2[i]) <= tolerance for i in range(len(values))]
        return all(res)

    def getPoints(self):
        x, y, w, h = self.bounding_rect
        return (x, y), (x + w, y + h)

    def draw(self, frame):
        color = (255, 0, 0)
        x, y, w, h = self.bounding_rect
        text_params = [cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(self.tag, text_params[0], text_params[1], text_params[3])
        cv2.rectangle(frame, (x, y), (x + text_w, y + text_h + baseline), color, -1)
        cv2.putText(frame, self.tag, (x, y + text_h), text_params[0], text_params[1], text_params[2], text_params[3] )

def get_mask(fr):
    gray_frame = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

    # Odjęcie tła (różnica absolutna)
    fgmask = cv2.absdiff(background_gray, gray_frame)

    # Progowanie, aby uzyskać binarną maskę
    _, fgmask = cv2.threshold(fgmask, 50, 255, cv2.THRESH_BINARY)

    # Usuwanie szumów za pomocą operacji morfologicznych
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    return fgmask

def filter_coutours(contours):
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 5000:
            filtered_contours.append(contour)
    return filtered_contours

# Wczytanie nagrania
video_path = 'recordings/1.AVI'
cap = cv2.VideoCapture(video_path)

# Sprawdzenie, czy plik został poprawnie otwarty
if not cap.isOpened():
    print("Nie można otworzyć pliku wideo.")
    exit()

# Pobranie pierwszej klatki jako tła
ret, background_frame = cap.read()
if not ret:
    print("Nie można odczytać pierwszej klatki.")
    cap.release()
    exit()

background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
cap.release()
video_path = 'recordings/4.AVI'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Nie można otworzyć pliku wideo.")
    exit()

boxes = {}
box_counter = 0

f_index = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konwersja bieżącej klatki do skali szarości
    fgmask = get_mask(frame)

    # Znajdowanie konturów
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = filter_coutours(contours)

    print("-------------")
    for contour in filtered_contours:
        found_key = None
        br = cv2.boundingRect(contour)
        for key in boxes:
            bounding_box = boxes[key]
            if bounding_box.isSuccesor(br):
                found_key = key
                break
        newbb = []
        if found_key is None:
            newbb = BoundingBox(str(box_counter), br)
            boxes[box_counter] = newbb
            box_counter += 1
        else:
            newbb = BoundingBox(str(found_key), br)
            boxes[found_key] = newbb

        print(cv2.contourArea(contour))
        newbb.draw(frame)
    print("-------------")

    cv2.putText(frame, f"Klatka: {f_index}", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

    # Wyświetlanie wyników
    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)
    f_index += 1

    # Przerwanie działania klawiszem 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Zwolnienie zasobów
cap.release()
cv2.destroyAllWindows()

print(box_counter)
