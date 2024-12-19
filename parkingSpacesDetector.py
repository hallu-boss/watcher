import cv2

class ParkingSpace:
    def __init__(self, pos, width, height):
        self.pos = pos
        self.width = width
        self.height = height

    def inMe(self, x, y):
        x1, y1 = self.pos
        x2, y2 = x1 + self.width, y1 + self.height
        return (x1 < x < x2) and (y1 < y < y2)

    def beginEnd(self):
        return self.pos, (self.pos[0] + self.width, self.pos[1] + self.height)


# Wymiary prostokąta
width, height = 65, 155
spacesPos = []

# Funkcja obsługująca kliknięcia myszy
def handler(event, x, y, flags, param):
    global spacesPos
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            w, h = height, width
        else:
            w, h = width, height
        spacesPos.append(ParkingSpace((x, y), w, h))
    elif event == cv2.EVENT_RBUTTONDOWN:  # Prawy przycisk myszy
        for i, parkingSpace in enumerate(spacesPos):
            if parkingSpace.inMe(x, y):  # Sprawdź, czy kliknięto w prostokąt
                spacesPos.pop(i)
                break  # Usuń tylko pierwszy znaleziony prostokąt


window_width, window_height = 1000, 600
# Wczytaj obraz i ustawienie okna
cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('image', window_width, window_height)

while True:
    img = cv2.imread('wynik.jpg')
    resized_img = cv2.resize(img, (window_width, window_height))
    cv2.imshow('image', resized_img)
    if img is None:
        print("Nie można załadować obrazu.")
        break

    # Rysowanie wszystkich prostokątów
    for parkingSpace in spacesPos:
        pos1, pos2 = parkingSpace.beginEnd()
        cv2.rectangle(img, pos1, pos2, (255, 0, 0), 2)

    # Wyświetl obraz i ustaw handlera myszy
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', handler)

    # Wyjście przy ESC
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
