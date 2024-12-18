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


spacesPos = []

drawing = False
dstart_x, dstart_y = -1, -1
temp_rect = None

def handler(event, x, y, flags, param):
    global spacesPos, drawing, temp_rect, dstart_x, dstart_y
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        dstart_x, dstart_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        width = x - dstart_x
        height = y - dstart_y
        temp_rect = ParkingSpace((dstart_x, dstart_y), width, height)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        spacesPos.append(temp_rect)
        temp_rect = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, ps in enumerate(spacesPos):
            if ps.inMe(x, y):
                spacesPos.pop(i)
                break


window_width, window_height = 1400, 900
cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('image', window_width, window_height)

while True:
    img = cv2.imread('images/test-image.jpg')
    resized_img = cv2.resize(img, (window_width, window_height))
    cv2.imshow('image', resized_img)
    if img is None:
        print("Nie można załadować obrazu.")
        break

    # Rysowanie wszystkich prostokątów
    for parkingSpace in spacesPos:
        pos1, pos2 = parkingSpace.beginEnd()
        cv2.rectangle(img, pos1, pos2, (255, 0, 0), 2)

    if temp_rect:
        pos1, pos2 = temp_rect.beginEnd()
        cv2.rectangle(img, pos1, pos2, (0, 255, 0), 2)

    # Wyświetl obraz i ustaw handlera myszy
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', handler)

    # Wyjście przy ESC
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
