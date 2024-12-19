import os
from email.policy import default

import cv2
import pickle
import utils
from utils import ParkingSpace
import structureDefinitions as sd
from argparse import ArgumentParser

default_iamge_path = "images/test-image.jpg"

spacesPos = []

drawing = False
dstart_x, dstart_y = -1, -1
temp_rect = None

def writeList(file_path, array):
    with open(file_path, "wb") as f:
        pickle.dump(array, f)
        print(f"Positions saved into {file_path}")

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
        if temp_rect:
            spacesPos.append(temp_rect)
        temp_rect = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, ps in enumerate(spacesPos):
            if ps.inMe(x, y):
                spacesPos.pop(i)
                break

def builder(path):
    winname = "Layout Builder"
    cv2.namedWindow(winname, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(winname, sd.window_width, sd.window_height)

    while True:
        img = cv2.imread(path)
        resized_img = cv2.resize(img, (sd.window_width, sd.window_height))
        cv2.imshow(winname, resized_img)
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
        cv2.imshow(winname, img)
        cv2.setMouseCallback(winname, handler)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

def parseArguments():
    parser = ArgumentParser()
    # TODO --path zamiast --image i --dir
    parser.add_argument("-p", "--path", dest="path", default=default_iamge_path, help="Ścieżka do obrazu [domyślnie :" + default_iamge_path + "]")

    return parser.parse_args()
if __name__ == "__main__":
    args = parseArguments()
    builder(args.path)
    print(f"{len(spacesPos)} spaces selected.")

    layout_file = sd.layout_path(args.path)

    actions = {"yes": lambda: writeList(layout_file, spacesPos), "no": lambda: None}
    text = "Do you want to save layout? [yes/no]: "
    if os.path.isfile(layout_file):
        text = "File already exists. Do you want to overwrite it? [yes/no]: "

    utils.prompt(text, actions)