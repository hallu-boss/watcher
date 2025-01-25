import pickle
from typing import List, Optional, Tuple, Dict
import cv2
from cv2 import BackgroundSubtractorKNN
from utils import ParkingSpace
import numpy as np

def box_center(box):
    x, y, w, h = box
    return x + w // 2, y + h // 2

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

class BoundingBox:
    dist_tolerance = 40
    area_tolerance = 0.3
    def __init__(self, tag, bounding_rect):
        self.bounding_rect = bounding_rect
        self.tag = tag


    def isSuccesor(self, bounding_rect):
        _, _, w1, h1 = bounding_rect
        _, _, w2, h2 = self.bounding_rect

        a1 = w1 * h1
        a2 = w2 * h2

        res1 = abs(a1 - a2) / max(a1, a2) <= BoundingBox.area_tolerance

        c1 = box_center(self.bounding_rect)
        c2 = box_center(bounding_rect)
        res2 = distance(c1, c2) <= BoundingBox.dist_tolerance and a1 / a2
        return res1 and res2

    def getPoints(self):
        x, y, w, h = self.bounding_rect
        return (x, y), (x + w, y + h)

    def getTag(self):
        return self.tag

    def getBoundingRect(self):
        return self.bounding_rect

    def parked(self, space: ParkingSpace):
        overlay = 0.4
        x1_1, y1_1, w, h = self.bounding_rect
        x2_1, y2_1 = x1_1 + w, y1_1 + h
        (x1_2, y1_2), (x2_2, y2_2) = space.beginEnd()

        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x1_inter < x2_inter and y1_inter < y2_inter:
            intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

            overlap1 = intersection_area / area1
            overlap2 = intersection_area / area2

            return overlap1 >= overlay or overlap2 >= overlay

        return False

    def setBR(self, bounding_rect):
        self.bounding_rect = bounding_rect

    def draw(self, frame):
        color = (255, 0, 0)
        x, y, w, h = self.bounding_rect
        text_params = [cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        (text_w, text_h), baseline = cv2.getTextSize(self.tag, text_params[0], text_params[1], text_params[3])
        cv2.rectangle(frame, (x, y), (x + text_w, y + text_h + baseline), color, -1)
        cv2.putText(frame, self.tag, (x, y + text_h), text_params[0], text_params[1], text_params[2], text_params[3])
        cv2.circle(frame, box_center(self.bounding_rect), radius=1, color=(0, 255, 255), thickness=2)
        cv2.circle(frame, box_center(self.bounding_rect), radius=BoundingBox.dist_tolerance, color=(255, 255, 0), thickness=2)

class CarDetector:
    entry_start = (130, 260)
    entry_end = (107, 444)

    exit_start = (812, 670)
    exit_end = (1046, 670)

    def __init__(self,
                 parking_spaces: List[ParkingSpace],
                 cars: Optional[List[Tuple[BoundingBox, Dict[int, int]]]] = None,
                 background_subtractor: Optional[BackgroundSubtractorKNN] = None):

        self.cars = None
        if background_subtractor is None:
            self.background_subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400,
                                                                           detectShadows=True)
        else:
            self.background_subtractor = background_subtractor
        self.total_frames = 0
        self.parking_spaces = parking_spaces
        self.t_cap = cv2.VideoCapture("recordings/rec1.AVI")

    def close(self):
        self.t_cap.release()
        cv2.destroyAllWindows()

    def go(self, video_path: str, expected: List[str], rg: Tuple[int, int] = (0, 2000000000), visualize: bool = True):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, rg[0])

        frame_idx = rg[0]
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.__clear_around(frame)

            mask = self.__getMask(frame)

            contours = self.__getContours(mask)

            self.__assocContours(contours, expected)

            self.__parkingManager()

            self.__popLeavingCars()

            hud = self.__getHUD(frame, frame_idx)
            cv2.drawContours(hud, contours, -1, (255, 0, 255), 2)



            self.total_frames += 1
            frame_idx += 1

            if visualize:
                cv2.imshow('frame', hud)

            if self.total_frames % 250 == 0:
                self.recalibration()

            if frame_idx >= rg[1]:
                break

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()

    def __popLeavingCars(self):
        if self.cars is not None:
            to_pop = []
            for i, (car, ocupying) in enumerate(self.cars):
                if len(ocupying.keys()) > 0:
                    continue
                if self.__on_exit(car.getBoundingRect()):
                    print(f"{car.getTag()} left parking")
                    to_pop.append(i)
            for i in to_pop:
                self.cars.pop(i)

    def __updateCarOcupation(self, tolerance):

        for car, ocupying in self.cars:
            keys = ocupying.keys()
            for i, space in enumerate(self.parking_spaces, start=1):
                if car.parked(space):
                    if i in keys:
                        if ocupying[i] < tolerance:
                            ocupying[i] += 1
                    else:
                        ocupying.update({i: 1})
                else:
                    if i in keys:
                        ocupying[i] -= 1



    def __parkingManager(self):
        tolerance = 5

        parked_val = 100
        ocupying_val = 150
        if self.cars is not None:
            self.__updateCarOcupation(tolerance)

            for car, ocupying in self.cars:
                keys = list(ocupying.keys())
                for key in keys:
                    if ocupying[key] == tolerance:
                        print(f"{car.getTag()} has parked at spot {key}")
                        ocupying[key] = parked_val

                if len(keys) > 1:
                    if all(parked_val - tolerance < value <= parked_val for value in ocupying.values()):
                        print(f"{car.getTag()} is ocupying {keys} !!!")
                        for key in keys:
                            ocupying[key] = ocupying_val

                for key in keys:
                    if ocupying[key] == parked_val - tolerance or ocupying[key] == ocupying_val - tolerance:
                        ocupying.pop(key)
                        print(f"{car.getTag()} left spot {key}")

    def __on_exit(self, bounding_rect):
        x, y, w, h = bounding_rect
        rec = (x, y, x + w, y + h)
        p1, p2 = (x,y), (x + w, y + h)

        res1, _, _ = cv2.clipLine(rec, CarDetector.exit_start, CarDetector.exit_end)

        res2 = distance(p1, CarDetector.exit_start) < 100

        return res1 and res2

    def __clear_around(self, fr):
        x1, y1 = 83, 37
        x2, y2 = 1266, 701

        fr[:y1, :] = (0, 0, 0)
        fr[y2:, :] = (0, 0, 0)
        fr[y1:y2, :x1] = (0, 0, 0)
        fr[y1:y2, x2:] = (0, 0, 0)

    def __on_entry(self, bounding_rect):
        x, y, w, h = bounding_rect
        rect = (x, y, x + w, y + h)

        res, _, _ = cv2.clipLine(rect, CarDetector.entry_start, CarDetector.entry_end)
        return res

    def __getMask(self, frame):
        mask = self.background_subtractor.apply(frame)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def __getContours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return [contour for contour in contours if 5000 < cv2.contourArea(contour) < 30000]

    def __assocContours(self, contours, expected):
        for con in contours:
            br = cv2.boundingRect(con)
            if len(expected) > 0:
                if self.__on_entry(br):
                    tag = expected.pop(0)
                    if self.cars is None:
                        self.cars = [(BoundingBox(tag, br), {})]
                    else:
                        self.cars.append((BoundingBox(tag, br), {}))
                continue
            if self.cars is not None:
                for car, _ in self.cars:
                    if car.isSuccesor(br):
                        car.setBR(br)
                        break

    def __getHUD(self, frame, idx):
        hud = frame.copy()

        # entry line
        cv2.line(hud, CarDetector.entry_start, CarDetector.entry_end, (0, 0, 255), 2)
        cv2.line(hud, CarDetector.exit_start, CarDetector.exit_end, (0, 0, 255), 2)

        for space in self.parking_spaces:
            pos1, pos2 = space.beginEnd()
            cv2.rectangle(hud, pos1, pos2, (100, 100, 100), 2)

        if self.cars is not None:
            for car, _ in self.cars:
                for space in self.parking_spaces:
                    if car.parked(space):
                        pos1, pos2 = space.beginEnd()
                        cv2.rectangle(hud, pos1, pos2, (2, 102, 242), 2)

            # cars
            for car, _ in self.cars:
                car.draw(hud)

        # frame counter
        cv2.putText(hud, f"Klatka: {self.total_frames}", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

        return hud

    def recalibration(self):
        self.t_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        max_frame = 70
        idx = 0
        while True:
            ret, frame = self.t_cap.read()
            if not ret:
                break

            self.background_subtractor.apply(frame)
            if idx >= max_frame:
                break
            idx += 1






if __name__ == '__main__':

    with open('layouts/rec1-layout', 'rb') as f:
        parking_spaces = pickle.load(f)

    vis = True
    path = 'recordings/'
    videos = ['rec2.AVI', 'rec3.AVI', 'rec4.AVI', 'rec5.AVI', 'rec6.AVI']
    exp = [['smart'], ['volvo'], [], ['cabrio'], []]
    frames = [(10, 500), (10, 540), (10, 290), (20, 410), (10, 540), (300, 20000)]
    carDetector = CarDetector(parking_spaces)

    for i, (nd, video, frame_pair) in enumerate(zip(exp, videos, frames)):
        carDetector.go(path+ video, nd, frame_pair, visualize=vis)
        # print(i)

    # input('Press enter to start...')

    path = 'recordings/rec7.AVI'
    carDetector.go(path, [], (280, 900))

    carDetector.close()