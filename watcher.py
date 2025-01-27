import pickle
import threading
import DataBaseConnection as db
from queue import Queue
from typing import Callable, List
from carDetector import CarDetector
from numberDetector import numberDetector

path = 'recordings/'

cd_sem = threading.Semaphore()

def parked(car: str, spot: int) -> None:
    database = db.DataBaseConnection()
    if not database.checkEmployeePlace(spot, car):
        simple_msg(car, f"parked at wrong spot [{spot}] !!!")
    else:
        print(f"{car} parked  at {spot}")
    del database

def left(car: str, spot: int) -> None:
    print(f"{car} left  at {spot}")

def simple_msg(car:str, msg:str) -> None:
    print(f"{car} -> {msg}")
    local_database = db.DataBaseConnection()
    local_database.insertEvent(car, f"{msg}")
    del local_database


def car_detector_thread(queue: Queue) -> None:
    recordings = ['rec2.AVI', 'rec3.AVI', 'rec4.AVI', 'rec5.AVI', 'rec6.AVI', 'rec7.AVI']
    frames = [(10, 500), (10, 540), (10, 290), (20, 410), (10, 540), (300, 20000)]

    with open('layouts/rec1-layout', 'rb') as f:
        parking_spaces = pickle.load(f)

    car_detector = CarDetector(parking_spaces, simple_msg, parked, left)

    for video, frame_pair in zip(recordings, frames):
        expecting = queue.get()
        car_detector.go(path + video, expecting, frame_pair)
        cd_sem.release()

    car_detector.close()

def licence_plate_reader_thread(queue: Queue,) -> None:
    # recordings = [1, 3, None, 6, None, None]
    recordings = ['ELAGF92', 'EL0GVA0', None, 'EZGB0AF', None, None]

    # nd = numberDetector()

    for rec in recordings:
        cd_sem.acquire()
        # TODO zamienić na numberDetector
        if rec is not None:
            queue.put([rec])
        else:
            queue.put([])


if __name__ == '__main__':
    lp_queue = Queue()
    lpr_thread = threading.Thread(target=licence_plate_reader_thread, args=(lp_queue,), daemon=True)
    lpr_thread.start()
    cd_thread = threading.Thread(target=car_detector_thread, args=(lp_queue,))
    cd_thread.start()

    cd_thread.join()
    lpr_thread.join()

    database = db.DataBaseConnection()
    database.displayEvents()
    database.clearEvents()