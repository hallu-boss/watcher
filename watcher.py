import pickle
import threading
from queue import Queue
from typing import Callable
from carDetector import CarDetector
from numberDetector import numberDetector

path = 'recordings/'

cd_sem = threading.Semaphore()

def save_to_db(car:str, msg: str) -> None:
    print(f"{car} -> {msg}")

def car_detector_thread(queue: Queue, handle_msg: Callable[[str, str], None]) -> None:
    recordings = ['rec2.AVI', 'rec3.AVI', 'rec4.AVI', 'rec5.AVI', 'rec6.AVI', 'rec7.AVI']
    frames = [(10, 500), (10, 540), (10, 290), (20, 410), (10, 540), (300, 20000)]

    with open('layouts/rec1-layout', 'rb') as f:
        parking_spaces = pickle.load(f)

    car_detector = CarDetector(parking_spaces)

    for video, frame_pair in zip(recordings, frames):
        expecting = queue.get()
        car_detector.go(path + video, expecting, frame_pair)
        cd_sem.release()

    car_detector.close()

def licence_plate_reader_thread(queue: Queue,) -> None:
    # recordings = [1, 3, None, 6, None, None]
    recordings = ['ELAGF9Z', 'EL0GVA0', None, 'EZGB0AF', None, None]

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
    cd_thread = threading.Thread(target=car_detector_thread, args=(lp_queue,save_to_db))
    cd_thread.start()