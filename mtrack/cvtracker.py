from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import pandas as pd
import numpy as np

class CVTracker:
    """
    CVTracker: Class to track objects using YOLO model with camera feed.

    Attributes:
        cap(cv2.VideoCapture): the camera object.
        yolo(YOLO): the YOLO object.
        device(int): the device to use for tracking.
        # current_track_res(List[ultralytics.engine.results.Results]): the current tracking results.
    
    Methods:
        track_next_frame(): Track the next frame and return the tracking results.
        annotate_frame(): Annotate the current frame with the tracking results.
        close(): Close the camera feed.
    """

    def __init__(self, camera_id: int, weight: str, device: int) -> None:
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

        self.yolo = YOLO(weight)
        self.device = device
        # self.current_track_res : list[Results] = None

    def track_next_frame(self, max_det: int, conf: float, half: bool, tracker: str = "bytetrack.yaml") -> None | tuple[pd.Timestamp, np.ndarray, list[Results]]:
        ret, frame = self.cap.read()
        # print("Shape of frame:", frame.shape)
        cur_time = pd.Timestamp.now()
        if not ret:
            return None
        # self.current_track_res = self.yolo.track(frame, persist=True, augment=False, device=self.device)
        track_res = self.yolo.track(frame, persist=True, augment=False, device=self.device, conf=conf, max_det=max_det, tracker=tracker, half=half, verbose=False)
        return  cur_time, frame, track_res

    # def annotate_frame(self):
    #     return self.current_track_res[0].plot()

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()