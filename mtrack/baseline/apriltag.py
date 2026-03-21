from ..data.cv import ObjectTrace
import pandas as pd
import numpy as np

class GtTracklet:
    def __init__(self, cv_id: int):
        self.cv_id = cv_id
        self.frames: list[int] = []
        self.x_lst: list[float] = []
        self.y_lst: list[float] = []
        self.w_lst: list[float] = []
        self.h_lst: list[float] = []

    def add_detection(self, frame: int, x: float, y: float, w: float, h: float):
        self.frames.append(frame)
        self.x_lst.append(x)
        self.y_lst.append(y)
        self.w_lst.append(w)
        self.h_lst.append(h)

    @property
    def start_frame(self) -> int:
        return self.frames[0] if self.frames else -1
    
    @property
    def end_frame(self) -> int:
        return self.frames[-1] if self.frames else -1

def load_gt(path: str):
    df = pd.read_csv(path, sep=',', names=['frame', 'id', 'x', 'y', 'w', 'h', "_", "__", "___"])
    tracklets: dict[int, GtTracklet] = {}
    for _, row in df.iterrows():
        frame = int(row['frame'])
        cv_id = int(row['id'])
        x = float(row['x'])
        y = float(row['y'])
        w = float(row['w'])
        h = float(row['h'])
        if cv_id not in tracklets:
            tracklets[cv_id] = GtTracklet(cv_id)
        tracklets[cv_id].add_detection(frame, x, y, w, h)
    return tracklets

class AprilTagResults:
    def __init__(self, path: str):
        self.df = pd.read_csv(path)
    
    def get_detections(self, start_frame: int, end_frame: int) -> dict[int, list[tuple[int, float, float]]]:
        detections: dict[int, list[tuple[int, float, float]]] = {}
        df_filtered = self.df[(self.df['frame'] >= start_frame) & (self.df['frame'] <= end_frame)]
        for _, row in df_filtered.iterrows():
            frame = int(row['frame'])
            tag_id = int(row['tag_id'])
            x = float(row['x'])
            y = float(row['y'])
            if frame not in detections:
                detections[frame] = []
            detections[frame].append((tag_id, x, y))

        return detections

DIS_THRESHOLD = 20

def generate_correct_id(gt_tracklets: dict[int, GtTracklet], apriltag_results: AprilTagResults):

    correct_id_map: dict[int, int] = {}

    for cv_id, tracklet in gt_tracklets.items():
        detections = apriltag_results.get_detections(tracklet.start_frame, tracklet.end_frame)
        
        id_count: dict[int, int] = {}
        for frame, x_gt, y_gt, w_gt, h_gt in zip(tracklet.frames, tracklet.x_lst, tracklet.y_lst, tracklet.w_lst, tracklet.h_lst):
            if frame not in detections:
                continue
            for tag_id, x_tag, y_tag in detections[frame]:
                cx_gt = x_gt + w_gt / 2
                cy_gt = y_gt + h_gt / 2
                dist = np.sqrt((cx_gt - x_tag) ** 2 + (cy_gt - y_tag) ** 2)
                if dist < DIS_THRESHOLD:
                    if tag_id not in id_count:
                        id_count[tag_id] = 0
                    id_count[tag_id] += 1
        if id_count:
            correct_id = max(id_count, key=id_count.get)
            print(f"CV ID {cv_id} is matched to AprilTag ID {correct_id} with {id_count[correct_id]} matches.")
            correct_id_map[cv_id] = correct_id
        else:
            print(f"CV ID {cv_id} has no matching AprilTag ID.")

    return correct_id_map