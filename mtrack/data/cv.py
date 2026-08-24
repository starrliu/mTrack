"""
Data structures for storing object traces.
"""

from dataclasses import dataclass

import pandas as pd
from ultralytics.engine.results import Results


@dataclass
class XYWH:
    """
    XYWH class is used to store the bounding box of an object.

    Attributes:
        x(float): the x coordinate of the top left corner.
        y(float): the y coordinate of the top left corner.
        w(float): the width of the bounding box.
        h(float): the height of the bounding box.
    """

    x: float
    y: float
    w: float
    h: float

    def get_center(self) -> tuple[float, float]:
        """
        Get the center of the bounding box.

        Returns:
            tuple[float, float]: the center of the bounding box.
        """
        return self.x + self.w / 2, self.y + self.h / 2

@dataclass
class TrackResult:
    """
    TrackResult class is used to store the result of tracking an object.

    Attributes:
    """

    id: list[int]  # List of object IDs
    xywh: list[XYWH]    # List of bounding boxes (XYWH)

    def load_from_dict(self, track_res: dict[int, XYWH]) -> None:
        """
        Initialize the TrackResult from a dictionary.
        """
        self.id = list(track_res.keys())
        self.xywh = [track_res[i] for i in self.id]

@dataclass
class TrackerMessage:
    """
    TrackerMessage class is used to store the message of tracking an object.

    Attributes:
        timestamp(pd.Timestamp): the timestamp of the message.
        trackresult(TrackResult): the result of tracking an object.
    """

    timestamp: pd.Timestamp
    trackresult: TrackResult

def ultralytics_result_to_trackresult(ultralytics_result: Results) -> TrackResult:
    """
    #TODO: To be test.
    Convert a Ultralytics Results object to a TrackResult object.

    Args:
        ultralytics_result(Results): the Ultralytics Results object.

    Returns:
        TrackResult: the TrackResult object.
    """
    res_cpu = ultralytics_result.cpu()
    boxes = res_cpu.boxes
    if boxes is None:
        return TrackResult(id=[], xywh=[])
    if boxes.id is None:
        return TrackResult(id=[], xywh=[])
    ids = boxes.id.tolist()
    xywh = boxes.xywh.tolist()
    xywh_objects = [XYWH(*xywh[i]) for i in range(len(xywh))]
    return TrackResult(id=ids, xywh=xywh_objects)

class ObjectTrace:
    """
    Efficient version of ObjectTrace using List.
    (Dataframe is slow for appending data)

    Attributes:
        object_id(int): the ID of the object.

        frames(list[int]): the list of frame numbers.
        timestamps(list[pd.Timestamp]): the list of timestamps.
        x_lst(list[float]): the list of x coordinates.
        y_lst(list[float]): the list of y coordinates.
        w_lst(list[float]): the list of widths.
        h_lst(list[float]): the list of heights.
    """

    def __init__(self, objid: int) -> None:
        self.object_id = objid
        self.frames = []
        self.timestamps = []
        self.x_lst = []
        self.y_lst = []
        self.w_lst = []
        self.h_lst = []

        self._timestamp_to_index = {}

    def append(self, timestamp: pd.Timestamp, data_dict: dict) -> None:
        """
        Append a new data point to the object trace.

        Args:
            timestamp(pd.Timestamp): the timestamp of the data point.
            data_dict(dict): the data point. "x", "y", "w", "h" and "frame" are required.
        """

        # Ensure the data is appended in order
        if len(self.timestamps) > 0 and timestamp < self.timestamps[-1]:
            raise ValueError("The timestamp is before the last data.")

        if timestamp in self._timestamp_to_index:
            raise ValueError("The timestamp is already in the object trace.")
        
        self._timestamp_to_index[timestamp] = len(self.timestamps)

        self.frames.append(data_dict["frame"])
        self.timestamps.append(timestamp)
        self.x_lst.append(data_dict["x"])
        self.y_lst.append(data_dict["y"])
        self.w_lst.append(data_dict["w"])
        self.h_lst.append(data_dict["h"])

    def get_pos(self, ts: pd.Timestamp, interpolate: bool = True) -> XYWH:
        """
        Get the position of the object at the timestamp.

        Args:
            ts(pd.Timestamp): the timestamp.
            interpolate(bool): whether to interpolate between timestamps.
                If False, will raise ValueError if timestamp not found.

        Returns:
            XYWH: the position of the object at the timestamp.

        Raises:
            ValueError: If no data in trace, timestamp out of range, or 
                      timestamp not found when interpolate=False
        """
        # Check if trace is empty
        if len(self.timestamps) == 0:
            raise ValueError("No data in the object trace.")

        # Check timestamp range
        if ts > self.timestamps[-1]:
            raise ValueError("The timestamp is after the last data.")
        if ts < self.timestamps[0]:
            raise ValueError("The timestamp is before the first data.")

        # Direct lookup if timestamp exists
        if ts in self._timestamp_to_index:
            idx = self._timestamp_to_index[ts]
            return XYWH(self.x_lst[idx], self.y_lst[idx], self.w_lst[idx], self.h_lst[idx])

        if not interpolate:
            raise ValueError("Timestamp not found and interpolation disabled")

        # Estimate initial position using linear interpolation
        # Convert timestamps to nanoseconds for numerical computation
        t0_ns = self.timestamps[0].value
        t1_ns = self.timestamps[-1].value
        ts_ns = ts.value
        n = len(self.timestamps)
        
        # Estimate the index based on uniform distribution assumption
        est_idx = int((ts_ns - t0_ns) / (t1_ns - t0_ns) * (n - 1))
        est_idx = max(0, min(est_idx, n - 2))  # Ensure we have space for interpolation
        
        # Define search range around estimated index
        # Use a window size that grows with the distance from uniform distribution
        window_size = 5  # Base window size
        if n > 1000:  # For large datasets, adjust window based on local uniformity
            # Check local uniformity by comparing actual and expected intervals
            local_interval = self.timestamps[est_idx + 1].value - self.timestamps[est_idx].value
            expected_interval = (t1_ns - t0_ns) / (n - 1)
            uniformity_factor = abs(local_interval - expected_interval) / expected_interval
            window_size = max(5, min(50, int(window_size * (1 + uniformity_factor))))
        
        # Set search boundaries
        left = max(0, est_idx - window_size)
        right = min(n - 1, est_idx + window_size)
        
        # Verify and adjust boundaries if needed
        while left > 0 and self.timestamps[left] > ts:
            left = max(0, left - window_size)
        while right < n - 1 and self.timestamps[right] < ts:
            right = min(n - 1, right + window_size)
            
        # Binary search in the refined range
        while left < right:
            mid = (left + right + 1) // 2
            if self.timestamps[mid] <= ts:
                left = mid
            else:
                right = mid - 1

        # Now timestamps[left] <= ts < timestamps[left + 1]
        if left == len(self.timestamps) - 1:
            raise ValueError("The timestamp is after the last data.")

        # Interpolate between left and left + 1
        alpha = (ts - self.timestamps[left]) / (self.timestamps[left + 1] - self.timestamps[left])
        x = self.x_lst[left] + alpha * (self.x_lst[left + 1] - self.x_lst[left])
        y = self.y_lst[left] + alpha * (self.y_lst[left + 1] - self.y_lst[left])
        w = self.w_lst[left] + alpha * (self.w_lst[left + 1] - self.w_lst[left])
        h = self.h_lst[left] + alpha * (self.h_lst[left + 1] - self.h_lst[left])
        return XYWH(x, y, w, h)
    
    @property
    def end_timestamp(self) -> pd.Timestamp | None:
        """
        Get the end timestamp of the object trace.

        Returns:
            pd.Timestamp: The end timestamp of the object trace.
            None: If the trace is empty.
        """
        if len(self.timestamps) == 0:
            return None
        return self.timestamps[-1]

    @property
    def start_timestamp(self) -> pd.Timestamp | None:
        """
        Get the start timestamp of the object trace.

        Returns:
            pd.Timestamp: The start timestamp of the object trace.
            None: If the trace is empty.
        """
        if len(self.timestamps) == 0:
            return None
        return self.timestamps[0]


    def __len__(self) -> int:
        return len(self.timestamps)

    def has_data_at(self, ts: pd.Timestamp) -> bool:
        """Check if there is data at the specified timestamp.

        Args:
            ts (pd.Timestamp): the timestamp to check

        Returns:
            bool: True if there is data at the timestamp
        """
        return ts in self._timestamp_to_index
