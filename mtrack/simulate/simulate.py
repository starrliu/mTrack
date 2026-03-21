import os

import pandas as pd
from ultralytics.engine.results import Results
import numpy as np
import torch

from ..data import RFIDData, TagData, ObjectTrace, RFIDMessage, TrackerMessage, TrackResult

# pylint: disable=too-many-instance-attributes
class SimulatorFromData:
    """Simulates RFID and camera data playback from recorded datasets."""

    def __init__(
        self,
        rfdata: dict[bytes, TagData],
        objdata: dict[int, ObjectTrace],
        cameradf: pd.DataFrame,
    ):
        self.rfdata = rfdata
        self.objdata = objdata
        self.cameradf = cameradf  # columns: ["TimeStamp", "FrameID"]

        self.start_time = None
        self.end_time = None
        self.prev_time = None
        self.current_time = None
        self.camera_idx = None  # Current row index of cameradf, int

    def set_time_range(self, start_time: pd.Timestamp, end_time: pd.Timestamp):
        """Set the time range for data playback.

        Args:
            start_time (pd.Timestamp): Start time of the playback range.
            end_time (pd.Timestamp): End time of the playback range.
        """
        self.start_time = start_time
        self.end_time = end_time

        self.current_time = None
        self.prev_time = None


    def read_data(self) -> tuple[list[RFIDMessage], TrackerMessage] | None:
        """Read the data from the RFID and camera.

        Returns:
            tuple[list[RFIDMessage], TrackerMessage: list of RFID data and tracker data.
            None: if the time range is not set or the data is not available.
        """
        if self.start_time is None or self.end_time is None:
            return None

        if self.current_time is None:
            # Choose the nearest time in cameradf
            timestamps = pd.to_datetime(self.cameradf["TimeStamp"])
            self.camera_idx = timestamps.sub(self.start_time).abs().idxmin()
            self.current_time = pd.Timestamp(
                self.cameradf.loc[self.camera_idx]["TimeStamp"]
            )

        if self.current_time > self.end_time:
            return None

        # read RFID data
        rfid_messages = []
        if self.prev_time is not None:
            for tagid, tagdata in self.rfdata.items():
                # Get data within time range using the new interface
                data_points = tagdata.get_data_in_range(self.prev_time, self.current_time)
                for ts, data in data_points:
                    rfid_messages.append(RFIDMessage(tagid, ts, data))
            # sort by time
            rfid_messages.sort(key=lambda x: x.timestamp)

        # read camera data
        obj_ids = []
        obj_boxes = []
        for key, obj_trace in self.objdata.items():
            if obj_trace.has_data_at(self.current_time):
                obj_ids.append(key)
                obj_boxes.append(obj_trace.get_pos(self.current_time, interpolate=False))

        track_result = TrackResult(id=obj_ids, xywh=obj_boxes)
        tracker_messages = TrackerMessage(self.current_time, track_result)

        # Update time pointers
        self.prev_time = self.current_time
        self.camera_idx += 1
        if self.camera_idx < len(self.cameradf):
            self.current_time = pd.Timestamp(
                self.cameradf.loc[self.camera_idx]["TimeStamp"]
            )
        else:
            self.current_time = self.end_time + pd.Timedelta(seconds=1)

        return rfid_messages, tracker_messages

    def read_data_ultralytics(self, imgdir: str, shape: tuple[int, int]) -> (
        tuple[
            list[tuple[bytes, pd.Timestamp, RFIDData]],
            tuple[pd.Timestamp, list[Results]],
        ]
        | None
    ):
        # TODO: to be modified
        if self.start_time is None or self.end_time is None:
            return None

        if self.current_time is None:
            # Choose the nearest time in cameradf
            timestamps = pd.to_datetime(self.cameradf["TimeStamp"])

            self.camera_idx = timestamps.sub(self.start_time).abs().idxmin()
            self.current_time = pd.Timestamp(
                self.cameradf.loc[self.camera_idx]["TimeStamp"]
            )

        if self.current_time > self.end_time:
            return None

        # read RFID data
        rfid_data = []
        if self.prev_time is not None:
            for tagid, tagdata in self.rfdata.items():
                rfdata = tagdata.dataframe.loc[self.prev_time : self.current_time]
                for i, row in rfdata.iterrows():
                    data = RFIDData(
                        rssi=row["rssi"],
                        phase=row["phase"],
                        frequency=row["frequency"],
                        antenna_id=row["antenna"],
                    )
                    rfid_data.append((tagid, i, data))
            # sort by time
            rfid_data.sort(key=lambda x: x[1])

        # read camera data
        boxes = []
        random_img = np.random.randint(0, 255, shape, dtype=np.uint8)
        for key in self.objdata.keys():
            if self.current_time in self.objdata[key].dataframe.index:
                x = self.objdata[key].dataframe.loc[self.current_time]["x"]
                y = self.objdata[key].dataframe.loc[self.current_time]["y"]
                w = self.objdata[key].dataframe.loc[self.current_time]["w"]
                h = self.objdata[key].dataframe.loc[self.current_time]["h"]

                x1, y1, x2, y2 = x, y, x + w, y + h
                boxes.append([x1, y1, x2, y2, key, 0, 0])

        boxes_tensor = torch.tensor(boxes)
        # Resize to (N, 7)
        boxes_tensor = boxes_tensor.view(-1, 7)
        img_path = os.path.join(
            imgdir, "%08d.jpg" % self.cameradf.loc[self.camera_idx]["FrameID"]
        )
        res = Results(random_img, img_path, None, boxes_tensor)

        self.prev_time = self.current_time
        self.camera_idx += 1
        if self.camera_idx < len(self.cameradf):
            self.current_time = pd.Timestamp(
                self.cameradf.loc[self.camera_idx]["TimeStamp"]
            )
        else:
            self.current_time = self.end_time + pd.Timedelta(seconds=1)

        return rfid_data, (self.prev_time, [res])

class TrackerSimulator:
    """Simulate tracker data from gt.txt file.

    Attributes:
        object_trace (dict[int, ObjectTrace]): object trace data.
    """

    def __init__(self, object_trace: dict[int, ObjectTrace]):
        self.object_trace = object_trace
        self.start_time = None
        self.end_time = None
        self.current_time = None

    def set_time_range(self, start_time: pd.Timestamp, end_time: pd.Timestamp):
        """Set the time range for data playback.

        Args:
            start_time (pd.Timestamp): Start time of the playback range.
            end_time (pd.Timestamp): End time of the playback range.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.current_time = start_time

    def read_data(self) -> TrackerMessage | None:
        """Read the data from the object trace.

        Returns:
            TrackerMessage: tracker data.
            None: if the time range is not set or the data is not available.
        """
        # Check if time range is set
        if self.start_time is None or self.end_time is None or self.current_time is None:
            return None

        # Check if current time is within range
        if self.current_time > self.end_time:
            return None

        # Get object positions at current time
        obj_ids = []
        obj_boxes = []
        for obj_id, obj_trace in self.object_trace.items():
            if obj_trace.has_data_at(self.current_time):
                obj_ids.append(obj_id)
                obj_boxes.append(obj_trace.get_pos(self.current_time, interpolate=False))

        # Create track result
        track_result = TrackResult(id=obj_ids, xywh=obj_boxes)
        tracker_message = TrackerMessage(self.current_time, track_result)

        # Update current time to next timestamp
        next_time = None
        for obj_trace in self.object_trace.values():
            for ts in obj_trace.timestamps:
                if ts > self.current_time:
                    if next_time is None or ts < next_time:
                        next_time = ts

        self.current_time = next_time if next_time is not None else self.end_time + pd.Timedelta(seconds=1)

        return tracker_message

def load_ultralytics_results(
    gt_txt: str, camera_csv: str, shape: tuple[int, int], imgdir: str
) -> list[tuple[pd.Timestamp, list[Results]]]:
    """
    Load ground truth data.

    Args:
        gt_txt (str): path to the ground truth txt file.
        camera_csv (str): path to the camera csv file.

    Returns:
        list[tuple[pd.Timestamp, list[Results]]]: list of ground truth data.
    """

    gt_data = pd.read_csv(
        gt_txt,
        header=None,
        names=["frame_id", "obj_id", "x", "y", "w", "h", "_", "__", "___"],
    )

    camera_data = pd.read_csv(camera_csv)

    ret = []
    # Random generate 3-channel image
    random_img = np.random.randint(0, 255, shape, dtype=np.uint8)

    for frame_id, group in gt_data.groupby("frame_id"):
        timestamp = pd.Timestamp(
            camera_data[camera_data["FrameID"] == frame_id]["TimeStamp"].values[0]
        )

        boxes = []
        for _, row in group.iterrows():
            x, y, w, h = row["x"], row["y"], row["w"], row["h"]
            obj_id = row["obj_id"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2, obj_id, 0, 0])
            # x1, y1, x2, y2, obj_conf, cls_conf, obj_id

        boxes_tensor = torch.tensor(boxes)
        img_path = os.path.join(imgdir, "%08d.jpg" % frame_id)
        res = Results(random_img, img_path, None, boxes_tensor)
        ret.append((timestamp, [res]))

    return ret
