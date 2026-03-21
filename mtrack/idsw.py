"""ID Switch Detector for Multi-Object Tracking

This module implements an ID switch detector that identifies potential ID switches
in multi-object tracking results. It uses a combination of IoU, detection frequency,
and distance metrics to determine if an ID switch may have occurred.
"""

# pylint: disable=no-member

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import cv2

from .utils import iou, distance_bbox
from .data import XYWH, TrackResult
from .config import IDSWConfig


class IDSWDetector:
    """ID Switch Detector.

    This class try to detect ID switch in tracking results.
    False positive & false negative are possible.
    It uses IoU, detection frequency, and distance to determine if an ID switch occurs.

    Attributes:
        config (IDSWConfig): Configuration for ID switch detection.
        state (dict): Internal state of the detector, including:
            - max_id: Maximum ID assigned so far.
            - id_map: Mapping from YOLO IDs to IDs without ID switch.
            - cur_frame: Current frame number.
            - inactive_id: Set of inactive IDs (IDs no longer actively tracked).
            - cur_res: Tracking results of the current frame.
            - prev_bbox: Previous bounding boxes for each ID. Usage: dict[id] = (frame, bbox).
            - det_freq: Detection frequency of each ID.
    """

    MAX_MISSING_FRAMES = 10

    def __init__(
        self, p2m: float, config: IDSWConfig = None
    ) -> None:
        """
        Initialize the ID switch detector.

        Args:
            p2m (float): Pixel to meter conversion factor.
            config (IDSWConfig): Configuration object. If None, uses default values.
        """
        if config is None:
            config = IDSWConfig()
        
        self.config = config
        self.p2m = p2m

        self.state = {
            "max_id": 0,
            "id_map": {},
            "cur_frame": 0,
            "inactive_id": set(),
            "cur_res": None,
            "prev_bbox": {},
            "det_freq": defaultdict(list),
        }

    def _map_yolo_id_to_mtrack_id(
        self, track_id: int
    ) -> tuple[int, list[int], list[tuple[int, int]]]:
        """Map YOLO ID to ID without ID switch.

        Args:
            track_id (int): ID from YOLO tracking results.

        Returns:
            tuple[int, list[int], list[tuple[int, int]]]:
                - int: ID without ID switch.
                - list[int]: List of new IDs in the current frame.
                - list[tuple[int, int]]: List of ID switches, each tuple is (prev_id, new_id).
        """
        # 如果id不在id_map中，说明是新的id，需要分配新的id
        if track_id not in self.state["id_map"]:
            self.state["max_id"] += 1
            self.state["id_map"][track_id] = self.state["max_id"]
            return self.state["max_id"], [self.state["max_id"]], []

        # 如果id在id_map中，但是对应的id是inactive的，说明发生了ID switch，需要重新分配新的id
        if self.state["id_map"][track_id] in self.state["inactive_id"]:
            self.state["max_id"] += 1
            prev_id = self.state["id_map"][track_id]
            self.state["id_map"][track_id] = self.state["max_id"]
            return (
                self.state["max_id"],
                [self.state["max_id"]],
                [(prev_id, self.state["max_id"])],
            )

        return self.state["id_map"][track_id], [], []

    def update(
        self, track_res: TrackResult
    ) -> tuple[
        dict[int, XYWH], list[tuple[int, XYWH]], list[int], list[tuple[int, int]]
    ]:
        """
        Update ID switch detector with tracking results.

        Args:
            track_res (list[Results]): tracking results of current frame.

        Returns:
            Tuple[Dict[int, XYWH], List[int], List[int]]:
                - Dict[int, XYWH]: tracking results without ID switch.
                - List[tuple[int, XYWH]]: list of ids that are inactive
                    in current frame and their bbox.
                - List[int]: list of ids that are newly added in current frame.
                - List[tuple[int, int]]: list of id switch, each tuple is (id1, id2),
                    which means id1 is switched to id2.
        """

        # Step 1: Map id of YOLO to id without ID switch, and update id_map if there are new ids.
        new_ids_in_cur_frame, idswitch_in_cur_frame, ids_without_switch = (
            self._update_step_1(track_res)
        )

        # Step 2: Update detection frequency of each id
        self._update_step_2(ids_without_switch)

        # Step 3: Calculate iou of each bbox with previous bbox
        ious = self._update_step_3(track_res, ids_without_switch)

        # Step 4: Check the possible ID switch
        inactive_ids_in_cur_frame = []
        # Step 4.1: Check if IOU satisfies the threshold.
        inactive_ids_in_cur_frame.extend(
            self._update_step_4_1(ious, ids_without_switch, track_res)
        )

        # Step 4.2: Check the detection frequency of each id.
        inactive_ids_in_cur_frame.extend(self._update_step_4_2())

        # Check all prev_bbox, if there is some bbox too old, then remove it.
        # Step 5: Clean up prev_bbox
        inactive_ids_in_cur_frame.extend(self._update_step_5())

        # Step 6: Return tracking results without ID switch
        ret_dict = {}
        for id_wo_switch, box in zip(ids_without_switch, track_res.xywh):
            ret_dict[id_wo_switch] = box

        self.state["cur_frame"] += 1
        self.state["cur_res"] = ret_dict

        return (
            ret_dict,
            inactive_ids_in_cur_frame,
            new_ids_in_cur_frame,
            idswitch_in_cur_frame,
        )

    def _update_step_1(
        self, track_res: TrackResult
    ) -> tuple[list[int], list[tuple[int, int]], list[int]]:
        """
        Update step 1 of ID switch detection.

        Args:
            track_res (TrackResult): Tracking results of the current frame.

        Returns:
            tuple[list[int], list[tuple[int, int]], list[int]]:
                - List of new IDs in the current frame.
                - List of ID switches, each tuple is (prev_id, new_id).
                - List of IDs without switch.
        """
        new_ids_in_cur_frame = []
        idswitch_in_cur_frame = []
        ids_without_switch = []

        for track_id in track_res.id:
            mapped_id, new_ids, idswitches = self._map_yolo_id_to_mtrack_id(track_id)
            if new_ids:
                new_ids_in_cur_frame.extend(new_ids)
            if idswitches:
                idswitch_in_cur_frame.extend(idswitches)
            ids_without_switch.append(mapped_id)

        return new_ids_in_cur_frame, idswitch_in_cur_frame, ids_without_switch

    def _update_step_2(self, ids_without_switch: list[int]) -> None:
        """
        Update step 2 of ID switch detection.

        Updates the detection frequency of each ID.

        Args:
            ids_without_switch (list[int]): List of IDs without ID switch.
        """

        # Update detection frequency of each id
        for id_wo_switch in ids_without_switch:
            if id_wo_switch not in self.state["det_freq"]:
                self.state["det_freq"][id_wo_switch] = [1]  # New ID detected
            else:
                if (
                    len(self.state["det_freq"][id_wo_switch])
                    >= self.config.det_fps_thres[1]
                ):
                    self.state["det_freq"][id_wo_switch].pop(0)
                self.state["det_freq"][id_wo_switch].append(1)

        # Update those IDs not detected in the current frame
        for id_wo_switch in self.state["det_freq"]:
            if id_wo_switch not in ids_without_switch:
                if (
                    len(self.state["det_freq"][id_wo_switch])
                    >= self.config.det_fps_thres[1]
                ):
                    self.state["det_freq"][id_wo_switch].pop(0)
                self.state["det_freq"][id_wo_switch].append(0)

    def _update_step_3(
        self, track_res: TrackResult, ids_without_switch: list[int]
    ) -> dict[int, float]:
        """Update step 3 of ID switch detection.

        Update the ious of each bounding box with the previous bounding box.
        """
        ious = {}

        for id_wo_switch, cur_bbox in zip(ids_without_switch, track_res.xywh):
            if id_wo_switch in self.state["prev_bbox"]:
                prev_bbox = self.state["prev_bbox"][id_wo_switch][1]

                ious[id_wo_switch] = iou(prev_bbox, cur_bbox)
                self.state["prev_bbox"][id_wo_switch] = (
                    self.state["cur_frame"],
                    cur_bbox,
                )
            else:
                self.state["prev_bbox"][id_wo_switch] = (
                    self.state["cur_frame"],
                    cur_bbox,
                )

        return ious

    def _is_not_isolated(self, possible_switch_id: int, cur_bbox: XYWH) -> bool:
        """Check if the distance between the current bounding box and
            previous bounding boxes is within the threshold.

        Args:
            possible_switch_id (int): ID to check for switch.
            cur_bbox (XYWH): Current bounding box.

        Returns:
            bool: True if the distance is within the threshold, False otherwise.
        """
        for prev_id, prev_bbox in self.state["prev_bbox"].items():
            if prev_id == possible_switch_id:
                continue
            if prev_bbox[0] > self.state["cur_frame"] - self.MAX_MISSING_FRAMES:
                if (
                    distance_bbox(prev_bbox[1], cur_bbox) * self.p2m
                    < self.config.dis_thres
                ):
                    return True
        return False

    def _update_step_4_1(
        self,
        ious: dict[int, float],
        ids_without_switch: list[int],
        track_res: TrackResult,
    ) -> list[tuple[int, XYWH]]:
        """Update step 4 of ID switch detection.

        Check if the IoU satisfies the threshold.
        This step is integrated into the `update` method for efficiency.
        """
        inactive_ids_in_cur_frame = []

        for idx, id_wo_switch in enumerate(ids_without_switch):
            if id_wo_switch in ious and ious[id_wo_switch] < self.config.iou_thres:
                # Check all prev_bbox, if there is only one bbox within distance self.dis_thres,
                # then not switch
                cur_bbox = track_res.xywh[idx]

                if self._is_not_isolated(id_wo_switch, cur_bbox):
                    self.state["inactive_id"].add(id_wo_switch)
                    inactive_ids_in_cur_frame.append(
                        (id_wo_switch, self.state["prev_bbox"][id_wo_switch][1])
                    )

                    self._remove_id_from_det_freq(id_wo_switch)
                    self._remove_id_from_prev_bbox(id_wo_switch)

        return inactive_ids_in_cur_frame

    def _update_step_4_2(self) -> list[tuple[int, XYWH]]:
        """Update step 4.2 of ID switch detection."""
        inactive_ids_in_cur_frame = []
        to_remove = []

        for id_wo_switch in self.state["det_freq"]:
            if (
                len(self.state["det_freq"][id_wo_switch])
                == self.config.det_fps_thres[1]
            ) and (
                sum(self.state["det_freq"][id_wo_switch]) < self.config.det_fps_thres[0]
            ):
                # Check all prev_bbox, if there is only one bbox within distance 10cm,
                # then not switch
                cur_bbox = self.state["prev_bbox"][id_wo_switch][1]

                if self._is_not_isolated(id_wo_switch, cur_bbox):
                    self.state["inactive_id"].add(id_wo_switch)
                    inactive_ids_in_cur_frame.append(
                        (id_wo_switch, self.state["prev_bbox"][id_wo_switch][1])
                    )

                    # Step 5: Remove inactive id from detection frequency and prev_bbox
                    to_remove.append(id_wo_switch)

        for id_ in to_remove:
            self._remove_id_from_det_freq(id_)
            self._remove_id_from_prev_bbox(id_)

        return inactive_ids_in_cur_frame

    def _update_step_5(self) -> list[tuple[int, XYWH]]:
        """Update step 5 of ID switch detection.

        If there are some identity too long not detected, then remove it.
        """
        inactive_ids_in_cur_frame = []
        to_remove = []
        for prev_id, prev_det in self.state["prev_bbox"].items():
            if prev_det[0] < self.state["cur_frame"] - self.MAX_MISSING_FRAMES:
                # If the bbox is too old, remove it
                to_remove.append(prev_id)

                if prev_id not in self.state["inactive_id"]:
                    self.state["inactive_id"].add(prev_id)
                    inactive_ids_in_cur_frame.append((prev_id, prev_det[1]))

        for tmp_id in to_remove:
            self._remove_id_from_det_freq(tmp_id)
            self._remove_id_from_prev_bbox(tmp_id)

        return inactive_ids_in_cur_frame

    def _remove_id_from_det_freq(self, id_wo_switch: int) -> None:
        """Remove an ID from the detection frequency.

        Args:
            id_wo_switch (int): ID to be removed.
        """
        if id_wo_switch in self.state["det_freq"]:
            del self.state["det_freq"][id_wo_switch]

    def _remove_id_from_prev_bbox(self, id_wo_switch: int) -> None:
        """Remove an ID from the previous bounding boxes.

        Args:
            id_wo_switch (int): ID to be removed.
        """
        if id_wo_switch in self.state["prev_bbox"]:
            del self.state["prev_bbox"][id_wo_switch]

    def annotate_current_frame(self, frame: np.ndarray) -> np.ndarray:
        """Annotate the current frame with tracking results.

        Args:
            frame (np.ndarray): Current frame to annotate.

        Returns:
            np.ndarray: Annotated frame with bounding boxes and IDs.
        """
        for id_, xywh in self.state["cur_res"].items():
            x, y, w, h = int(xywh.x), int(xywh.y), int(xywh.w), int(xywh.h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                str(id_),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            f"Frame: {self.state['cur_frame']}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return frame
