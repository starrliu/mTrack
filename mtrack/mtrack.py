# pylint: disable=no-member
from dataclasses import dataclass
from bidict import bidict
from typing import Dict

import numpy as np
import cv2

from .match import MaxLikelihoodMatch
from .select import SelectiveRead
from .idsw import IDSWDetector
from .data import XYWH, TrackerMessage, RFIDMessage, TrackResult
from .trackletgraph import TrackletGraph, TagIDState
from .checker import GlobalChecker
from .config import MTrackConfig

@dataclass
class MTrackResult:
    """
    MTrack result.

    Attributes:
        frame: The frame number.
        track_res: The tracking results. cv id -> xywh.
        matched_ids: The matched mice. list of (tag epcid, cv id).
        bc_res: The backward identified mice. list of (tag epcid, cv id).
        mismatched_tags: The mismatched mice. list of tag epcid.
    """
    frame: int
    track_res: dict[int, XYWH]
    matched_ids: list[tuple[bytes, int]]
    bc_res: list[tuple[bytes, int]]
    mismatched_tags: list[bytes]

class MTrack:
    """
    MTrack tracks multiple mice with RFID and video data.
    """

    def __init__(
        self,
        tags: dict[bytes, int],
        antpos: dict[int, tuple[float, float, float]],
        p2m: float,
        config: MTrackConfig = None,
        no_sel: bool = False,
    ) -> None:
        if config is None:
            config = MTrackConfig.default()
        
        self.config = config
        self.match = MaxLikelihoodMatch(antpos, p2m, config.match)
        self.idsw = IDSWDetector(p2m, config.idsw)
        tag_lst = list(tags.keys())

        if no_sel:
            self.select = None
        else:
            self.select = SelectiveRead(tag_lst, config=config.select)

        self.tracklet_graph = TrackletGraph(p2m, tag_lst, config.tracklet_graph)

        max_checking_tags = len(tag_lst)

        self.global_checker = GlobalChecker(max_checking_tags, antpos, p2m, config.checker)

        self.tags = tags
        self._cur_res = None
        self._cur_match = bidict()  # type: bidict[int, bytes]
        self._inv_match: Dict[int, bytes] = self._cur_match.inv
        self._cur_frame = 0

    def track(
        self,
        track_res: TrackerMessage,
        rfid_msg: list[RFIDMessage],
        select=True,
        global_check=True,
    ) -> tuple[
        int,
        dict[int, XYWH],
        list[tuple[bytes, int]],
        list[tuple[bytes, int]],
        list[bytes],
    ]:
        """
        Update the RF-MAT state with new data.
        And return the tracking results in this frame.

        Args:
            track_res: The tracking results from the video data.
            rfid_msg: The RFID messages from the RFID reader.

        Returns:
            tuple[int, dict[int, XYWH], list[tuple[bytes, int]], list[tuple[bytes, int]], list[bytes]]:
                - The frame number.
                - The tracking results. cv id -> xywh.
                - The matched mice. list of (tag epcid, cv id).
                - The backward identified mice. list of (tag epcid, cv id).
                - The mismatched mice. list of tag epcid.
        """

        self._cur_frame += 1

        # Selective read
        if select and self.select is not None:
            # rfid_msg = self.select.update(rfid_msg)
            self.select.update(track_res.timestamp, rfid_msg)

        # IDSW
        ts, _ = track_res.timestamp, track_res.trackresult
        res_wo_idsw, inactive_ids, new_ids, idsw = self.idsw.update(track_res.trackresult)

        # Tracklet graph
        self.tracklet_graph.update_from_idswdetector(
            ts, res_wo_idsw, inactive_ids, new_ids, idsw
        )
        candidates = self.tracklet_graph.get_matching_candidates()

        # Match
        inactive_ids_list = [data[0] for data in inactive_ids]
        self.match.update_from_idsm(candidates, inactive_ids_list)
        
        track_res_wo_idsw = TrackResult(id=[], xywh=[])
        track_res_wo_idsw.load_from_dict(res_wo_idsw)
        track_res_wo_idsw = TrackerMessage(ts, track_res_wo_idsw)

        self.match.update_likelihood(rfid_msg, track_res_wo_idsw)
        matched_ids = self.match.get_best_match()

        # Update tracklet graph
        bc_res = self.tracklet_graph.update_from_matchalgo(matched_ids)

        # Global chercker
        mismatched_tags = list()
        if global_check:
            self.global_checker.update_from_graph(
                self.tracklet_graph.matched_tid2vid,
                self.tracklet_graph.active_visual_ids,
            )
            mismatched_tags = self.global_checker.update_from_data(
                rfid_msg, track_res_wo_idsw
            )

            # Update tracklet graph from global checker
            if len(mismatched_tags) > 0:
                self.tracklet_graph.update_from_globalchecker(mismatched_tags)

        # Get unmatched RFID tags and checking tags
        unmatched_tags = self.tracklet_graph.match_status_mgr.tid_state.get_ids_by_status(
            TagIDState.UNMATCHED
        )

        if global_check:
            self.global_checker.update_max_checking_tags(num_of_unmatched_tags=len(unmatched_tags))
            self.global_checker.update_checking_tags()

            checking_tags = set(self.global_checker.checking_tags)
            selected_tags = unmatched_tags.union(checking_tags)
        else:
            selected_tags = unmatched_tags

        if select and self.select is not None:
            # print(f"Selected tags: {[tag.hex() for tag in selected_tags]}")
            self.select.set_selective_tags(selected_tags)

        # Update results
        self._cur_res = res_wo_idsw
        self._cur_match = self.tracklet_graph.match_status_mgr.matched_tid2vid
        self._inv_match = self._cur_match.inv  # 更新反向映射

        mtrack_res = MTrackResult(
            frame=self._cur_frame,
            track_res=res_wo_idsw,
            matched_ids=matched_ids,
            bc_res=bc_res,
            mismatched_tags=mismatched_tags,
        )

        return mtrack_res

    def annotate_frame(
        self,
        frame: np.ndarray,
        plot_tags: bool = True,
        plot_select: bool = True,
    ) -> np.array:
        """
        Annotate the frame with the tracking results.
        Content to be annotated:
            - Bounding boxes of the tracked mice.
            - The ID of each mouse
            - The RFID tag of each matched mouse
            - Unmatched RFID tags
            - Reading tags
        Args:
            frame: The frame to be annotated.

        Returns:
            np.array: The annotated frame.
        """

        # Draw bounding boxes, matched: green, unmatched: red
        for vid, xywh in self._cur_res.items():
            color = (0, 255, 0) if vid in self._inv_match else (0, 0, 255)
            cv2.rectangle(
                frame,
                (int(xywh.x), int(xywh.y)),
                (int(xywh.x + xywh.w), int(xywh.y + xywh.h)),
                color,
                2,
            )
            cv2.putText(
                frame,
                str(vid),
                (int(xywh.x), int(xywh.y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )
            if vid in self._inv_match and plot_tags:
                tag_idx = self.tags[self._inv_match[vid]]
                cv2.putText(
                    frame,
                    str(tag_idx),
                    (int(xywh.x), int(xywh.y + xywh.h)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )

        # If plot_select, draw the reading tags and reading state.
        if plot_select and self.select is not None:
            reading_tags = list(self.select.reading_tags.keys())
            reading_tags = [self.tags[tag] for tag in reading_tags]

            reading_state = self.select.current_reading_state

            cur_timeslot = self.select.cur_timeslot

            cv2.putText(
                frame,
                f"Reading state: {reading_state}",
                (10, 900),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Reading tags: {reading_tags}",
                (10, 930),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Current timeslot: {cur_timeslot}",
                (10, 960),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        return frame

    def close(self):
        """
        Close the RF-MAT.
        """
        if self.select is not None:
            self.select.close()

    def snapshot(self) -> str:
        """
        Get the snapshot of the RF-MAT.
        """

        match_scores = str(self.match)
        global_checker = str(self.global_checker)

        return f"{match_scores}\n{global_checker}"

    def __str__(self) -> str:
        output = "MTrack: \n"
        output += f"Current frame: {self._cur_frame}\n"
        if self.select is not None:
            reading_tags = self.select.reading_tags.keys()
            reading_tags = [tag.hex() for tag in reading_tags]
            output += "Select:\n"
            for tag in reading_tags:
                output += "    {tag}\n"
        return output
