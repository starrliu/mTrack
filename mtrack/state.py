from enum import Enum
from .utils import StatusManager, distance_bbox
from bidict import bidict
from rfmat.utils import Multi2MultiMapping
from collections import defaultdict
import pandas as pd
from .data import XYWH
from .logger import LOGGER
import numpy as np
import cv2

class VisualIDState(Enum):
    UNMATCHED = 0
    MATCHED = 1
    INACTIVE = 2

class TagIDState(Enum):
    UNMATCHED = 0
    MATCHED = 1

class IDStateMachine:
    """
    ID state machine.
    """

    def __init__(self, p2m: float, tags: list[bytes]) -> None:
        self.frame = 0
        self.p2m = p2m # pixel to meter ratio

        self.tags = set(tags)

        # States
        self.vid_state : StatusManager = StatusManager()
        self.tid_state : StatusManager = StatusManager()
        for tag in tags:
            self.tid_state.add_object(tag, TagIDState.UNMATCHED)
    
        self.matched_tid2vid : bidict = bidict()   # Mapping from tag ID to visual ID

        self.tag_unmathed_records = {}  # Tag ID to where and when it was unmatched

        self.unmatched_tid2vid = Multi2MultiMapping(bytes, int)  # Unmatched tag ID to visual ID
        for tag in tags:
            self.unmatched_tid2vid.add(tag, -1) # -1 means tag should be matched any unmatched visual ID

        # thresholds
        self._recent_time_ms = pd.Timedelta('1s')
        self._near_distance_m = 0.20

        self._recent_new_vids : dict[int, (pd.Timestamp, XYWH)] = {}  # Visual IDs that are recently detected
        self._recent_inactive_vids : dict[int, (pd.Timestamp, XYWH)] = {}  # Visual IDs that are recently inactive
        self._inactive_vids_to_tags : dict[int, set[bytes]] = {}  # Visual IDs that are inactive and their possible tag IDs

        # Record but not used
        self.vid_predecessors = defaultdict(set)  # Visual ID to its predecessors

    def update_from_idswdetector(self, ts: pd.Timestamp, mot_data: dict[int, XYWH], inactive_ids: list[tuple[int, XYWH]], 
                                                       new_ids: list[int], idsw: list[tuple[int, int]]):
        """
        Update ID state machine from IDSW detector results.

        Args:
            ts: Timestamp of the current frame.
            mot_data: MOT data from IDSW detector. mot_data[i] is the bounding box of object i.
            inactive_ids: List of inactive IDs in the current frame.
            new_ids: List of new IDs in the current frame.
            idsw: List of IDSW results. idsw[i] is the tuple of (old_id, new_id).

        Returns: the output of this function is the input of the matching algorithm.
            List of tuples of (tag_id, visual_id): The matching candidates.
            List of inactive visual IDs: The visual IDs that are inactive in the current frame.
        """

        # Algorithm:
        # 1. If a new visual ID is detected:
        #    - Try to find its possible predecessors in the recent inactive IDs.
        #    - If found, its possible tag IDs contains the same tag IDs as the predecessors.
        #    - If not found in time limit, its possible tag IDs are all the unmatched tags. (set to b'0')
        #    - For tag with candidate -1, add the visual ID to the candidate list.
        # 2. If a visual ID is inactive:
        #    - If it is matched, update its state to unmatched. And update the corresponding tag ID state to unmatched.
        #       - Record the time and location where it was unmatched.
        #       - If the tag did not find candidate in time limit, its possible visual ID is all the unmatched visual IDs. (set to -1)
        #    - If it is unmatched, update its state to inactive.
        # 3. Returns:
        #    - List of tuples of (tag_id, visual_id): The matching candidates generated from self.unmatched_tid2vid.

        self.frame += 1

        # 1. If a inactivated visual ID is detected:
        for vid, xywh in inactive_ids:
            prev_state = self.vid_state.get_status(vid)
            if prev_state == VisualIDState.MATCHED:
                print("Matched visual ID {} is inactive.".format(vid))
                tid = self.matched_tid2vid.inv[vid]
                self.tid_state.add_object(tid, TagIDState.UNMATCHED)
                self.matched_tid2vid.pop(tid)
                self.vid_state.add_object(vid, VisualIDState.INACTIVE)

                self.tag_unmathed_records[tid] = (ts, xywh)
        
                self._inactive_vids_to_tags[vid] = {tid}

                self._recent_inactive_vids[vid] = (ts, xywh)
            elif prev_state == VisualIDState.UNMATCHED:
                self.vid_state.add_object(vid, VisualIDState.INACTIVE)
                
                self._recent_inactive_vids[vid] = (ts, xywh)

                possible_tids = self.unmatched_tid2vid.get_inverse(vid) # may be b'0'
                self._inactive_vids_to_tags[vid] = possible_tids

                self.unmatched_tid2vid.remove_key_b(vid)

        # 2. If a new visual ID is detected:
        for vid in new_ids:
            self.vid_state.add_object(vid, VisualIDState.UNMATCHED)
            self._recent_new_vids[vid] = (ts, mot_data[vid])
            
        # 3. Matching candidates
        for old_vid, new_vid in idsw:
            self.vid_predecessors[new_vid].add(old_vid)
            # print("Status of new_vid: ", self.vid_state.get_status(new_vid))
            # print("Status of old_vid: ", self.vid_state.get_status(old_vid))
            if self.vid_state.get_status(new_vid) == VisualIDState.UNMATCHED and self.vid_state.get_status(old_vid) == VisualIDState.INACTIVE:
                possible_tids = {}
                if old_vid in self._inactive_vids_to_tags:
                    possible_tids = self._inactive_vids_to_tags[old_vid]
                    # print("Predecessor found.")
                    # print("new_vid: ", new_vid)
                    # print("old_vid: ", old_vid)
                    # print("possible_tids: ", possible_tids)
                else:
                    pass
                    # print("Predecessor not found.")
                    # print("new_vid: ", new_vid)
                    # print("old_vid: ", old_vid)
                
                for tid in possible_tids:
                    if tid == b'0':
                        self.unmatched_tid2vid.add(tid, new_vid)
                    elif self.tid_state.get_status(tid) != TagIDState.MATCHED:
                        self.unmatched_tid2vid.add(tid, new_vid)
            
        for new_id in self._recent_new_vids:
            # Find possible predecessors
            for inactive_id in self._recent_inactive_vids:
                if inactive_id in self.vid_predecessors[new_id]:
                    continue
                if inactive_id in self._recent_new_vids:
                    # If recent inactive ID is born after the new ID, it cannot be the predecessor
                    if self._recent_inactive_vids[inactive_id][0] > self._recent_new_vids[new_id][0]:
                        continue

                # Check if the distance is near
                new_xywh = self._recent_new_vids[new_id][1]
                inactive_xywh = self._recent_inactive_vids[inactive_id][1]

                dis = distance_bbox(new_xywh, inactive_xywh) * self.p2m

                if dis < self._near_distance_m:
                    self.vid_predecessors[new_id].add(inactive_id)

                    if self.vid_state.get_status(new_id) == VisualIDState.UNMATCHED:

                        possible_tids = self._inactive_vids_to_tags[inactive_id]

                        for tid in possible_tids:
                            if tid == b'0':
                                self.unmatched_tid2vid.add(tid, new_id)
                            elif self.tid_state.get_status(tid) != TagIDState.MATCHED:
                                self.unmatched_tid2vid.add(tid, new_id)

        # Clear recent records if they are too old
        cur_time = ts
        for vid, (ts_, data) in list(self._recent_new_vids.items()):
            if cur_time - ts_ > self._recent_time_ms:
                self._recent_new_vids.pop(vid)
            # If the visual ID is not matched and no possible tag ID is found, add it to the candidate list
            if self.vid_state.get_status(vid) == VisualIDState.UNMATCHED:
                possible_tids = self.unmatched_tid2vid.get_inverse(vid)
                if len(possible_tids) == 0:
                    self.unmatched_tid2vid.add(b'0', vid)

        for vid, (ts_, data) in list(self._recent_inactive_vids.items()):
            if cur_time - ts_ > self._recent_time_ms:
                self._recent_inactive_vids.pop(vid)
                self._inactive_vids_to_tags.pop(vid) 

        # If a tag has no candidate in time limit, set it to all the unmatched visual IDs
        for tid, (ts_, data) in list(self.tag_unmathed_records.items()):
            if cur_time - ts_ > self._recent_time_ms:
                vid_candidates = self.unmatched_tid2vid.get(tid)
                if len(vid_candidates) == 0:
                    self.unmatched_tid2vid.add(tid, -1)
                self.tag_unmathed_records.pop(tid)

        # Step 3: Generate matching candidates
        # If a tag has candidate -1, add all the unmatched visual IDs to the candidate list
        # If a visual ID has candidate b'0', add all the unmatched tag IDs to the candidate list
        matching_candidates = []
        
        vid_with_candidate_b0 = self.unmatched_tid2vid.get(b'0')
        for vid in vid_with_candidate_b0:
            for tid in self.tid_state.get_ids_by_status(TagIDState.UNMATCHED):
                matching_candidates.append((tid, vid))
        
        tid_with_candidate_minus1 = self.unmatched_tid2vid.get_inverse(-1)
        for tid in tid_with_candidate_minus1:
            for vid in self.vid_state.get_ids_by_status(VisualIDState.UNMATCHED):
                if vid not in vid_with_candidate_b0:
                    matching_candidates.append((tid, vid))

        for tid, candidates in self.unmatched_tid2vid.items():
            if tid == b'0' :
                continue
            for vid in candidates:
                if vid == -1:
                    continue
                matching_candidates.append((tid, vid))
        
        return matching_candidates
    
    def update_from_matchalgo(self, match_res: list[tuple[bytes, int]]):
        """
        Update ID state machine from matching algorithm results.

        Args:
            match_res: Matching results. match_res[i] is the tuple of (tag_id, visual_id).
        """

        for tid, vid in match_res:
            if tid not in self.tags:
                raise ValueError("Tag ID not in the tag list.")

            self.unmatched_tid2vid.remove_key_a(tid)
            self.unmatched_tid2vid.remove_key_b(vid)
            self.matched_tid2vid[tid] = vid
            self.tid_state.add_object(tid, TagIDState.MATCHED)
            self.vid_state.add_object(vid, VisualIDState.MATCHED)

            if tid in self.tag_unmathed_records:
                self.tag_unmathed_records.pop(tid)

    @property
    def matched_tid(self) -> set[bytes]:
        """
        Get matched tag IDs.
        """
        return self.tid_state.get_ids_by_status(TagIDState.MATCHED)

    @property
    def unmatched_tid(self) -> set[bytes]:
        """
        Get unmatched tag IDs.
        """
        return self.tid_state.get_ids_by_status(TagIDState.UNMATCHED)

    def __str__(self) -> str:
        output = ""
        output += f"Frame: {self.frame}\n"
        output += "Visual ID states:\n"
        for status, vids in self.vid_state.inverse_items():
            output += f"  {status.name}: {vids}\n"

        output += "Tag ID states:\n"
        for status, tids in self.tid_state.inverse_items():
            output += f"  {status.name}: {[tid.hex() for tid in tids]}\n"
        
        output += "Matched tag ID to visual ID:\n"
        for tid, vid in self.matched_tid2vid.items():
            output += f"  {tid.hex()}: {vid}\n"
        
        output += "Unmatched tag ID to visual ID:\n"
        for tid, vids in self.unmatched_tid2vid.items():
            output += f"  {tid.hex()}: {vids}\n"

        output += "Inactive visual ID to tag ID:\n"
        for vid, tids in self._inactive_vids_to_tags.items():
            output += f"  {vid}: {[tid.hex() for tid in tids]}\n"
        
        return output
    
    def annotate_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Annotate the frame with the ID state machine results.

        Args:
            frame: The frame to be annotated.

        Returns:
            np.ndarray: The annotated frame.
        """

        # Put text of unmatched_tid2vid at the left bottom
        num_rows = len(self.unmatched_tid2vid.map)
        for i, (tid, vids) in enumerate(self.unmatched_tid2vid.items()):
            text = f"{tid.hex()}: {vids}"
            cv2.putText(frame, text, (10, frame.shape[0] - 10 - 35 * (num_rows - i)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame