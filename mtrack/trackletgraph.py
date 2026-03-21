"""Tracklet graph module for managing visual ID and tag ID matching."""

from enum import Enum
from typing import Optional

import networkx as nx
import pandas as pd
from bidict import bidict

from .data import XYWH
from .utils import Multi2MultiMapping, StatusManager, distance_bbox
from .config import TrackletGraphConfig
from .logger import LOGGER


class VisualIDState(Enum):
    """Visual ID states in the tracking system."""

    UNMATCHED = 0
    MATCHED = 1
    INACTIVE_UNMATCHED = 2
    INACTIVE_MATCHED = 3


class TagIDState(Enum):
    """Tag ID states in the tracking system."""

    UNMATCHED = 0
    MATCHED = 1


class TrackletAttributes:
    """Attributes for a tracklet in the system."""

    def __init__(
        self, tracklet_id: int, state: VisualIDState = VisualIDState.UNMATCHED
    ) -> None:
        """Initialize tracklet attributes.

        Args:
            tracklet_id: The ID of the tracklet
            state: Initial state of the tracklet
        """
        self.tracklet_id = tracklet_id
        self.cur_state = state
        self.possible_tag_ids: set[bytes] = set()
        self.end_time: Optional[pd.Timestamp] = None  # valid only when inactive

    def add_tag_id(self, tag_id: bytes) -> None:
        """Add a possible tag ID to this tracklet."""
        self.possible_tag_ids.add(tag_id)

    def set_status(self, state: VisualIDState) -> None:
        """Update the state of this tracklet."""
        self.cur_state = state


class TrackletGraphStructure:
    """Manages the tracklet graph structure and its attributes.

    This class encapsulates the graph structure that records spatial-temporal relationships
    between tracklets and their attributes.

    Attributes:
        graph (nx.DiGraph): Directed graph recording tracklet relationships
        attributes (dict[int, TrackletAttributes]): Attributes for each tracklet
    """

    def __init__(self) -> None:
        """Initialize the tracklet graph structure."""
        self.graph = nx.DiGraph()
        self.attributes: dict[int, TrackletAttributes] = {}

    def add_tracklet(
        self, vid: int, state: VisualIDState = VisualIDState.UNMATCHED
    ) -> None:
        """Add a new tracklet to the graph.

        Args:
            vid: Visual ID of the tracklet
            state: Initial state of the tracklet
        """
        self.graph.add_node(vid)
        self.attributes[vid] = TrackletAttributes(vid, state)

    def add_edge(self, old_vid: int, new_vid: int) -> None:
        """Add a directed edge between two tracklets.

        Args:
            old_vid: Source tracklet visual ID
            new_vid: Target tracklet visual ID
        """
        self.graph.add_edge(old_vid, new_vid)

    def get_predecessors(self, vid: int) -> list[int]:
        """Get predecessor tracklets of a given tracklet.

        Args:
            vid: Visual ID of the tracklet

        Returns:
            List of predecessor tracklet IDs
        """
        return list(self.graph.predecessors(vid))

    def get_successors(self, vid: int) -> list[int]:
        """Get successor tracklets of a given tracklet.

        Args:
            vid: Visual ID of the tracklet

        Returns:
            List of successor tracklet IDs
        """
        return list(self.graph.successors(vid))

    def update_tracklet_status(
        self, vid: int, state: VisualIDState, end_time: Optional[pd.Timestamp] = None
    ) -> None:
        """Update the status of a tracklet.

        Args:
            vid: Visual ID of the tracklet
            state: New state
            end_time: Optional end time for inactive tracklets
        """
        self.attributes[vid].set_status(state)
        if end_time is not None:
            self.attributes[vid].end_time = end_time

    def update_possible_tags(self, vid: int, tag_ids: set[bytes]) -> None:
        """Update possible tag IDs for a tracklet.

        Args:
            vid: Visual ID of the tracklet
            tag_ids: Set of possible tag IDs
        """
        self.attributes[vid].possible_tag_ids = tag_ids

    def union_possible_tags(self, vid: int, tag_ids: set[bytes]) -> None:
        """Union possible tag IDs for a tracklet.

        Args:
            vid: Visual ID of the tracklet
            tag_ids: Set of possible tag IDs
        """
        self.attributes[vid].possible_tag_ids.update(tag_ids)

    def add_possible_tag(self, vid: int, tag_id: bytes) -> None:
        """Add a possible tag ID to a tracklet.

        Args:
            vid: Visual ID of the tracklet
            tag_id: Tag ID to add
        """
        self.attributes[vid].possible_tag_ids.add(tag_id)


class MatchStatusManager:
    """Manages matching status between visual IDs and tag IDs.

    Attributes:
        vid_state (StatusManager): Status manager for visual IDs.
        tid_state (StatusManager): Status manager for tag IDs.
        matched_tid2vid (bidict[bytes, int]): Bidirectional mapping between tag IDs and visual IDs.
        unmatched_tid2vid (Multi2MultiMapping[bytes, int]):
            Multi-to-multi mapping between unmatched tag IDs and visual IDs
            (only for active visual IDs).
        tag_unmatched_records (dict[bytes, tuple[pd.Timestamp, XYWH]]):
            Records of unmatched tag IDs and their timestamps and bounding boxes
            when they are unmatched.
    """

    def __init__(self, tags: list[bytes]) -> None:
        """Initialize the match status manager.

        Args:
            tags: List of tag IDs to manage
        """
        self.vid_state: StatusManager = StatusManager()
        self.tid_state: StatusManager = StatusManager()
        self.matched_tid2vid: bidict[bytes, int] = bidict()
        self.unmatched_tid2vid = Multi2MultiMapping(bytes, int)
        self.tag_unmatched_records: dict[bytes, tuple[pd.Timestamp, XYWH]] = {}

        # Initialize tag states
        for tag in tags:
            self.tid_state.add_object(tag, TagIDState.UNMATCHED)
            self.tag_unmatched_records[tag] = (pd.Timestamp.min, XYWH(0, 0, 0, 0))

    def update_matched_status(self, tid: bytes, vid: int) -> None:
        """Update status when a match is found."""
        self.tid_state.add_object(tid, TagIDState.MATCHED)
        self.vid_state.add_object(vid, VisualIDState.MATCHED)
        self.matched_tid2vid[tid] = vid
        if tid in self.tag_unmatched_records:
            self.tag_unmatched_records.pop(tid)
        self.unmatched_tid2vid.remove_key_a(tid)
        self.unmatched_tid2vid.remove_key_b(vid)

    def update_unmatched_status(
        self, tid: bytes, vid: int, ts: pd.Timestamp, xywh: XYWH
    ) -> None:
        """Update status when a match is lost."""
        self.tid_state.add_object(tid, TagIDState.UNMATCHED)
        self.vid_state.add_object(vid, VisualIDState.UNMATCHED)
        if tid in self.matched_tid2vid:
            self.matched_tid2vid.pop(tid)
        if ts != None and xywh != None:
            self.tag_unmatched_records[tid] = (ts, xywh)

    def update_visual_id_status(self, vid: int, state: VisualIDState) -> None:
        """Update status when a visual ID becomes inactive."""
        self.vid_state.add_object(vid, state)


class RecentStateManager:
    """Manages recent state information for visual IDs.

    Attributes:
        recent_new_vids (dict[int, tuple[pd.Timestamp, XYWH]]):
            Records of newly detected visual IDs and their timestamps and bounding boxes.
        recent_inactive_vids (dict[int, tuple[pd.Timestamp, XYWH]]):
            Records of inactive visual IDs and their timestamps and bounding boxes.
        recent_time_thres (pd.Timedelta): Time threshold for considering states as recent.
    """

    def __init__(self, recent_time_thres: pd.Timedelta) -> None:
        """Initialize the recent state manager.

        Args:
            recent_time_thres: Time threshold for considering states as recent
        """
        self.recent_new_vids: dict[int, tuple[pd.Timestamp, XYWH]] = {}
        self.recent_inactive_vids: dict[int, tuple[pd.Timestamp, XYWH]] = {}
        self.recent_time_thres = recent_time_thres

    def add_new_vid(self, vid: int, ts: pd.Timestamp, xywh: XYWH) -> None:
        """Add a newly detected visual ID."""
        self.recent_new_vids[vid] = (ts, xywh)

    def add_inactive_vid(self, vid: int, ts: pd.Timestamp, xywh: XYWH) -> None:
        """Add an inactive visual ID."""
        self.recent_inactive_vids[vid] = (ts, xywh)

    def clean_old_records(self, cur_time: pd.Timestamp) -> None:
        """Clean records older than the time threshold."""
        for vid, (ts, _) in list(self.recent_new_vids.items()):
            if cur_time - ts > self.recent_time_thres:
                self.recent_new_vids.pop(vid)

        for vid, (ts, _) in list(self.recent_inactive_vids.items()):
            if cur_time - ts > self.recent_time_thres:
                self.recent_inactive_vids.pop(vid)


class TrackletGraph:
    """Graph structure for managing tracklet relationships and matching."""

    def __init__(self, p2m: float, tags: list[bytes], config: TrackletGraphConfig = None) -> None:
        """Initialize the tracklet graph.

        Args:
            p2m: Pixels to meters conversion factor
            tags: List of tag IDs to manage
            config: TrackletGraph configuration object
        """
        if config is None:
            config = TrackletGraphConfig()
        
        self.config = config
        self.frame = 0
        self.p2m = p2m
        self.tags = set(tags)

        # Initialize managers
        self.match_status_mgr = MatchStatusManager(tags)
        self.recent_state_mgr = RecentStateManager(pd.Timedelta("1s"))
        self.tracklet_structure = TrackletGraphStructure()

        # Thresholds
        self._too_old_time_thres = pd.Timedelta("5min")
        self._near_distance_m = 0.20

    def update_from_idswdetector(
        self,
        ts: pd.Timestamp,
        mot_data: dict[int, XYWH],
        inactive_ids: list[tuple[int, XYWH]],
        new_ids: list[int],
        idsw: list[tuple[int, int]],
    ) -> None:
        """Update the tracklet graph from the IDSW detector.

        Args:
            ts: The timestamp of the current frame
            mot_data: The MOT data of the current frame
            inactive_ids: The inactive visual IDs and their bounding boxes
            new_ids: The new visual IDs
            idsw: The IDSW data of the current frame

        Algorithm:
            1. If an inactivated visual ID is detected:
                - If matched, update state to INACTIVE_MATCHED
                - If unmatched, update state to inactive
            2. If a new visual ID is detected:
                - Try to find possible predecessors in recent inactive IDs
                - If found, possible tag IDs contain same tag IDs as predecessors
                - If not found in time limit, possible tag IDs are all current unmatched tags
            3. Clean inactive nodes that are too old
        """

        def step1_inactive_matched(vid: int, xywh: XYWH) -> None:
            """Handle matched visual ID becoming inactive."""
            tid = self.match_status_mgr.matched_tid2vid.inv[vid]
            self.match_status_mgr.update_unmatched_status(tid, vid, ts, xywh)
            self.match_status_mgr.update_visual_id_status(
                vid, VisualIDState.INACTIVE_MATCHED
            )
            self.recent_state_mgr.add_inactive_vid(vid, ts, xywh)
            self.tracklet_structure.update_possible_tags(vid, {tid})
            self.tracklet_structure.update_tracklet_status(
                vid, VisualIDState.INACTIVE_MATCHED, ts
            )

        def step1_inactive_unmatched(vid: int, xywh: XYWH) -> None:
            """Handle unmatched visual ID becoming inactive."""
            self.match_status_mgr.update_visual_id_status(
                vid, VisualIDState.INACTIVE_UNMATCHED
            )
            possible_tids = self.match_status_mgr.unmatched_tid2vid.get_inverse(vid)
            self.recent_state_mgr.add_inactive_vid(vid, ts, xywh)
            self.tracklet_structure.update_possible_tags(vid, possible_tids)
            self.tracklet_structure.update_tracklet_status(
                vid, VisualIDState.INACTIVE_UNMATCHED, ts
            )
            self.match_status_mgr.unmatched_tid2vid.remove_key_b(vid)

        def step2_add_new_vid(vid: int) -> None:
            """Add a new visual ID to the system."""
            self.match_status_mgr.update_visual_id_status(vid, VisualIDState.UNMATCHED)
            self.recent_state_mgr.add_new_vid(vid, ts, mot_data[vid])
            self.tracklet_structure.add_tracklet(vid, VisualIDState.UNMATCHED)

        def step3_update_from_idsw(old_vid: int, new_vid: int) -> None:
            """Update possible tag IDs from IDSW information."""
            self.tracklet_structure.add_edge(old_vid, new_vid)

            if self.match_status_mgr.vid_state.get_status(
                new_vid
            ) == VisualIDState.UNMATCHED and self.match_status_mgr.vid_state.get_status(
                old_vid
            ) in {
                VisualIDState.INACTIVE_UNMATCHED,
                VisualIDState.INACTIVE_MATCHED,
            }:
                old_possible_tag_ids = self.tracklet_structure.attributes[
                    old_vid
                ].possible_tag_ids

                self.tracklet_structure.union_possible_tags(
                    new_vid, old_possible_tag_ids
                )

                for tid in self.tracklet_structure.attributes[new_vid].possible_tag_ids:
                    if (
                        self.match_status_mgr.tid_state.get_status(tid)
                        == TagIDState.UNMATCHED
                    ):
                        self.match_status_mgr.unmatched_tid2vid.add(tid, new_vid)

        def step3_find_predecessors() -> None:
            """Find possible predecessor nodes for new visual IDs."""
            for new_id, (
                new_ts,
                new_xywh,
            ) in self.recent_state_mgr.recent_new_vids.items():
                for inactive_id, (
                    _,
                    inactive_xywh,
                ) in self.recent_state_mgr.recent_inactive_vids.items():
                    if new_id == inactive_id:
                        continue

                    if (
                        inactive_id in self.recent_state_mgr.recent_new_vids
                        and self.recent_state_mgr.recent_new_vids[inactive_id][0]
                        > new_ts
                    ):
                        continue

                    distance = distance_bbox(new_xywh, inactive_xywh) * self.p2m

                    if distance < self._near_distance_m:
                        self.tracklet_structure.add_edge(inactive_id, new_id)

                        if (
                            self.match_status_mgr.vid_state.get_status(new_id)
                            == VisualIDState.UNMATCHED
                        ):
                            possible_tids = self.tracklet_structure.attributes[
                                inactive_id
                            ].possible_tag_ids
                            self.tracklet_structure.union_possible_tags(
                                new_id, possible_tids
                            )

                            for tid in possible_tids:
                                if (
                                    self.match_status_mgr.tid_state.get_status(tid)
                                    == TagIDState.UNMATCHED
                                ):
                                    self.match_status_mgr.unmatched_tid2vid.add(
                                        tid, new_id
                                    )

        def step5_update_unmatched_vids() -> None:
            """Update possible tags for unmatched visual IDs."""
            for vid, _ in list(self.recent_state_mgr.recent_new_vids.items()):
                if (
                    self.match_status_mgr.vid_state.get_status(vid)
                    == VisualIDState.UNMATCHED
                ):
                    possible_tids = self.match_status_mgr.unmatched_tid2vid.get_inverse(
                        vid
                    )
                    if not possible_tids:
                        all_unmatched_tids = (
                            self.match_status_mgr.tid_state.get_ids_by_status(
                                TagIDState.UNMATCHED
                            )
                        )
                        self.tracklet_structure.update_possible_tags(
                            vid, all_unmatched_tids
                        )
                        for tid in all_unmatched_tids:
                            self.match_status_mgr.unmatched_tid2vid.add(tid, vid)

        def step6_handle_no_candidates() -> None:
            """Handle tags that have no candidate visual IDs."""
            for tid, (unmatched_ts, _) in list(
                self.match_status_mgr.tag_unmatched_records.items()
            ):
                vid_candidates = self.match_status_mgr.unmatched_tid2vid.get(tid)
                if not vid_candidates:
                    if (
                        unmatched_ts == pd.Timestamp.min
                        or ts - unmatched_ts > self.recent_state_mgr.recent_time_thres
                    ):
                        unmatched_vids = (
                            self.match_status_mgr.vid_state.get_ids_by_status(
                                VisualIDState.UNMATCHED
                            )
                        )
                        for vid in unmatched_vids:
                            self.tracklet_structure.add_possible_tag(vid, tid)
                            self.match_status_mgr.unmatched_tid2vid.add(tid, vid)

                        self.match_status_mgr.tag_unmatched_records.pop(tid)

        self.frame += 1

        # Step 1: Handle inactive visual IDs
        for vid, xywh in inactive_ids:
            prev_state = self.match_status_mgr.vid_state.get_status(vid)
            if prev_state == VisualIDState.MATCHED:
                step1_inactive_matched(vid, xywh)
            elif prev_state == VisualIDState.UNMATCHED:
                step1_inactive_unmatched(vid, xywh)

        # Step 2: Handle new visual IDs
        for vid in new_ids:
            step2_add_new_vid(vid)

        # Step 3: Update from IDSW and find predecessors
        for old_vid, new_vid in idsw:
            step3_update_from_idsw(old_vid, new_vid)
        step3_find_predecessors()

        # Step 4: Clean old records
        self.recent_state_mgr.clean_old_records(ts)

        # Step 5: Update unmatched visual IDs
        step5_update_unmatched_vids()

        # Step 6: Handle tags with no candidates
        step6_handle_no_candidates()

    def get_matching_candidates(self) -> list[tuple[bytes, int]]:
        """Get all possible matching candidates between tags and visual IDs.

        Returns:
            List of (tag_id, visual_id) tuples representing possible matches
        """
        if self.config.graph_on:
            matching_candidates = []
            for tag, candidates in self.match_status_mgr.unmatched_tid2vid.items():
                for vid in candidates:
                    matching_candidates.append((tag, vid))
            return matching_candidates

        # For testing without graph functionality
        unmatched_tids = self.match_status_mgr.tid_state.get_ids_by_status(
            TagIDState.UNMATCHED
        )
        unmatched_vids = self.match_status_mgr.vid_state.get_ids_by_status(
            VisualIDState.UNMATCHED
        )
        return [(tid, vid) for tid in unmatched_tids for vid in unmatched_vids]

    def update_from_matchalgo(
        self, match_res: list[tuple[bytes, int]]
    ) -> list[tuple[bytes, int]]:
        """Update tracklet graph based on matching algorithm results.

        Args:
            match_res: List of (tag_id, visual_id) tuples representing matches

        Returns:
            List of additional matches found through backward identification

        Raises:
            ValueError: If a tag ID is not in the managed tag set
        """
        if not match_res:
            return []

        for tid, vid in match_res:
            if tid not in self.tags:
                raise ValueError("The tag ID is not in the tag set.")

            self.match_status_mgr.unmatched_tid2vid.remove_key_a(tid)
            self.match_status_mgr.unmatched_tid2vid.remove_key_b(vid)
            self.match_status_mgr.update_matched_status(tid, vid)

            self.tracklet_structure.attributes[vid].set_status(VisualIDState.MATCHED)
            self.tracklet_structure.attributes[vid].possible_tag_ids = {tid}

        backward_matches = self.backward_identify([vid for _, vid in match_res])

        return backward_matches

    def update_from_globalchecker(self, mismatched_tags: list[bytes]) -> None:
        """Update tracklet graph based on global checker results.

        Args:
            mismatched_tags: List of tag IDs that were incorrectly matched
        """
        for tid in mismatched_tags:
            if tid not in self.match_status_mgr.matched_tid2vid:
                continue

            vid = self.match_status_mgr.matched_tid2vid[tid]

            # Update status
            self.match_status_mgr.update_unmatched_status(tid, vid, None, None)

            # Update possible tag IDs
            all_unmatched_tids = self.match_status_mgr.tid_state.get_ids_by_status(
                TagIDState.UNMATCHED
            )
            self.tracklet_structure.attributes[vid].cur_state = VisualIDState.UNMATCHED
            self.tracklet_structure.attributes[vid].possible_tag_ids = (
                all_unmatched_tids
            )

            for unmatched_tid in all_unmatched_tids:
                self.match_status_mgr.unmatched_tid2vid.add(unmatched_tid, vid)

            # Update unmatched visual IDs
            unmatched_vids = self.match_status_mgr.vid_state.get_ids_by_status(
                VisualIDState.UNMATCHED
            )
            for unmatched_vid in unmatched_vids:
                self.tracklet_structure.union_possible_tags(unmatched_vid, {tid})
                self.match_status_mgr.unmatched_tid2vid.add(tid, unmatched_vid)

    def backward_identify(
        self, newly_matched_vid: list[int]
    ) -> list[tuple[bytes, int]]:
        """Identify matches by looking at predecessors of newly matched visual IDs.

        Args:
            newly_matched_vid: List of visual IDs that were just matched

        Returns:
            List of additional (tag_id, visual_id) matches found

        Raises:
            ValueError: If a visual ID is not inactive when expected
        """
        max_depth = 1000

        matched_lst = []

        debug = 0
        if 159 in newly_matched_vid:
            debug = 1

        # Find predecessors of newly matched visual IDs
        predecessors = {
            predecessor
            for vid in newly_matched_vid
            for predecessor in self.tracklet_structure.graph.predecessors(vid)
            if self.match_status_mgr.vid_state.get_status(predecessor)
            == VisualIDState.INACTIVE_UNMATCHED
        }
        if debug:
            predecessor_of_159 = self.tracklet_structure.get_predecessors(159)
            LOGGER.info(f"predecessor_of_159: {predecessor_of_159}")

        # Check if predecessors have all successors matched
        to_be_back_identified_queue = list(predecessors)

        depth = 0
        while to_be_back_identified_queue:
            if depth > max_depth:
                raise RuntimeError(
                    "The depth of the backward identification is too large."
                )

            depth += 1
            if debug:
                LOGGER.info(f"Backward the vid: {to_be_back_identified_queue[0]}")
            vid = to_be_back_identified_queue.pop(0)
            successors = self.tracklet_structure.get_successors(vid)
            if debug:
                LOGGER.info(f"successors: {successors}")

            if all(
                self.tracklet_structure.attributes[successor].cur_state
                in {VisualIDState.MATCHED, VisualIDState.INACTIVE_MATCHED}
                for successor in successors
            ):
                # Backward identify
                possible_tids = set()
                for successor in successors:
                    possible_tids.update(
                        self.tracklet_structure.attributes[successor].possible_tag_ids
                    )
                if debug:
                    LOGGER.info("possible_tids: " + str([tid.hex() for tid in possible_tids]))

                cur_possible_tids = self.tracklet_structure.attributes[
                    vid
                ].possible_tag_ids
                if debug:
                    LOGGER.info("cur_possible_tids: " + str([tid.hex() for tid in cur_possible_tids]))

                # Find intersection
                possible_tids = possible_tids.intersection(cur_possible_tids)
                if len(possible_tids) == 1:
                    if debug:
                        LOGGER.info("matched_vid: " + str(vid))
                    self.tracklet_structure.update_possible_tags(vid, possible_tids)
                    self.tracklet_structure.update_tracklet_status(
                        vid, VisualIDState.INACTIVE_MATCHED
                    )

                    matched_lst.append((list(possible_tids)[0], vid))

                    # Add new predecessors to queue
                    new_predecessors = {
                        predecessor
                        for predecessor in self.tracklet_structure.get_predecessors(vid)
                        if self.match_status_mgr.vid_state.get_status(predecessor)
                        == VisualIDState.INACTIVE_UNMATCHED
                    }
                    if debug:
                        LOGGER.info("new_predecessors: " + str([predecessor for predecessor in new_predecessors]))
                    to_be_back_identified_queue.extend(new_predecessors)
                else:
                    if debug:
                        LOGGER.info("Backward identification failed")
        return matched_lst

    def __str__(self) -> str:
        """Return a string representation of the tracklet graph.

        Returns:
            String describing the current state of unmatched and matched tag IDs
        """
        unmatched_tags = [
            tid.hex()
            for tid in self.match_status_mgr.tid_state.get_ids_by_status(
                TagIDState.UNMATCHED
            )
        ]
        matched_tags = [
            tid.hex()
            for tid in self.match_status_mgr.tid_state.get_ids_by_status(
                TagIDState.MATCHED
            )
        ]

        return (
            "Tracklet graph:\n"
            f"Unmatched tag IDs:\n    {chr(10).join(unmatched_tags)}\n"
            f"Matched tag IDs:\n    {chr(10).join(matched_tags)}\n"
        )

    @property
    def matched_tid2vid(self) -> bidict:
        """Get the matched tag IDs to visual IDs dictionary."""
        return self.match_status_mgr.matched_tid2vid

    @property
    def unmatched_tid2vid(self) -> bidict:
        """Get the unmatched tag IDs to visual IDs dictionary."""
        return self.match_status_mgr.unmatched_tid2vid

    @property
    def active_visual_ids(self) -> set[int]:
        """Get the active visual IDs."""
        active_ids = set()
        active_ids.update(
            self.match_status_mgr.vid_state.get_ids_by_status(VisualIDState.UNMATCHED)
        )
        active_ids.update(
            self.match_status_mgr.vid_state.get_ids_by_status(VisualIDState.MATCHED)
        )
        return active_ids
