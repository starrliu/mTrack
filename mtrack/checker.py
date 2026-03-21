"""Global checker for online matching quality check.

This module provides functionality for real-time quality assessment of 
RFID tag and visual ID matching.
It implements a global checking mechanism to validate the correctness of matches between RFID tags
and visual tracking IDs.
"""

import pandas as pd
import numpy as np
from bidict import bidict

from .match import (
    ManhattanMatcher,
    RFDataPoint,
    CalculatedFlag,
    MatchScoreManager,
)
from .phase import PhaseCalculator
from .data import RFIDMessage, TrackerMessage
from .logger import LOGGER
from .config import CheckerConfig


class MatchScoreMonitor(MatchScoreManager):
    """
    Monitor the match scores of all tags and visual IDs.
    """
    def __init__(self, window_size: int = 60) -> None:
        """Initialize the MatchScoreMonitor.

        Args:
            window_size (int, optional): Size of the monitoring window in seconds. Defaults to 60.
        """
        super().__init__(window_size)
        self.checking_tags: set[bytes] = set()
        self.active_visual_ids: set[int] = set()

    def update_active_visual_ids(self, active_visual_ids: set[int]):
        """Update the set of active visual IDs and initialize data structures for new ones.

        Args:
            active_visual_ids (set[int]): Set of currently active visual IDs.

        Note:
            This method performs the following operations:
            1. Removes visual IDs that are no longer active
            2. Adds new active visual IDs
            3. Initializes data structures for new visual IDs
            4. Creates pairs between new visual IDs and all checking tags
        """
        # Remove no longer active visual ids
        inactive_visual_ids = self.active_visual_ids - active_visual_ids
        for vid in inactive_visual_ids:
            self.remove_visual_id_from_all_tags(vid)

        # Add new active visual ids
        new_visual_ids = active_visual_ids - self.active_visual_ids
        for vid in new_visual_ids:
            # Initialize visual data
            self.add_visual_data(vid)
            # Create pairs with all checking tags
            for tag_id in self.checking_tags:
                self.add_tag_visual_pair(tag_id, vid)

        # Update active visual ids set
        self.active_visual_ids = active_visual_ids.copy()

class RoundRobinScheduler:
    """
    Schedule the tags to be checked in a round-robin manner.
    """
    def __init__(
        self,
        max_checking_tags: int,
        min_checking_tags: int,
        max_checking_time: pd.Timedelta,
    ) -> None:
        """Initialize the RoundRobinScheduler.

        Args:
            max_checking_tags (int): Maximum number of tags that can be checked simultaneously.
            min_checking_tags (int): Minimum number of tags that should be checked simultaneously.
            max_checking_time (pd.Timedelta): Maximum time allowed for checking a single tag.
        """
        self.max_checking_tags = max_checking_tags
        self.min_checking_tags = min_checking_tags
        self.max_checking_time = max_checking_time

        self.current_checking_tags: list[bytes] = []
        self.waiting_tags: list[bytes] = []
        self.current_time: pd.Timestamp = pd.Timestamp.min

        self.check_start_time: dict[bytes, pd.Timestamp] = {}

    def set_max_checking_tags(self, max_checking_tags: int):
        """Update the maximum number of tags that can be checked simultaneously.
        
        Args:
            max_checking_tags (int): The new maximum number of tags allowed for checking.
        """
        self.max_checking_tags = max_checking_tags

    def set_current_time(self, current_time: pd.Timestamp):
        """Update the current time for scheduler operations.

        Args:
            current_time (pd.Timestamp): The new current time to set.
        """
        self.current_time = current_time

    def add_tag_to_waiting(self, tag_id: bytes):
        """Add a tag to the waiting queue if not already present.

        Args:
            tag_id (bytes): The ID of the tag to add to the waiting queue.
        """
        if tag_id not in self.waiting_tags and tag_id not in self.current_checking_tags:
            self.waiting_tags.append(tag_id)

    def remove_tag_completely(self, tag_id: bytes):
        """Remove a tag from both waiting queue and checking list.

        Args:
            tag_id (bytes): The ID of the tag to remove.
        """
        if tag_id in self.waiting_tags:
            self.waiting_tags.remove(tag_id)

        if tag_id in self.current_checking_tags:
            self.current_checking_tags.remove(tag_id)

        if tag_id in self.check_start_time:
            del self.check_start_time[tag_id]

    def move_tag_from_checking_to_waiting(self, tag_id: bytes):
        """Move a tag from checking list back to waiting queue.

        Args:
            tag_id (bytes): The ID of the tag to move.
        """
        if tag_id in self.current_checking_tags:
            self.current_checking_tags.remove(tag_id)

        if tag_id in self.check_start_time:
            del self.check_start_time[tag_id]

        if tag_id not in self.waiting_tags:
            self.waiting_tags.append(tag_id)

    def get_expired_tags(self) -> list[bytes]:
        """Get list of tags that have exceeded maximum checking time.

        Returns:
            list[bytes]: List of expired tag IDs.
        """
        expired_tags = []
        for tag_id in self.current_checking_tags:
            if tag_id in self.check_start_time:
                elapsed_time = self.current_time - self.check_start_time[tag_id]
                if elapsed_time > self.max_checking_time:
                    expired_tags.append(tag_id)
        return expired_tags

    def update_checking_tags(self) -> tuple[list[bytes], list[bytes]]:
        """Update checking tags by moving from waiting to checking.

        This method manages the transition of tags from the waiting queue to the checking list
        while maintaining the minimum and maximum constraints.

        Returns:
            tuple[list[bytes], list[bytes]]: A tuple containing:
                - List of newly added tag IDs
                - List of expired tag IDs
        """
        newly_added = []

        # Move tags from waiting to checking based on capacity
        available_slots = self.max_checking_tags - len(self.current_checking_tags)

        while available_slots > 0 and len(self.waiting_tags) > 0:
            tag_id = self.waiting_tags.pop(0)
            self._add_tag_to_checking(tag_id)
            newly_added.append(tag_id)
            available_slots -= 1

        # Ensure minimum checking tags
        while (
            len(self.current_checking_tags) < self.min_checking_tags
            and len(self.waiting_tags) > 0
        ):
            tag_id = self.waiting_tags.pop(0)
            self._add_tag_to_checking(tag_id)
            newly_added.append(tag_id)

        # Get expired tags
        expired_tags = self.get_expired_tags()

        return newly_added, expired_tags

    def _add_tag_to_checking(self, tag_id: bytes):
        """Internal method to add a tag to checking list.

        Args:
            tag_id (bytes): The ID of the tag to add to checking list.
        """
        if tag_id not in self.current_checking_tags:
            self.current_checking_tags.append(tag_id)
            self.check_start_time[tag_id] = self.current_time

    def get_checking_tags(self) -> list[bytes]:
        """Get current list of checking tags.

        Returns:
            list[bytes]: Copy of the current checking tags list.
        """
        return self.current_checking_tags.copy()

    def is_tag_being_checked(self, tag_id: bytes) -> bool:
        """Check if a tag is currently being checked.

        Args:
            tag_id (bytes): The ID of the tag to check.

        Returns:
            bool: True if the tag is being checked, False otherwise.
        """
        return tag_id in self.current_checking_tags

    @property
    def tags_required_to_check(self) -> list[bytes]:
        """Get the tags that are required to be checked.

        Returns:
            list[bytes]: List of all tags that need to be checked 
                (both waiting and currently checking).
        """
        return self.current_checking_tags + self.waiting_tags


class GlobalChecker:
    """
    Global checker for online matching quality check.

    This module provides functionality for real-time quality assessment of
    RFID tag and visual ID matching.
    It implements a global checking mechanism to validate the correctness
    of matches between RFID tags
    and visual tracking IDs.
    """
    def __init__(
        self,
        max_reading_tags: int,
        antpos: dict[int, tuple[float, float, float]],
        p2m: float,
        config: CheckerConfig = None,
        min_checking_tags: int = 1,
    ) -> None:
        """Initialize the GlobalChecker.

        Args:
            max_reading_tags (int):
                Maximum number of tags that can be read simultaneously.
            antpos (dict[int, tuple[float, float, float]]): 
                Dictionary mapping antenna IDs to their 3D positions.
            p2m (float): Pixels to meters conversion factor.
            config (CheckerConfig): Configuration object. If None, uses default values.
            min_checking_tags (int, optional):
                Minimum number of tags to check. Defaults to 1.
        """
        if config is None:
            config = CheckerConfig()
        
        self.config = config
        self.matcher = ManhattanMatcher()
        self.matched_tid2vid: bidict = bidict()
        self.match_score_monitor = MatchScoreMonitor(window_size=self.config.window_size)
        self.phase_calculator = PhaseCalculator(antpos, p2m)
        self.max_reading_tags = max_reading_tags
        
        max_checking_time = pd.Timedelta(f"{self.config.max_checking_time}ms")
        self.scheduler = RoundRobinScheduler(
            max_reading_tags, min_checking_tags, max_checking_time
        )
        self.current_time: pd.Timestamp = pd.Timestamp.min

    def update_max_checking_tags(self, num_of_unmatched_tags: int):
        """Update the maximum number of tags that can be checked based on unmatched tags.

        Args:
            num_of_unmatched_tags (int): Number of currently unmatched tags.
        """
        self.scheduler.set_max_checking_tags(
            self.max_reading_tags - num_of_unmatched_tags
        )

    def update_from_graph(self, matched_tid2vid: bidict, active_visual_ids: set[int]):
        """Update the current matched pairs and active visual IDs.

        This method handles the following operations:
        1. Adds new matched pairs to the waiting queue
        2. Removes obsolete matched pairs
        3. Updates the matched pairs dictionary
        4. Updates active visual IDs

        Args:
            matched_tid2vid (bidict): Bidirectional dictionary mapping tag IDs to visual IDs.
            active_visual_ids (set[int]): Set of currently active visual IDs.
        """
        self._add_new_matched_pairs(matched_tid2vid)
        self._remove_obsolete_matched_pairs(matched_tid2vid)
        self._update_matched_pairs(matched_tid2vid)
        self._update_active_visual_ids(active_visual_ids)

    def _add_new_matched_pairs(self, matched_tid2vid: bidict):
        """Add new matched pairs to the waiting queue.

        Args:
            matched_tid2vid (bidict): Bidirectional dictionary mapping tag IDs to visual IDs.
        """
        for tid, _ in matched_tid2vid.items():
            if tid not in self.matched_tid2vid:
                self.scheduler.add_tag_to_waiting(tid)

    def _remove_obsolete_matched_pairs(self, matched_tid2vid: bidict):
        """Remove matched pairs that are no longer valid.

        Args:
            matched_tid2vid (bidict): Bidirectional dictionary mapping tag IDs to visual IDs.
        """
        obsolete_tags = []
        for tid in self.matched_tid2vid:
            if tid not in matched_tid2vid:
                obsolete_tags.append(tid)

        for tid in obsolete_tags:
            self.scheduler.remove_tag_completely(tid)
            self._cleanup_tag_data(tid)

    def _update_matched_pairs(self, matched_tid2vid: bidict):
        """Update the matched pairs dictionary.

        Args:
            matched_tid2vid (bidict): Bidirectional dictionary mapping tag IDs to visual IDs.
        """
        self.matched_tid2vid = matched_tid2vid.copy()

    def _update_active_visual_ids(self, active_visual_ids: set[int]):
        """Update active visual IDs and remove inactive ones.

        Args:
            active_visual_ids (set[int]): Set of currently active visual IDs.
        """
        self.match_score_monitor.update_active_visual_ids(active_visual_ids)

    def _cleanup_tag_data(self, tid: bytes):
        """Clean up all data associated with a tag.

        Args:
            tid (bytes): The ID of the tag to clean up.
        """
        if tid in self.match_score_monitor.checking_tags:
            self.match_score_monitor.checking_tags.remove(tid)
        self.match_score_monitor.remove_tag(tid)

    def update_from_data(
        self, rfdata: list[RFIDMessage], visualdata: TrackerMessage
    ) -> list[bytes]:
        """Update the checking state with new RFID and visual tracking data.

        This method performs the following steps:
        1. Updates visual data for all active visual IDs
        2. Processes RFID data and updates match scores
        3. Calculates match scores for all uncalculated data points
        4. Checks for mismatched tags
        5. Handles expired tags

        Unlike RF-mat's checker which only calculates match scores between matched pairs,
        this checker calculates scores between all checking tags and all active visual IDs.
        For example, if tag A is matched with visual ID 1, and visual IDs 2, 3, 4 are also active,
        the checker calculates scores between tag A and all visual IDs 1, 2, 3, 4.
        If tag A's match score with visual ID 1 is significantly lower than its best match score
        among all visual IDs, the match is considered incorrect.

        Args:
            rfdata (list[RFIDMessage]): List of RFID messages.
            visualdata (TrackerMessage): Visual tracking data.

        Returns:
            list[bytes]: List of mismatched tag IDs.
        """
        self.current_time = visualdata.timestamp
        self.scheduler.set_current_time(self.current_time)

        self._update_visual_data(visualdata)
        self._process_rfid_data(rfdata)
        self._calculate_uncalculated_scores()
        mismatched_tags = self._check_for_mismatched_tags()

        # expired_tags = self.scheduler.get_expired_tags()
        # for tag_id in expired_tags:
        #     self.scheduler.move_tag_from_checking_to_waiting(tag_id)
        #     self._cleanup_tag_data(tag_id)

        return mismatched_tags

    def _update_visual_data(self, visualdata: TrackerMessage):
        """Update visual data for all active visual IDs.

        Args:
            visualdata (TrackerMessage):
                Visual tracking data containing positions of tracked objects.
        """
        for obj_id, xywh in zip(visualdata.trackresult.id, visualdata.trackresult.xywh):
            if obj_id in self.match_score_monitor.active_visual_ids:
                self.match_score_monitor.update_visual_data(
                    obj_id, visualdata.timestamp, xywh
                )

    def _process_rfid_data(self, rfdata: list[RFIDMessage]):
        """Process RFID data and calculate match scores for checking tags.

        Args:
            rfdata (list[RFIDMessage]): List of RFID messages to process.
        """
        checking_tags = self.scheduler.get_checking_tags()
        for msg in rfdata:
            tag_id = msg.tag_id
            if tag_id not in checking_tags:
                continue

            self._process_single_rfid_message(msg)

    def _process_single_rfid_message(self, msg: RFIDMessage):
        """Process a single RFID message and update match scores.

        Args:
            msg (RFIDMessage): RFID message to process.
        """
        tag_id = msg.tag_id
        timestamp = msg.timestamp
        freq = msg.data.frequency
        antid = msg.data.antenna_id
        phase = msg.data.phase

        try:
            prev_timestamp, prev_phase, ref_freq = (
                self.match_score_monitor.get_tag_ref_data(tag_id, antid)
            )

            self.match_score_monitor.update_tag_ref_data(
                tag_id, antid, timestamp, phase, freq
            )

            if ref_freq != freq:
                return

            data_point = RFDataPoint(
                prev_timestamp, timestamp, prev_phase, phase, freq, antid
            )

            self.match_score_monitor.add_rfid_data_point(tag_id, data_point)

        except KeyError:
            self.match_score_monitor.update_tag_ref_data(
                tag_id, antid, timestamp, phase, freq
            )

    def _calculate_uncalculated_scores(self):
        """Calculate match scores for all uncalculated data points.

        This method processes all data points that haven't had their match scores calculated yet.
        For each point, it:
        1. Gets the corresponding visual trace data
        2. Checks if the data point is within the valid time range
        3. Calculates predicted and actual phase differences
        4. Computes the match score
        """
        uncalculated_points = self.match_score_monitor.iter_uncalculated_points()

        for tag_id, visual_id, idx, data_point in uncalculated_points:
            visual_trace = self.match_score_monitor.get_visual_data(visual_id)

            start_ts = visual_trace.start_timestamp
            end_ts = visual_trace.end_timestamp

            if start_ts is None or end_ts is None:
                continue

            if data_point.ts1 < start_ts:
                pair = self.match_score_monitor.get_pair(tag_id, visual_id)
                pair.update_calculated_flag(idx, CalculatedFlag.NOT_VALID)
                continue

            if data_point.ts2 > end_ts:
                continue

            pos1 = visual_trace.get_pos(data_point.ts1)
            pos2 = visual_trace.get_pos(data_point.ts2)

            pred_delta_phase = self.phase_calculator.calculate_predicted_phase(
                pos1, pos2, data_point.antenna, data_point.freq
            )

            actual_delta_phase = self.phase_calculator.calculate_actual_phase(
                data_point.phase1, data_point.phase2
            )

            score = self.matcher.predict(actual_delta_phase, pred_delta_phase)

            pair = self.match_score_monitor.get_pair(tag_id, visual_id)
            pair.finish_calculation(idx, score)

    def _check_for_mismatched_tags(self) -> list[bytes]:
        """Check for mismatched tags based on score comparison.

        This method identifies tags that are likely mismatched by comparing their
        match scores with their current visual ID against their best possible match scores.
        A tag is considered mismatched if:
        1. Its match score with the current visual ID is significantly 
            lower than its best match score
        2. Its match score is below an absolute threshold

        Returns:
            list[bytes]: List of mismatched tag IDs.
        """
        mismatched_tags = []
        mismatch_threshold_best_visual = self.config.mismatch_threshold_best_visual  # Threshold for comparing with best visual score
        mismatch_threshold_other_tag = self.config.mismatch_threshold_other_tag      # Threshold for comparing with other tag's score
        num_data_threshold = self.config.num_data_threshold
        abs_threshold = self.config.abs_threshold

        checking_tags = self.scheduler.get_checking_tags()
        if not checking_tags:
            return mismatched_tags

        tag_ids, visual_ids, match_matrix, num_matrix = (
            self.match_score_monitor.get_match_matrix()
        )

        for tag_id in checking_tags:
            if tag_id not in self.matched_tid2vid:
                continue

            try:
                tag_idx = tag_ids.index(tag_id)
                matched_visual_id = self.matched_tid2vid[tag_id]
                visual_idx = visual_ids.index(matched_visual_id)

                # Get valid visual IDs indices where we have enough data
                valid_visual_indices = [
                    i for i, num in enumerate(num_matrix[tag_idx]) 
                    if num >= num_data_threshold
                ]

                if visual_idx not in valid_visual_indices:
                    continue

                valid_visual_scores = [
                    match_matrix[tag_idx, i] for i in valid_visual_indices
                ]
                best_visual_score = max(valid_visual_scores)

                # Get valid tag indices where we have enough data
                valid_tag_indices = [
                    i for i, num in enumerate(num_matrix[:, visual_idx]) 
                    if num >= num_data_threshold
                ]

                if tag_idx not in valid_tag_indices:
                    continue

                best_visual_idxs_for_all_tags = np.argmax(match_matrix, axis=1)

                if match_matrix[tag_idx, visual_idx] < abs_threshold:
                    LOGGER.info(
                        "Mismatched tag: %s due to absolute threshold", 
                        tag_id.hex()
                    )                
                    mismatched_tags.append(tag_id)
                    self.scheduler.remove_tag_completely(tag_id)
                    self._cleanup_tag_data(tag_id)
                    continue

                if (
                    match_matrix[tag_idx, visual_idx]
                    < (1 + mismatch_threshold_best_visual) * best_visual_score
                ):
                    for other_tag_idx in valid_tag_indices:
                        if other_tag_idx == tag_idx:
                            continue
                        
                        best_score_for_other_tag = match_matrix[other_tag_idx, best_visual_idxs_for_all_tags[other_tag_idx]]
                        if match_matrix[tag_idx, visual_idx] > (1 + mismatch_threshold_other_tag) * best_score_for_other_tag:
                            LOGGER.info(
                                "Mismatched tag: %s due to score difference with tag %s", 
                                tag_id.hex(),
                                tag_ids[other_tag_idx].hex()
                            )
                            mismatched_tags.append(tag_id)
                            self.scheduler.remove_tag_completely(tag_id)
                            self._cleanup_tag_data(tag_id)
                            break
                    else:
                        continue

            except (ValueError, IndexError) as e:
                LOGGER.warning(
                    "Error processing tag %s: %s", 
                    tag_id.hex(), 
                    str(e)
                )
                continue

        return mismatched_tags

    def update_checking_tags(self):
        """Update the checking tags.

        This method manages the checking tags similar to RF-mat's checker:
        1. Moves tags from waiting queue to checking list
        2. Initializes data for newly added tags
        3. Cleans up data for expired tags
        """
        newly_added, expired_tags = self.scheduler.update_checking_tags()

        for tag_id in newly_added:
            self._initialize_tag_for_checking(tag_id)

        for tag_id in expired_tags:
            self.scheduler.move_tag_from_checking_to_waiting(tag_id)
            self._cleanup_tag_data(tag_id)

    def _initialize_tag_for_checking(self, tag_id: bytes):
        """Initialize data structures for a tag that starts being checked.

        Args:
            tag_id (bytes): The ID of the tag to initialize.
        """
        self.match_score_monitor.checking_tags.add(tag_id)

        for visual_id in self.match_score_monitor.active_visual_ids:
            self.match_score_monitor.add_tag_visual_pair(tag_id, visual_id)

    @property
    def checking_tags(self) -> list[bytes]:
        """Get the checking tags."""
        return self.scheduler.get_checking_tags()

    def __str__(self):
        output = "GlobalChecker\n"
        output += "match_score_monitor:\n"

        tag_ids, visual_ids, match_matrix, num_matrix = (
            self.match_score_monitor.get_match_matrix()
        )

        output += f"{'':<28}"
        for visual_id in visual_ids:
            output += f"{visual_id:<10}"
        output += "\n"

        for i, tag_id in enumerate(tag_ids):
            output += f"{tag_id.hex():<28}"
            for j, visual_id in enumerate(visual_ids):
                output += f"{match_matrix[i, j]:<10.2f}"
            output += "\n"

        output += "num_data_matrix:\n"

        output += f"{'':<28}"
        for visual_id in visual_ids:
            output += f"{visual_id:<10}"
        output += "\n"

        for i, tag_id in enumerate(tag_ids):
            output += f"{tag_id.hex():<28}"
            for j, visual_id in enumerate(visual_ids):
                output += f"{num_matrix[i, j]:<10}"
            output += "\n"

        output += "waiting_tags:\n"
        for tag_id in self.scheduler.waiting_tags:
            output += f"{tag_id.hex():<28}"
        output += "\n"

        output += "checking_tags:\n"
        for tag_id in self.scheduler.current_checking_tags:
            output += f"{tag_id.hex():<28}"
        output += "\n"

        return output