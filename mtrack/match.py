"""
This file implements the matching algorithm.
"""
import math
from dataclasses import dataclass
from enum import IntEnum


import numpy as np
import pandas as pd

from .data import ObjectTrace, RFIDMessage, TrackerMessage, XYWH
from .utils import norm_phase_to_half_circle, sub_phase_to_half_circle
from .phase import PhaseCalculator
from .config import MatchConfig

@dataclass
class RFDataPoint:
    """
    A data point of RFID data.
    """
    ts1: pd.Timestamp
    ts2: pd.Timestamp
    phase1: float
    phase2: float
    freq: float
    antenna: int


class CalculatedFlag(IntEnum):
    """
    Enumeration type for calculation status flags
    """

    NOT_CALCULATED = 0  # not calculated
    CALCULATED = 1  # calculated
    NOT_VALID = 2  # not valid


@dataclass
class TagVisualPair:
    """
    Manages matching data between tag ID and visual ID.

    Attributes:
        tag_id (bytes): RFID tag ID
        visual_id (int): Visual tracking ID
        scores (list[float]): Sliding window of matching scores
        calculated_flags (list[CalculatedFlag]): Calculation status flags
        window_size (int): Size of sliding window
    """

    tag_id: bytes
    visual_id: int
    window_size: int = 60

    def __post_init__(self):
        self.scores = [0] * self.window_size
        self.calculated_flags = [CalculatedFlag.NOT_VALID] * self.window_size

    def add_calculated_flag(self, flag: CalculatedFlag) -> None:
        """Add a new calculation status flag"""
        if len(self.calculated_flags) == self.window_size:
            self.calculated_flags.pop(0)
            self.scores.pop(0)
        self.calculated_flags.append(flag)
        self.scores.append(0)

    def get_valid_scores(self) -> list[float]:
        """Get all valid matching scores"""
        return [
            score
            for score, flag in zip(self.scores, self.calculated_flags)
            if flag == CalculatedFlag.CALCULATED
        ]

    def update_calculated_flag(self, idx: int, flag: CalculatedFlag) -> None:
        """Update the calculation status flag"""
        if idx >= len(self.calculated_flags):
            raise IndexError(f"Index {idx} out of range for calculated_flags")
        self.calculated_flags[idx] = flag

    def finish_calculation(self, idx: int, score: float) -> None:
        """Update the calculation status flag and score after calculation"""
        self.update_calculated_flag(idx, CalculatedFlag.CALCULATED)
        self.scores[idx] = score

class MatchScoreManager:
    """
    Manages matching scores and window data.

    Attributes:
        window_size (int): Size of sliding window
        tag_visual_pairs (dict[bytes, dict[int, TagVisualPair]]): Tag and visual ID matching pairs
        slide_windows (dict[bytes, list[RFDataPoint]]): RFID data points sliding window
        tag_ref_data (dict[bytes, dict[int, tuple[pd.Timestamp, float, float]]]): Tag reference data
        visual_data (dict[int, ObjectTrace]): Visual tracking data
    """

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.tag_visual_pairs: dict[bytes, dict[int, TagVisualPair]] = {}
        self.slide_windows: dict[bytes, list[RFDataPoint]] = {}
        self.tag_ref_data: dict[bytes, dict[int, tuple[pd.Timestamp, float, float]]] = (
            {}
        )
        self.visual_data: dict[int, ObjectTrace] = {}

    def add_visual_data(self, visual_id: int) -> None:
        """Add new visual tracking data"""
        if visual_id not in self.visual_data:
            self.visual_data[visual_id] = ObjectTrace(visual_id)

    def __remove_visual_data(self, visual_id: int) -> None:
        """Remove visual tracking data"""
        if visual_id in self.visual_data:
            del self.visual_data[visual_id]

    def __add_tag(self, tag_id: bytes) -> None:
        """Add new tag ID"""
        if tag_id not in self.tag_visual_pairs:
            self.tag_visual_pairs[tag_id] = {}
            self.slide_windows[tag_id] = []
            self.tag_ref_data[tag_id] = {}

    def __add_visual_id(self, tag_id: bytes, visual_id: int) -> None:
        """Add new visual ID to tag"""
        if (
            tag_id in self.tag_visual_pairs
            and visual_id not in self.tag_visual_pairs[tag_id]
        ):
            self.tag_visual_pairs[tag_id][visual_id] = TagVisualPair(
                tag_id, visual_id, self.window_size
            )

            self.add_visual_data(visual_id)

    def add_tag_visual_pair(self, tag_id: bytes, visual_id: int) -> None:
        """Add new tag and visual ID pair"""
        self.__add_tag(tag_id)
        self.__add_visual_id(tag_id, visual_id)

    def remove_tag(self, tag_id: bytes) -> None:
        """Remove tag and all its related data"""
        if tag_id in self.tag_visual_pairs:
            del self.tag_visual_pairs[tag_id]
            del self.slide_windows[tag_id]
            del self.tag_ref_data[tag_id]

    def remove_visual_id_from_all_tags(self, visual_id: int) -> None:
        """Remove visual ID from all tags"""
        for _, pairs in self.tag_visual_pairs.items():
            if visual_id in pairs:
                del pairs[visual_id]

        self.__remove_visual_data(visual_id)

    def remove_tag_visual_pair(self, tag_id: bytes, visual_id: int) -> None:
        """Remove visual ID from tag"""
        if (
            tag_id in self.tag_visual_pairs
            and visual_id in self.tag_visual_pairs[tag_id]
        ):
            del self.tag_visual_pairs[tag_id][visual_id]

            # Check if this visual_id is still used by other tag_ids
            visual_id_in_use = False
            for other_tag_id, other_pairs in self.tag_visual_pairs.items():
                if other_tag_id != tag_id and visual_id in other_pairs:
                    visual_id_in_use = True
                    break

            # If visual_id is no longer used by any tag, remove it from visual_data
            if not visual_id_in_use:
                self.__remove_visual_data(visual_id)

            # If tag_id has no visual_ids, delete the tag_id
            if len(self.tag_visual_pairs[tag_id]) == 0:
                self.remove_tag(tag_id)

    def add_rfid_data_point(self, tag_id: bytes, data_point: RFDataPoint) -> None:
        """Add new RFID data point"""
        if tag_id in self.slide_windows:
            if len(self.slide_windows[tag_id]) == self.window_size:
                self.slide_windows[tag_id].pop(0)
            self.slide_windows[tag_id].append(data_point)

            # Update calculation status for all related visual IDs
            if tag_id in self.tag_visual_pairs:
                for pair in self.tag_visual_pairs[tag_id].values():
                    pair.add_calculated_flag(CalculatedFlag.NOT_CALCULATED)
            else:
                raise KeyError(f"Tag ID {tag_id.hex()} not found in tag_visual_pairs")
        else:
            raise KeyError(f"Tag ID {tag_id.hex()} not found in slide_windows")

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-positional-arguments
    def update_tag_ref_data(
        self,
        tag_id: bytes,
        antenna_id: int,
        ts: pd.Timestamp,
        phase: float,
        freq: float,
    ) -> None:
        """Update tag reference data"""
        if tag_id in self.tag_ref_data:
            self.tag_ref_data[tag_id][antenna_id] = (ts, phase, freq)
        if tag_id not in self.tag_ref_data:
            raise KeyError(f"Tag ID {tag_id.hex()} not found in tag_ref_data")

    def get_tag_ref_data(
        self, tag_id: bytes, antenna_id: int
    ) -> tuple[pd.Timestamp, float, float]:
        """Get tag reference data"""
        if tag_id not in self.tag_ref_data:
            raise KeyError(f"Tag ID {tag_id.hex()} not found in tag_ref_data")
        if antenna_id not in self.tag_ref_data[tag_id]:
            raise KeyError(
                f"Antenna ID {antenna_id} not found in tag_ref_data for tag ID {tag_id.hex()}"
            )
        return self.tag_ref_data[tag_id][antenna_id]

    def get_match_matrix(self) -> tuple[list[bytes], list[int], np.ndarray, np.ndarray]:
        """Generate matching matrix"""
        _min_score = -20

        tag_ids = list(self.tag_visual_pairs.keys())
        visual_ids = []
        for tag_pairs in self.tag_visual_pairs.values():
            visual_ids.extend(tag_pairs.keys())
        visual_ids = list(set(visual_ids))

        match_matrix = np.zeros((len(tag_ids), len(visual_ids)), dtype=float)
        num_matrix = np.zeros((len(tag_ids), len(visual_ids)), dtype=int)

        for i, tag_id in enumerate(tag_ids):
            for j, visual_id in enumerate(visual_ids):
                if visual_id in self.tag_visual_pairs[tag_id]:
                    valid_scores = self.tag_visual_pairs[tag_id][
                        visual_id
                    ].get_valid_scores()
                    match_matrix[i, j] = np.mean(valid_scores) if valid_scores else _min_score
                    num_matrix[i, j] = len(valid_scores)
                else:
                    match_matrix[i, j] = _min_score  # default score
                    num_matrix[i, j] = -1

        return tag_ids, visual_ids, match_matrix, num_matrix

    def update_visual_data(
        self, visual_id: int, timestamp: pd.Timestamp, pos: XYWH
    ) -> None:
        """Update visual tracking data"""
        if visual_id in self.visual_data:
            try:
                self.visual_data[visual_id].append(
                    timestamp,
                    {"x": pos.x, "y": pos.y, "w": pos.w, "h": pos.h, "frame": -1},
                )
            except ValueError as ex:
                print(f"Error in appending visual data for visual ID {visual_id}.")
                print(f"ts_vis: {timestamp}, pos: {pos.x} {pos.y} {pos.w} {pos.h}")
                raise ValueError("Error in appending visual data") from ex

    def get_visual_data(self, visual_id: int) -> ObjectTrace:
        """Get visual tracking data"""
        if visual_id not in self.visual_data:
            raise KeyError(f"Visual ID {visual_id} not found in visual_data")
        return self.visual_data[visual_id]

    def remove_inactive_visual_data(self, inactive_ids: list[int]) -> None:
        """Remove inactive visual tracking data"""
        for vid in inactive_ids:
            self.__remove_visual_data(vid)

    def iter_pairs(self) -> list[tuple[bytes, int]]:
        """Get all tag-visual pairs"""
        pairs = []
        for tag_id, visual_dict in self.tag_visual_pairs.items():
            for visual_id in visual_dict:
                pairs.append((tag_id, visual_id))
        return pairs

    def iter_uncalculated_points(self) -> list[tuple[bytes, int, int, RFDataPoint]]:
        """
        Get all uncalculated data points.

        Returns:
            list[tuple[bytes, int, int, RFDataPoint]]:
                Each element is a tuple (tag_id, visual_id, window_idx, data_point)
        """
        uncalculated = []
        for tag_id, visual_dict in self.tag_visual_pairs.items():
            for visual_id, pair in visual_dict.items():
                for idx, flag in enumerate(pair.calculated_flags):
                    if flag == CalculatedFlag.NOT_CALCULATED and idx < len(
                        self.slide_windows[tag_id]
                    ):
                        uncalculated.append(
                            (tag_id, visual_id, idx, self.slide_windows[tag_id][idx])
                        )
        return uncalculated

    def get_pair(self, tag_id: bytes, visual_id: int) -> TagVisualPair:
        """Get specific tag-visual pair"""
        if (
            tag_id not in self.tag_visual_pairs
            or visual_id not in self.tag_visual_pairs[tag_id]
        ):
            raise KeyError(
                f"Pair (tag_id: {tag_id.hex()}, visual_id: {visual_id}) not found"
            )
        return self.tag_visual_pairs[tag_id][visual_id]

    def has_pair(self, tag_id: bytes, visual_id: int) -> bool:
        """Check if specific tag-visual pair exists"""
        return (
            tag_id in self.tag_visual_pairs
            and visual_id in self.tag_visual_pairs[tag_id]
        )

    def get_data_point(self, tag_id: bytes, window_idx: int) -> RFDataPoint:
        """Get data point for specific tag at specific window index"""
        if tag_id not in self.slide_windows:
            raise KeyError(f"Tag ID {tag_id.hex()} not found in slide_windows")
        if window_idx >= len(self.slide_windows[tag_id]):
            raise IndexError(
                f"Window index {window_idx} out of range for tag ID {tag_id.hex()}"
            )
        return self.slide_windows[tag_id][window_idx]

    def get_ref_data(
        self, tag_id: bytes, antenna_id: int
    ) -> tuple[pd.Timestamp, float, float]:
        """Get reference data for specific tag and antenna"""
        if (
            tag_id not in self.tag_ref_data
            or antenna_id not in self.tag_ref_data[tag_id]
        ):
            raise KeyError(
                f"Reference data not found for tag ID {tag_id.hex()} and antenna ID {antenna_id}"
            )
        return self.tag_ref_data[tag_id][antenna_id]

    def __len__(self) -> int:
        """Return total number of tag-visual pairs"""
        return sum(len(visual_dict) for visual_dict in self.tag_visual_pairs.values())

    def contains_tag(self, tag_id: bytes) -> bool:
        """Check if tag exists"""
        return tag_id in self.tag_visual_pairs

class ManhattanMatcher:
    """
    Manhattan distance matcher.
    """

    def predict(self, actual_delta_phase: float, predicted_delta_phase: float):
        """
        Predict the distance based on the phase data.

        Args:
            actual_delta_phase(float): the actual phase difference.
            predicted_delta_phase(float): the predicted phase difference.
        """

        actual_delta_phase = (actual_delta_phase + np.pi / 2) % np.pi
        predicted_delta_phase = (predicted_delta_phase + np.pi / 2) % np.pi

        delta_phase = sub_phase_to_half_circle(
            actual_delta_phase, predicted_delta_phase
        )

        return -abs(delta_phase)

    def predict_batch(self, data: list[tuple[float, float]]):
        """Predict the distance based on the phase data in batch

        Args:
            data(list[tuple[float, float]]): a list of tuples of actual phase difference
                and predicted phase difference.

        Returns:
            list[float]: a list of predicted distances.
        """
        return [
            self.predict(actual_delta_phase, predicted_delta_phase)
            for actual_delta_phase, predicted_delta_phase in data
        ]


class MaxLikelihoodMatch:
    """
    Maximum likelihood matching algorithm for matching tag IDs and visual IDs.
    MLM mantains a matrix of likelihoods of matching a tag ID to a visual ID.

    Attributes:
        - match_score_manager(MatchScoreManager): a manager of match score.
        - phase_calculator(PhaseCalculator): a calculator of phase.
        - matcher(ManhattanMatcher): a matcher of Manhattan distance.
    """

    def __init__(
        self, antpos: dict[int, tuple[float, float, float]], p2m: float, config: MatchConfig = None
    ) -> None:
        if config is None:
            config = MatchConfig()
        
        self.config = config
        self.matcher = ManhattanMatcher()

        self.match_score_manager = MatchScoreManager(self.config.window_size)

        self.phase_calculator = PhaseCalculator(antpos, p2m)

    def update_from_idsm(
        self,
        candidates: list[tuple[bytes, int]],
        inactive_ids: list[int],
    ):
        """
        Update the tag data from ID state machine.

        Args:
            candidates(list[tuple[bytes, int]]): a list of tuples of tag IDs and visual IDs.
            inactive_ids(list[int]): a list of inactive visual IDs.
        """

        # If (tagid, visualid) is in candidates but not in match_score, add it to match_score
        # If (tagid, visualid) is in match_score but not in candidates, remove it from match_score

        for tagid, visualid in candidates:
            self.match_score_manager.add_tag_visual_pair(tagid, visualid)
            self.match_score_manager.add_visual_data(visualid)

        for tagid, visualid in self.match_score_manager.iter_pairs():
            if (tagid, visualid) not in candidates:
                self.match_score_manager.remove_tag_visual_pair(tagid, visualid)

        self.match_score_manager.remove_inactive_visual_data(inactive_ids)

    def update_likelihood(
        self,
        rfdata: list[RFIDMessage],
        visualdata: TrackerMessage,
    ):
        """
        Update the likelihood matrix of matching tag IDs to visual IDs.

        Args:
            rfdata(list[RFIDMessage]): a list of RFID messages.
            visualdata(TrackerMessage): the visual tracking message.
        """

        def step1():
            # Step 1: Update the visual data
            for objid, xywh in zip(
                visualdata.trackresult.id, visualdata.trackresult.xywh
            ):
                self.match_score_manager.update_visual_data(
                    objid, visualdata.timestamp, xywh
                )

        def check_rfid_timestamps():
            # Check if RFID timestamps are monotonically increasing
            timestamps = [msg.timestamp for msg in rfdata]
            if len(timestamps) > 1:
                for curr_ts, prev_ts in zip(timestamps[1:], timestamps[:-1]):
                    if curr_ts < prev_ts:
                        raise ValueError(
                            "RFID timestamps must be monotonically increasing"
                        )

        def step2():
            for msg in rfdata:
                tag_id, cur_timestamp = msg.tag_id, msg.timestamp
                freq, antid, phase = (
                    msg.data.frequency,
                    msg.data.antenna_id,
                    msg.data.phase,
                )

                if not self.match_score_manager.contains_tag(tag_id):
                    continue

                if antid not in self.match_score_manager.tag_ref_data[tag_id]:
                    self.match_score_manager.update_tag_ref_data(
                        tag_id, antid, cur_timestamp, phase, freq
                    )
                    continue

                prev_timestamp, phase1, ref_freq = (
                    self.match_score_manager.get_tag_ref_data(tag_id, antid)
                )
                self.match_score_manager.update_tag_ref_data(
                    tag_id, antid, cur_timestamp, phase, freq
                )

                if ref_freq != freq:
                    # If the frequency is different
                    continue

                new_data_point = RFDataPoint(
                    prev_timestamp, cur_timestamp, phase1, phase, freq, antid
                )

                self.match_score_manager.add_rfid_data_point(tag_id, new_data_point)

        def step3():
            datapoint_idxs = []
            datapoints = []

            # 获取所有未计算的数据点
            uncalculated_points = self.match_score_manager.iter_uncalculated_points()

            for tag_id, visual_id, idx, data_point in uncalculated_points:
                # 获取视觉跟踪数据的时间范围
                visual_trace = self.match_score_manager.get_visual_data(visual_id)
                start_ts, end_ts = (
                    visual_trace.start_timestamp,
                    visual_trace.end_timestamp,
                )

                if start_ts is None or end_ts is None:
                    continue  # Wait for the next update

                # 检查时间戳是否在有效范围内
                if data_point.ts1 < start_ts:
                    pair = self.match_score_manager.get_pair(tag_id, visual_id)
                    pair.update_calculated_flag(idx, CalculatedFlag.NOT_VALID)
                    continue

                if data_point.ts2 > end_ts:
                    continue  # Wait for the next update

                # 获取两个时间点的位置数据
                pos1 = visual_trace.get_pos(data_point.ts1)
                pos2 = visual_trace.get_pos(data_point.ts2)

                # 计算预测的相位差
                pred_delta_phase = self.phase_calculator.calculate_predicted_phase(
                    pos1, pos2, data_point.antenna, data_point.freq
                )

                # 计算实际的相位差
                actual_delta_phase = self.phase_calculator.calculate_actual_phase(
                    data_point.phase1, data_point.phase2
                )

                # 收集数据点用于批量预测
                datapoint_idxs.append((tag_id, visual_id, idx))
                datapoints.append([actual_delta_phase, pred_delta_phase])

            # 批量预测匹配分数并更新
            if len(datapoints) > 0:
                scores = self.matcher.predict_batch(datapoints)
                for i, (tag_id, visual_id, idx) in enumerate(datapoint_idxs):
                    score = scores[i]
                    pair = self.match_score_manager.get_pair(tag_id, visual_id)
                    pair.finish_calculation(idx, score)

        # Step 1: Update the visual data
        step1()

        # Check if RFID timestamps are monotonically increasing
        check_rfid_timestamps()

        # Step 2: Update the sliding windows
        step2()

        # Step 3: Try to update the match score if there are some calculated_windows[i][j] == 0
        step3()

    def get_best_match(self) -> list[tuple[bytes, int]]:
        """
        Get the best match of tag IDs to visual IDs.

        This function implements a matching algorithm that:
        1. Waits until all visual_ids for a tag_id have enough data points (>60)
        2. Finds tag-visual pairs that satisfy both maximal and stability conditions
        3. Continues until no more valid pairs can be found

        Returns:
            list[tuple[bytes, int]]: List of matched (tag_id, visual_id) pairs.
        """
        # Constants for matching conditions
        num_data_threshold = self.config.num_data_threshold  # Minimum number of data points required
        score_threshold = self.config.score_threshold  # Minimum score for a valid match
        alpha = self.config.alpha  # Stability condition parameter
        single_match_threshold = self.config.single_match_threshold  # Score threshold for single tag-visual pairs

        matched_pairs = []

        while True:
            # Get current state of matching scores
            tag_ids, visual_ids, match_matrix, num_data_matrix = (
                self.match_score_manager.get_match_matrix()
            )

            # Exit if no more tags or visuals to match
            if len(tag_ids) == 0 or len(visual_ids) == 0:
                break

            # Find tags ready for matching (all visuals have enough data)
            ready_tags = []
            for i, tag_id in enumerate(tag_ids):
                if np.logical_or(
                    num_data_matrix[i, :] >= num_data_threshold,
                    num_data_matrix[i, :] == -1,
                ).all():
                    ready_tags.append((i, tag_id))

            # Handle special cases based on number of tags/visuals
            matched_tag = None
            matched_visual = None

            if len(visual_ids) == 1 and len(tag_ids) > 1:
                # Wait until number of visuals equals number of tags
                break
            if len(tag_ids) == 1 and len(visual_ids) > 1:
                # Wait until number of visuals equals number of tags
                break
            if len(tag_ids) == 1 and len(visual_ids) == 1:
                # Single tag and visual case
                if (0, tag_ids[0]) in ready_tags and match_matrix[
                    0, 0
                ] > single_match_threshold:
                    matched_tag, matched_visual = tag_ids[0], visual_ids[0]
            else:
                # Multiple tags and visuals case
                max_visual_indices = np.argmax(match_matrix, axis=1)

                # Check each ready tag for matching conditions
                for tag_idx, tag_id in ready_tags:
                    tag_scores = match_matrix[tag_idx, :]

                    if len(tag_scores) == 1:
                        # Single visual case
                        if tag_scores[0] > score_threshold:
                            matched_tag, matched_visual = tag_id, visual_ids[0]
                            break
                    else:
                        # Multiple visuals case
                        sorted_indices = np.argsort(tag_scores)
                        best_visual_idx = sorted_indices[-1]
                        second_best_idx = sorted_indices[-2]

                        best_score = tag_scores[best_visual_idx]
                        second_best_score = tag_scores[second_best_idx]

                        # Check maximal condition
                        if best_score > score_threshold and (
                            best_score - second_best_score
                        ) > alpha * abs(best_score):
                            if not self.config.rule2:
                                matched_tag = tag_id
                                matched_visual = visual_ids[best_visual_idx]
                                break

                            # Check stability condition
                            stable = True
                            for other_idx in range(len(tag_ids)):
                                if other_idx == tag_idx:
                                    continue

                                competing_score = match_matrix[
                                    other_idx, best_visual_idx
                                ]
                                other_best_score = match_matrix[
                                    other_idx, max_visual_indices[other_idx]
                                ]

                                if competing_score > (1 + alpha) * other_best_score:
                                    stable = False
                                    break

                            if stable:
                                matched_tag = tag_id
                                matched_visual = visual_ids[best_visual_idx]
                                break

            # If no match found, exit loop
            if matched_tag is None:
                break

            # Add match and remove matched pair from consideration
            matched_pairs.append((matched_tag, matched_visual))
            self.match_score_manager.remove_tag(matched_tag)
            self.match_score_manager.remove_visual_id_from_all_tags(matched_visual)

        return matched_pairs

    def __str__(self):
        output = "MaxLikelihoodMatch\n"
        output += "match_score:\n"

        tag_ids, visual_ids, match_matrix, num_data_matrix = (
            self.match_score_manager.get_match_matrix()
        )

        # Print as a table
        # Print the header
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
                output += f"{num_data_matrix[i, j]:<10}"
            output += "\n"

        return output
