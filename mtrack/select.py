"""
This module implements selective reading functionality for RFID tags.
It provides mechanisms for both slow and high speed reading modes,
and includes utilities for managing tag selection and data collection.
"""

from enum import Enum
from typing import List

import numpy as np
import pandas as pd

from .batch_select import FilterManager
from .data import TagData, RFIDData, RFIDMessage
from .utils import sub_phase_to_half_circle, norm_phase_to_half_circle
from .config import SelectConfig

def _cusum(antid_to_data: dict):
    """
    Using CUSUM algorithm to help check if the RF data is static.

    Args:
        antid_to_data: The RF data. key: antenna ID, value: dict of timestamps, phases, rssis,
            and frequencies.
    """

    cusum_each_ant = {}
    for antid, data in antid_to_data.items():

        freqs = data["freqs"]

        if len(freqs) == 0:
            continue

        phases = [norm_phase_to_half_circle(p) for p in data["phases"]]
        prev_freq = freqs[0]
        prev_phase = phases[0]
        prev_ts = data["timestamps"][0]
        cusum = [0]

        for i in range(1, len(freqs)):
            freq = freqs[i]
            if freq == prev_freq:
                delta_phase = sub_phase_to_half_circle(prev_phase, phases[i])
                delta_ts = (data["timestamps"][i] - prev_ts).total_seconds()

                prev_freq = freqs[i]
                prev_phase = phases[i]
                prev_ts = data["timestamps"][i]

                if delta_ts == 0:
                    v = 0
                else:
                    v = delta_phase / delta_ts * (3 * 10**8 / freq) / (4 * np.pi)

                cusum.append(v + cusum[-1])

            prev_freq = freqs[i]
            prev_phase = phases[i]
            prev_ts = data["timestamps"][i]

        cusum_each_ant[antid] = cusum

    return cusum_each_ant


def is_static(antid_to_data, threshold: float = 0.4):
    """
    Check if the RF data is static.

    Args:
        antid_to_data: The RF data. key: antenna ID, value: dict of timestamps, phases, rssis,
            and frequencies.
        threshold: The threshold to determine if the RF data is static.
    """

    cusum_each_ant = _cusum(antid_to_data)

    max_cusum = 0
    for _, cusum in cusum_each_ant.items():
        max_cusum = np.max([max_cusum, np.max(np.abs(cusum))])
        if np.max(np.abs(cusum)) > threshold:
            return False

    return True


class ReadingState(Enum):
    """Enum class representing the possible states of RFID tag reading."""
    SLOW_SPEED = 0
    HIGH_SPEED = 1


class SelectiveRead:
    """
    A class that manages selective reading of RFID tags.
    
    This class implements a dual-mode reading strategy with slow and high speed modes.
    It manages tag selection, scheduling, and data collection while optimizing reading performance.
    """

    def __init__(
        self,
        tags: list[bytes],
        address: str = "tcp://*:5556",
        config: SelectConfig = None,
    ):
        if config is None:
            config = SelectConfig()
        
        self.config = config
        self.filter_manager = FilterManager(tags, address)

        self.tags = tags

        self.current_scheduling_tags: set[bytes] = (
            set()
        )  # The current scheduling tags. Set by API: set_selective_tags.
        self.to_be_add_tags: set[bytes] = (
            set()
        )  # The tags that are waiting to be added to the scheduling tags.
        self.to_be_remove_tags: set[bytes] = (
            set()
        )  # The tags that are waiting to be removed from the scheduling tags.

        self.reading_tags: dict[bytes, TagData] = (
            {}
        )  # Tag ID to (start_ts, TagDatadf)

        self.cur_timeslot = None
        self.slow_speed_reading_timeslot = pd.Timedelta(
            f"{self.config.slow_speed_reading_timeslot}ms"
        )
        self.max_high_speed_reading_timeslot = pd.Timedelta(
            f"{self.config.max_high_speed_reading_timeslot}ms"
        )
        self.min_high_speed_reading_timeslot = pd.Timedelta("1000ms")

        self.start_ts = None  # When current ts - start_ts > timeslot or start_ts = None, goto next state.
        self.cur_ts = None
        self.current_reading_state = None
        self.next_reading_state = ReadingState.SLOW_SPEED

    def set_selective_tags(self, tags: set[bytes]):
        """
        Set the selective tags.

        Args:
            tags: The set of tags to be selected.
        """

        self.to_be_add_tags = tags - self.current_scheduling_tags

        self.to_be_remove_tags = self.current_scheduling_tags - tags

    def _is_static(self, tag: bytes):
        """
        Check if the RF data of a tag is static.

        Args:
            tag: Tag ID.
        """
        return is_static(
            self.reading_tags[tag].antid_to_data, self.config.static_threshold
        )

    def update(
        self, cur_ts: pd.Timestamp, msg: List[RFIDMessage]
    ):
        """
        Update the filter manager with new incoming data.

        Args:
            cur_ts: Current timestamp.
            msg: List of RFIDMessage objects containing tag data.

        Returns:
            Nothing.
        """
        self.cur_ts = cur_ts

        if self.start_ts is None or self.cur_ts - self.start_ts > self.cur_timeslot:
            # Start a new reading stage
            self._start_new_reading_stage(msg)
            self.start_ts = self.cur_ts
        else:
            # Update the reading tags
            for rfid_msg in msg:
                if rfid_msg.tag_id in self.reading_tags:
                    data_dict = {
                        "phase": rfid_msg.data.phase,
                        "rssi": rfid_msg.data.rssi,
                        "frequency": rfid_msg.data.frequency,
                        "antenna": rfid_msg.data.antenna_id,
                    }
                    self.reading_tags[rfid_msg.tag_id].append(rfid_msg.timestamp, data_dict)

    def _start_new_reading_stage(self, msg: List[RFIDMessage]):
        if self.next_reading_state == ReadingState.SLOW_SPEED:
            self._start_slow_speed_reading(msg)
        elif self.next_reading_state == ReadingState.HIGH_SPEED:
            self._start_high_speed_reading(msg)

    def _start_slow_speed_reading(
        self, msg: List[RFIDMessage]
    ):
        """
        Start slow speed reading mode to read all tags in the scheduling tags.

        Args:
            msg: List of RFIDMessage objects containing tag data.
        """
        # Step 0: Update the state variables
        self.cur_timeslot = self.slow_speed_reading_timeslot
        self.current_reading_state = ReadingState.SLOW_SPEED
        self.next_reading_state = ReadingState.HIGH_SPEED

        # Clean the reading tags
        self.reading_tags.clear()

        # Step 1: Update the scheduling tags
        self.current_scheduling_tags = self.current_scheduling_tags.union(
            self.to_be_add_tags
        )
        self.current_scheduling_tags = (
            self.current_scheduling_tags - self.to_be_remove_tags
        )
        self.to_be_add_tags.clear()
        self.to_be_remove_tags.clear()

        # Step 2: Update the FilterManager
        self.filter_manager.set_filters(self.current_scheduling_tags)
        self.filter_manager.send()

        # Step 3: Update the reading tags
        for tag in self.current_scheduling_tags:
            self.reading_tags[tag] = TagData(tag)

        # Step 4: Update the data
        for rfid_msg in msg:
            if rfid_msg.tag_id in self.reading_tags:
                data_dict = {
                    "phase": rfid_msg.data.phase,
                    "rssi": rfid_msg.data.rssi,
                    "frequency": rfid_msg.data.frequency,
                    "antenna": rfid_msg.data.antenna_id,
                }
                self.reading_tags[rfid_msg.tag_id].append(rfid_msg.timestamp, data_dict)

    def _start_high_speed_reading(
        self, msg: List[RFIDMessage]
    ):
        """
        Start high speed reading mode to read moving tags.

        Args:
            msg: List of RFIDMessage objects containing tag data.
        """
        # High speed reading: read the tags that are moving.
        # The state of tags are determined by the data collected in the slow speed reading.

        # Step 0: Remove the to_be_remove_tags from the scheduling tags.
        self.current_scheduling_tags = (
            self.current_scheduling_tags - self.to_be_remove_tags
        )
        self.to_be_remove_tags.clear()

        # Step 1: Get the tags that are moving.
        moving_tags = set()
        for tag in self.reading_tags:
            if not self._is_static(tag) and tag in self.current_scheduling_tags:
                moving_tags.add(tag)

        if len(moving_tags) == 0:
            # start slow speed reading
            self._start_slow_speed_reading(msg)
            return

        # Step 2: Update the state variables
        reading_rags_ratio = len(moving_tags) / len(self.current_scheduling_tags)

        self.cur_timeslot = max(
            self.min_high_speed_reading_timeslot,
            self.max_high_speed_reading_timeslot * reading_rags_ratio,
        )
        self.current_reading_state = ReadingState.HIGH_SPEED
        self.next_reading_state = ReadingState.SLOW_SPEED

        # Clean the reading tags
        self.reading_tags.clear()

        # Step 3: Update the FilterManager.
        self.filter_manager.set_filters(moving_tags)
        self.filter_manager.send()

        # Step 4: Update the reading tags.
        for tag in moving_tags:
            self.reading_tags[tag] = TagData(
                tag
            )  # Seems no need to update the data here.

        # Step 5: Update the data.
        # No need to update the data in the high speed reading stage.

    def close(self):
        """Close the selective read."""
        self.filter_manager.close()

    def __str__(self) -> str:
        """Return a string representation of the SelectiveRead instance."""
        output = ""
        reading_tags = self.reading_tags.keys()
        reading_tags = [tag.hex() for tag in reading_tags]
        output += "Select reading tags:\n"
        for tag in reading_tags:
            output += f"    {tag}\n"

        output += "Select scheduling tags:\n"
        for tag in self.current_scheduling_tags:
            output += f"    {tag.hex()}\n"

        output += "Select to be add tags:\n"
        for tag in self.to_be_add_tags:
            output += f"    {tag.hex()}\n"

        output += "Select to be remove tags:\n"
        for tag in self.to_be_remove_tags:
            output += f"    {tag.hex()}\n"

        output += f"Select current reading state: {self.current_reading_state}\n"
        output += f"Select next reading state: {self.next_reading_state}\n"

        output += f"Select rest time: {self.cur_timeslot - (self.cur_ts - self.start_ts)}\n"
        return output
