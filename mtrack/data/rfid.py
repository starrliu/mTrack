"""
Data structures for storing RFID tag data.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class RFIDData:
    """
    RFIDData class is used to store the data item of RFID tag.

    Attributes:
        rssi (float): the RSSI of the tag.
        phase (float): the phase of the tag.
        antenna_id (int): which antenna the tag is read by.
        frequency (float): which frequency the tag is read at.
    """

    rssi: float
    phase: float
    antenna_id: int
    frequency: float

@dataclass
class RFIDMessage:
    """
    RFIDMessage class is used to store the message of RFID tag.

    Attributes:
        tag_id (bytes): the ID of the tag.
        timestamp (pd.Timestamp): the timestamp of the message.
        data (RFIDData): the data of the message.
    """
    
    tag_id: bytes
    timestamp: pd.Timestamp
    data: RFIDData

class TagData:
    def __init__(self, tag_id: bytes) -> None:
        self.tag_id = tag_id
        self.antid_to_data: dict[int, dict[str, list]] = {}

    def append(self, timestamp: pd.Timestamp, data: dict) -> None:
        """Append data to the tag.

        Args:
            timestamp (pd.Timestamp): the timestamp of the data point.
            data (dict): a dictionary containing the RFID data with keys:
            "phase", "rssi", "frequency", and "antenna".
        """
        phase = data["phase"]
        rssi = data["rssi"]
        freq = data["frequency"]
        antenna = data["antenna"]

        if antenna not in self.antid_to_data:
            self.antid_to_data[antenna] = {
                "timestamps": [],
                "phases": [],
                "rssis": [],
                "freqs": [],
            }

        self.antid_to_data[antenna]["timestamps"].append(timestamp)
        self.antid_to_data[antenna]["phases"].append(phase)
        self.antid_to_data[antenna]["rssis"].append(rssi)
        self.antid_to_data[antenna]["freqs"].append(freq)

    def _binary_search_left(self, timestamps: list, target: pd.Timestamp) -> int:
        """Binary search to find the leftmost index where timestamps[index] >= target.

        Args:
            timestamps (list): Sorted list of timestamps
            target (pd.Timestamp): Target timestamp

        Returns:
            int: Index of the first element >= target, or len(timestamps) if not found
        """
        left, right = 0, len(timestamps)
        while left < right:
            mid = (left + right) // 2
            if timestamps[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left

    def _binary_search_right(self, timestamps: list, target: pd.Timestamp) -> int:
        """Binary search to find the rightmost index where timestamps[index] < target.

        Args:
            timestamps (list): Sorted list of timestamps
            target (pd.Timestamp): Target timestamp

        Returns:
            int: Index of the last element < target
        """
        left, right = 0, len(timestamps)
        while left < right:
            mid = (left + right) // 2
            if timestamps[mid] >= target:
                right = mid
            else:
                left = mid + 1
        return left

    def get_data_in_range(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> list[tuple[pd.Timestamp, RFIDData]]:
        """Get data points within the specified time range using binary search.

        Args:
            start_time (pd.Timestamp): start time (inclusive)
            end_time (pd.Timestamp): end time (inclusive)

        Returns:
            list[tuple[pd.Timestamp, RFIDData]]: list of (timestamp, data) pairs within the range
        """
        result = []
        for antenna, data in self.antid_to_data.items():
            timestamps = data["timestamps"]
            if not timestamps:  # Skip empty lists
                continue
                
            # Find start and end indices using binary search
            start_idx = self._binary_search_left(timestamps, start_time)
            end_idx = self._binary_search_right(timestamps, end_time)
            
            # Collect data points within the range
            for i in range(start_idx, end_idx):
                rfid_data = RFIDData(
                    rssi=data["rssis"][i],
                    phase=data["phases"][i],
                    frequency=data["freqs"][i],
                    antenna_id=antenna
                )
                result.append((timestamps[i], rfid_data))
        
        # Sort by timestamp
        result.sort(key=lambda x: x[0])
        return result
