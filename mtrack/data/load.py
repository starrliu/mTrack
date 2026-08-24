"""
Load the data from the CSV files.
"""
import json

import pandas as pd

from ..utils import reverse_phase
from .rfid import TagData
from .cv import ObjectTrace


def load_object_trace(object_csv: str, camera_csv: str) -> dict[int, ObjectTrace]:
    """
    Load the object trace data from the CSV file.

    Args:
        object_csv(str): the path to the CSV file.
        camera_csv(str): the path to the camera timestamp file.

    Returns:
        dict[int, ObjectTrace]: the object trace data, with the object ID as the key.
    """
    # Read the object and camera CSV files
    object_df = pd.read_csv(
        object_csv,
        header=None,
        names=[
            "frame",
            "id",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "x",
            "y",
            "z",
        ],
    )
    camera_df = pd.read_csv(camera_csv)

    # Convert the TimeStamp column to pandas Timestamp
    camera_df["TimeStamp"] = pd.to_datetime(camera_df["TimeStamp"])
    
    # Create a frame to timestamp mapping dictionary for faster lookup
    frame_to_timestamp = dict(zip(camera_df["FrameID"], camera_df["TimeStamp"]))

    # Initialize the dictionary to store ObjectTrace objects
    object_traces = {}

    # Pre-process the data using vectorized operations
    for obj_id, group in object_df.groupby("id"):
        trace = ObjectTrace(obj_id)
        
        # Get timestamps for all frames at once
        timestamps = [frame_to_timestamp[frame] for frame in group["frame"]]
        
        # Create data dictionaries efficiently
        data_dicts = [
            {
                "frame": int(row["frame"]),
                "x": float(row["bb_left"]),
                "y": float(row["bb_top"]),
                "w": float(row["bb_width"]),
                "h": float(row["bb_height"]),
            }
            for _, row in group.iterrows()
        ]
        
        # Append data in bulk
        for ts, data in zip(timestamps, data_dicts):
            trace.append(ts, data)
            
        object_traces[obj_id] = trace

    return object_traces

def load_tag_data(rfid_csv: str) -> dict[bytes, TagData]:
    """
    Load the tag data from the CSV file.

    Args:
        rfid_csv(str): the path to the CSV file.

    Returns:
        dict[bytes, TagData]: the tag data, with the tag ID as the key.
    """
    # Read the RFID CSV file, ensure EpcId is read as string
    rfid_df = pd.read_csv(rfid_csv, dtype={"EpcId": str})

    # Convert the TimeStamp column to pandas Timestamp
    rfid_df["TimeStamp"] = pd.to_datetime(rfid_df["TimeStamp"])

    # Initialize the dictionary to store TagData objects
    tag_data = {}

    # Group by tag ID and process each tag's data
    for tag_id, group in rfid_df.groupby("EpcId"):
        # Convert tag ID string to bytes
        tag_id_bytes = bytes.fromhex(tag_id)

        if tag_id_bytes not in tag_data:
            tag_data[tag_id_bytes] = TagData(tag_id_bytes)

        # Process each row of data for this tag
        for _, row in group.iterrows():
            data_dict = {
                "phase": reverse_phase(row["Phase"]),  # Reverse the phase
                "rssi": float(row["RSSI"]),
                "frequency": float(row["Frequency"] * 1e6),
                "antenna": int(row["AntennaID"]),
            }
            tag_data[tag_id_bytes].append(row["TimeStamp"], data_dict)

    return tag_data

def load_config(config_path: str) -> dict:
    
    with open(config_path, 'r') as f:
        config = json.load(f)

        # weight = config['weight']
        # rfid_config_path = config['rfid_config_path']
        tmp_tags = config['tags']
        tags = {}
        for tag, cvid in tmp_tags.items():
            tags[bytes.fromhex(tag)] = int(cvid)

        tmp_antpos = config['ant_pos']
        antpos = {}
        for ant_id, pos in tmp_antpos.items():
            antpos[int(ant_id)] = tuple(pos)

        p2m = config['p2m']

        # max_det = config['max_det']
        # half = bool(config['half'])
        # conf = float(config['conf_thres'])

    return {
        # "weight": weight,
        # "rfid_config_path": rfid_config_path,
        "tags": tags,
        "ant_pos": antpos,
        "p2m": p2m,
        # "max_det": max_det,
        # "half": half,
        # "conf_thres": conf
    }    