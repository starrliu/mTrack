# pylint: disable=no-member

from .data import XYWH
import os
import pandas as pd
import cv2
from .select import ReadingState

class MTrackDumper:
    def __init__(self, dump_dir, video=False) -> None:
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)

        cv_trace_path = os.path.join(dump_dir, 'cv_trace.csv')
        camera_data_path = os.path.join(dump_dir, 'camera_data.csv')
        matched_id_path = os.path.join(dump_dir, 'matched_id.csv')
        bc_path = os.path.join(dump_dir, 'bc.csv')
        gc_path = os.path.join(dump_dir, 'gc.csv')
        state_path = os.path.join(dump_dir, 'state.csv')
        video_path = os.path.join(dump_dir, 'video.mp4')
        select_trace_path = os.path.join(dump_dir, 'select_trace.csv')

        self.cv_trace_path = cv_trace_path
        self.camera_data_path = camera_data_path
        self.matched_id_path = matched_id_path
        self.bc_path = bc_path
        self.gc_path = gc_path
        self.state_path = state_path
        self.video_path = video_path
        self.select_trace_path = select_trace_path

        self.cv_trace_handle = open(cv_trace_path, 'w')
        self.camera_data_handle = open(camera_data_path, 'w')
        self.camera_data_handle.write("TimeStamp,FrameID\n")
        self.matched_id_handle = open(matched_id_path, 'w')
        self.bc_handle = open(bc_path, 'w')
        self.gc_handle = open(gc_path, 'w')
        self.state_handle = open(state_path, 'w')
        self.select_trace_path = open(select_trace_path, 'w')

        if video:
            self.video_handle = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))

    def dump_mtrack_select_trace(self, frame: int, cur_stage: ReadingState, reading_tags: list[bytes]):
        self.select_trace_path.write(f"{frame},{cur_stage},{','.join([tag.hex() for tag in reading_tags])}\n")

    def dump_mtrack_state(self, frame: int, state: str) -> None:
        self.state_handle.write(f"Frame {frame}:\n{state}\n\n")

    def dump_mtrack_res(self, frame: int, ts: pd.Timestamp, cv_trace: dict[int, XYWH],
                          matched_ids: list[tuple[bytes, int]], bc_res: list[tuple[bytes, int]],
                          gc_res: list[bytes]) -> None:
        
        # Dump cv_trace in gt format
        for cv_id, xywh in cv_trace.items():
            self.cv_trace_handle.write(f"{frame},{cv_id},{xywh.x},{xywh.y},{xywh.w},{xywh.h},1,1,1\n")

        # Dump camera data
        self.camera_data_handle.write(f"{ts},{frame}\n")
        # Dump matched_ids
        for tag, cv_id in matched_ids:
            self.matched_id_handle.write(f"{frame},{tag.hex()},{cv_id}\n")

        # Dump bc_res
        for tag, cv_id in bc_res:
            self.bc_handle.write(f"{frame},{tag.hex()},{cv_id}\n")

        # Dump gc_res
        for tag in gc_res:
            self.gc_handle.write(f"{frame},{tag.hex()}\n")

    def dump_img(self, img) -> None:
        self.video_handle.write(img)

    def close(self) -> None:
        self.cv_trace_handle.close()
        self.matched_id_handle.close()
        self.bc_handle.close()
        self.gc_handle.close()
        self.state_handle.close()
        self.select_trace_path.close()
        self.camera_data_handle.close
        
        if hasattr(self, 'video_handle'):
            self.video_handle.release()