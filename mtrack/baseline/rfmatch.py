from ..data.cv import ObjectTrace
from ..data.rfid import TagData
from ..match import ManhattanMatcher
from ..utils import norm_phase_to_half_circle, sub_phase_to_half_circle
import pandas as pd
import numpy as np
import cvxpy as cp
import time

class MatchSolver:
    def __init__(self, tags: dict[bytes, int], object_data: dict[int, ObjectTrace], camera_csv: pd.DataFrame, rfid_data: dict[bytes, TagData],
                 ant_pos: dict[int, tuple[float, float, float]], p2m: float):
        self.object_data = object_data
        self.rfid_data = rfid_data
        self.camera_csv = camera_csv
        self.ant_pos = ant_pos
        self.p2m = p2m
        self.tags = tags
        self.matcher = ManhattanMatcher()

    def _get_ids(self, start_time: pd.Timestamp, end_time: pd.Timestamp):
        objids = []
        for obj_id in self.object_data:
            obj_trace = self.object_data[obj_id]
            
            # Check if object trace has data in the time range
            if (obj_trace.start_timestamp is not None and 
                obj_trace.end_timestamp is not None and
                obj_trace.start_timestamp <= end_time and 
                obj_trace.end_timestamp >= start_time):
                objids.append(obj_id)
        
        tagids = list(self.tags.keys())
        
        pairs = []
        for obj_id in objids:
            for tagid in tagids:
                pairs.append((obj_id, tagid))

        return pairs
    
    def _calculate_match_score(self, obj_trace: ObjectTrace, rfid_data: list[tuple[pd.Timestamp, dict]]):
        prev_freq = None
        prev_data = {}

        scores = []

        # Get the first timestamp from rfid_data
        if len(rfid_data) == 0:
            return scores
        
        start_t = max(obj_trace.start_timestamp, rfid_data[0][0])

        for timestamp, rfid_item in rfid_data:

            if timestamp < start_t:
                continue

            phase = norm_phase_to_half_circle(rfid_item['phase'])
            channel = rfid_item["frequency"] 
            antenna = rfid_item["antenna_id"]

            try:
                pos = obj_trace.get_pos(timestamp)
            except ValueError:
                # print("Error: ", timestamp)
                # print("Object ID: ", obj_trace.object_id)
                continue

            if prev_freq != channel:
                prev_data = {}
                prev_freq = channel

            if antenna not in prev_data:
                cx = pos.x + pos.w/2
                cy = pos.y + pos.h/2

                z = 0

                dis = np.sqrt((cx - self.ant_pos[antenna][0])**2 + 
                              (cy - self.ant_pos[antenna][1])**2 + 
                              (z - self.ant_pos[antenna][2])**2) * self.p2m
                
                prev_data[antenna] = (phase, (cx, cy, z), dis, timestamp)
            else:
                prev_phase, prev_pos, prev_dis, prev_timestamp = prev_data[antenna]

                cx = pos.x + pos.w/2
                cy = pos.y + pos.h/2
                z = 0
                dis = np.sqrt((cx - self.ant_pos[antenna][0])**2 + 
                              (cy - self.ant_pos[antenna][1])**2 + 
                              (z - self.ant_pos[antenna][2])**2) * self.p2m
                
                actual_delta_phase = sub_phase_to_half_circle(prev_phase, phase)
                delta_dis = dis - prev_dis
                predicted_delta_phase = 4 * np.pi * delta_dis * channel / 3e8

                score = self.matcher.predict(actual_delta_phase, predicted_delta_phase)

                scores.append(score)

                prev_data[antenna] = (phase, (cx, cy, z), dis, timestamp)

        return scores

    def _calculate_match_scores(self, pairs: list[tuple[int, bytes]], start_time: pd.Timestamp, end_time: pd.Timestamp):
        scores = []
        weights = []

        for obj_id, tagid in pairs:
            obj_trace = self.object_data[obj_id]
            rfid_data = self.rfid_data[tagid]

            # Get RFID data in range - returns list of (timestamp, RFIDData) tuples
            raw_data = rfid_data.get_data_in_range(start_time, end_time)
            
            # Convert to format expected by _calculate_match_score
            data = []
            for timestamp, rfid_item in raw_data:
                data_dict = {
                    'phase': rfid_item.phase,
                    'frequency': rfid_item.frequency,
                    'antenna_id': rfid_item.antenna_id
                }
                data.append((timestamp, data_dict))

            tmp_scores = self._calculate_match_score(obj_trace, data)
            if len(tmp_scores) == 0:
                score = 0
            else:
                score = np.mean(tmp_scores)
            # print(tmp_scores)
            weight = np.min([len(tmp_scores)/60, 1])

            scores.append(score)
            weights.append(weight)

        # 将scores转换为对角线矩阵
        scores = np.diag(scores)

        # 将weights转换为向量
        weights = np.array(weights)

        return scores, weights

    def _calculate_constrains(self, pairs: list[tuple[int, bytes]], start_time: pd.Timestamp, end_time: pd.Timestamp):
        
        # 限制：
        # 1. 每个object只能匹配一个tag
        # 2. 同一个tag不能分配给时间重叠的object

        # 初始化限制矩阵
        constrains = np.zeros((len(pairs), len(pairs)))

        for idx1 in range(len(pairs)):
            for idx2 in range(idx1+1, len(pairs)):
                obj_id1, tagid1 = pairs[idx1]
                obj_id2, tagid2 = pairs[idx2]

                # 检查条件：1
                if obj_id1 == obj_id2:
                    constrains[idx1, idx2] = 1
                    constrains[idx2, idx1] = 1
                    continue

                # # 检查条件：2
                if tagid1 == tagid2 and obj_id1 != obj_id2:
                    # 检查时间重叠
                    obj_trace1 = self.object_data[obj_id1]
                    obj_trace2 = self.object_data[obj_id2]

                    # Use the actual trace timestamps to check overlap
                    start_time_1 = max(obj_trace1.start_timestamp, start_time)
                    end_time_1 = min(obj_trace1.end_timestamp, end_time)
                    start_time_2 = max(obj_trace2.start_timestamp, start_time) 
                    end_time_2 = min(obj_trace2.end_timestamp, end_time)

                    if start_time_1 < end_time_2 and start_time_2 < end_time_1:
                        constrains[idx1, idx2] = 1
                        constrains[idx2, idx1] = 1
                        continue

        return constrains

    def solve_interval(self, start_time: pd.Timestamp, end_time: pd.Timestamp, mu: float = 0.5, timeit: bool = False):
        
        pairs = self._get_ids(start_time, end_time)

        scores, weights = self._calculate_match_scores(pairs, start_time, end_time)

        constrains = self._calculate_constrains(pairs, start_time, end_time)

        x = cp.Variable(len(pairs), boolean=True)
        # objective = cp.Minimize(cp.quad_form(x, -scores) - mu * cp.sum(cp.multiply(weights, x)))
        scores = abs(scores)
        weights = -weights
        objective = cp.Minimize(x.T @ scores @ x + mu * x.T @ weights)
        constraints = []
        for i in range(len(pairs)):
            for j in range(len(pairs)):
                if constrains[i,j] == 1:
                    constraints.append(x[i] + x[j] <= 1)

        if timeit:
            t1 = time.time()
            
        prob = cp.Problem(objective, constraints)
        # prob = cp.Problem(objective)
        prob.solve()

        if timeit:
            t2 = time.time()
            print("Time: ", t2 - t1)

        return x.value, prob, pairs, scores, weights, constrains
    
    def solve(self, frame_interval: int=9000, mu: float = 0.5):
        start_frame = self.camera_csv["FrameID"].min()
        end_frame = self.camera_csv["FrameID"].max()
        
        results = []    # (old_id, new_id, start_f, end_f)

        for frame_idx in range(start_frame, end_frame, frame_interval):
            start_time = self.camera_csv[self.camera_csv["FrameID"] == frame_idx]["TimeStamp"].values[0]
            end_f = min(frame_idx + frame_interval, end_frame)
            end_time = self.camera_csv[self.camera_csv["FrameID"] == end_f]["TimeStamp"].values[0]

            start_time, end_time = pd.Timestamp(start_time), pd.Timestamp(end_time)

            x, prob, pairs, _, _, _ = self.solve_interval(start_time, end_time, mu=mu, timeit=False)

            matched_idx = [idx for idx in range(len(pairs)) if x[idx] == 1]

            for idx in matched_idx:
                obj_id, tagid = pairs[idx]
                results.append((obj_id, self.tags[tagid], frame_idx, end_f-1))

        return results
