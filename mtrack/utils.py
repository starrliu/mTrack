import numpy as np
import pandas as pd
from collections import defaultdict
from bidict import bidict
from typing import Any
from .data import XYWH

def distance_bbox(xywh1: XYWH, xywh2: XYWH):
    cx1, cy1 = xywh1.x + xywh1.w/2, xywh1.y + xywh1.h/2
    cx2, cy2 = xywh2.x + xywh2.w/2, xywh2.y + xywh2.h/2

    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

def iou(xywh1: XYWH, xywh2: XYWH):
    x1, y1, w1, h1 = xywh1.x, xywh1.y, xywh1.w, xywh1.h
    x2, y2, w2, h2 = xywh2.x, xywh2.y, xywh2.w, xywh2.h
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = w1 * h1
    boxBArea = w2 * h2
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

class Multi2MultiMapping:
    """
    Multi-to-multi mapping.

    map[ida1] = [idb1, idb2, ...]
    inverse_map[idb1] = [ida1, ida2, ...]
    """

    def __init__(self, key_type_a, key_type_b):
        self.map : dict[Any, set] = {}
        self.inverse_map : dict[Any, set] = {}
        self.key_type_a = key_type_a
        self.key_type_b = key_type_b

    def add(self, key_a, key_b):
        if not isinstance(key_a, self.key_type_a) or not isinstance(key_b, self.key_type_b):
            raise ValueError("Key type mismatch.")
    
        if key_a not in self.map:
            self.map[key_a] = set()
        if key_b not in self.inverse_map:
            self.inverse_map[key_b] = set()

        self.map[key_a].add(key_b)
        self.inverse_map[key_b].add(key_a)

    def remove(self, key_a, key_b):
        if not isinstance(key_a, self.key_type_a) or not isinstance(key_b, self.key_type_b):
            raise ValueError("Key type mismatch.")
        
        if key_a in self.map:
            self.map[key_a].remove(key_b)
        if key_b in self.inverse_map:
            self.inverse_map[key_b].remove(key_a)

    def remove_key_a(self, key_a):
        if not isinstance(key_a, self.key_type_a):
            raise ValueError("Key type mismatch.")
        
        if key_a in self.map:
            for key_b in self.inverse_map:
                if key_a in self.inverse_map[key_b]:
                    self.inverse_map[key_b].remove(key_a)
            del self.map[key_a]

    def remove_key_b(self, key_b):
        if not isinstance(key_b, self.key_type_b):
            raise ValueError("Key type mismatch.")
        
        if key_b in self.inverse_map:
            for key_a in self.map:
                if key_b in self.map[key_a]:
                    self.map[key_a].remove(key_b)
            del self.inverse_map[key_b]

    def get(self, key_a):
        if not isinstance(key_a, self.key_type_a):
            raise ValueError("Key type mismatch.")
        
        return self.map.get(key_a, set())
    
    def get_inverse(self, key_b):
        if not isinstance(key_b, self.key_type_b):
            raise ValueError("Key type mismatch.")
        
        return self.inverse_map.get(key_b, set())
    
    def __iter__(self):
        """
        迭代器方法，遍历 map 中的键值对。
        """
        for key_a, values in self.map.items():
            yield key_a, values

    def items(self):
        """
        返回 map 中的所有键值对。
        """
        return self.map.items()

    def inverse_items(self):
        """
        返回 inverse_map 中的所有键值对。
        """
        return self.inverse_map.items()

class StatusManager:
    def __init__(self) -> None:
        self.id_to_status = {}
        self.status_to_ids = defaultdict(set)
    
    def add_object(self, obj_id, status):
        """
        Add an object to the StatusManager.

        Args:
            obj_id: the id of the object.
            status: the status of the object.

        Note:
            If the object already exists in the StatusManager,
            the status will be updated.
        """
        # Remove the object from the previous status set
        if obj_id in self.id_to_status:
            prev_status = self.id_to_status[obj_id]
            self.status_to_ids[prev_status].remove(obj_id)
        
        # Update the object status
        self.id_to_status[obj_id] = status
        self.status_to_ids[status].add(obj_id)
    
    def get_status(self, obj_id):
        """
        Get the status of the object.

        Args:
            obj_id: the id of the object.

        Returns:
            status: the status of the object.
        """
        return self.id_to_status[obj_id]

    def get_ids_by_status(self, status):
        """
        Get the ids of objects with the specified status.

        Args:
            status: the status of the objects.

        Returns:
            ids: the ids of the objects with the specified status.
        """
        return self.status_to_ids[status].copy()

    def items(self):
        """
        Return all items in the StatusManager.
        """
        return self.id_to_status.items()

    def inverse_items(self):
        """
        Return all items in the inverse StatusManager.
        """
        return self.status_to_ids.items()

# def mean_angle_half_circle(angles_rad: np.ndarray):
#     # Check if angles are all within [0, π]
#     if np.any(angles_rad < 0) or np.any(angles_rad > np.pi):
#         raise ValueError("The angles should be within [0, π]")

#     # 计算单位向量的和
#     two_angles = 2 * angles_rad
#     sin_sum = np.sum(np.sin(two_angles))
#     cos_sum = np.sum(np.cos(two_angles))
    
#     # 计算均值角度
#     mean_angle_rad = np.arctan2(sin_sum, cos_sum)
        
#     # 将角度转换为正值
#     if mean_angle_rad < 0:
#         mean_angle_rad += 2 * np.pi

#     return mean_angle_rad/2

def sub_phase_to_half_circle(phase1: float, phase2: float):
    """
    计算两个相位之间的差值，返回值在-pi/2到pi/2之间
    
    Args:  
        phase1: 第一个相位
        phase2: 第二个相位
        其中phase1和phase2的范围是0到pi。
    
    Returns:
        相位差值，范围是-pi/2到pi/2

    Example:
        >>> sub_phase_to_half_circle(0.1, 0.2)
        0.1
        >>> sub_phase_to_half_circle(0.1, 3.14)
        -0.1
    """ 

    if phase1 < 0 or phase1 > np.pi or phase2 < 0 or phase2 > np.pi:
        raise ValueError("Phase value out of range")

    delta_phase_1 = phase2 - phase1
    delta_phase_2 = (phase2 - np.pi) - phase1
    delta_phase_3 = (phase2 + np.pi) - phase1

    delta_phases = [delta_phase_1, delta_phase_2, delta_phase_3]
    delta_phase = min(delta_phases, key=lambda x: abs(x))

    return delta_phase

def norm_phase_to_half_circle(phase):
    """
    Normalize the phase (or phases if an array or a scalar is provided) to be between 0 and 3.14 radians
    """
    # Check if the input is a scalar (not an array)
    is_scalar = np.isscalar(phase)
    
    # Convert scalar to np.ndarray for uniform processing
    phase = np.asarray(phase)
    
    # Normalize phase values to be within [0, π]
    normalized_phase = phase % np.pi
    # Ensure we only attempt item assignment if it's an array
    if not is_scalar:
        normalized_phase[normalized_phase < 0] += np.pi
    else:
        if normalized_phase < 0:
            normalized_phase += np.pi
    
    # If the input was a scalar, convert back to scalar
    if is_scalar:
        return normalized_phase.item()
    else:
        return normalized_phase
    
def reverse_phase(phase: float)-> float:
    """
    Reverse the phase.
    """

    if phase < 0 or phase > 2 * np.pi:
        raise ValueError("The phase should be within [0, 2π]")
    
    return 2 * np.pi - phase