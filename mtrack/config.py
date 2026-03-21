"""Configuration system for MTrack components.

This module provides configuration classes and loading utilities for all MTrack components,
allowing parameters to be loaded from YAML files instead of being hardcoded.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, Union
import yaml
import pandas as pd


@dataclass
class IDSWConfig:
    """Configuration for ID Switch Detector."""
    iou_thres: float = 0.6
    det_fps_thres: Tuple[int, int] = (3, 5)
    dis_thres: float = 0.05


@dataclass
class MatchConfig:
    """Configuration for Maximum Likelihood Match algorithm."""
    window_size: int = 60
    num_data_threshold: int = 60
    score_threshold: float = -0.55
    alpha: float = 0.1
    single_match_threshold: float = -0.5
    rule2: bool = True


@dataclass
class TrackletGraphConfig:
    """Configuration for Tracklet Graph."""
    graph_on: bool = True


@dataclass
class SelectConfig:
    """Configuration for Selective Reading."""
    static_threshold: float = 0.4
    slow_speed_reading_timeslot: int = 1000  # milliseconds
    max_high_speed_reading_timeslot: int = 1000  # milliseconds


@dataclass
class CheckerConfig:
    """Configuration for Global Checker."""
    window_size: int = 60
    max_checking_time: int = 20000  # milliseconds
    mismatch_threshold_best_visual: float = 0.3  # Threshold for comparing with best visual score
    mismatch_threshold_other_tag: float = 0.3    # Threshold for comparing with other tag's score
    num_data_threshold: int = 60
    abs_threshold: float = -0.55


@dataclass
class MTrackConfig:
    """Complete configuration for MTrack system."""
    idsw: IDSWConfig
    match: MatchConfig
    tracklet_graph: TrackletGraphConfig
    select: SelectConfig
    checker: CheckerConfig

    @classmethod
    def default(cls) -> 'MTrackConfig':
        """Create default configuration."""
        return cls(
            idsw=IDSWConfig(),
            match=MatchConfig(),
            tracklet_graph=TrackletGraphConfig(),
            select=SelectConfig(),
            checker=CheckerConfig()
        )


class ConfigLoader:
    """YAML configuration loader for MTrack."""

    @staticmethod
    def load_from_yaml(yaml_path: Union[str, Path]) -> MTrackConfig:
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            MTrackConfig: Loaded configuration object
            
        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration validation fails
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML file {yaml_path}: {e}") from e
        
        return ConfigLoader._dict_to_config(config_dict)

    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> MTrackConfig:
        """Convert dictionary to MTrackConfig object.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            MTrackConfig: Configuration object
        """
        # Handle IDSW configuration
        idsw_dict = config_dict.get('idsw', {})
        idsw_config = IDSWConfig(
            iou_thres=idsw_dict.get('iou_thres', 0.6),
            det_fps_thres=tuple(idsw_dict.get('det_fps_thres', [3, 5])),
            dis_thres=idsw_dict.get('dis_thres', 0.05)
        )

        # Handle Match configuration
        match_dict = config_dict.get('match', {})
        match_config = MatchConfig(
            window_size=match_dict.get('window_size', 60),
            num_data_threshold=match_dict.get('num_data_threshold', 60),
            score_threshold=match_dict.get('score_threshold', -0.55),
            alpha=match_dict.get('alpha', 0.1),
            single_match_threshold=match_dict.get('single_match_threshold', -0.5),
            rule2=match_dict.get('rule2', True)
        )

        # Handle TrackletGraph configuration
        tracklet_dict = config_dict.get('tracklet_graph', {})
        tracklet_config = TrackletGraphConfig(
            graph_on=tracklet_dict.get('graph_on', True)
        )

        # Handle Select configuration
        select_dict = config_dict.get('select', {})
        select_config = SelectConfig(
            static_threshold=select_dict.get('static_threshold', 0.4),
            slow_speed_reading_timeslot=select_dict.get('slow_speed_reading_timeslot', 1000),
            max_high_speed_reading_timeslot=select_dict.get('max_high_speed_reading_timeslot', 1000)
        )

        # Handle Checker configuration
        checker_dict = config_dict.get('checker', {})
        checker_config = CheckerConfig(
            window_size=checker_dict.get('window_size', 60),
            max_checking_time=checker_dict.get('max_checking_time', 20000),
            mismatch_threshold_best_visual=checker_dict.get('mismatch_threshold_best_visual', 0.3),
            mismatch_threshold_other_tag=checker_dict.get('mismatch_threshold_other_tag', 0.3),
            num_data_threshold=checker_dict.get('num_data_threshold', 60),
            abs_threshold=checker_dict.get('abs_threshold', -0.55)
        )

        return MTrackConfig(
            idsw=idsw_config,
            match=match_config,
            tracklet_graph=tracklet_config,
            select=select_config,
            checker=checker_config
        )

    @staticmethod
    def save_to_yaml(config: MTrackConfig, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration object to save
            yaml_path: Path where to save the YAML file
        """
        yaml_path = Path(yaml_path)
        
        config_dict = {
            'idsw': {
                'iou_thres': config.idsw.iou_thres,
                'det_fps_thres': list(config.idsw.det_fps_thres),
                'dis_thres': config.idsw.dis_thres
            },
            'match': {
                'window_size': config.match.window_size,
                'num_data_threshold': config.match.num_data_threshold,
                'score_threshold': config.match.score_threshold,
                'alpha': config.match.alpha,
                'single_match_threshold': config.match.single_match_threshold,
                'rule2': config.match.rule2
            },
            'tracklet_graph': {
                'graph_on': config.tracklet_graph.graph_on
            },
            'select': {
                'static_threshold': config.select.static_threshold,
                'slow_speed_reading_timeslot': config.select.slow_speed_reading_timeslot,
                'max_high_speed_reading_timeslot': config.select.max_high_speed_reading_timeslot
            },
            'checker': {
                'window_size': config.checker.window_size,
                'max_checking_time': config.checker.max_checking_time,
                'mismatch_threshold_best_visual': config.checker.mismatch_threshold_best_visual,
                'mismatch_threshold_other_tag': config.checker.mismatch_threshold_other_tag,
                'num_data_threshold': config.checker.num_data_threshold,
                'abs_threshold': config.checker.abs_threshold
            }
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    @staticmethod
    def create_default_config_file(yaml_path: Union[str, Path]) -> None:
        """Create a default configuration YAML file.
        
        Args:
            yaml_path: Path where to create the default config file
        """
        default_config = MTrackConfig.default()
        ConfigLoader.save_to_yaml(default_config, yaml_path)


# Convenience function for quick loading
def load_config(yaml_path: Union[str, Path]) -> MTrackConfig:
    """Convenience function to load configuration from YAML file.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        MTrackConfig: Loaded configuration object
    """
    return ConfigLoader.load_from_yaml(yaml_path)