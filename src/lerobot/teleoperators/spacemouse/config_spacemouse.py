#!/usr/bin/env python

from dataclasses import dataclass
from typing import TypeAlias

from ..config import TeleoperatorConfig


@dataclass
class SpacemouseConfig:
    """Base configuration class for Spacemouse teleoperators."""

    # Sensitivity scaling for translation and rotation
    max_translation: float = 1.0
    max_rotation: float = 1.0
    
    # Deadzone to filter small movements
    deadzone: float = 0.05
    
    # Whether to control gripper via buttons
    use_gripper: bool = False


@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpacemouseTeleopConfig(TeleoperatorConfig, SpacemouseConfig):
    pass