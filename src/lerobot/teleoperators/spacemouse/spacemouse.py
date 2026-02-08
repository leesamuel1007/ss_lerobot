#!/usr/bin/env python

from curses import raw
import logging
import multiprocessing
import time
import numpy as np
# Assuming pyspacemouse is available in the python path or same directory
# If it is a local file, you might need: from . import pyspacemouse
from . import pyspacemouse 

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from ..teleoperator import Teleoperator
from .config_spacemouse import SpacemouseTeleopConfig

logger = logging.getLogger(__name__)


class Spacemouse(Teleoperator):
    """Spacemouse teleoperator implementation matching LeRobot structure."""

    config_class = SpacemouseTeleopConfig
    name = "spacemouse"

    def __init__(self, config: SpacemouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        
        # Shared state for multiprocessing (from spacemouse_expert.py)
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["buttons"] = [0, 0]
        
        self.process = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "roll", "pitch", "yaw", "reward"],
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self) -> None:
        # Open device connection (must be done in the main process first or inside the worker depending on lib)
        # spacemouse_expert.py calls open() globally, but here we do it safely.
        try:
            success = pyspacemouse.open()
            if not success:
                raise IOError("Failed to open Spacemouse.")
        except Exception as e:
            logger.error(f"Could not open Spacemouse: {e}")
            raise

        self._is_connected = True
        
        # Start the background reader process
        self.process = multiprocessing.Process(target=self._read_spacemouse, args=(self.latest_data,))
        self.process.daemon = True
        self.process.start()
        logger.info(f"Spacemouse connected and listening.")

    @staticmethod
    def _read_spacemouse(shared_data):
        """Background process loop to read HID data, adapted from spacemouse_expert.py."""
        # Ensure open in this process if needed, though usually HID needs to be opened here
        # if the library is not process-safe. pyspacemouse usually requires open() called
        # but expert example called it in init. We assume open() persists or we re-verify.
        
        while True:
            state = pyspacemouse.read_all()
            
            # Logic from spacemouse_expert.py
            # Default zero action
            action = [0.0] * 6
            buttons = [0, 0]

            if len(state) > 0:
                # Use the first device found (Index 0)
                s = state[0]
                
                # Mapping from spacemouse_expert.py:
                # [-y, x, z, -roll, -pitch, -yaw]
                action = [
                    -s.y, s.x, s.z,
                    -s.roll, -s.pitch, -s.yaw
                ]
                buttons = s.buttons

            # Update shared dictionary
            shared_data["action"] = action
            shared_data["buttons"] = buttons
            
            # Small sleep to prevent 100% CPU usage in loop if read_all is non-blocking
            time.sleep(0.001)

    @check_if_not_connected
    def get_action(self) -> np.ndarray:
        """Returns the latest action vector [x, y, z, r, p, y]."""
        # Retrieve from shared memory
        raw_action = self.latest_data["action"]
        raw_buttons = self.latest_data["buttons"]
        
        # Apply sensitivity from config
        scale_trans = self.config.max_translation
        scale_rot = self.config.max_rotation
        
        # Apply simple deadzone
        def dz(val):
            return 0.0 if abs(val) < self.config.deadzone else val
        
        reward_val = 0.0
        if len(raw_buttons)>1:
            if raw_buttons[0]:  # Assuming button 0 is for reward
                reward_val = 1.0
            else:
                reward_val = 0.0
        
        processed_action = np.array([
            dz(raw_action[0]) * scale_trans,
            dz(raw_action[1]) * scale_trans,
            dz(raw_action[2]) * scale_trans,
            dz(raw_action[3]) * scale_rot,
            dz(raw_action[4]) * scale_rot,
            dz(raw_action[5]) * scale_rot,
            reward_val
        ], dtype=np.float32)

        return processed_action

    def send_feedback(self, feedback: dict) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.join()
            self.process = None
        
        pyspacemouse.close()
        self._is_connected = False
        logger.info("Spacemouse disconnected.")