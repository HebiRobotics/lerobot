#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import hebi
import logging
import time
import threading
from functools import cached_property
from typing import Any
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from numpy import nan

from ..robot import Robot
from .config_hebi import HebiConfig

logger = logging.getLogger(__name__)


class HebiArm(Robot):
    """
    6-DoF arm designed by HEBI Robotics, Inc.
    """

    config_class = HebiConfig
    name = "hebi_arm"

    def __init__(self, config: HebiConfig):
        super().__init__(config)
        self.config = config
        self.arm_config = hebi.config.load_config(config.config_file)
        self.arm_connected = False
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        motors_ft = {}
        for joint_name in self.arm_config.names:
            motors_ft.update({f"{joint_name}.pos": float })
        has_gripper = self.arm_config.gains.get("gripper", None) != None
        if has_gripper:
            motors_ft.update({f"gripper.pos": float })
        return motors_ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.arm_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        lookup = hebi.Lookup()
        time.sleep(2.0)  # wait for lookup to find modules

        self.arm = hebi.arm.create_from_config(self.arm_config, lookup=lookup)
        while not self.arm:
            self.arm = hebi.arm.create_from_config(self.arm_config)
            time.sleep(1.0)

        gripper_group = lookup.get_group_from_names(self.arm_config.families, ["gripperSpool"])
        while not gripper_group:
            gripper_group = lookup.get_group_from_names(self.arm_config.families, ["gripperSpool"])
            time.sleep(1.0)

        self.gripper = hebi.arm.Gripper(gripper_group, -5, 1)
        self.gripper.load_gains(os.path.join(self.arm_config.config_location, self.arm_config.gains["gripper"]))

        self.arm.update()
        self.gripper.open()
        self.gripper.send()
        self.arm_connected = True
        logger.info("HEBI arm connected.")

        for cam in self.cameras.values():
            cam.connect()
        
        if self.config.send_action:
            self.configure()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.arm.group.command_lifetime = 0
        self.arm.send()
        
        # Run a separate thread to update the arm and gripper and send commands
        if hasattr(self, '_update_thread') and self._update_thread.is_alive():
            return  # Already running
        self._running = True

        def update_loop():
            while self._running:
                try:
                    self.arm_connected = self.arm.update()
                    self.arm.send()
                    self.gripper.send()
                except Exception as e:
                    logger.error(f"Update thread error: {e}")
                time.sleep(0.01)
        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()

        self.home_arm()

    def home_arm(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        logger.info(f"{self} homing arm...")

        home_pos = self.arm_config.user_data.get("home_position", None)
        if home_pos is None:
            raise ValueError(f"{self} home position not specified in config file.")
        
        assert(len(home_pos) == self.arm.size)
        goal = hebi.arm.Goal(self.arm.size)
        goal.add_waypoint(t = 5.0, position = home_pos)
        self.arm.set_goal(goal)

        logger.info(f"arm homed.")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        self.arm.update()
        positions = self.arm.last_feedback.position
        assert(len(positions) == len(self.arm_config.names))
        obs_dict = {f"{joint_name}.pos": positions[i] for i, joint_name in enumerate(self.arm_config.names)}
        gripper_pos = self.gripper._group.get_next_feedback().position[0]
        gripper_pos = max(-0.65, min(-0.4, gripper_pos))
        gripper_val = 1.0 - (gripper_pos + 0.65) / 0.25
        obs_dict.update({f"gripper.pos": float(round(gripper_val))})
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        if self.config.send_action:
            # Send action to the arm
            position_cmd = [action[f"{joint_name}.pos"] for joint_name in self.arm_config.names]
            goal = hebi.arm.Goal(self.arm.size)
            goal.add_waypoint(t = 0.3, position = position_cmd)
            self.arm.set_goal(goal)
            gripper_val = action.get("gripper.pos", 0.0)
            gripper_val = bool(gripper_val > 1e-2)
            if gripper_val:
                self.gripper.close()
            else:
                self.gripper.open()
            return action
        else:
            position_cmd = self.arm.last_feedback.position_command
            assert(len(position_cmd) == len(self.arm_config.names))
            goal_pos = {f"{joint_name}.pos": position_cmd[i] for i, joint_name in enumerate(self.arm_config.names)}
            gripper_pos_cmd = self.gripper._group.get_next_feedback().position[0]
            gripper_pos_cmd = max(-0.65, min(-0.4, gripper_pos_cmd))
            gripper_val = 1.0 - (gripper_pos_cmd + 0.65) / 0.25
            goal_pos.update({f"gripper.pos": float(bool(gripper_val > 1e-2))})

            return goal_pos

    def disconnect(self):
        if hasattr(self, '_running'):
            self._running = False
        if hasattr(self, '_update_thread') and self._update_thread.is_alive():
            self._update_thread.join(timeout=5)

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
