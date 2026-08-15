from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class SO101Motion:
    def __init__(self, port: str, urdf: str, target_frame: str = "gripper_frame_link", robot_id: str = "keyboard_follower"):
        self.port = port
        self.urdf = str(Path(urdf))
        self.target_frame = target_frame
        self.robot_id = robot_id
        self.robot = None
        self.ik = None
        self.safety_bounds = {
            "x": (0.15, 0.45),
            "y": (-0.20, 0.20),
            "z": (-0.10, 0.15),
        }

    def _import_robot_classes(self):
        try:
            from lerobot.robots.so_follower import SO101Follower, SOFollowerRobotConfig
            return SO101Follower, SOFollowerRobotConfig
        except Exception:
            pass
        try:
            from lerobot.robots.so_follower.so_follower import SO101Follower
            from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
            return SO101Follower, SOFollowerRobotConfig
        except Exception:
            pass
        from lerobot.robots.so_follower.so_follower import SO100Follower as SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO100FollowerConfig as SOFollowerRobotConfig
        return SO101Follower, SOFollowerRobotConfig

    def connect(self) -> None:
        if self.robot is not None:
            return
        from lerobot.model.kinematics import RobotKinematics

        SO101Follower, SOFollowerRobotConfig = self._import_robot_classes()
        try:
            cfg = SOFollowerRobotConfig(port=self.port, id=self.robot_id, use_degrees=True)
        except TypeError:
            try:
                cfg = SOFollowerRobotConfig(port=self.port, id=self.robot_id)
            except TypeError:
                cfg = SOFollowerRobotConfig(port=self.port)
        self.robot = SO101Follower(cfg)
        self.robot.connect()
        self.ik = RobotKinematics(self.urdf, self.target_frame)

    def disconnect(self) -> None:
        if self.robot is not None:
            try:
                self.robot.disconnect()
            finally:
                self.robot = None
                self.ik = None

    def get_q(self) -> np.ndarray:
        if self.robot is None:
            raise RuntimeError("Robot is not connected.")
        obs = self.robot.get_observation()
        return np.array([obs[f"{m}.pos"] for m in self.robot.bus.motors], dtype=float)

    def get_ee_xyz(self) -> np.ndarray:
        if self.ik is None:
            raise RuntimeError("Kinematics is not initialized.")
        q_now = self.get_q()
        pose = self.ik.forward_kinematics(q_now)
        return np.asarray(pose[:3, 3], dtype=float)

    def _assert_safe(self, xyz: np.ndarray) -> None:
        x, y, z = np.asarray(xyz, dtype=float)
        for name, value in (("x", x), ("y", y), ("z", z)):
            lo, hi = self.safety_bounds[name]
            if not lo <= value <= hi:
                raise ValueError(f"Unsafe target {name}={value:.4f}; expected {lo:.4f}..{hi:.4f}")

    def move_ee_xyz(
        self,
        xyz: np.ndarray,
        duration: float = 0.6,
        steps: int | None = None,
    ) -> None:
        """
        Smooth quintic interpolation in XYZ.
        """
        if self.robot is None or self.ik is None:
            raise RuntimeError("Robot is not connected.")
        target = np.asarray(xyz, dtype=float).reshape(3)
        self._assert_safe(target)
        start = self.get_ee_xyz()
        steps = steps or max(2, int(duration * 50))
        dt = duration / max(steps - 1, 1)
        s = np.linspace(0.0, 1.0, steps)
        blend = 10 * s**3 - 15 * s**4 + 6 * s**5
        waypoints = start + (target - start) * blend[:, None]

        for waypoint in waypoints:
            q_now = self.get_q()
            pose = self.ik.forward_kinematics(q_now)
            pose[:3, 3] = waypoint
            q_target = self.ik.inverse_kinematics(q_now, pose)
            action = {f"{m}.pos": float(q_target[i]) for i, m in enumerate(self.robot.bus.motors)}
            if "gripper" in self.robot.bus.motors:
                action["gripper.pos"] = float(q_now[list(self.robot.bus.motors).index("gripper")])
            self.robot.send_action(action)
            time.sleep(dt)

    def press_key(
        self,
        target_xyz: np.ndarray,
        press_cfg: dict,
        hover_only: bool = False,
    ) -> None:
        """
        hover -> descend -> dwell -> retract.
        """
        target = np.asarray(target_xyz, dtype=float).reshape(3)
        surface_z = float(press_cfg["keyboard_surface_z_m"])
        hover = target.copy()
        hover[2] = surface_z + float(press_cfg.get("hover_offset_m", 0.035))
        press = target.copy()
        press[2] = surface_z + float(press_cfg.get("press_offset_m", -0.004))

        self.move_ee_xyz(hover, duration=0.5)
        if not hover_only:
            self.move_ee_xyz(press, duration=0.2)
            time.sleep(float(press_cfg.get("dwell_s", 0.12)))
            self.move_ee_xyz(hover, duration=0.25)
