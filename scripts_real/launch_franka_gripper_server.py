"""
Zerorpc server that exposes Franka Hand gripper controls.
Run this on the robot control PC (same place as polymetis RobotServer).
"""
from typing import Optional

import zerorpc
import numpy as np
from polymetis import RobotInterface


class FrankaGripperInterface:
    def __init__(self, robot_host: str = "localhost"):
        self.robot = RobotInterface(robot_host)
        if self.robot.gripper is None:
            raise RuntimeError("RobotInterface does not expose a gripper instance")
        self.gripper = self.robot.gripper

    def homing(self):
        # Franka hand supports homing to learn current width limits
        return self.gripper.home()

    def goto(self, width: float, speed: float = 0.1, force: Optional[float] = None):
        # speed is in meters/second, width in meters
        kwargs = {"width": float(width), "speed": float(speed)}
        if force is not None:
            kwargs["force"] = float(force)
        return self.gripper.goto(**kwargs)

    def stop_motion(self):
        return self.gripper.stop()

    def get_state(self):
        state = self.gripper.get_state()
        # state is a dataclass-like struct; fall back to attrs if missing
        width = float(getattr(state, "width", 0.0))
        max_width = float(getattr(state, "max_width", np.nan))
        is_grasped = bool(getattr(state, "is_grasped", False))
        velocity = float(getattr(state, "speed", 0.0))
        force = float(getattr(state, "force", 0.0))
        timestamp = float(getattr(state, "timestamp", 0.0))
        return {
            "width": width,
            "max_width": max_width,
            "is_grasped": is_grasped,
            "velocity": velocity,
            "force": force,
            "timestamp": timestamp,
        }


def main():
    server = zerorpc.Server(FrankaGripperInterface())
    server.bind("tcp://0.0.0.0:4243")
    server.run()


if __name__ == "__main__":
    main()
