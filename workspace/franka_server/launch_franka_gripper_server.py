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

    # TODO: 确认下这里的state是否正确
    def get_state(self):
        state_ = self.gripper.get_state()
        # state is a dataclass-like struct; fall back to attrs if missing
        state = dict()
        state['width'] = state_.width
        state['is_moving'] = state_.is_moving
        state['is_grasped'] = state_.is_grasped
        state['timestamp'] = state_.timestamp.seconds + state_.timestamp.nanos * 1e-9
        state['prev_command_successful'] = state_.prev_command_successful
        return state


def main():
    server = zerorpc.Server(FrankaGripperInterface())
    server.bind("tcp://0.0.0.0:4242")
    server.run()


if __name__ == "__main__":
    main()
