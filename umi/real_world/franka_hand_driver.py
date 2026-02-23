"""
Lightweight RPC client for controlling Franka Hand (parallel gripper)
via a zerorpc server (see scripts_real/launch_franka_gripper_server.py).
"""

from typing import Optional

import zerorpc


class FrankaHandDriver:
    def __init__(self, hostname="192.168.0.5", port=4242, heartbeat=20):
        self.hostname = hostname
        self.port = port
        self.heartbeat = heartbeat
        self.client = None

    # ========= lifecycle ===========
    def start(self):
        # 对于Franka Hand，我们使用zerorpc连接
        # 和franka_interpolation_controller.py中的连接方式一样
        self.client = zerorpc.Client(heartbeat=self.heartbeat)
        self.client.connect(f"tcp://{self.hostname}:{self.port}")

    def stop(self):
        if self.client is None:
            return
        self.client.close()
        self.client = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= high level API ===========
    # TODO: 这里确认下要有哪些API？参考fairo中的Franka Hand接口设计
    # 1. homing()
    # 2. goto()
    # 3. get_state()

    def homing(self):
        # fairo的franka没有专门的homing接口，手动调用goto来实现
        self.goto(width=1, speed=0.2, force=0.1)

    def goto(self, width: float, speed: float, force: Optional[float] = None):
        return self.client.goto(width, speed, force)

    def get_state(self):
        return self.client.get_state()

    def grasp(
        self,
        speed: float,
        force: float,
        grasp_width: float = 0.0,
        epsilon_inner: float = -1.0,
        epsilon_outer: float = -1.0,
    ):
        self.client.grasp(speed, force, grasp_width, epsilon_inner, epsilon_outer)
