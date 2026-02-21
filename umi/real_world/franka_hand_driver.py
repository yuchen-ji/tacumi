"""
Lightweight RPC client for controlling Franka Hand (parallel gripper)
via a zerorpc server (see scripts_real/launch_franka_gripper_server.py).
"""
from typing import Optional

import zerorpc


class FrankaHandDriver:
    def __init__(self, hostname="127.0.0.1", port=4243, heartbeat=20):
        self.hostname = hostname
        self.port = port
        self.heartbeat = heartbeat
        self.client = None

    # ========= lifecycle ===========
    def start(self):
        self.client = zerorpc.Client(heartbeat=self.heartbeat)
        self.client.connect(f"tcp://{self.hostname}:{self.port}")

    def stop(self):
        if self.client is None:
            return
        try:
            # best-effort stop of any ongoing motion
            self.client.stop_motion()
        finally:
            self.client.close()
            self.client = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= high level API ===========
    def homing(self):
        return self.client.homing()

    def goto(self, width: float, speed: float, force: Optional[float] = None):
        return self.client.goto(width, speed, force)

    def stop_motion(self):
        return self.client.stop_motion()

    def get_state(self):
        return self.client.get_state()
