"""
Controller process for the Franka Hand gripper.
Mirrors the interface of WSGController: schedule waypoints in a queue,
publish measured state via shared memory ring buffer.
"""
import time
import enum
import multiprocessing as mp
from typing import Optional
from multiprocessing.managers import SharedMemoryManager

from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from umi.real_world.franka_hand_driver import FrankaHandDriver


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2


class FrankaHandController(mp.Process):
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        hostname: str,
        port: int = 4243,
        frequency: float = 30.0,
        home_on_start: bool = True,
        move_max_speed: float = 0.1,
        move_force: Optional[float] = None,
        get_max_k: Optional[int] = None,
        command_queue_size: int = 1024,
        launch_timeout: float = 3.0,
        receive_latency: float = 0.0,
        use_meters: bool = True,
        verbose: bool = False,
    ):
        super().__init__(name="FrankaHandController")
        self.hostname = hostname
        self.port = port
        self.frequency = frequency
        self.home_on_start = home_on_start
        self.move_max_speed = move_max_speed
        self.move_force = move_force
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        # Franka hand expects meters; set scale so that input/output units follow use_meters flag
        self.scale = 1.0 if use_meters else 1e-3
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        # build input queue
        example = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pos": 0.0,
            "target_time": 0.0,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size,
        )

        # build ring buffer
        example = {
            "gripper_state": 0,
            "gripper_position": 0.0,
            "gripper_velocity": 0.0,
            "gripper_force": 0.0,
            "gripper_measure_timestamp": time.time(),
            "gripper_receive_timestamp": time.time(),
            "gripper_timestamp": time.time(),
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait: bool = True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaHandController] Controller process spawned at {self.pid}")

    def stop(self, wait: bool = True):
        message = {"cmd": Command.SHUTDOWN.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pos": pos,
            "target_time": target_time,
        }
        self.input_queue.put(message)

    def restart_put(self, start_time: float):
        self.input_queue.put({
            "cmd": Command.RESTART_PUT.value,
            "target_time": start_time,
        })

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        try:
            with FrankaHandDriver(hostname=self.hostname, port=self.port) as gripper:
                if self.verbose:
                    print(f"[FrankaHandController] Connect to gripper: {self.hostname}:{self.port}")

                if self.home_on_start:
                    gripper.homing()

                info = gripper.get_state()
                curr_pos = float(info.get("width", 0.0)) / self.scale
                curr_t = time.monotonic()
                last_waypoint_time = curr_t
                pose_interp = PoseTrajectoryInterpolator(
                    times=[curr_t],
                    poses=[[curr_pos, 0, 0, 0, 0, 0]],
                )

                keep_running = True
                t_start = time.monotonic()
                iter_idx = 0
                while keep_running:
                    t_now = time.monotonic()
                    dt = 1.0 / self.frequency
                    t_target = t_now
                    target_pos = pose_interp(t_target)[0]
                    target_vel = (target_pos - pose_interp(t_target - dt)[0]) / dt

                    speed = min(self.move_max_speed, abs(target_vel))
                    gripper.goto(
                        width=target_pos * self.scale,
                        speed=speed,
                        force=self.move_force,
                    )

                    info = gripper.get_state() or {}
                    state = {
                        "gripper_state": int(bool(info.get("is_grasped", False))),
                        "gripper_position": float(info.get("width", 0.0)) / self.scale,
                        "gripper_velocity": float(info.get("velocity", 0.0)) / self.scale,
                        "gripper_force": float(info.get("force", 0.0)),
                        "gripper_measure_timestamp": info.get("timestamp", time.time()),
                        "gripper_receive_timestamp": time.time(),
                        "gripper_timestamp": time.time() - self.receive_latency,
                    }
                    self.ring_buffer.put(state)

                    try:
                        commands = self.input_queue.get_all()
                        n_cmd = len(commands["cmd"])
                    except Empty:
                        n_cmd = 0

                    for i in range(n_cmd):
                        command = {key: value[i] for key, value in commands.items()}
                        cmd = command["cmd"]

                        if cmd == Command.SHUTDOWN.value:
                            keep_running = False
                            break
                        if cmd == Command.SCHEDULE_WAYPOINT.value:
                            target_pos = command["target_pos"] * self.scale
                            target_time = command["target_time"]
                            target_time = time.monotonic() - time.time() + target_time
                            curr_time = t_now
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=[target_pos, 0, 0, 0, 0, 0],
                                time=target_time,
                                max_pos_speed=self.move_max_speed,
                                max_rot_speed=self.move_max_speed,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time,
                            )
                            last_waypoint_time = target_time
                        elif cmd == Command.RESTART_PUT.value:
                            t_start = command["target_time"] - time.time() + time.monotonic()
                            iter_idx = 1
                        else:
                            keep_running = False
                            break

                    if iter_idx == 0:
                        self.ready_event.set()
                    iter_idx += 1

                    t_end = t_start + (1.0 / self.frequency) * iter_idx
                    precise_wait(t_end=t_end, time_func=time.monotonic)
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[FrankaHandController] Disconnected from gripper: {self.hostname}")
