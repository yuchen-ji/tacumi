import zerorpc
from polymetis import RobotInterface
from polymetis import GripperInterface
import scipy.spatial.transform as st
import numpy as np
import torch

class FrankaInterface:
    def __init__(self):
        self.robot = RobotInterface('localhost')
        self.gripper = GripperInterface('localhost')

    ####################################################
    # Robot Interface
    ####################################################

    def get_ee_pose(self):
        data = self.robot.get_ee_pose()
        pos = data[0].numpy()
        quat_xyzw = data[1].numpy()
        rot_vec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
        return np.concatenate([pos, rot_vec]).tolist()
    
    def get_joint_positions(self):
        return self.robot.get_joint_positions().numpy().tolist()
    
    def get_joint_velocities(self):
        return self.robot.get_joint_velocities().numpy().tolist()
    
    def move_to_joint_positions(self, positions, time_to_go):
        self.robot.move_to_joint_positions(
            positions=torch.Tensor(positions),
            time_to_go=time_to_go
        )
    
    def start_cartesian_impedance(self, Kx, Kxd):
        self.robot.start_cartesian_impedance(
            Kx=torch.Tensor(Kx),
            Kxd=torch.Tensor(Kxd)
        )

    def update_desired_ee_pose(self, pose):
        pose = np.asarray(pose)
        self.robot.update_desired_ee_pose(
            position=torch.Tensor(pose[:3]),
            orientation=torch.Tensor(st.Rotation.from_rotvec(pose[3:]).as_quat())
        )

    def terminate_current_policy(self):
        self.robot.terminate_current_policy()


    ####################################################
    # Gripper Interface
    ####################################################

    def get_state(self):
        state_ = self.gripper.get_state()
        state = dict()
        state['timestamp'] = state_.timestamp.seconds + state_.timestamp.nanoseconds * 1e-9
        state['width'] = state_.width
        state['is_moving'] = state_.is_moving
        state['is_grasped'] = state_.is_grasped
        state['prev_command_successful'] = state_.prev_command_successful
        return state

    # TODO: 这里看一下GripperInface里的blocking参数的作用
    # 能不能实现实时的夹爪控制
    def goto(self, width: float, speed: float, force: float):
        self.gripper.goto(
            width=width,
            speed=speed,
            force=force,
            blocking=True
        )

    # TODO: 这里需要指定epsilon_inner和epsilon_outer
    # 当 grasp_width - epsilon_inner < final_width < grasp_width + epsilon_outer 时，gripper才会停止施加力
    # ==> 如果没有达到这个范围，线程会阻塞么？
    def grasp(self, speed, force, grasp_width, epsilon_inner, epsilon_outer):
        self.gripper.grasp(
            speed=speed,
            force=force,
            grasp_width=grasp_width,
            epsilon_inner=epsilon_inner,
            epsilon_outer=epsilon_outer,
            blocking=True
        )


s = zerorpc.Server(FrankaInterface())
s.bind("tcp://0.0.0.0:4242")
s.run()