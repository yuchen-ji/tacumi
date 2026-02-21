# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from umi.common.precise_sleep import precise_wait
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController


# %%
@click.command()
@click.option('-rh', '--robot_hostname', default='192.168.0.26')
@click.option('-f', '--frequency', type=float, default=30)
def main(robot_hostname, frequency):
    max_pos_speed = 0.25
    max_rot_speed = 0.6
    max_gripper_width = 90.
    cube_diag = np.linalg.norm([1,1,1])
    tcp_offset = 0.13
    # tcp_offset = 0
    dt = 1/frequency
    command_latency = dt / 2

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
        FrankaInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_hostname,
            frequency=100,
            Kx_scale=5.0,
            Kxd_scale=2.0,
            verbose=False
        ) as controller:
            print('Ready!')
            # to account for recever interfance latency, use target pose
            # to init buffer.
            state = controller.get_state()
            # target_pose = state['TargetTCPPose']
            target_pose = state['ActualTCPPose']

            # print(target_pose)
            # exit()
        
            # target_pose = np.array([ 0.40328411,  0.00620825,  0.29310859, -2.26569407,  2.12426248, -0.00934497])
            # controller.servoL(target_pose, 5)
            # time.sleep(8)
            # exit()
        
            t_start = time.monotonic()
            
            iter_idx = 0
            stop = False
            while not stop:
                state = controller.get_state()
                # print(target_pose - state['ActualTCPPose'])
                s = time.time()
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # handle key presses
                press_events = key_counter.get_press_events()
                dpos = np.array([0.0, 0.0, 0.0])
                drot_xyz = np.array([0.0, 0.0, 0.0])
                gripper_cmd = 0  # For Franka gripper control
                
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                    # Position control (arrow keys for X-Y, W/S for Z)
                    elif key_stroke == Key.up:  # Move forward (X+)
                        dpos[0] = max_pos_speed / frequency
                    elif key_stroke == Key.down:  # Move backward (X-)
                        dpos[0] = -max_pos_speed / frequency
                    elif key_stroke == Key.left:  # Move left (Y+)
                        dpos[1] = max_pos_speed / frequency
                    elif key_stroke == Key.right:  # Move right (Y-)
                        dpos[1] = -max_pos_speed / frequency
                    elif key_stroke == KeyCode(char='w'):  # Move up (Z+)
                        dpos[2] = max_pos_speed / frequency
                    elif key_stroke == KeyCode(char='s'):  # Move down (Z-)
                        dpos[2] = -max_pos_speed / frequency
                    # Rotation control (R/F for Roll, T/G for Pitch, Y/H for Yaw)
                    elif key_stroke == KeyCode(char='r'):  # Roll+
                        drot_xyz[0] = max_rot_speed / frequency
                    elif key_stroke == KeyCode(char='f'):  # Roll-
                        drot_xyz[0] = -max_rot_speed / frequency
                    elif key_stroke == KeyCode(char='t'):  # Pitch+
                        drot_xyz[1] = max_rot_speed / frequency
                    elif key_stroke == KeyCode(char='g'):  # Pitch-
                        drot_xyz[1] = -max_rot_speed / frequency
                    elif key_stroke == KeyCode(char='y'):  # Yaw+
                        drot_xyz[2] = max_rot_speed / frequency
                    elif key_stroke == KeyCode(char='h'):  # Yaw-
                        drot_xyz[2] = -max_rot_speed / frequency
                    # Franka gripper control
                    elif key_stroke == KeyCode(char='c'):  # Close gripper
                        gripper_cmd = 1.0
                    elif key_stroke == KeyCode(char='o'):  # Open gripper
                        gripper_cmd = 0.0
                
                precise_wait(t_sample)

                drot = st.Rotation.from_euler('xyz', drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (drot * \
                        st.Rotation.from_rotvec(target_pose[3:])
                ).as_rotvec()

                controller.schedule_waypoint(target_pose, 
                    t_command_target-time.monotonic()+time.time())
                # TODO: Add Franka gripper control here if needed
                # gripper_cmd can be 0.0 (open) or 1.0 (close)

                precise_wait(t_cycle_end)
                iter_idx += 1
                # print(1/(time.time() -s))


    controller.terminate_current_policy()
# %%
if __name__ == '__main__':
    main()