import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


# 用来根据时间对pos进行插值, 返回的是一个插值函数，输入t，输出该t对应的x
# 默认为线性插值
def get_interp1d(t, x):
    gripper_interp = si.interp1d(
        t, x, 
        axis=0, bounds_error=False, 
        fill_value=(x[0], x[-1]))   # 用于设置t越界时的位姿
    return gripper_interp


class PoseInterpolator:
    def __init__(self, t, x):
        pos = x[:,:3]
        rot = st.Rotation.from_rotvec(x[:,3:])
        self.pos_interp = get_interp1d(t, pos)  # t和pos是序列
        self.rot_interp = st.Slerp(t, rot)
    
    @property
    def x(self):
        return self.pos_interp.x
    
    def __call__(self, t):
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)

        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        rvec = rot.as_rotvec()
        pose = np.concatenate([pos, rvec], axis=-1)
        return pose

def get_gripper_calibration_interpolator(
        aruco_measured_width, 
        aruco_actual_width):
    """
    Assumes the minimum width in aruco_actual_width
    is measured when the gripper is fully closed
    and maximum width is when the gripper is fully opened
    """
    aruco_measured_width = np.array(aruco_measured_width)
    aruco_actual_width = np.array(aruco_actual_width)
    assert len(aruco_measured_width) == len(aruco_actual_width)
    assert len(aruco_actual_width) >= 2
    aruco_min_width = np.min(aruco_actual_width)
    gripper_actual_width = aruco_actual_width - aruco_min_width
    # aruco_measured_width：[0.04, 0.12]
    # gripper_actual_width: [0.0, 0.08]
    # 这个操作看起来是将gripper_width的下限映射为0（夹爪一定会闭合）
    interp = get_interp1d(aruco_measured_width, gripper_actual_width)
    return interp
