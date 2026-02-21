import argparse
import ast
import math
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface



@dataclass
class Gains:
    k_trans: float = 125.0  # N/m
    d_trans: float = 25.0   # N·s/m
    k_rot: float = 8.0      # Nm/rad
    d_rot: float = 0.5      # Nm·s/rad


def pose_error(des_pose: np.ndarray, cur_pose: np.ndarray) -> np.ndarray:
    """计算位置和姿态误差"""
    pd = des_pose[:3]
    rd = des_pose[3:]
    pc = cur_pose[:3]
    rc = cur_pose[3:]
    e_pos = pd - pc

    Rd = R.from_rotvec(rd).as_matrix()
    Rc = R.from_rotvec(rc).as_matrix()
    Re = Rd @ Rc.T
    e_rot = R.from_matrix(Re).as_rotvec()

    return e_pos, e_rot


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="UR5 笛卡尔阻抗控制(force_mode)")
    ap.add_argument("--ip", default="192.168.12.21", help="UR 控制器 IP 地址，例如 192.168.1.10")
    ap.add_argument("--duration", type=float, default=600.0, help="控制时长（秒），默认 20")
    ap.add_argument("--axes", type=list, default=[1, 1, 1, 1, 1, 1],
                    help="force_mode 选择向量, 1=力控/顺从, 0=位控。例如 1,1,1,0,0,0")
    ap.add_argument("--limits", type=list, default=[0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                    help="force_mode 速度限制 [m/s, rad/s]，如 0.05,0.05,0.05,0.5,0.5,0.5")
    ap.add_argument("--offset", type=list, default=[0, 0, 0, 0, 0, 0], help="相对当前 TCP 的目标偏置 (m,rad), 如 0,0,-0.03,0,0,0")
    ap.add_argument("--k-trans", type=float, default=600.0, help="平移刚度 K (N/m)")
    ap.add_argument("--d-trans", type=float, default=100.0, help="平移阻尼 D (N·s/m)")
    ap.add_argument("--k-rot", type=float, default=16.0, help="姿态刚度 K (Nm/rad)")
    ap.add_argument("--d-rot", type=float, default=1.0, help="姿态阻尼 D (Nm·s/rad)")
    ap.add_argument("--f-max", type=float, default=120.0, help="力饱和上限 |F|max (N)")
    ap.add_argument("--tau-max", type=float, default=12.0, help="力矩饱和上限 |Tau|max (Nm)")
    ap.add_argument("--type", type=int, default=2, choices=[1, 2], help="URScript force_mode typ, 通常 2 表示力控（目标扭矩/力）")
    ap.add_argument("--rate", type=float, default=125.0, help="下发频率 Hz, 默认 125")
    # 负载/重力补偿相关
    ap.add_argument("--payload-mass", type=float, default=-2.0, help="末端负载质量 (kg), 用于UR内置重力补偿")
    ap.add_argument("--payload-cog", type=list, default=[0.0, 0.0, 0.0], help="末端负载质心相对TCP的坐标(m) [x,y,z]")
    # 手动重力补偿，不通过设置payload的方式
    ap.add_argument("--use-gravity-ff", default=False, help="启用软件重力前馈补偿 (当未在UR设置负载时可开启)")
    ap.add_argument("--gravity", type=float, default=9.80665, help="重力加速度 (m/s^2)，仅用于软件前馈")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    gains = Gains(
        k_trans=args.k_trans,
        d_trans=args.d_trans,
        k_rot=args.k_rot,
        d_rot=args.d_rot,
    )

    # 连接机器人
    rtde_c = RTDEControlInterface(args.ip)
    rtde_r = RTDEReceiveInterface(args.ip)

    # 设置UR负载（推荐）：让控制器在伺服层进行重力补偿
    # 注意：质量/质心需包含工具夹具+工件的总值，质心坐标为相对TCP
    # rtde_c.setTcp(args.payload_cog)
    try:
        if args.payload_mass > 0.0:
            rtde_c.setPayload(args.payload_mass, args.payload_cog)
    except Exception as e:
        print(f"[WARN] setPayload 失败: {e}. 将仅依赖软件控制侧补偿 (如已开启 --use-gravity-ff)。")

    # 设定任务坐标系：base frame（全零向量）
    task_frame = [0.0] * 6  
    sel = args.axes
    limits = args.limits

    # 获取当前 TCP 位置，计算目标位置
    cur_pose = np.array(rtde_r.getActualTCPPose())
    desired_pose = cur_pose + np.array(args.offset)
    # 控制循环
    dt = 1.0 / args.rate
    # 饱和上限
    f_sat = np.array([args.f_max, args.f_max, args.f_max])
    tau_sat = np.array([args.tau_max, args.tau_max, args.tau_max])
    # 微分噪声抑制
    alpha = 0.2
    wrench_prev = np.zeros(6)

    next_t = t0 = time.perf_counter()
    try:
        while True:
            now = time.perf_counter()
            if now - t0 > args.duration:
                break
            # 保持特定的控制频率
            if now < next_t:
                time.sleep(next_t - now)
            next_t += dt

            # 当前 TCP 位置和姿态
            cur_pose = np.array(rtde_r.getActualTCPPose())
            tcp_speed = np.array(rtde_r.getActualTCPSpeed())
            # 位置和姿态误差
            e_pos, e_rot = pose_error(desired_pose, cur_pose)
            # 期望速度 = 0, 实际速度 = tcp_speed
            v_trans = tcp_speed[:3]
            v_rot = tcp_speed[3:]
            # 虚拟弹簧阻尼（基坐标系）
            F = gains.k_trans * e_pos - gains.d_trans * v_trans
            Tau = gains.k_rot * e_rot - gains.d_rot * v_rot

            # 可选：重力前馈（基坐标系→任务坐标系）。若已通过 setPayload 配置，通常无需再加前馈。
            if args.use_gravity_ff and args.payload_mass > 0.0:
                # 基坐标系下的重力向上补偿（与重力相反方向）
                Fg_base = np.array([0.0, 0.0, args.payload_mass * args.gravity])
                # 由质心产生的力矩（关于 TCP）。r 为 TCP->CoG 在基坐标系下的向量
                R_tcp = R.from_rotvec(cur_pose[3:]).as_matrix()
                cog_tcp = np.array(args.payload_cog)
                r_tcp_to_cog_base = R_tcp @ cog_tcp
                tau_base = np.cross(r_tcp_to_cog_base, Fg_base)
                # 将前馈力/矩转换到 task_frame（wrench 定义在任务坐标系）
                task_R = R.from_rotvec(np.array(task_frame[3:])).as_matrix()
                Fg_task = task_R.T @ Fg_base  # base->task
                tau_task = task_R.T @ tau_base
                F = F + Fg_task
                Tau = Tau + tau_task

            # 轴选择：力控/顺从=1, 位控=0
            sel_vec = np.array(sel)
            wrench_vec = np.hstack((F, Tau)) * sel_vec
            # 饱和
            wrench_vec[:3] = np.clip(wrench_vec[:3], -f_sat, f_sat)
            wrench_vec[3:] = np.clip(wrench_vec[3:], -tau_sat, tau_sat)
            # 简单滤波，避免抖动
            wrench_vec = alpha * wrench_vec + (1 - alpha) * wrench_prev
            wrench_prev = wrench_vec.copy()
            # 使用力控模式，下发控制
            rtde_c.forceMode(task_frame, sel, wrench_vec.tolist(), args.type, limits)
            # print(f"{[v for v in np.round(wrench_vec, 3)]}", end="\n")
            print(f"{[v for v in np.round(e_pos, 3)]}", end="\n")

    finally:
        rtde_c.forceModeStop()
        rtde_c.disconnect()
        rtde_r.disconnect()


if __name__ == "__main__":
    main()