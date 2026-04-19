import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import cv2
import zarr
import pathlib
import math
import scipy.spatial.transform as st
from diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.diffusion_policy.common.precise_sleep import precise_wait
from pynput import keyboard
from diffusion_policy.diffusion_policy.real_world.keystroke_counter import KeystrokeCounter, Key, KeyCode

# ---- 控制参数 ----
CONTROL_HZ = 15  # 控制频率 (Hz) - 提高到15Hz以获得更好的响应
DT = 1 / CONTROL_HZ

# 末端最大移动速度 (m/step)
MAX_POS_SPEED = 0.3  # 降低速度使控制更平滑
MAX_ROT_SPEED = 0.5

IMG_H, IMG_W = 360, 480  # 提高分辨率但保持性能

# ==========================================
# One Euro Filter (防抖动)
# ==========================================
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=None, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0) if isinstance(x0, (float, int)) else np.array(x0, dtype=np.float64)
        self.dx_prev = float(dx0) if dx0 is not None else np.zeros_like(self.x_prev)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev

        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    def reset(self, t0, x0):
        self.x_prev = np.array(x0, dtype=np.float64)
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev = t0

# ==========================================
# 虚拟 SpaceMouse (支持大窗口触控板模式)
# ==========================================
class VirtualSpaceMouse:
    def __init__(self, window_name, sensitivity=1.5):
        self.window_name = window_name
        self.sensitivity = sensitivity
        self.dx = 0
        self.dy = 0
        self.last_x = None
        self.last_y = None
        self.left_btn = False
        self.right_btn = False
        self.shift = False
        self.ctrl = False
        cv2.setMouseCallback(window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        self.shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        self.ctrl = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
        self.left_btn = bool(flags & cv2.EVENT_FLAG_LBUTTON)
        self.right_btn = bool(flags & cv2.EVENT_FLAG_RBUTTON)

        if self.last_x is None:
            self.last_x = x
            self.last_y = y
            return

        if event == cv2.EVENT_MOUSEMOVE:
             if self.left_btn: 
                self.dx += (x - self.last_x)
                self.dy += (y - self.last_y)
        
        self.last_x = x
        self.last_y = y

    def get_motion_state_transformed(self):
        cur_dx = self.dx
        cur_dy = self.dy
        self.dx = 0
        self.dy = 0
        
        scale = 0.01 * self.sensitivity
        vx, vy, vz, vr, vp, vyaw = 0, 0, 0, 0, 0, 0
        
        if self.shift:
            vz = -cur_dy * scale 
        elif self.ctrl:
            vyaw = cur_dx * scale * 2.0
        else:
            vx = cur_dy * scale 
            vy = cur_dx * scale 
            
        state = np.array([vx, vy, vz, vr, vp, vyaw])
        return np.clip(state, -1.0, 1.0)
    
    def draw_feedback(self, img):
        h, w = img.shape[:2]
        cv2.line(img, (w//2, h//2-20), (w//2, h//2+20), (50, 50, 50), 2)
        cv2.line(img, (w//2-20, h//2), (w//2+20, h//2), (50, 50, 50), 2)
        
        status = "Mode: Move XY"
        color = (0, 255, 0)
        if self.shift: 
            status = "Mode: Move Z (Height)"
            color = (0, 255, 255)
        elif self.ctrl: 
            status = "Mode: Rotate (Yaw)"
            color = (0, 100, 255)
            
        if self.left_btn:
            cv2.circle(img, (self.last_x, self.last_y), 10, color, -1)
            cv2.putText(img, "DRAGGING", (self.last_x + 15, self.last_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(img, "Hold L-Click to Move", (w//2 - 100, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        cv2.putText(img, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(img, "Shift: Z-Axis | Ctrl: Rotate", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def get_ik(model, data, target_pos, target_mat, initial_qpos):
    """6-DOF 阻尼最小二乘逆运动学（位置 + 姿态，保持推杆垂直）。"""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    jac = np.zeros((6, model.nv))

    # 保存完整物理状态，防止 mj_forward 污染仿真
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()

    qpos = initial_qpos.copy()
    data.qpos[:6] = qpos
    mujoco.mj_forward(model, data)

    for _ in range(IK_MAX_STEPS):
        # 位置误差
        pos_err = target_pos - data.site_xpos[site_id]

        # 姿态误差（轴角）：使推杆始终垂直
        R_cur = data.site_xmat[site_id].reshape(3, 3)
        R_err = target_mat @ R_cur.T
        cos_a = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
        angle = np.arccos(cos_a)
        if abs(angle) < 1e-7:
            ori_err = np.zeros(3)
        else:
            ori_err = (angle / (2 * np.sin(angle))) * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1],
            ])

        # 合并误差（姿态权重 0.25，优先保证位置精度）
        err = np.concatenate([pos_err, 0.25 * ori_err])
        if np.linalg.norm(err) < IK_TOLERANCE:
            break

        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
        jac6 = jac[:, :6]  # 完整 6×6 雅可比
        dq = jac6.T @ np.linalg.solve(
            jac6 @ jac6.T + IK_DAMPING * np.eye(6), err
        )
        qpos += dq
        data.qpos[:6] = qpos
        mujoco.mj_forward(model, data)

    result = qpos.copy()
    # 恢复物理状态，mj_step 从正确状态继续
    data.qpos[:] = saved_qpos
    data.qvel[:] = saved_qvel
    mujoco.mj_forward(model, data)
    return result


def collect_data():
    xml_path = os.path.join(os.path.dirname(__file__), "ur5e_push_t.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, IMG_H, IMG_W)

    mocap_id = model.body("mocap_target").mocapid[0]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    # ---- 数据存储 ----
    output_dir = './data'
    zarr_path = pathlib.Path(output_dir)
    if not zarr_path.exists():
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=str(zarr_path), mode='a')
    current_episode = {}
    is_recording = False

    # ---- 初始化环境 ----
    def reset_env():
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        # 同步 Mocap
        ep = data.site_xpos[site_id].copy()
        
        # ====== 初始高度下降0.05，保证推杆垂直于桌面 ======
        ep[2] -= 0.37
        
        em = data.site_xmat[site_id].reshape(3, 3)
        eq = np.zeros(4)
        mujoco.mju_mat2Quat(eq, em.flatten())
        data.mocap_pos[mocap_id] = ep.copy()
        data.mocap_quat[mocap_id] = eq.copy()
        # 随机放置 T-block
        data.qpos[6:9] = [np.random.uniform(0.3, 0.7),
                          np.random.uniform(-0.3, 0.3), 0.42]
        yaw0 = np.random.uniform(0, 2*np.pi)
        rot = st.Rotation.from_euler('z', yaw0)
        quat_xyzw = rot.as_quat()
        mj_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        data.qpos[9:13] = mj_quat
        mujoco.mj_forward(model, data)
        return ep.copy()

    smooth_target = reset_env()

    print("=" * 58)
    print("  Push-T 数据采集 (快速版)")
    print("=" * 58)
    print("  【鼠标控制】")
    print("    左键拖拽移动 XY")
    print("    Shift + 左键拖拽移动 Z")
    print("    Ctrl + 左键拖拽旋转")
    print("  【命令】")
    print("    R: 开始录制 | S: 保存 | 空格: 删除并重置 | Q: 退出")
    print("=" * 58)

    cv2.namedWindow("Control Pad", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Control Pad", 1200, 900)
    cv2.namedWindow("Robot View", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Wrist View", cv2.WINDOW_AUTOSIZE)

    try:
        with KeystrokeCounter() as key_counter:
            with VirtualSpaceMouse(window_name="Control Pad", sensitivity=1.5) as sm:
                # 初始化滤波器
                t_start = time.monotonic()
                pos_filter = OneEuroFilter(t0=t_start, x0=np.zeros(3), min_cutoff=0.5, beta=0.1)
                rot_filter = OneEuroFilter(t0=t_start, x0=np.zeros(3), min_cutoff=0.5, beta=0.1)

                with mujoco.viewer.launch_passive(model, data) as viewer:
                    viewer.cam.lookat = np.array([0.5, 0, 0.8])
                    viewer.cam.distance = 2.0
                    viewer.cam.azimuth = 180
                    viewer.cam.elevation = -90

                    state = {'TargetTCPPose': np.concatenate([smooth_target, np.array([np.pi, 0, 0])])}
                    target_pose = state['TargetTCPPose']

                    iter_idx = 0
                    stop = False

                    while not stop and viewer.is_running():
                        current_time = time.monotonic()
                        t_cycle_end = t_start + (iter_idx + 1) * DT

                        press_events = key_counter.get_press_events()
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='q') or key_stroke == KeyCode(char='Q'): stop = True
                            elif key_stroke == KeyCode(char='r') or key_stroke == KeyCode(char='R'): 
                                is_recording = True
                                current_episode = {}
                                print("Recording started...")
                            elif key_stroke == KeyCode(char='s') or key_stroke == KeyCode(char='S'): 
                                if current_episode:
                                    episode_data = {k: np.array(v) for k, v in current_episode.items()}
                                    replay_buffer.add_episode(episode_data)
                                    current_episode = {}
                                    print(f"Episode saved! Total: {replay_buffer.n_episodes}")
                                is_recording = False
                                
                                smooth_target = reset_env()
                                pos_filter.reset(current_time, np.zeros(3))
                                rot_filter.reset(current_time, np.zeros(3))
                                state = {'TargetTCPPose': np.concatenate([smooth_target, np.array([np.pi, 0, 0])])}
                                target_pose = state['TargetTCPPose']
                                print("Environment reset after save.")
                            elif key_stroke == Key.space: 
                                if is_recording:
                                    # 如果正在录制，则废弃当前这条录了一半的数据
                                    is_recording = False
                                    current_episode = {}
                                    print("【清空当前】正在录制的数据已作废并重置。")
                                else:
                                    # 如果没在录制，按下空格则从 Zarr 里删掉（撤销）最新保存的一整条数据
                                    if replay_buffer.n_episodes > 0:
                                        replay_buffer.drop_episode()
                                        print(f"【撤销保存】已删除上一条存入的数据！当前总条数: {replay_buffer.n_episodes}")
                                    else:
                                        print("【提示】数据空空如也，无可删除。")
                                
                                smooth_target = reset_env()
                                pos_filter.reset(current_time, np.zeros(3))
                                rot_filter.reset(current_time, np.zeros(3))
                                state = {'TargetTCPPose': np.concatenate([smooth_target, np.array([np.pi, 0, 0])])}
                                target_pose = state['TargetTCPPose']
                                print("Episode dropped and reset.")

                        # 获取鼠标输入
                        raw_state = sm.get_motion_state_transformed()
                        raw_dpos = raw_state[:3] * (MAX_POS_SPEED / CONTROL_HZ)
                        raw_drot = raw_state[3:] * (MAX_ROT_SPEED / CONTROL_HZ)

                        # 应用滤波
                        smooth_dpos = pos_filter(current_time, raw_dpos)
                        smooth_drot = rot_filter(current_time, raw_drot)

                        # 死区和最大速度限制
                        if np.linalg.norm(smooth_dpos) < 0.0005: smooth_dpos = np.zeros(3)
                        if np.linalg.norm(smooth_drot) < 0.0005: smooth_drot = np.zeros(3)
                        smooth_dpos = np.clip(smooth_dpos, -0.02, 0.02)

                        # 更新目标姿态
                        target_pose[:3] += smooth_dpos
                        drot = st.Rotation.from_euler('xyz', smooth_drot)
                        curr_rot = st.Rotation.from_rotvec(target_pose[3:])
                        target_pose[3:] = (drot * curr_rot).as_rotvec()
                        target_pose[2] = np.clip(target_pose[2], 0.4, 1.1)

                        # 边界限制
                        target_pose[0] = np.clip(target_pose[0], 0.0, 1.0)
                        target_pose[1] = np.clip(target_pose[1], -0.7, 0.7)

                        # 执行动作
                        data.mocap_pos[mocap_id] = target_pose[:3]
                        rot = st.Rotation.from_rotvec(target_pose[3:])
                        q = rot.as_quat()
                        data.mocap_quat[mocap_id] = [q[3], q[0], q[1], q[2]]

                        sim_steps = int(DT / model.opt.timestep)
                        for _ in range(sim_steps):
                            mujoco.mj_step(model, data)

                        # 记录数据
                        if is_recording:
                            obs = {}
                            renderer.update_scene(data, camera="front_camera")
                            front_img = renderer.render()
                            obs['front_image'] = np.moveaxis(front_img, -1, 0)

                            renderer.update_scene(data, camera="wrist_camera")
                            wrist_img = renderer.render()
                            obs['wrist_image'] = np.moveaxis(wrist_img, -1, 0)

                            data_dict = {
                                'action': target_pose[:2].copy(),
                                'robot_eef_pose': target_pose[:2].copy(),
                                'front_image': obs['front_image'],
                                'wrist_image': obs['wrist_image'],
                                'robot_joint': data.qpos[0:6].copy(),
                                'robot_joint_vel': data.qvel[0:6].copy(),
                                'timestamp': np.array([time.time()])
                            }
                            for key, value in data_dict.items():
                                if key not in current_episode: current_episode[key] = []
                                current_episode[key].append(value)

                        # 显示
                        renderer.update_scene(data, camera="front_camera")
                        vis_img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                        if is_recording:
                            cv2.circle(vis_img, (20, 20), 10, (0, 0, 255), -1)
                            cv2.putText(vis_img, "REC", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow("Robot View", vis_img)

                        renderer.update_scene(data, camera="wrist_camera")
                        wrist_vis_img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                        if is_recording:
                            cv2.circle(wrist_vis_img, (20, 20), 10, (0, 0, 255), -1)
                            cv2.putText(wrist_vis_img, "REC", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow("Wrist View", wrist_vis_img)

                        control_img = np.zeros((900, 1200, 3), dtype=np.uint8)
                        sm.draw_feedback(control_img)
                        cv2.imshow("Control Pad", control_img)
                        cv2.pollKey()

                        viewer.sync()
                        precise_wait(t_cycle_end)
                        iter_idx += 1

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            renderer.close()
        except:
            pass
        cv2.destroyAllWindows()
        print("程序已退出")


if __name__ == "__main__":
    collect_data()
