import os
import sys

# 将内层 diffusion_policy 目录加入系统路径，使得所有的 'diffusion_policy.xxx' 导入能找到正确包
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusion_policy'))

import time
import collections
import cv2
import dill
import hydra
import torch
import threading
import numpy as np
import scipy.spatial.transform as st
import mujoco
import mujoco.viewer
from diffusion_policy.common.pytorch_util import dict_apply

# 禁用 Hydra 默认处理命令行参数，避免冲突
sys.argv = sys.argv[:1]

class AsyncPolicy:
    """定义一个异步策略线程，独立于仿真渲染，完美解决掉帧卡顿。"""
    def __init__(self, policy):
        self.policy = policy
        self.obs_queue = collections.deque(maxlen=1)
        self.result_queue = collections.deque(maxlen=1)
        self.stop_event = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        
    def _run(self):
        while not self.stop_event.is_set():
            if len(self.obs_queue) > 0:
                obs_dict, step_id = self.obs_queue.pop()
                with torch.no_grad():
                    result = self.policy.predict_action(obs_dict)
                    action_pred = result['action_pred'][0].detach().cpu().numpy()
                self.result_queue.append((step_id, action_pred))
            else:
                time.sleep(0.01)

    def put(self, obs_dict, step_id):
        self.obs_queue.append((obs_dict, step_id))

    def get(self):
        if len(self.result_queue) > 0:
            return self.result_queue[-1]
        return None, None

def process_img(img_hwc):
    """
    将 MuJoCo 渲染的(H,W,C)图像调整为(C,96,96)的(0-1)浮点张量格式，与训练集预处理对齐。
    """
    img_resized = cv2.resize(img_hwc, (96, 96), interpolation=cv2.INTER_AREA)
    img_chw = np.moveaxis(img_resized, -1, 0).astype(np.float32) / 255.0
    return img_chw

def reset_env(model, data, site_id, mocap_id):
    # 每次重置前打乱随机数种子，否则由于环境重置会永远固定在同一点
    np.random.seed(int(time.time() * 1000) % (2**32 - 1))
    
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    
    ep = data.site_xpos[site_id].copy()
    # ====== 初始高度下降0.37，保证推杆垂直于桌面，保持和采集一致 ======
    ep[2] -= 0.37
    
    em = data.site_xmat[site_id].reshape(3, 3)
    eq = np.zeros(4)
    mujoco.mju_mat2Quat(eq, em.flatten())
    data.mocap_pos[mocap_id] = ep.copy()
    data.mocap_quat[mocap_id] = eq.copy()
    
    # 随机放置 T-block，扩大活动范围
    data.qpos[6:9] = [np.random.uniform(0.35, 0.65), np.random.uniform(-0.25, 0.05), 0.42]
    # 随机yaw
    yaw0 = np.random.uniform(0, 2*np.pi)
    rot = st.Rotation.from_euler('z', yaw0)
    quat_xyzw = rot.as_quat()
    mj_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    data.qpos[9:13] = mj_quat
    
    mujoco.mj_forward(model, data)
    return ep

def main():
    checkpoint_path = "/home/uraanus/文档/push_T/epoch=0250-train_loss=0.002.ckpt"
    if not os.path.exists(checkpoint_path):
        print(f"找不到权重文件：{checkpoint_path}")
        return
        
    print(f"正在加载模型权重: {checkpoint_path}")
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)
    
    # 将原本缓慢的 DDPM 替换为支持跳步的 DDIM
    # Diffusion Policy 默认在训练时使用 100 步的 DDPM。如果在评估时也跑100步，
    # GPU 很容易耗流 1秒左右，导致的巨大异步延迟会让机械臂“预测旧位置”从而剧烈抽搐！
    # 我们用 DDIM 截断到 10 步，可以直接实现近乎实时的 10Hz+ 推理体验！
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    if hasattr(policy, 'noise_scheduler'):
        # 继承模型的原版噪声规模配置，但改用加速采样器
        policy.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon"
        )
        policy.num_inference_steps = 10  # 10次迭代足矣，极大消除计算延迟的相位差！

    policy.reset() # 初始化内部状态 (如需要)
    print("模型加载完毕并已移动至:", device)

    # 提取超参数与训练时一致
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    
    CONTROL_HZ = 15  
    DT = 1.0 / CONTROL_HZ

    # 初始化 MuJoCo 仿真环境
    xml_path = os.path.join(os.path.dirname(__file__), "ur5e_push_t.xml")
    if not os.path.exists(xml_path):
        # 尝试备用路径
        xml_path = os.path.join(os.path.dirname(__file__), "mujoco_ur5e", "pusht.xml")
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 360, 480)
    
    mocap_id = model.body("mocap_target").mocapid[0]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    
    print("启动环境与可视化...")
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.distance = 2.0
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -90

    # 获取初始状态并重置
    initial_tcp_pos = reset_env(model, data, site_id, mocap_id)
    target_pose = np.concatenate([initial_tcp_pos, np.array([np.pi, 0, 0])])
    
    # 定义历史观测缓存
    obs_deque = collections.deque(maxlen=n_obs_steps)
    
    def get_obs():
        # 截取画面
        renderer.update_scene(data, camera="front_camera")
        front_img = renderer.render()
        f_chw = process_img(front_img)
        
        renderer.update_scene(data, camera="wrist_camera")
        wrist_img = renderer.render()
        w_chw = process_img(wrist_img)
        
        # EEF position XY
        eef_pose = target_pose[:2].copy()
        
        return {
            'front_image': f_chw,
            'wrist_image': w_chw,
            'robot_eef_pose': eef_pose.astype(np.float32)
        }

    # 填充初始观测队列
    for _ in range(n_obs_steps):
        obs_deque.append(get_obs())
        
    step_count = 0
    max_steps = 6000
    
    # 辅助打包函数的定义
    def get_obs_dict_tensor():
        obs_dict = {
            'front_image': np.stack([o['front_image'] for o in obs_deque]),
            'wrist_image': np.stack([o['wrist_image'] for o in obs_deque]),
            'robot_eef_pose': np.stack([o['robot_eef_pose'] for o in obs_deque])
        }
        return dict_apply(obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

    # 初始化并拉起异步推理线程
    async_policy = AsyncPolicy(policy)
    async_policy.put(get_obs_dict_tensor(), 0)
    
    print("======== 初始化：等待首次视觉推断完成 ========")
    # 阻塞主线程以确保拿到第一帧的预测动作
    while async_policy.get()[0] is None:
        time.sleep(0.01)

    start_step = n_obs_steps - 1
    current_chunk_id = -1
    
    # 【完美对齐】：建立动作缓冲字典，实现学术界标准的 Temporal Ensembling (时序融合)
    action_history = collections.defaultdict(list)
    last_executed_action = None

    print("======== 开始自主推理测试（多线程异步推流） ========")
    print("在弹出的图像窗口中按 'q' 或 'ESC' 退出测试")
    try:
        while viewer.is_running() and step_count < max_steps:
            loop_start = time.time()
            
            # 1. 检查推理线程是否吐出了最新的“新决策块”
            latest_chunk_id, latest_action_pred = async_policy.get()
            
            # 使用 ACT 论文标准 Temporal Ensembling（滑动平均滤波融合）
            # 拿到新的推测动作序列后，不是把旧的全盘抛弃，而是将 [0, n] 步的预测全部存入一个字典的绝对时间轴里。
            # 执行时用多条预测轨迹在此时刻的期望值做滑动平均（Mean），自动完美过滤掉高频生生成白噪声，彻底消除抽搐！
            if latest_chunk_id is not None and latest_chunk_id != current_chunk_id:
                if not (np.isnan(latest_action_pred).any() or np.isinf(latest_action_pred).any()):
                    current_chunk_id = latest_chunk_id
                    # 把新拿到的一条长动作序列全部塞进未来时间历表中！
                    # start_step（即队列最后那张最新的图像）对应的就是真实物理界的 latest_chunk_id 时刻
                    for i in range(start_step, len(latest_action_pred)):
                        global_step = latest_chunk_id + (i - start_step)
                        action_history[global_step].append(latest_action_pred[i])
            
            # 2. 从轨迹组中获取当前步骤的目标（融合取平均）
            if len(action_history[step_count]) > 0:
                # 对当前时刻所有的“曾预测记录”取均值，这个 mean 就是学术级的高级消抖滤波！
                action = np.mean(action_history[step_count], axis=0)
                last_executed_action = action
            else:
                # 若模型算得实在太慢产生了数据空窗缓冲，就顺滑保持最新状态！
                action = last_executed_action if last_executed_action is not None else np.zeros(2)
            
            # 及时清理字典，防止随着执行步数增长爆内存
            for old_step in list(action_history.keys()):
                if old_step < step_count:
                    del action_history[old_step]
            

            # 3. 将推理出的位置直传给控制器
            # 【深度对齐 1】：数据采集时，保存的动作已经是经过了 OneEuro Filter 的平滑目标点，DP 学习的也是这条丝滑曲线。
            # 如果再次通过 EMA 指数平滑处理，会人为造成响应延迟（相位滞后），机械臂会一直慢半拍且动作严重走样。
            desired_x = np.clip(action[0], 0.1, 0.9)
            desired_y = np.clip(action[1], -0.7, 0.7)
            
            # 【深度对齐 2】：与 collect_data.py 保持完全一致的物理安全保护限速（防止单帧意外发散）
            MAX_POS_SPEED = 0.3  # m/s
            max_delta = MAX_POS_SPEED * DT 
            
            delta_pos = np.array([desired_x, desired_y]) - target_pose[:2]
            dist = np.linalg.norm(delta_pos)
            if dist > max_delta:
                delta_pos = delta_pos * (max_delta / dist)
                desired_x = target_pose[0] + delta_pos[0]
                desired_y = target_pose[1] + delta_pos[1]

            target_pose[0] = desired_x
            target_pose[1] = desired_y

            # 【深度对齐 3】：去除微步插值。
            # 在 collect_data.py 里，程序是直接抛给 mocap_pos 后直接运行 33 次 mujoco_step，
            # 由 MuJoCo 内部原生的 PD 控制器牵引。手动施加微步插值会导致机器人的加速度特征与数据集完全不一致！
            data.mocap_pos[mocap_id] = target_pose[:3]
            rot = st.Rotation.from_rotvec(target_pose[3:])
            q = rot.as_quat()
            data.mocap_quat[mocap_id] = [q[3], q[0], q[1], q[2]]

            sim_steps = int(DT / model.opt.timestep)
            for _ in range(sim_steps):
                mujoco.mj_step(model, data)
                
            viewer.sync()
            
            # 5. 更新最新画面，并以最新环境姿态发起【下一轮推断命令】投喂给后台线程
            obs_deque.append(get_obs())
            async_policy.put(get_obs_dict_tensor(), step_count)
            
            # 6. 渲染视觉监控
            renderer.update_scene(data, camera="front_camera")
            vis_img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.putText(vis_img, f"Step: {step_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Diffusion Policy Inference", vis_img)
            key = cv2.waitKey(1)
            
            step_count += 1
            if key in [ord('q'), 27]:
                print("测试被手动中止。")
                break
                
            # 7. **控制循环的时钟，保证机器人动作在地球时间是流畅连续的**
            elapsed = time.time() - loop_start
            if elapsed < DT:
                time.sleep(DT - elapsed)

    except KeyboardInterrupt:
        print("\n中止程序。")
        
    print("======== 测试结束 ========")
    cv2.destroyAllWindows()
    viewer.close()

if __name__ == "__main__":
    main()
