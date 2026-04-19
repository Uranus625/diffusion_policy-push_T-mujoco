# MuJoCo UR5e Push-T: Visuomotor Diffusion Policy

本项目是在 MuJoCo 仿真环境下，基于 UR5e 机械臂复现的 Push-T（推 T 型块）任务。本项目包含从**数据采集**、**模型训练配置**到**异步部署评估**的完整 Pipeline，使用基于视觉的扩散策略 (Visuomotor Diffusion Policy) 控制机械臂完成精细的物理接触任务。

## 🎥 效果演示

[点击此处查看训练部署效果演示视频](./9980e4d8b7e3eb03f24a20724cd7b8d1.mp4)

## 📂 核心代码结构

开源的核心部分主要包含以下三个组件：

1. **`collect_data.py` (数据采集)**
   - 实现了基于 MuJoCo 的实时交互环境。
   - 使用异步多线程将用户的控制信号（键鼠/空间鼠标等）映射到机械臂末端 Mocap 上。
   - 数据被无缝流式写入 `.zarr` 数据集中，避免了图像保存时的硬盘 I/O 阻塞。

2. **`Dataset` (Zarr 格式流式数据集)**
   - 包含的字段集：`wrist_image` (腕部相机图像), `front_image` (全局相机图像), `robot_eef_pose` (末端位姿), `robot_joint` (关节角), `action` (专家动作)。
   - 具有极其友好的按块(Chunk)压缩存储特性，极大提升 Diffusion 训练时的 DataLoader 读取效率。

3. **`diffusion_policy/config/train_my_pusht_image.yaml` (超详细训练配置)**
   - 经过全中文详细注释的 Hydra 配置文件。
   - 包含了对 `shape_meta` (输入输出观测空间), `horizon` (预测步长), AdamW 优化器参数, EMA (指数移动平均) 以及 DDPM 噪声调度等核心参数的深度解析。

4. **`eval_my_pusht.py` (评估与推理引擎)**
   - 搭载了训练完毕的 Diffusion Policy 进行实时推理。
   - **解决了视觉策略在物理引擎中由于计算延迟导致的严重物理崩溃问题（如卡顿、瞬移、把物体撞飞）**。

---

## 🚀 核心技术亮点：评估与推理引擎 (`eval_my_pusht.py`)

在从仿真物理空间迁移到神经网络，再部署回物理空间的过程中，本项目重点攻克了**推理延迟**与**动作执行抖动**的难题，使用了如下核心技术：

### 1. 与训练环境严格对齐 (Training-Inference Alignment)
神经网络是严格依据历史设定运行的，任何细微的数据错位都会导致模型崩溃。我们确保了：
- **输入频率与队列对齐**：使用 Ring Buffer (环形缓冲区) 精确维持 `n_obs_steps=2`（仅提取最近两帧的历史视野作为观测输入）。
- **空间归一化对齐**：使用训练时计算好的 `min-max` 统计量在预处理阶段对图像 `[0, 1]` 计算以及本体感受数据进行相同的放缩，并在模型输出后执行精准的逆归一化 (unnormalize)。

### 2. 时序集成与动作块 (Temporal Ensemble & Action Chunking)
- **Receding Horizon (滚动时域控制)**：Diffusion 策略一次性预测未来 16 步（`horizon=16`）的完整运动轨迹。但我们仅提取并执行其前 8 步（`n_action_steps=8`）。
- **Temporal Ensemble**：通过使用时间维度上的动作平滑，模型避免了动作的极大跳跃。在每个控制周期，当前步的指令不仅受当前预测影响，还能与前序预测的后续延展进行时间上的平滑融合。这种重叠执行机制大幅增加了原本开环预测的抗扰动能力和控制柔性。

### 3. 多线程异步推理 (Asynchronous Inference)
扩散模型的去噪过程（Denoising steps）极其消耗 GPU 计算时间（通常 > 30-50ms）。如果将其与物理引擎放置于同一主线程：
- 会导致 MuJoCo 物理帧率暴跌。
- 时间戳发生严重脱节（Time-rewind 现象）。
**解决方案**：我们实现了一个专门的 `AsyncPolicy` 类将神经网络推理放入独立的后台线程 (Background Thread) 中。主线程仅需非阻塞地从事件队列 `action_queue` 里消费计算好的动作块，使得 MuJoCo 的画面渲染和物理步进依旧保持如丝般顺滑的流畅。

---

## 🛠️ 如何运行 (Quick Start)

### 1. 准备 Diffusion Policy 源码与环境
本项目依赖于官方的 Diffusion Policy 代码库作为基础模型框架。请先克隆并配置环境：

```bash
# 1. 克隆官方 Diffusion Policy 仓库（如果本项目未直接包含核心代码）
git clone https://github.com/real-stanford/diffusion_policy.git
cd diffusion_policy

# 2. 创建 Conda 环境并安装依赖
conda env create -f conda_environment.yaml
conda activate robomimic_venv
```
*(注：请将本仓库中的自定义脚本如 `eval_my_pusht.py`, `collect_data.py` 以及配置文件放入对应目录中。)*

### 2. 数据采集
```bash
python collect_data.py
```
*启动后，在弹出的渲染窗口内交互获取数据，结束时将自动封装进入 Zarr 数据集。*

### 启动训练
*(修改 `diffusion_policy/config/train_my_pusht_image.yaml` 确保数据集路径正确)*
```bash
python train.py --config-name=train_my_pusht_image
```

### 部署与测试
```bash
python eval_my_pusht.py --checkpoint /path/to/your/checkpoint.ckpt
```

## 📝 License
MIT License
