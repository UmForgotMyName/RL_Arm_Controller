# 🦾 RL Arm Controller (Isaac Lab 5.1 + RSL-RL PPO)

<div align="center">

![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1-76b900?style=for-the-badge&logo=nvidia)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-5.1-0ea5e9?style=for-the-badge&logo=nvidia)
![RSL-RL](https://img.shields.io/badge/RSL--RL-PPO-f97316?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=for-the-badge&logo=python&logoColor=white)

**Obstacle-aware reaching for a FANUC LR Mate 200iC/5L + SG2 tool using Isaac Lab DirectRLEnv.**  
Train a PPO policy to reach a target TCP pose while avoiding randomized obstacles, including partially or fully infeasible layouts.

</div>

---

## 📌 Table of contents
- [Overview](#-overview)
- [Key features](#-key-features)
- [Components](#-components)
- [System architecture](#-system-architecture)
- [Requirements](#-requirements)
- [Quick start](#-quick-start)
- [Run options](#-run-options)
- [Verify your setup](#-verify-your-setup)
- [Task details](#-task-details)
- [Environment design](#-environment-design)
- [Invalid scenes and metrics](#-invalid-scenes-and-metrics)
- [Key configuration files](#-key-configuration-files)
- [Asset conversion (URDF to USD)](#-asset-conversion-urdf-to-usd)
- [Repo layout](#-repo-layout)
- [Roadmap](#-roadmap)
- [References](#-references)

---

## 🎯 Overview

**Task ID:** `Isaac-Reach-Fanuc-v0`

This repository implements a **direct-workflow RL manipulation task** (Isaac Lab `DirectRLEnv`) for a 6-DOF industrial arm:
- Reach a sampled 3D target TCP pose
- Avoid collisions with 1 to 2 rigid obstacles
- Produce smooth, bounded joint motion under randomized scene layouts

The design explicitly addresses a common manipulation RL reality:
randomized scenes can create targets that are kinematically reachable in free space but geometrically blocked once obstacles are introduced.

---

## ✨ Key features
✅ Direct workflow environment (custom reset, reward, dones, sampling logic)  
✅ PPO training using RSL-RL scripts  
✅ Parallel training friendly (headless mode supported)  
✅ Scene randomization (targets + obstacles) with retry budgets  
✅ Curriculum staging (0 obstacles to 2 obstacles) driven by success rate  
✅ Invalid-scene handling to prevent silent training corruption  
✅ URDF to USD conversion workflow for Isaac Sim articulations  

---

## 🧩 Components

| Component | Purpose |
|---|---|
| FANUC LR Mate 200iC/5L (sim) | 6-DOF industrial arm used for reaching |
| SG2 tool (sim) | End-effector geometry and TCP frame correctness |
| Isaac Sim 5.1 | Physics (PhysX), contacts, rigid bodies, USD pipeline |
| Isaac Lab 5.1 | RL environment scaffolding (DirectRLEnv) and vectorized execution |
| RSL-RL (PPO) | Training loop, logging, checkpointing, policy rollout |
| USD assets | Robot articulation asset used in the environment |

## 🧭 Typical workflow
1. Install the project as an editable Isaac Lab extension
2. Confirm the task is registered
3. Run a no-learning sanity check (zero-agent)
4. Train PPO headless for speed
5. Play back a checkpoint to validate behavior
6. Iterate on sampling, rewards, curriculum, and constraints

---

## 🧱 System architecture
```text
┌──────────────────────────────────────────┐
│            Isaac Sim (PhysX)             │
│  Robot articulation + rigid obstacles    │
└───────────────────────┬──────────────────┘
                        │ (sim step)
                        ▼
┌──────────────────────────────────────────┐
│          Isaac Lab DirectRLEnv            │
│  reset() / dones() / reward() / obs()     │
│  target + obstacle sampling + validity    │
└───────────────────────┬──────────────────┘
                        │ (vectorized env API)
                        ▼
┌──────────────────────────────────────────┐
│               RSL-RL PPO                 │
│  rollout -> optimize -> checkpoint/logs   │
└───────────────────────┬──────────────────┘
                        │ (actions)
                        ▼
                 Joint targets / torques
```

---

## 🧰 Requirements

### Software
- Isaac Sim 5.1 + Isaac Lab installed
- Python 3.11 (Isaac Sim 5.x requirement)
- Conda environment assumed: `env_isaaclab`

### Hardware (recommended)
- NVIDIA GPU for parallel environments and faster training
- Enough VRAM to scale `num_envs` safely

---

## 🚀 Quick start

From the repository root:

### 1) Activate environment
```bash
conda activate env_isaaclab
```

### 2) Install project (editable)
```bash
python -m pip install -e source/RL_Arm_Controller
```

### 3) Verify task registration
```bash
python scripts/list_envs.py
```

### 4) Sanity check (no learning)
```bash
python scripts/zero_agent.py --task Isaac-Reach-Fanuc-v0
```

### 5) Train PPO (headless recommended)
```bash
python scripts/rsl_rl/train.py --task Isaac-Reach-Fanuc-v0 --headless
```

### 6) Play a trained checkpoint
```bash
python scripts/rsl_rl/play.py --task Isaac-Reach-Fanuc-v0 --num_envs 32 --load_run <run_dir> --checkpoint <model.pt>
```

---

## 🧭 Run options

### Debug with a viewport (slower, useful early)
Remove `--headless` to watch rollouts and confirm resets, collisions, and reward shaping behave as intended.

### Train headless (faster)
```bash
python scripts/rsl_rl/train.py --task Isaac-Reach-Fanuc-v0 --headless
```

### Windows fallback if Isaac Lab is not on your Python path
```bash
C:\repos\IsaacLab\isaaclab.bat -p scripts\rsl_rl\train.py --task Isaac-Reach-Fanuc-v0 --headless
```

python scripts/rsl_rl/play.py --task Isaac-Reach-Fanuc-v0 --num_envs 32 --checkpoint logs/rsl_rl/reach_fanuc/2025-12-31_19-58-01/model_499.pt

---

## ✅ Verify your setup

### Confirm your task is discoverable
```bash
python scripts/list_envs.py
```

### Run a zero-action baseline (should be stable, no crashes)
```bash
python scripts/zero_agent.py --task Isaac-Reach-Fanuc-v0
```

### Run a random-action baseline (expect lots of failures, but should reset cleanly)
```bash
python scripts/random_agent.py --task Isaac-Reach-Fanuc-v0
```

### GPU sanity check (optional)
```bash
nvidia-smi
```

---

## 🎯 Task details

**TCP reference**
- Path: `/lrmate200ic5l_with_sg2/link_6/flange/tool0/sg2/tcp`
- Forward axis: `+Z`

High-level objectives:
- Minimize TCP-to-target distance
- Avoid collisions with obstacles
- Encourage smooth, bounded joint motion

---

## 🧠 Environment design

This task uses the Isaac Lab **direct workflow** (DirectRLEnv), meaning environment logic is implemented explicitly:
- scene setup
- reset logic
- termination logic
- reward computation
- observation computation

Conceptually, this is closest to how you would write a custom env for real manipulation RL research when you need full control over sampling and edge cases.

### Scene sampling
- Targets are sampled within bounded workspace limits
- A reachability check is applied before obstacle placement (IK ignoring obstacles)
- Obstacles are spawned as rigid bodies with contact sensing
- Rewards include distance-to-target, smoothness penalties, and collision penalties

### Curriculum learning
Difficulty increases when the rolling success rate reaches thresholds:
- Stage 1: large tolerance, easy targets, 0 obstacles
- Stage 2: wider workspace, tighter tolerance, 1 obstacle
- Stage 3: full workspace, tight tolerance, 2 obstacles

Config location:
```
source/RL_Arm_Controller/RL_Arm_Controller/tasks/direct/rl_arm_controller/rl_arm_controller_env_cfg.py
```

---

## 🚧 Invalid scenes and metrics

Randomized scenes can be invalid (no feasible placement after retries). When sampling fails:
- Target is placed near the current TCP
- Obstacles are moved outside the active workspace
- Episode is timed out with zero reward

Logged metrics (via `self.extras`) include:
- `Stats/success_rate`
- `Stats/curriculum_stage`
- `Stats/invalid_env_fraction`
- `Stats/invalid_env_count`
- `Stats/reset_count`

These are important because parallel RL can silently waste samples if invalid resets are not tracked.

---

## 🧩 Key configuration files
- `rl_arm_controller_env_cfg.py`
- `rl_arm_controller_env.py`
- `fanuc_cfg.py`
- `agents/rsl_rl_ppo_cfg.py`

---

## 🧱 Asset conversion (URDF to USD)

Convert the robot URDF to a USD articulation:

```bash
cd C:\repos\IsaacLab
isaaclab.bat -p scripts\tools\convert_urdf.py ^
  C:\repos\RL_arm_controller\RL_Arm_Controller\assets\Robots\FANUC\urdf\fanuc200ic5l_w_sg2.urdf ^
  C:\repos\RL_arm_controller\RL_Arm_Controller\assets\Robots\FANUC\usd\fanuc200ic5l_sg2.usd ^
  --fix-base ^
  --merge-joints ^
  --joint-stiffness 0.0 ^
  --joint-damping 0.0 ^
  --joint-target-type none
```

---

## 🗂 Repo layout
```text
source/RL_Arm_Controller/
  RL_Arm_Controller/
    tasks/direct/rl_arm_controller/
      rl_arm_controller_env.py
      rl_arm_controller_env_cfg.py
      fanuc_cfg.py
      agents/
        rsl_rl_ppo_cfg.py

assets/Robots/FANUC/
  urdf/
  usd/

scripts/
  list_envs.py
  zero_agent.py
  random_agent.py
  rsl_rl/
    train.py
    play.py
```

---

## 🛠 Roadmap

High priority:
- [ ] Add a small evaluation script that reports success rate and collision rate over N episodes
- [ ] Add an option to export a trained policy (TorchScript or ONNX) for deployment experiments

Medium priority:
- [ ] Add richer domain randomization (sensor noise, friction, mass)
- [ ] Add a "reachability with obstacles" filter for data efficiency (optional toggle)

Low priority:
- [ ] Integrate the learned policy into a ROS 2 control loop for hardware-style command publishing
- [ ] Benchmark against a classical planner baseline (MoveIt, OMPL) for comparable scene sets

---

## 🔗 References
- Isaac Lab docs (DirectRLEnv vs ManagerBasedRLEnv): https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html
- Isaac Lab task workflows overview: https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html
- Isaac Lab RSL-RL wrapper utilities: https://isaac-sim.github.io/IsaacLab/main/source/api/lab_rl/isaaclab_rl.html
- Isaac Lab installation notes (Python version compatibility): https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
- Isaac Sim Python environment installation: https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html
- NVIDIA Isaac Lab blog example (sim-to-real RL workflow patterns): https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/
