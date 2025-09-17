# Standalone Sim2Sim

A standalone robot simulation environment based on Genesis and Warp, supporting G1 humanoid robot simulation and control.

## Project Overview

This project implements a complete sim2sim simulation pipeline, including:
- G1 humanoid robot simulation
- LiDAR sensor simulation
- Real-time obstacle detection and avoidance
- Keyboard control interface
- Policy network inference

## Environment Setup

### System Requirements

- Ubuntu 20.04
- Python 3.9
- NVIDIA GPU (CUDA 11.0+ support)

### Dependencies Installation

#### 1. Basic Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install basic tools
sudo apt install -y git wget curl build-essential
```

#### 2. Python Environment

We recommend using conda to manage Python environment:

```bash
# Create new conda environment
conda create -n sim2sim python=3.9
conda activate sim2sim
```

#### 3. Core Dependencies

Based on the requirements.txt, install the following packages with specific versions:

```bash
# Install PyTorch (CUDA 12.4 compatible)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Install Genesis simulation engine
pip install genesis-world==0.2.0

# Install Warp (NVIDIA's high-performance physics simulation library)
pip install warp-lang==1.6.0

# Install other core dependencies
pip install numpy==1.26.4
pip install matplotlib==3.9.4
pip install trimesh==4.6.4
pip install keyboard==0.13.5

# Install additional visualization and processing libraries
pip install open3d==0.19.0
pip install opencv-python==4.10.0.84
pip install pillow==10.4.0
pip install scikit-image==0.24.0
pip install pyvista==0.44.2
```

#### 4. Optional Dependencies

For extended functionality:

```bash
# Machine learning and data processing
pip install scikit-learn==1.6.1
pip install pandas==2.2.3
pip install scipy==1.13.1

# Visualization and GUI
pip install plotly==5.24.1
pip install dash==2.18.2

# 3D processing
pip install pymeshlab==2023.12.post2
pip install tetgen==0.6.4
```

#### 5. Installation Verification

```python
# Test Genesis
python -c "import genesis as gs; print('Genesis installed successfully')"

# Test Warp
python -c "import warp as wp; print('Warp installed successfully')"

# Test PyTorch GPU support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Test other core libraries
python -c "import trimesh, matplotlib, keyboard; print('Core libraries installed successfully')"
```

### Project Structure

```
standalone_sim2sim/
├── assets/                    # Asset files
│   ├── g1_new/               # G1 robot model
│   │   ├── g1_29dof_fakehand.urdf
│   │   └── meshes/           # Mesh files
│   ├── humanoid/             # Humanoid obstacle models
│   ├── plane.urdf            # Ground plane model
│   └── policy.jit            # Pre-trained policy network
├── envs/                     # Environment definitions
│   ├── base_task.py          # Base task class
│   ├── g1_with_hand_robot.py # G1 robot environment (main implementation)
│   └── g1_with_inspire_config.py # Configuration file
├── sensors/                  # Sensor modules
│   ├── base_sensor.py
│   ├── imu_sensor.py
│   └── warp/                 # Warp sensor implementations
│       └── warp_kernels/
├── utils/                    # Utility functions
├── sim2sim.py               # Main execution script (simplified)
├── requirements.txt         # Python dependencies
└── README.md
```

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd standalone_sim2sim
```

### 2. Install Dependencies

```bash
# Activate environment
conda activate sim2sim

# Install from requirements file
pip install -r requirements.txt
```

### 3. Run Simulation

```bash
python sim2sim.py
```

### 4. Control Instructions

- **Arrow Keys**: Control robot movement
  - ↑: Forward (speed: 0.5 m/s)
  - ↓: Backward (speed: -0.4 m/s)  
  - ←: Left (speed: 0.4 m/s)
  - →: Right (speed: -0.4 m/s)

- **Number Keys**: Control rotation
  - 1: Turn left (angular velocity: 0.5 rad/s)
  - 2: Turn right (angular velocity: -0.5 rad/s)
  - 3: Display LiDAR polar plot

## Configuration Options

### Sensor Configuration

Configure in `envs/g1_with_inspire_config.py`:

- LiDAR parameters (FOV, resolution, range)
- Update frequency
- Visualization options

### Simulation Parameters

- Physics engine parameters
- Robot DOF settings
- Obstacle count and distribution
- Terrain settings
