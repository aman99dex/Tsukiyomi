# PBR-NeRF (C++ Implementation)

A high-performance C++ implementation of **PBR-NeRF** (Physics-Based Rendering Neural Radiance Fields), built on top of [LibTorch](https://pytorch.org/cppdocs/).

## 📚 Inspiration & Background

This project is inspired by the CVPR paper **"PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields"** (and similar works like NeRFactor and PhySG).
*   **Original Paper**: [PBR-NeRF (Zhang et al.)](https://pbr-nerf.github.io/) / [Official Repo](https://github.com/s3anwu/pbrnerf)
*   **Base Codebase**: [cNeRF](https://github.com/rafaelanderka/cNeRF) (A minimal C++ NeRF implementation).

### The Problem: Baked-in Lighting
Standard NeRFs (Mildenhall et al., 2020) learn a "Radiance Field" that maps `(Position, Direction) -> (Color, Density)`.
This "bakes" the lighting into the color. If you train on a scene with a shadow, that shadow is painted onto the object. You cannot move the light or change the material.

### The Solution: Inverse Rendering
PBR-NeRF solves this by **decomposing** the scene into intrinsic properties, effectively "un-baking" the lighting.
Instead of predicting just `Color`, our neural network predicts:
1.  **Geometry (Density)**: The shape of the object.
2.  **Surface Normal**: The orientation of the surface at each point.
3.  **Albedo**: The base color of the material (independent of lighting).
4.  **Roughness**: How shiny or matte the surface is.

We then use a **Physics-Based Shader** (Cook-Torrance BRDF) and a **Neural Incident Light Field (NeILF)** to physically simulate light transport.

### Comparison with Original PBR-NeRF
While this project replicates the core results of the original paper, there are some implementation differences:
*   **Lighting Model**: The original paper builds upon **NeILF++** (often separating Sun/Sky components). We implement a unified **NeILF** (single MLP) for simplicity and performance, which still captures high-frequency environment lighting.
*   **Loss Functions**: We implement the **Energy Conservation Loss** as described. For the specular term, we use a **Roughness-Weighted Regularization** (`Specular * Roughness`) to disentangle materials, whereas the original paper uses a specific **NDF-weighted** formulation.
*   **Architecture**: This is a **pure C++** implementation focused on speed and portability, whereas the original is a Python/PyTorch research codebase.

---

## ⚙️ Technical Implementation

### 1. Neural Network Architecture (`src/model.cpp`)
We use two separate networks:
1.  **NeRF Model**: Predicts geometry and material properties.
    *   **Albedo**: `Sigmoid` (0-1). Diffuse color.
    *   **Roughness**: `Sigmoid` (0-1). Specular spread.
    *   **Metallic**: `Sigmoid` (0-1). Metalness.
    *   **Normal**: `Tanh` (-1 to 1). Surface orientation.
    *   **Density**: `Softplus` (0 to inf). Opacity.
2.  **NeILF Model**: A **Neural Incident Light Field** that predicts incoming radiance for any `(Position, Direction)`. This replaces the simple point light, allowing for complex, spatially-varying environmental lighting.

### 2. Physics-Based Rendering (`src/renderer.cpp`)
We use **Monte Carlo Integration** to compute the final pixel color.
For every sample point along a ray:
1.  **Sample Lights**: We sample `N` random light directions on the sphere.
2.  **Query NeILF**: We ask the NeILF model "how much light is coming from this direction?".
3.  **Compute BRDF**: We evaluate the **Cook-Torrance BRDF** (GGX Distribution, Smith Geometry, Schlick Fresnel) for each light direction.
4.  **Integrate**: `Color = Sum(Li * BRDF * dot(N, L))`
5.  **Volume Rendering**: The shaded colors are composited using standard NeRF alpha blending.

### 3. Optimization & Regularization
To achieve high-quality decomposition, we implement advanced loss functions:
*   **MSE Loss**: Standard reconstruction loss.
*   **Normal Consistency Loss**: Penalizes normals that point away from the camera.
*   **Roughness Entropy Loss**: Encourages the network to commit to smooth or rough surfaces (binary entropy).
*   **Energy Conservation Loss**: Penalizes the BRDF if it reflects more energy than it receives (`Integral(BRDF) <= 1`).
*   **Specular Regularization**: Penalizes high specular highlights on rough surfaces to encourage disentanglement.

### 4. Relighting & Material Demo
Because we have separated Material from Lighting, we can:
*   **Relight**: Rotate the environment map or change the NeILF.
*   **Material Override**: Render the learned geometry with forced materials (e.g., "Gold", "Plastic") to prove the geometry is disentangled from the appearance.
The project includes demos for both.

---

## 🚀 Features

*   **Pure C++**: No Python runtime required for training or rendering.
*   **High Performance**:
    *   **Mac MPS Support**: Fully accelerated on Apple Silicon (M1/M2/M3/M4) GPUs using Metal Performance Shaders.
    *   **Multithreading**: Uses OpenMP for parallel CPU operations.
    *   **Ray Batching**: Implements stochastic ray sampling and batched rendering for memory efficiency.
*   **Interactive Visualization**:
    *   **Real-time Preview**: View training progress and rendered scene in real-time.
    *   **GUI Controls**: Adjust camera, training parameters, and visualization settings on the fly using ImGui.
*   **True PBR Pipeline**:
    *   **NeILF**: Neural Incident Light Field for realistic lighting.
    *   **Monte Carlo**: Physically accurate rendering integration.
    *   **Disentanglement**: Geometry, Albedo, Roughness, Normal, Metallic.
*   **Cross-Platform**: Compatible with **macOS (Apple Silicon/Intel)**, **Linux**, and **Windows**.
*   **LibTorch Backend**: Uses PyTorch's C++ frontend for automatic differentiation and tensor operations.

---

## 📦 Installation

### Prerequisites

1.  **CMake** (>= 3.20)
2.  **C++ Compiler** (Clang, GCC, or MSVC) supporting C++17/20.
3.  **LibTorch** (PyTorch C++ Library)
4.  **OpenCV** (For image I/O)
5.  **OpenMP** (For CPU multithreading)
6.  **GLFW** (For windowing and input)

### Step-by-Step Setup

#### 1. Download LibTorch
Download the **Pre-cxx11 ABI** version (unless you know what you are doing) from [pytorch.org](https://pytorch.org/get-started/locally/).
*   **Mac**: Download the `libtorch-macos-*.zip`.
*   **Windows**: Download the `libtorch-win-*.zip` (Release version).
*   **Linux**: Download the `libtorch-cxx11-abi-*.zip`.

Extract the zip file into the project root directory so you have a folder named `libtorch`.
Structure should look like:
```
cNeRF/
  ├── libtorch/
  ├── src/
  ├── include/
  ├── CMakeLists.txt
  ...
```

#### 2. Install Dependencies

**macOS (Homebrew)**
```bash
brew install cmake opencv libomp glfw
```

**Ubuntu/Debian**
```bash
sudo apt-get install cmake libopencv-dev libomp-dev libglfw3-dev
```

**Windows**
*   Install [CMake](https://cmake.org/download/).
*   Install OpenCV (e.g., using `vcpkg` or pre-built binaries). Set `OpenCV_DIR` environment variable if needed.

#### 3. Build the Project

```bash
mkdir build
cd build
cmake ..
make  # On Windows, open the generated .sln file or use 'cmake --build .'
```

---

## 🏃 Usage

### 1. Prepare Data
The application expects data in a pre-processed `.pt` (PyTorch Tensor) format.
If you have a standard NeRF dataset (Blender format, `transforms.json`), use the provided converter script.

**Using the Converter Script (Requires Python):**
```bash
pip install torch numpy opencv-python
python scripts/convert_data.py /path/to/nerf_synthetic/lego ./data/lego --half_res
```
This will generate `images.pt`, `poses.pt`, and `focal.pt` in `./data/lego`.

### 2. Run Training & Rendering
Run the executable with the data directory and output directory.

```bash
# Syntax
./build/cNeRF <path_to_data_folder> <path_to_output_folder>

# Example
./build/cNeRF ./data/lego ./output
```

### 3. Interactive Controls
The application launches a window with a real-time preview and control panel.

**Control Panel:**
*   **Training**: Pause/Resume training.
*   **Target Iterations**: Set the number of iterations to train for (default: 50,000).
*   **Save Checkpoint**: Manually save the current model state.

**Camera Controls:**
*   **Azimuth / Elevation**: Rotate the camera around the object.
*   **Radius**: Zoom in/out.
*   **Near / Far Plane**: Adjust the clipping planes to see inside or crop the scene.
*   **Flip Axes**: Toggle this if the rendering looks upside down or mirrored (common with Blender vs. OpenCV coordinates).

**Viewport Interaction:**
*   **Rotate**: Click and drag with the **Left Mouse Button**.
*   **Zoom**: Use the **Mouse Wheel**.

### 4. Output
The program will produce:
*   **Training Logs**: Loss values in the console (updates every iteration).
*   **Preview Frames**: `frame_*.png` (saved every 50 iterations).
*   **Dataset Comparisons**: `dataset_view_*.png` (saved at the end).
    *   Side-by-side comparison of Ground Truth vs. Rendered View.
*   **Material Overrides**: `material_gold.png`, `material_plastic.png` (saved at the end).
    *   Demonstrates geometry disentanglement by rendering the object with forced materials.
*   **3D Point Cloud**: `model.ply` (saved at the end).
    *   Exported geometry that can be viewed in MeshLab or Blender.

---

## 🛠 Project Structure

*   `src/main.cpp`: Entry point. Handles data loading, training loop, and demo generation.
*   `src/model.cpp`: Defines the Neural Networks (`NeRFModel` and `NeILFModel`).
*   `src/renderer.cpp`: Implements the Volumetric Rendering, Monte Carlo Integration, and PBR Shading.
*   `scripts/convert_data.py`: Helper to convert datasets to `.pt` format.

---

## 📄 License
MIT License