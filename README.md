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

We then use a **Physics-Based Shader** (Blinn-Phong) during rendering to combine these properties with a light source to produce the final image.

---

## ⚙️ Technical Implementation

### 1. Neural Network Architecture (`src/model.cpp`)
We modified the standard MLP (Multi-Layer Perceptron) to output **8 channels**:
*   **Channels 0-2 (Albedo)**: `Sigmoid` activation (0-1 range). Represents diffuse color.
*   **Channel 3 (Roughness)**: `Sigmoid` activation (0-1 range). Controls specular highlight size.
*   **Channels 4-6 (Normal)**: `Tanh` activation (-1 to 1 range). Normalized to unit vector. Represents surface orientation.
*   **Channel 7 (Density)**: `ReLU` activation (0 to infinity). Represents opacity.

### 2. Physics-Based Rendering (`src/renderer.cpp`)
Standard NeRF uses Volume Rendering to accumulate color. We inject the **Blinn-Phong Reflection Model** into this process.
For every sample point along a ray:
1.  **Light Direction (`L`)**: Vector from point to light source.
2.  **View Direction (`V`)**: Vector from point to camera.
3.  **Half Vector (`H`)**: Bisector of `L` and `V`.
4.  **Diffuse Term**: `Albedo * max(0, dot(Normal, L))`
5.  **Specular Term**: `pow(max(0, dot(Normal, H)), 1/Roughness)`
6.  **Final Color**: `Diffuse + Specular`

This color is then composited using the standard alpha-blending formula:
`Pixel_Color = Sum(Transmittance_i * Alpha_i * Shaded_Color_i)`

### 3. Relighting Demo
Because we have separated Material from Lighting, we can change the `Light Position` at runtime without retraining the network.
The project includes a demo mode that orbits a point light around the object, showcasing dynamic specular highlights that move correctly across the surface.

---

## 🚀 Features

*   **Pure C++**: No Python runtime required for training or rendering.
*   **PBR Pipeline**: Disentangles geometry and materials for relighting.
*   **Cross-Platform**: Compatible with **macOS (Apple Silicon/Intel)**, **Linux**, and **Windows**.
*   **LibTorch Backend**: Uses PyTorch's C++ frontend for automatic differentiation and tensor operations.

---

## 📦 Installation

### Prerequisites

1.  **CMake** (>= 3.20)
2.  **C++ Compiler** (Clang, GCC, or MSVC) supporting C++17/20.
3.  **LibTorch** (PyTorch C++ Library)
4.  **OpenCV** (For image I/O)

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
brew install cmake opencv libomp
```

**Ubuntu/Debian**
```bash
sudo apt-get install cmake libopencv-dev
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

### 3. Output
The program will produce:
*   **Training Logs**: Loss values in the console.
*   **Preview Frames**: `frame_*.png` (saved periodically).
*   **Relighting Demo**: `light_orbit_*.png` (saved at the end).
    *   These images show the object with a **moving light source**, demonstrating the PBR capabilities.

---

## 🛠 Project Structure

*   `src/main.cpp`: Entry point. Handles data loading, training loop, and demo generation.
*   `src/model.cpp`: Defines the Neural Network.
    *   **Input**: (x, y, z) coordinate + viewing direction.
    *   **Output**: 8 channels (Albedo RGB, Roughness, Normal XYZ, Density).
*   `src/renderer.cpp`: Implements the Volumetric Rendering and PBR Shading.
    *   Uses **Blinn-Phong** model for specular highlights.
*   `scripts/convert_data.py`: Helper to convert datasets to `.pt` format.

---

## 🖥 Windows Support
This project is fully compatible with Windows.
1.  Ensure you download the **Windows** version of LibTorch.
2.  Use `cmake -G "Visual Studio 16 2019"` (or your version) to generate the solution.
3.  Open the solution in Visual Studio and build the **Release** configuration.
4.  Ensure `opencv_world*.dll` and `torch_cpu.dll` (and others) are in your PATH or next to the executable.

---

## 📄 License
MIT License