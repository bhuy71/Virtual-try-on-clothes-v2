# Installing DensePose / Detectron2 (Optional)

DensePose is **optional** for this project. The project works with **MediaPipe** for pose estimation, which is much easier to install.

However, if you need DensePose for higher quality pose estimation, follow these steps:

## Prerequisites

DensePose requires PyTorch and build tools to be installed **before** installing detectron2.

## Installation Steps

### Step 1: Install PyTorch and Build Tools First

```bash
# Install build tools
pip install wheel ninja setuptools

# For macOS (CPU only)
pip install torch torchvision

# For Linux with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For Linux with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Install Detectron2

```bash
# For Linux with CUDA
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

# For macOS (use clang compiler + no-build-isolation)
CC=clang CXX=clang++ pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

### Step 3: Install DensePose

The pip subdirectory method has bugs, so install from a local clone:

```bash
# Clone detectron2 repo
git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2_temp

# Install DensePose using setup.py
cd /tmp/detectron2_temp/projects/DensePose
python setup.py install

# Install video support (optional)
pip install av
```

## Troubleshooting

### Error: "No module named 'torch'"
**Solution**: Install PyTorch and build tools first before detectron2:
```bash
pip install wheel ninja setuptools
pip install torch torchvision
CC=clang CXX=clang++ pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

### Error: "invalid command 'bdist_wheel'"
**Solution**: Install wheel package:
```bash
pip install wheel
```

### Error: Build fails on macOS
**Solution**: Use clang compiler with --no-build-isolation:
```bash
CC=clang CXX=clang++ pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
```

### Alternative: Use Pre-built Wheels (Linux only)

```bash
# For CUDA 11.8 and PyTorch 2.0
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

## Using MediaPipe Instead (Recommended for Easy Setup)

If you have trouble installing DensePose, the project works perfectly with **MediaPipe**:

```bash
pip install mediapipe
```

MediaPipe is:
- ✅ Easy to install (pure pip)
- ✅ Works on macOS, Linux, Windows
- ✅ Fast inference
- ✅ Good enough for most virtual try-on tasks

The preprocessing code automatically uses MediaPipe if DensePose is not available.
