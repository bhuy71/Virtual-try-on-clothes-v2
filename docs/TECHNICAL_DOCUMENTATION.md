# Virtual Try-On: Complete Technical Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Data Pipeline](#data-pipeline)
3. [Method 1: CP-VTON (Traditional)](#method-1-cp-vton-traditional)
4. [Method 2: HR-VITON (State-of-the-Art)](#method-2-hr-viton-state-of-the-art)
5. [Shared Network Components](#shared-network-components)
6. [Loss Functions](#loss-functions)
7. [Training Procedures](#training-procedures)
8. [Inference Pipeline](#inference-pipeline)
9. [Evaluation Metrics](#evaluation-metrics)

---

## Introduction

### Problem Definition

**Virtual Try-On** is the task of synthesizing an image of a person wearing a target clothing item, given:
- **Input 1**: Image of a person (reference person)
- **Input 2**: Image of a clothing item (target garment)
- **Output**: Realistic image of the person wearing the target garment

### Key Challenges

1. **Geometric Transformation**: The flat clothing must be warped to match the person's body shape and pose
2. **Texture Preservation**: Fine patterns, logos, and textures must remain intact after warping
3. **Occlusion Handling**: Body parts (arms, hands) may occlude parts of the clothing
4. **Boundary Blending**: Seamless integration between warped clothing and person's body
5. **High Resolution**: Maintaining quality at high resolutions (1024×768+)

### Project Overview

This project implements two methodologies:

| Method | Paper | Year | Key Innovation |
|--------|-------|------|----------------|
| **CP-VTON** | ECCV 2018 | 2018 | Characteristic-preserving TPS warping |
| **HR-VITON** | ECCV 2022 | 2022 | Misalignment-aware high-resolution synthesis |

---

## Data Pipeline

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Raw Images                                                      │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              PREPROCESSING STAGE                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │   │
│  │  │    Pose     │  │   Human     │  │   Cloth     │       │   │
│  │  │ Estimation  │  │  Parsing    │  │   Mask      │       │   │
│  │  │  (OpenPose) │  │  (LIP/ATR)  │  │ Generation  │       │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘       │   │
│  │         │                │                │               │   │
│  │         ▼                ▼                ▼               │   │
│  │    18 Keypoints    20-class Map    Binary Mask           │   │
│  └──────────────────────────────────────────────────────────┘   │
│      │                                                           │
│      ▼                                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              DATASET CONSTRUCTION                         │   │
│  │                                                           │   │
│  │  person_image ─────┐                                      │   │
│  │  cloth_image ──────┤                                      │   │
│  │  cloth_mask ───────┼──▶ DataLoader ──▶ Training Batch    │   │
│  │  pose_heatmap ─────┤                                      │   │
│  │  parse_map ────────┤                                      │   │
│  │  agnostic_image ───┘                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Pose Estimation

**Purpose**: Extract body joint locations to understand person's pose.

**Method**: OpenPose or similar pose estimation network

**Output**: 18 body keypoints

```
Keypoint Index Mapping:
 0: Nose           6: Left Elbow      12: Right Hip
 1: Neck           7: Left Wrist      13: Right Knee
 2: Right Shoulder 8: Right Hip       14: Right Ankle
 3: Right Elbow    9: Right Knee      15: Left Eye
 4: Right Wrist   10: Right Ankle     16: Right Eye
 5: Left Shoulder 11: Left Hip        17: Left Ear
                                      18: Right Ear
```

**Representation**:
```python
# Keypoints: [18, 2] array of (x, y) coordinates
# Heatmaps: [18, H, W] tensor with Gaussian blobs at each keypoint

def generate_pose_heatmap(keypoints, height, width, sigma=6):
    """Generate Gaussian heatmaps from keypoints."""
    heatmaps = np.zeros((18, height, width))
    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:  # Valid keypoint
            # Create Gaussian centered at (x, y)
            xx, yy = np.meshgrid(np.arange(width), np.arange(height))
            heatmaps[i] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    return heatmaps
```

### 2. Human Parsing (Semantic Segmentation)

**Purpose**: Segment the person into semantic body parts.

**Method**: LIP (Look Into Person) or ATR parser

**Output**: 20-class segmentation map

```
Class Labels:
 0: Background      7: Left Arm       14: Left Leg
 1: Hat             8: Right Arm      15: Right Leg
 2: Hair            9: Left Hand      16: Left Shoe
 3: Sunglasses     10: Right Hand     17: Right Shoe
 4: Upper Clothes  11: Skirt          18: Bag
 5: Dress          12: Pants          19: Scarf
 6: Coat           13: Belt
```

**Usage in Virtual Try-On**:
```python
# Create agnostic representation by masking clothing regions
CLOTHING_LABELS = [4, 5, 6]  # Upper clothes, dress, coat

def create_agnostic(person_image, parse_map):
    """Remove original clothing from person image."""
    agnostic = person_image.copy()
    
    # Create mask for clothing regions
    clothing_mask = np.zeros_like(parse_map)
    for label in CLOTHING_LABELS:
        clothing_mask[parse_map == label] = 1
    
    # Fill clothing region with gray
    agnostic[clothing_mask > 0] = 128
    
    return agnostic
```

### 3. Cloth Mask Generation

**Purpose**: Isolate the clothing item from its background.

**Methods**:

```python
def generate_cloth_mask(cloth_image, method='threshold'):
    """Generate binary mask for clothing item."""
    
    if method == 'threshold':
        # Simple thresholding (assumes white background)
        gray = cv2.cvtColor(cloth_image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
    elif method == 'grabcut':
        # GrabCut segmentation (more robust)
        mask = np.zeros(cloth_image.shape[:2], np.uint8)
        rect = (10, 10, cloth_image.shape[1]-20, cloth_image.shape[0]-20)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(cloth_image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
    # Clean up with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask * 255
```

### 4. Dataset Classes

**File**: `data/dataset.py`

```python
class VITONDataset(Dataset):
    """Dataset for standard resolution (256×192)."""
    
    def __getitem__(self, index):
        # Load all components
        sample = {
            'image': self.load_image(person_path),      # [3, 256, 192]
            'cloth': self.load_image(cloth_path),       # [3, 256, 192]
            'cloth_mask': self.load_mask(mask_path),    # [1, 256, 192]
            'agnostic': self.load_image(agnostic_path), # [3, 256, 192]
            'parse': self.load_parse(parse_path),       # [20, 256, 192]
            'pose': self.load_pose(pose_path),          # [18, 256, 192]
        }
        return sample

class VITONHDDataset(Dataset):
    """Dataset for high resolution (1024×768)."""
    # Same structure, different resolution
```

### Supported Datasets

| Dataset | Resolution | Pairs | Categories |
|---------|------------|-------|------------|
| VITON | 256×192 | 14,221 | Upper body |
| VITON-HD | 1024×768 | 13,679 | Upper body |
| DressCode | 1024×768 | 48,392 | Upper/Lower/Dress |

---

## Method 1: CP-VTON (Traditional)

### Paper Reference
> Wang et al., "Toward Characteristic-Preserving Image-based Virtual Try-On Network", ECCV 2018

### Architecture Overview

CP-VTON uses a **two-stage approach**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CP-VTON ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STAGE 1: GEOMETRIC MATCHING MODULE (GMM)                               │
│  ════════════════════════════════════════                               │
│                                                                          │
│  ┌──────────────┐        ┌──────────────┐                               │
│  │   Person     │        │    Cloth     │                               │
│  │ Representation│       │    Image     │                               │
│  └──────┬───────┘        └──────┬───────┘                               │
│         │                       │                                        │
│         ▼                       ▼                                        │
│  ┌──────────────┐        ┌──────────────┐                               │
│  │  Feature     │        │   Feature    │                               │
│  │  Extractor A │        │  Extractor B │                               │
│  └──────┬───────┘        └──────┬───────┘                               │
│         │                       │                                        │
│         └───────────┬───────────┘                                        │
│                     ▼                                                    │
│              ┌─────────────┐                                            │
│              │ Correlation │                                            │
│              │   Layer     │                                            │
│              └──────┬──────┘                                            │
│                     ▼                                                    │
│              ┌─────────────┐                                            │
│              │ Regression  │                                            │
│              │  Network    │                                            │
│              └──────┬──────┘                                            │
│                     ▼                                                    │
│              ┌─────────────┐      ┌─────────────┐                       │
│              │    TPS      │ ───▶ │   Warped    │                       │
│              │ Transform   │      │   Cloth     │                       │
│              └─────────────┘      └─────────────┘                       │
│                                                                          │
│  STAGE 2: TRY-ON MODULE (TOM)                                           │
│  ═════════════════════════════                                          │
│                                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                     │
│  │  Agnostic    │ │   Warped     │ │    Pose      │                     │
│  │   Person     │ │   Cloth      │ │   Heatmap    │                     │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘                     │
│         │                │                │                              │
│         └────────────────┼────────────────┘                              │
│                          ▼                                               │
│                   ┌─────────────┐                                       │
│                   │   U-Net     │                                       │
│                   │  Generator  │                                       │
│                   └──────┬──────┘                                       │
│                          │                                               │
│              ┌───────────┴───────────┐                                  │
│              ▼                       ▼                                   │
│       ┌─────────────┐         ┌─────────────┐                           │
│       │  Rendered   │         │ Composition │                           │
│       │   Image     │         │    Mask     │                           │
│       └──────┬──────┘         └──────┬──────┘                           │
│              │                       │                                   │
│              └───────────┬───────────┘                                  │
│                          ▼                                               │
│                   ┌─────────────┐                                       │
│                   │   Final     │                                       │
│                   │   Output    │                                       │
│                   │ = M·Warp +  │                                       │
│                   │ (1-M)·Render│                                       │
│                   └─────────────┘                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Geometric Matching Module (GMM)

**File**: `models/cp_vton/gmm.py`

#### Purpose
Learn a spatial transformation to warp the flat clothing image to match the person's body shape and pose.

#### Architecture Details

```python
class GeometricMatchingModule(nn.Module):
    """
    GMM Architecture:
    
    Input A (Person): Agnostic + Pose + Parse → [B, 22, 256, 192]
    Input B (Cloth):  Cloth + Mask → [B, 4, 256, 192]
    
    Feature Extractor (shared architecture):
    ├── Conv2d(in, 64, 4, stride=2, padding=1) + ReLU
    ├── Conv2d(64, 128, 4, stride=2, padding=1) + BatchNorm + ReLU
    ├── Conv2d(128, 256, 4, stride=2, padding=1) + BatchNorm + ReLU
    └── Conv2d(256, 512, 4, stride=2, padding=1) + BatchNorm + ReLU
    
    Output: [B, 512, 16, 12]
    """
```

#### Correlation Layer

The correlation layer computes the similarity between person and cloth features at each spatial location:

```python
def correlation(feature_A, feature_B):
    """
    Compute correlation between two feature maps.
    
    Args:
        feature_A: [B, C, H, W] - Person features
        feature_B: [B, C, H, W] - Cloth features
    
    Returns:
        correlation: [B, H*W, H, W] - Correlation volume
    """
    B, C, H, W = feature_A.shape
    
    # L2 normalize features
    feature_A = F.normalize(feature_A, dim=1)
    feature_B = F.normalize(feature_B, dim=1)
    
    # Reshape for matrix multiplication
    feature_A = feature_A.view(B, C, -1)  # [B, C, H*W]
    feature_B = feature_B.view(B, C, -1)  # [B, C, H*W]
    
    # Compute correlation
    correlation = torch.bmm(feature_A.transpose(1, 2), feature_B)  # [B, H*W, H*W]
    correlation = correlation.view(B, H*W, H, W)
    
    return correlation
```

#### TPS (Thin-Plate Spline) Transformation

TPS is a smooth interpolation technique that warps an image based on control point displacements:

```python
class TPSTransformation(nn.Module):
    """
    Thin-Plate Spline transformation.
    
    Uses a 5×5 grid of control points (25 total).
    Each control point has an (x, y) displacement.
    
    The transformation ensures:
    1. Control points move to their target positions
    2. The transformation is smooth everywhere
    3. Bending energy is minimized
    """
    
    def __init__(self, grid_size=5):
        super().__init__()
        self.grid_size = grid_size
        self.num_points = grid_size * grid_size  # 25
        
        # Create uniform grid of control points
        self.register_buffer('control_points', 
                            self._create_grid(grid_size))
    
    def forward(self, theta, image):
        """
        Apply TPS transformation.
        
        Args:
            theta: [B, 2, 5, 5] - Control point displacements
            image: [B, C, H, W] - Image to warp
        
        Returns:
            warped: [B, C, H, W] - Warped image
        """
        # Get target control points
        target_points = self.control_points + theta.view(-1, 2, 25).permute(0, 2, 1)
        
        # Compute TPS coefficients
        coefficients = self._solve_tps(self.control_points, target_points)
        
        # Generate sampling grid
        grid = self._generate_grid(coefficients, image.shape)
        
        # Sample from image
        warped = F.grid_sample(image, grid, mode='bilinear', 
                               padding_mode='border', align_corners=True)
        
        return warped
```

**TPS Mathematical Formulation**:

For a point $(x, y)$, the TPS transformation is:
$$f(x, y) = a_0 + a_x \cdot x + a_y \cdot y + \sum_{i=1}^{n} w_i \cdot U(||(x, y) - p_i||)$$

Where:
- $U(r) = r^2 \log(r)$ is the TPS radial basis function
- $p_i$ are the control points
- $w_i$ are the weights (solved from control point correspondences)
- $a_0, a_x, a_y$ are affine transformation parameters

### Stage 2: Try-On Module (TOM)

**File**: `models/cp_vton/tom.py`

#### Purpose
Synthesize the final try-on image by blending the warped clothing with the person image.

#### Architecture Details

```python
class TryOnModule(nn.Module):
    """
    U-Net based generator with skip connections.
    
    Input channels: 22
    ├── Agnostic person: 3
    ├── Warped cloth: 3
    ├── Warped cloth mask: 1
    ├── Pose heatmap: 18
    └── Parse map (one-hot): ...
    
    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                     ENCODER                          │
    ├─────────────────────────────────────────────────────┤
    │ Layer 1: Conv(22, 64, 4, s=2) + LeakyReLU           │
    │          Output: [B, 64, 128, 96]                   │
    │                                                      │
    │ Layer 2: Conv(64, 128, 4, s=2) + BN + LeakyReLU     │
    │          Output: [B, 128, 64, 48]                   │
    │                                                      │
    │ Layer 3: Conv(128, 256, 4, s=2) + BN + LeakyReLU    │
    │          Output: [B, 256, 32, 24]                   │
    │                                                      │
    │ Layer 4: Conv(256, 512, 4, s=2) + BN + LeakyReLU    │
    │          Output: [B, 512, 16, 12]                   │
    │                                                      │
    │ Layer 5: Conv(512, 512, 4, s=2) + BN + LeakyReLU    │
    │          Output: [B, 512, 8, 6]                     │
    └─────────────────────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │                   BOTTLENECK                         │
    ├─────────────────────────────────────────────────────┤
    │ ResBlock(512) × 2                                    │
    │ Output: [B, 512, 8, 6]                              │
    └─────────────────────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │                     DECODER                          │
    ├─────────────────────────────────────────────────────┤
    │ Layer 1: Upsample + Conv(512+512, 512) + BN + ReLU  │
    │          Skip connection from Encoder Layer 4       │
    │          Output: [B, 512, 16, 12]                   │
    │                                                      │
    │ Layer 2: Upsample + Conv(512+256, 256) + BN + ReLU  │
    │          Skip connection from Encoder Layer 3       │
    │          Output: [B, 256, 32, 24]                   │
    │                                                      │
    │ Layer 3: Upsample + Conv(256+128, 128) + BN + ReLU  │
    │          Skip connection from Encoder Layer 2       │
    │          Output: [B, 128, 64, 48]                   │
    │                                                      │
    │ Layer 4: Upsample + Conv(128+64, 64) + BN + ReLU    │
    │          Skip connection from Encoder Layer 1       │
    │          Output: [B, 64, 128, 96]                   │
    │                                                      │
    │ Layer 5: Upsample + Conv(64, 64) + BN + ReLU        │
    │          Output: [B, 64, 256, 192]                  │
    └─────────────────────────────────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────────┐
    │                   OUTPUT HEADS                       │
    ├─────────────────────────────────────────────────────┤
    │ Render Head: Conv(64, 3) + Tanh                     │
    │              Output: [B, 3, 256, 192] (rendered)    │
    │                                                      │
    │ Mask Head: Conv(64, 1) + Sigmoid                    │
    │            Output: [B, 1, 256, 192] (mask)          │
    └─────────────────────────────────────────────────────┘
    """
```

#### Composition

The final output is composed using the predicted mask:

```python
def compose(self, warped_cloth, rendered_image, mask):
    """
    Compose final try-on image.
    
    Args:
        warped_cloth: [B, 3, H, W] - Warped clothing from GMM
        rendered_image: [B, 3, H, W] - U-Net rendered image
        mask: [B, 1, H, W] - Composition mask (0-1)
    
    Returns:
        output: [B, 3, H, W] - Final try-on image
    
    Formula:
        output = mask * warped_cloth + (1 - mask) * rendered_image
    
    Intuition:
        - Where mask ≈ 1: Use warped cloth (clothing region)
        - Where mask ≈ 0: Use rendered image (body, background)
    """
    output = mask * warped_cloth + (1 - mask) * rendered_image
    return output
```

---

## Method 2: HR-VITON (State-of-the-Art)

### Paper Reference
> Lee et al., "High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions", ECCV 2022

### Architecture Overview

HR-VITON uses an **end-to-end approach** with three main components:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          HR-VITON ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    CONDITION GENERATOR                              │ │
│  ├────────────────────────────────────────────────────────────────────┤ │
│  │                                                                     │ │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐                              │ │
│  │   │ Person  │ │  Cloth  │ │  Pose   │                              │ │
│  │   │  Parse  │ │  Image  │ │ Heatmap │                              │ │
│  │   └────┬────┘ └────┬────┘ └────┬────┘                              │ │
│  │        │           │           │                                    │ │
│  │        └───────────┼───────────┘                                    │ │
│  │                    ▼                                                │ │
│  │            ┌──────────────┐                                         │ │
│  │            │    Shared    │                                         │ │
│  │            │   Encoder    │                                         │ │
│  │            └──────┬───────┘                                         │ │
│  │                   │                                                 │ │
│  │        ┌──────────┴──────────┐                                      │ │
│  │        ▼                     ▼                                      │ │
│  │  ┌───────────┐        ┌───────────┐                                │ │
│  │  │  Flow     │        │   Seg     │                                │ │
│  │  │ Decoder   │        │ Decoder   │                                │ │
│  │  └─────┬─────┘        └─────┬─────┘                                │ │
│  │        │                    │                                       │ │
│  │        ▼                    ▼                                       │ │
│  │  ┌───────────┐        ┌───────────┐                                │ │
│  │  │ Appearance│        │ Predicted │                                │ │
│  │  │   Flow    │        │   Parse   │                                │ │
│  │  └─────┬─────┘        └───────────┘                                │ │
│  │        │                                                            │ │
│  │        ▼                                                            │ │
│  │  ┌───────────┐                                                      │ │
│  │  │  Warped   │                                                      │ │
│  │  │   Cloth   │                                                      │ │
│  │  └───────────┘                                                      │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      SPADE GENERATOR                                │ │
│  ├────────────────────────────────────────────────────────────────────┤ │
│  │                                                                     │ │
│  │   Input: Warped cloth + Agnostic + Predicted parse                 │ │
│  │                                                                     │ │
│  │   ┌────────────────────────────────────────────────────────────┐   │ │
│  │   │                    SPADE ResBlocks                          │   │ │
│  │   │                                                             │   │ │
│  │   │   ┌─────────────────────────────────────────────────────┐  │   │ │
│  │   │   │  SPADE Block:                                        │  │   │ │
│  │   │   │                                                      │  │   │ │
│  │   │   │  x ──▶ InstanceNorm ──▶ × γ(seg) + β(seg) ──▶ out   │  │   │ │
│  │   │   │                           ▲          ▲               │  │   │ │
│  │   │   │                           │          │               │  │   │ │
│  │   │   │  segmentation ──▶ Conv ───┘          │               │  │   │ │
│  │   │   │  segmentation ──▶ Conv ──────────────┘               │  │   │ │
│  │   │   │                                                      │  │   │ │
│  │   │   └─────────────────────────────────────────────────────┘  │   │ │
│  │   │                                                             │   │ │
│  │   │   × 4 blocks with upsampling                               │   │ │
│  │   │                                                             │   │ │
│  │   └────────────────────────────────────────────────────────────┘   │ │
│  │                                                                     │ │
│  │   Output: [B, 3, 1024, 768]                                        │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                  MULTI-SCALE DISCRIMINATOR                          │ │
│  ├────────────────────────────────────────────────────────────────────┤ │
│  │                                                                     │ │
│  │   ┌───────────┐   ┌───────────┐   ┌───────────┐                    │ │
│  │   │  Scale 1  │   │  Scale 2  │   │  Scale 3  │                    │ │
│  │   │ 1024×768  │   │  512×384  │   │  256×192  │                    │ │
│  │   │           │   │  (↓2×)    │   │  (↓4×)    │                    │ │
│  │   └─────┬─────┘   └─────┬─────┘   └─────┬─────┘                    │ │
│  │         │               │               │                           │ │
│  │         ▼               ▼               ▼                           │ │
│  │   ┌───────────┐   ┌───────────┐   ┌───────────┐                    │ │
│  │   │ PatchGAN  │   │ PatchGAN  │   │ PatchGAN  │                    │ │
│  │   └───────────┘   └───────────┘   └───────────┘                    │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component 1: Condition Generator

**File**: `models/hr_viton/condition_generator.py`

#### Appearance Flow Network

Unlike TPS (which uses fixed control points), appearance flow predicts **per-pixel** displacement:

```python
class AppearanceFlowNetwork(nn.Module):
    """
    Predicts dense 2D flow field to warp clothing.
    
    Advantages over TPS:
    1. Per-pixel flexibility (vs 25 control points)
    2. Better handles complex deformations
    3. Preserves fine details like patterns
    
    Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │ Encoder:                                                        │
    │   Conv(C_in, 64, 3, s=1) → Conv(64, 128, 4, s=2) →             │
    │   Conv(128, 256, 4, s=2) → Conv(256, 512, 4, s=2) →            │
    │   Conv(512, 512, 4, s=2)                                        │
    │   Output: [B, 512, H/16, W/16]                                  │
    │                                                                 │
    │ Flow Decoder:                                                   │
    │   Upsample → Conv(512, 256) → Upsample → Conv(256, 128) →      │
    │   Upsample → Conv(128, 64) → Upsample → Conv(64, 2)            │
    │   Output: [B, 2, H, W] (dx, dy flow)                           │
    └────────────────────────────────────────────────────────────────┘
    """
    
    def warp_with_flow(self, image, flow):
        """
        Warp image using predicted flow.
        
        Args:
            image: [B, C, H, W] - Source image (cloth)
            flow: [B, 2, H, W] - Flow field (dx, dy)
        
        Returns:
            warped: [B, C, H, W] - Warped image
        """
        B, C, H, W = image.shape
        
        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0)  # [2, H, W]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)   # [B, 2, H, W]
        
        # Add flow to grid
        sampling_grid = grid + flow
        
        # Rearrange for grid_sample: [B, H, W, 2]
        sampling_grid = sampling_grid.permute(0, 2, 3, 1)
        
        # Bilinear sampling
        warped = F.grid_sample(
            image, sampling_grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        return warped
```

#### Segmentation Generator

Predicts the target segmentation map for the try-on result:

```python
class SegmentationGenerator(nn.Module):
    """
    Predicts semantic segmentation for the try-on result.
    
    Purpose:
    1. Provides spatial guidance for SPADE generator
    2. Handles occlusion by predicting where body parts will be
    3. Enables region-specific processing
    
    Output: [B, 20, H, W] - 20-class segmentation logits
    """
```

### Component 2: SPADE Generator

**File**: `models/hr_viton/generator.py`

#### SPADE Normalization

SPADE (Spatially-Adaptive Denormalization) is the key innovation for high-quality synthesis:

```python
class SPADENorm(nn.Module):
    """
    Spatially-Adaptive Denormalization.
    
    Unlike standard normalization (BatchNorm, InstanceNorm) which use
    learned global γ and β, SPADE learns SPATIAL γ and β from the
    segmentation map.
    
    This allows different regions (face, arms, clothing) to be
    processed differently.
    
    Formula:
        out = γ(seg) * InstanceNorm(x) + β(seg)
    
    Where γ(seg) and β(seg) are 2D convolutions of the segmentation map.
    """
    
    def __init__(self, norm_channels, seg_channels):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(norm_channels, affine=False)
        
        hidden_channels = 128
        self.shared = nn.Sequential(
            nn.Conv2d(seg_channels, hidden_channels, 3, padding=1),
            nn.ReLU(True)
        )
        self.gamma = nn.Conv2d(hidden_channels, norm_channels, 3, padding=1)
        self.beta = nn.Conv2d(hidden_channels, norm_channels, 3, padding=1)
    
    def forward(self, x, segmentation):
        """
        Args:
            x: [B, C, H, W] - Feature map to normalize
            segmentation: [B, S, H', W'] - Segmentation map
        
        Returns:
            out: [B, C, H, W] - Spatially-normalized features
        """
        # Resize segmentation to match x
        if segmentation.shape[2:] != x.shape[2:]:
            segmentation = F.interpolate(segmentation, size=x.shape[2:], 
                                         mode='nearest')
        
        # Instance normalize
        normalized = self.instance_norm(x)
        
        # Compute spatial modulation parameters
        shared_features = self.shared(segmentation)
        gamma = self.gamma(shared_features)  # [B, C, H, W]
        beta = self.beta(shared_features)    # [B, C, H, W]
        
        # Apply spatial denormalization
        out = gamma * normalized + beta
        
        return out
```

#### SPADE ResBlock

```python
class SPADEResBlock(nn.Module):
    """
    Residual block with SPADE normalization.
    
    Structure:
        x ──┬──▶ SPADE → ReLU → Conv → SPADE → ReLU → Conv ──┬──▶ out
            │                                                  │
            └──────────────────── + ◀──────────────────────────┘
    """
    
    def forward(self, x, segmentation):
        residual = x
        
        out = self.spade1(x, segmentation)
        out = F.leaky_relu(out, 0.2)
        out = self.conv1(out)
        
        out = self.spade2(out, segmentation)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        
        if self.learned_shortcut:
            residual = self.spade_shortcut(residual, segmentation)
            residual = self.conv_shortcut(residual)
        
        return out + residual
```

#### Full Generator Architecture

```python
class SPADEGenerator(nn.Module):
    """
    High-resolution generator using SPADE normalization.
    
    Architecture:
    ┌────────────────────────────────────────────────────────────────┐
    │ Initial:                                                        │
    │   Input: Warped cloth + Agnostic [B, 6, 1024, 768]             │
    │   DownSample to [B, 512, 64, 48]                               │
    │                                                                 │
    │ SPADE Blocks (conditioned on segmentation):                     │
    │   SPADEResBlock(512, 512) → [B, 512, 64, 48]                   │
    │   Upsample + SPADEResBlock(512, 512) → [B, 512, 128, 96]       │
    │   Upsample + SPADEResBlock(512, 256) → [B, 256, 256, 192]      │
    │   Upsample + SPADEResBlock(256, 128) → [B, 128, 512, 384]      │
    │   Upsample + SPADEResBlock(128, 64) → [B, 64, 1024, 768]       │
    │                                                                 │
    │ Output:                                                         │
    │   Conv(64, 3) + Tanh → [B, 3, 1024, 768]                       │
    └────────────────────────────────────────────────────────────────┘
    """
```

### Component 3: Multi-Scale Discriminator

**File**: `models/hr_viton/discriminator.py`

```python
class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator.
    
    Operates at 3 scales to capture both fine details and global structure.
    
    Scale 1 (1024×768): Fine textures, patterns, fabric details
    Scale 2 (512×384):  Medium features, folds, shape consistency
    Scale 3 (256×192):  Global structure, pose alignment
    
    Each scale is a PatchGAN that outputs a grid of predictions
    rather than a single real/fake decision.
    """
    
    def __init__(self, input_channels=3, num_scales=3):
        super().__init__()
        
        self.discriminators = nn.ModuleList()
        for _ in range(num_scales):
            self.discriminators.append(PatchGANDiscriminator(input_channels))
        
        # Downsampling between scales
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, 1024, 768] - Input image
        
        Returns:
            predictions: List of prediction maps at each scale
            features: List of intermediate features (for feature matching loss)
        """
        predictions = []
        features = []
        
        for i, disc in enumerate(self.discriminators):
            pred, feat = disc(x)
            predictions.append(pred)
            features.append(feat)
            
            # Downsample for next scale
            if i < len(self.discriminators) - 1:
                x = self.downsample(x)
        
        return predictions, features


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator.
    
    Instead of outputting a single real/fake probability,
    outputs a grid of probabilities. Each cell corresponds
    to a patch of the input image.
    
    Architecture:
        Conv(3, 64, 4, s=2) → LeakyReLU                    # [B, 64, H/2, W/2]
        Conv(64, 128, 4, s=2) → InstanceNorm → LeakyReLU   # [B, 128, H/4, W/4]
        Conv(128, 256, 4, s=2) → InstanceNorm → LeakyReLU  # [B, 256, H/8, W/8]
        Conv(256, 512, 4, s=1) → InstanceNorm → LeakyReLU  # [B, 512, H/8, W/8]
        Conv(512, 1, 4, s=1)                                # [B, 1, H/8, W/8]
    
    Receptive field: ~70×70 pixels (hence "70×70 PatchGAN")
    """
```

---

## Shared Network Components

**File**: `models/networks/base_networks.py`

### Convolutional Block

```python
class ConvBlock(nn.Module):
    """
    Standard convolutional block.
    
    Conv → Norm → Activation
    
    Configurable normalization: BatchNorm, InstanceNorm, None
    Configurable activation: ReLU, LeakyReLU, None
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, norm='batch', activation='relu'):
        super().__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, 
                           stride, padding, bias=norm is None)]
        
        if norm == 'batch':
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == 'instance':
            layers.append(nn.InstanceNorm2d(out_channels))
        
        if activation == 'relu':
            layers.append(nn.ReLU(True))
        elif activation == 'leaky':
            layers.append(nn.LeakyReLU(0.2, True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)
```

### Residual Block

```python
class ResBlock(nn.Module):
    """
    Standard residual block.
    
    x ──┬──▶ Conv → Norm → ReLU → Conv → Norm ──┬──▶ ReLU → out
        │                                        │
        └────────────────── + ◀──────────────────┘
    """
```

### U-Net Components

```python
class UNetEncoder(nn.Module):
    """
    U-Net encoder with skip connections.
    
    Progressively downsamples while capturing multi-scale features.
    Returns intermediate features for skip connections.
    """

class UNetDecoder(nn.Module):
    """
    U-Net decoder with skip connections.
    
    Progressively upsamples while incorporating encoder features
    through skip connections.
    """
```

### Feature Pyramid Network

```python
class FeaturePyramidNetwork(nn.Module):
    """
    Multi-scale feature extraction.
    
    Builds a pyramid of features at different resolutions.
    Used for capturing both fine details and global context.
    """
```

---

## Loss Functions

**File**: `models/networks/losses.py`

### L1 Loss (Reconstruction)

```python
def l1_loss(output, target):
    """
    Pixel-wise L1 loss.
    
    L1 = mean(|output - target|)
    
    Properties:
    - Encourages pixel-level similarity
    - More robust to outliers than L2
    - Can cause blurriness in high-frequency regions
    """
    return F.l1_loss(output, target)
```

### Perceptual Loss (VGG)

```python
class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    
    Instead of comparing pixels, compare high-level features
    extracted from a pretrained VGG network.
    
    L_perceptual = Σ_l λ_l * ||VGG_l(output) - VGG_l(target)||_1
    
    Layers used: relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
    
    Properties:
    - Captures semantic similarity
    - Better preserves textures and patterns
    - Reduces blur compared to pure L1
    """
    
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        
        # Extract features at specific layers
        self.slice1 = nn.Sequential(*list(vgg[:4]))   # relu1_2
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # relu2_2
        self.slice3 = nn.Sequential(*list(vgg[9:18])) # relu3_3
        self.slice4 = nn.Sequential(*list(vgg[18:27]))# relu4_3
        self.slice5 = nn.Sequential(*list(vgg[27:36]))# relu5_3
        
        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False
        
        # Layer weights
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
```

### Adversarial Loss (GAN)

```python
def adversarial_loss(discriminator_output, target_is_real):
    """
    GAN adversarial loss.
    
    For Generator (wants fake to look real):
        L_G = -E[log(D(G(z)))]
    
    For Discriminator (wants to distinguish real/fake):
        L_D = -E[log(D(real))] - E[log(1 - D(G(z)))]
    
    We use hinge loss variant for stability:
        L_D = E[ReLU(1 - D(real))] + E[ReLU(1 + D(fake))]
        L_G = -E[D(fake)]
    """
```

### Feature Matching Loss

```python
def feature_matching_loss(fake_features, real_features):
    """
    Feature matching loss for GAN training stability.
    
    Instead of directly optimizing discriminator output,
    match intermediate features.
    
    L_FM = Σ_l ||D_l(fake) - D_l(real)||_1
    
    Properties:
    - Stabilizes GAN training
    - Provides richer gradient signal
    - Reduces mode collapse
    """
    loss = 0
    for fake_feat, real_feat in zip(fake_features, real_features):
        loss += F.l1_loss(fake_feat, real_feat.detach())
    return loss
```

### Total Variation Loss

```python
def total_variation_loss(x):
    """
    Total variation loss for smoothness.
    
    L_TV = Σ|x[i+1,j] - x[i,j]| + Σ|x[i,j+1] - x[i,j]|
    
    Used for:
    - Flow field regularization (smooth warping)
    - Mask smoothness
    - Reducing artifacts
    """
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return tv_h + tv_w
```

---

## Training Procedures

### CP-VTON Training (Two-Stage)

**File**: `train/train_cp_vton.py`

```python
"""
Stage 1: Train GMM
==================
- Input: Person representation, Cloth image
- Output: TPS parameters → Warped cloth
- Loss: L1(warped_cloth, ground_truth_warped) + λ_reg * smoothness
- Duration: ~100 epochs
- LR: 1e-4, decay every 50 epochs

Stage 2: Train TOM (GMM frozen)
===============================
- Input: Agnostic + Warped cloth (from GMM) + Pose
- Output: Final try-on image + Composition mask
- Loss: λ_l1 * L1 + λ_vgg * Perceptual + λ_mask * MaskReg
- Duration: ~200 epochs
- LR: 1e-4, decay every 100 epochs
"""

# Training loop (simplified)
for epoch in range(num_epochs):
    for batch in dataloader:
        if stage == 'gmm':
            # GMM forward
            warped_cloth = gmm(person_repr, cloth)
            loss = l1_loss(warped_cloth, gt_warped_cloth)
            
        elif stage == 'tom':
            # GMM forward (no grad)
            with torch.no_grad():
                warped_cloth = gmm(person_repr, cloth)
            
            # TOM forward
            rendered, mask = tom(agnostic, warped_cloth, pose)
            output = mask * warped_cloth + (1 - mask) * rendered
            
            # Losses
            loss_l1 = l1_loss(output, target)
            loss_vgg = perceptual_loss(output, target)
            loss_mask = mask_regularization(mask)
            loss = λ_l1 * loss_l1 + λ_vgg * loss_vgg + λ_mask * loss_mask
        
        loss.backward()
        optimizer.step()
```

### HR-VITON Training (End-to-End GAN)

**File**: `train/train_hr_viton.py`

```python
"""
End-to-End GAN Training
=======================
All components trained jointly with adversarial loss.

Generator: Condition Generator + SPADE Generator
Discriminator: Multi-Scale PatchGAN

Losses:
- L1 reconstruction
- VGG perceptual
- Adversarial (hinge)
- Feature matching
- Flow regularization (TV)
- Segmentation (cross-entropy)

Duration: ~300 epochs
LR_G: 1e-4, LR_D: 4e-4 (TTUR)
"""

for epoch in range(num_epochs):
    for batch in dataloader:
        # ============ Train Discriminator ============
        optimizer_D.zero_grad()
        
        # Generate fake image
        with torch.no_grad():
            fake = generator(inputs)
        
        # Discriminator predictions
        real_pred, _ = discriminator(real_image)
        fake_pred, _ = discriminator(fake)
        
        # Hinge loss
        d_loss = hinge_d_loss(real_pred, fake_pred)
        d_loss.backward()
        optimizer_D.step()
        
        # ============ Train Generator ============
        optimizer_G.zero_grad()
        
        # Generate fake image
        output = generator(inputs)
        
        # Discriminator on fake
        fake_pred, fake_features = discriminator(output)
        _, real_features = discriminator(real_image)
        
        # Generator losses
        loss_adv = hinge_g_loss(fake_pred)
        loss_l1 = l1_loss(output, real_image)
        loss_vgg = perceptual_loss(output, real_image)
        loss_fm = feature_matching_loss(fake_features, real_features)
        loss_flow = tv_loss(flow)
        loss_seg = ce_loss(pred_seg, gt_seg)
        
        g_loss = (λ_adv * loss_adv + λ_l1 * loss_l1 + 
                  λ_vgg * loss_vgg + λ_fm * loss_fm +
                  λ_flow * loss_flow + λ_seg * loss_seg)
        
        g_loss.backward()
        optimizer_G.step()
```

---

## Inference Pipeline

**File**: `inference/inference.py`

```python
class VITONInference:
    """
    Unified inference interface for both methods.
    
    Usage:
        model = VITONInference(method='hr_viton', checkpoint='best.pth')
        result = model.try_on(person_image, cloth_image)
    """
    
    def try_on(self, person_image, cloth_image):
        """
        Perform virtual try-on.
        
        Pipeline:
        1. Preprocess images (resize, normalize)
        2. Generate cloth mask
        3. Generate agnostic representation
        4. Run model inference
        5. Postprocess output
        
        Returns:
            PIL Image of result
        """
```

---

## Evaluation Metrics

**File**: `utils/metrics.py`

### SSIM (Structural Similarity Index)

```python
"""
Measures structural similarity between images.

SSIM = (2μ_x μ_y + C1)(2σ_xy + C2) / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))

Range: [-1, 1], higher is better (1.0 = identical)

Good for: Overall structural quality
Limitation: May not capture perceptual quality well
"""
```

### LPIPS (Learned Perceptual Image Patch Similarity)

```python
"""
Learned perceptual metric using deep features.

LPIPS = Σ_l w_l * ||φ_l(x) - φ_l(y)||²

Where φ_l are features from a pretrained network (VGG/AlexNet).

Range: [0, ∞), lower is better (0 = identical)

Good for: Perceptual quality, texture similarity
Better correlates with human perception than SSIM.
"""
```

### FID (Fréchet Inception Distance)

```python
"""
Measures distance between feature distributions.

FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^(1/2))

Where μ, Σ are mean and covariance of Inception features.

Range: [0, ∞), lower is better

Good for: Overall generation quality
Measures diversity and quality together.
"""
```

---

## Summary Comparison

| Aspect | CP-VTON | HR-VITON |
|--------|---------|----------|
| **Paper** | ECCV 2018 | ECCV 2022 |
| **Resolution** | 256×192 | 1024×768 |
| **Warping** | TPS (25 points) | Per-pixel flow |
| **Generator** | U-Net | SPADE ResNet |
| **Discriminator** | None | Multi-scale PatchGAN |
| **Training** | 2-stage | End-to-end |
| **Parameters** | ~20M | ~85M |
| **Inference** | ~30ms | ~150ms |
| **SSIM** | ~0.78 | ~0.86 |
| **FID** | ~15.8 | ~9.2 |
| **Best For** | Real-time, resource-limited | High-quality results |

---

## References

1. Han et al., "VITON: An Image-based Virtual Try-on Network", CVPR 2018
2. Wang et al., "Toward Characteristic-Preserving Image-based Virtual Try-On Network", ECCV 2018
3. Choi et al., "VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization", CVPR 2021
4. Lee et al., "High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions", ECCV 2022
5. Park et al., "Semantic Image Synthesis with Spatially-Adaptive Normalization", CVPR 2019
6. Bookstein, "Principal Warps: Thin-Plate Splines and the Decomposition of Deformations", PAMI 1989
