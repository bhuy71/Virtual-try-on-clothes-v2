# Virtual Try-On Project

A comprehensive virtual try-on system that allows users to visualize how clothing items would look on them. This project implements two methodologies:

1. **CP-VTON (Traditional Method)**: Characteristic-Preserving Virtual Try-On Network
2. **HR-VITON (State-of-the-Art Method)**: High-Resolution Virtual Try-On with Misalignment and Occlusion Handling

## 📚 Research Background

### Key Papers Referenced:
1. **VITON** (CVPR 2018): "An Image-based Virtual Try-on Network" - The foundational work
2. **CP-VTON** (ECCV 2018): "Toward Characteristic-Preserving Image-based Virtual Try-On Network" - Improved warping
3. **VITON-HD** (CVPR 2021): "High-Resolution Virtual Try-On via Misalignment-Aware Normalization"
4. **HR-VITON** (ECCV 2022): "High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions"
5. **GP-VTON** (CVPR 2023): "Towards General Purpose Virtual Try-on via Collaborative Local-Flow Global-Parsing Learning"
6. **StableVITON** (CVPR 2024): "Learning Semantic Correspondence with Latent Diffusion Model for Virtual Try-On"

## 🏗️ Project Structure

```
project3/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   ├── cp_vton_config.yaml      # Configuration for CP-VTON
│   └── hr_viton_config.yaml     # Configuration for HR-VITON
├── data/
│   ├── download_dataset.py      # Script to download datasets
│   ├── preprocess.py            # Data preprocessing pipeline
│   └── dataset.py               # PyTorch Dataset classes
├── models/
│   ├── __init__.py
│   ├── cp_vton/                 # Method 1: Traditional CP-VTON
│   │   ├── __init__.py
│   │   ├── gmm.py               # Geometric Matching Module
│   │   ├── tom.py               # Try-On Module
│   │   └── cp_vton.py           # Full CP-VTON model
│   ├── hr_viton/                # Method 2: State-of-the-Art HR-VITON
│   │   ├── __init__.py
│   │   ├── condition_generator.py
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── hr_viton.py          # Full HR-VITON model
│   └── networks/
│       ├── __init__.py
│       ├── base_networks.py     # Shared network components
│       ├── attention.py         # Attention mechanisms
│       └── losses.py            # Loss functions
├── train/
│   ├── trainer.py               # Base trainer class with checkpointing
│   ├── train_cp_vton.py         # Training script for CP-VTON
│   └── train_hr_viton.py        # Training script for HR-VITON
├── inference/
│   ├── __init__.py
│   └── inference.py             # Unified inference pipeline
├── utils/
│   ├── __init__.py
│   └── metrics.py               # SSIM, FID, LPIPS, evaluation metrics
├── examples/
│   └── README.md                # Sample images guide
└── demo.py                      # Interactive demo script
```

## 📊 Datasets

### VITON-HD Dataset (Recommended)
- High-resolution (1024×768) paired images
- Contains 13,679 image pairs
- Includes: person images, cloth images, cloth masks, pose keypoints, parsing maps

### DressCode Dataset
- Multi-category (upper-body, lower-body, dresses)
- Over 48,000 pairs
- High resolution images

### Download Instructions:
```bash
python data/download_dataset.py --dataset viton-hd
```

## 🔧 Installation

```bash
# Clone the repository
cd project3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 🚀 Quick Start

### 1. Prepare Data
```bash
python data/preprocess.py --input_dir ./raw_data --output_dir ./processed_data
```

### 2. Train CP-VTON (Method 1)
```bash
# Train Geometric Matching Module
python train/train_cp_vton.py --stage gmm --config configs/cp_vton_config.yaml

# Train Try-On Module
python train/train_cp_vton.py --stage tom --config configs/cp_vton_config.yaml
```

### 3. Train HR-VITON (Method 2)
```bash
python train/train_hr_viton.py --config configs/hr_viton_config.yaml
```

### 4. Inference
```bash
# Single image inference
python demo.py --person examples/person.jpg --cloth examples/cloth.jpg --output result.jpg

# With specific method and checkpoint
python demo.py --method hr_viton --checkpoint checkpoints/hr_viton/best.pth \
    --person examples/person.jpg --cloth examples/cloth.jpg --output result.jpg

# Batch processing
python demo.py batch --person_dir ./persons --cloth_dir ./clothes --output_dir ./results

# Evaluate results
python demo.py eval --generated_dir ./results --real_dir ./ground_truth
```

## 🔬 Python API Usage

```python
from inference import VITONInference

# Initialize HR-VITON (state-of-the-art)
model = VITONInference(
    method='hr_viton',
    checkpoint_path='checkpoints/hr_viton/best.pth',
    config_path='configs/hr_viton_config.yaml'
)

# Run virtual try-on
result = model.try_on(
    person_image='examples/person.jpg',
    cloth_image='examples/cloth.jpg'
)
result.save('output.jpg')

# Get all intermediate results
results = model.try_on(person_image, cloth_image, return_all=True)
# Returns: output, warped_cloth, agnostic, cloth_mask

# Batch processing
model.batch_try_on(
    person_images=['person1.jpg', 'person2.jpg'],
    cloth_images=['cloth.jpg'],  # Same cloth for all
    output_dir='results/'
)
```

### Evaluation Metrics
```python
from utils.metrics import VITONMetrics

metrics = VITONMetrics(device='cuda')
results = metrics.evaluate_from_paths(
    generated_paths=['output1.jpg', 'output2.jpg'],
    real_paths=['gt1.jpg', 'gt2.jpg']
)
print(f"SSIM: {results['ssim']:.4f}")
print(f"LPIPS: {results['lpips']:.4f}")
print(f"FID: {results['fid']:.2f}")
```

## 📈 Methodology Comparison

| Feature | CP-VTON (Traditional) | HR-VITON (SOTA) |
|---------|----------------------|-----------------|
| Resolution | 256×192 | 1024×768 |
| Approach | Two-stage (GMM + TOM) | End-to-end with conditions |
| Warping | Thin-Plate Spline (TPS) | Appearance Flow |
| Occlusion Handling | Basic | Advanced with parsing |
| Training Time | ~24 hours | ~72 hours |
| FID Score | ~15.8 | ~9.2 |

## 🎯 Method 1: CP-VTON (Traditional)

### Architecture:
1. **Geometric Matching Module (GMM)**:
   - Takes person representation and target cloth as input
   - Predicts TPS transformation parameters
   - Warps clothing to match person's pose

2. **Try-On Module (TOM)**:
   - Synthesizes the final try-on result
   - Uses U-Net architecture with skip connections
   - Generates composition mask for blending

### Key Features:
- Characteristic preservation through perceptual loss
- Coarse-to-fine warping strategy
- Robust to various poses

## 🎯 Method 2: HR-VITON (State-of-the-Art)

### Architecture:
1. **Condition Generator**:
   - Generates segmentation map and misalignment-aware conditions
   - Handles occlusion through parsing-based attention

2. **Image Generator**:
   - High-resolution synthesis using alias-free generator
   - Multi-scale discriminator for realistic details
   - Appearance flow for accurate texture transfer

### Key Features:
- Misalignment-aware normalization
- Occlusion handling with semantic parsing
- High-resolution output (1024×768)
- Superior texture preservation

## 📊 Evaluation Metrics

- **SSIM**: Structural Similarity Index
- **FID**: Fréchet Inception Distance
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **IS**: Inception Score

## 🔗 References

```bibtex
@inproceedings{han2018viton,
  title={VITON: An Image-based Virtual Try-on Network},
  author={Han, Xintong and Wu, Zuxuan and Wu, Zhe and Yu, Ruichi and Davis, Larry S},
  booktitle={CVPR},
  year={2018}
}

@inproceedings{wang2018cpvton,
  title={Toward Characteristic-Preserving Image-based Virtual Try-On Network},
  author={Wang, Bochao and Zheng, Huabin and Liang, Xiaodan and Chen, Yimin and Lin, Liang},
  booktitle={ECCV},
  year={2018}
}

@inproceedings{choi2021vitonhd,
  title={VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization},
  author={Choi, Seunghwan and Park, Sunghyun and Lee, Minsoo and Choo, Jaegul},
  booktitle={CVPR},
  year={2021}
}

@inproceedings{lee2022hrviton,
  title={High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions},
  author={Lee, Sangyun and Gu, Gyojung and Park, Sunghyun and Choi, Seunghwan and Choo, Jaegul},
  booktitle={ECCV},
  year={2022}
}

@inproceedings{xie2023gpvton,
  title={GP-VTON: Towards General Purpose Virtual Try-on via Collaborative Local-Flow Global-Parsing Learning},
  author={Xie, Zhenyu and Huang, Zaiyu and Dong, Xin and Zhao, Fuwei and Dong, Haoye and Zhang, Xijin and Zhu, Feida and Liang, Xiaodan},
  booktitle={CVPR},
  year={2023}
}
```

## 📝 License

This project is for educational and research purposes.
