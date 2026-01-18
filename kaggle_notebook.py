# =============================================================================
# VIRTUAL TRY-ON - KAGGLE NOTEBOOK
# =============================================================================
# Copy nội dung này vào Kaggle Notebook
# Đảm bảo chọn GPU accelerator: GPU P100 hoặc T4 x2
# =============================================================================

# %% [markdown]
# # Virtual Try-On Training on Kaggle
# 
# This notebook trains CP-VTON and HR-VITON models on the VITON-HD dataset.
# 
# **Requirements:**
# - GPU: P100 or T4 (select in Settings → Accelerator)
# - Internet: ON (for cloning repo and downloading data)

# %% [code]
# ============== CELL 1: Clone Repository ==============
!git clone https://github.com/bhuy71/Virtual-try-on-clothes-v2.git
%cd Virtual-try-on-clothes-v2

# %% [code]
# ============== CELL 2: Install Dependencies ==============
# Kaggle đã có sẵn: torch, torchvision, numpy, PIL, cv2, scipy, matplotlib, tqdm, pyyaml, scikit-image, tensorboard

# Cài các thư viện còn thiếu (được sử dụng trong code)
!pip install -q gdown          # download_dataset.py - download từ Google Drive
!pip install -q lpips          # utils/metrics.py - Learned Perceptual Loss
!pip install -q einops         # models/networks/attention.py - tensor operations
!pip install -q mediapipe      # data/preprocess.py - pose estimation (thay thế OpenPose)

# Optional: Cài thêm nếu cần
# !pip install -q wandb        # train/trainer.py - experiment tracking (tùy chọn)
# !pip install -q gradio       # demo.py - web demo interface (tùy chọn)
# !pip install -q timm         # pretrained models (tùy chọn cho HR-VITON)

# (Tùy chọn) Cài DensePose nếu dùng HR-VITON với dense pose
# Kaggle có thể đã có detectron2, nếu không:
# !pip install -q 'git+https://github.com/facebookresearch/detectron2.git'
# !git clone https://github.com/facebookresearch/detectron2.git /tmp/detectron2
# !pip install -e /tmp/detectron2/projects/DensePose

# Verify environment
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Verify imports
import cv2, yaml, scipy, PIL
print(f"OpenCV: {cv2.__version__}, PyYAML: {yaml.__version__}")
print("✓ All pre-installed packages OK")

# %% [code]
# ============== CELL 3: Link VITON-HD Dataset ==============
# Sử dụng Kaggle Dataset: "high-resolution-viton-zalando-dataset"
# Thêm dataset này vào notebook: Add Data → Search → "viton zalando"

import os

# Create dataset directory OUTSIDE the 'data' module to avoid Python import conflicts
# Using 'datasets/' instead of 'data/' to prevent module shadowing
os.makedirs('datasets', exist_ok=True)

# Tạo symlink đến Kaggle dataset
KAGGLE_DATASET = '/kaggle/input/high-resolution-viton-zalando-dataset'

# Kiểm tra dataset tồn tại
if os.path.exists(KAGGLE_DATASET):
    # Tạo symlink ở thư mục datasets/ thay vì data/
    if not os.path.exists('datasets/viton_hd'):
        os.symlink(KAGGLE_DATASET, 'datasets/viton_hd')
    print(f"✓ Dataset linked: datasets/viton_hd -> {KAGGLE_DATASET}")
else:
    print(f"✗ Dataset not found at {KAGGLE_DATASET}")
    print("Please add 'high-resolution-viton-zalando-dataset' to your notebook:")
    print("  1. Click 'Add Data' on the right panel")
    print("  2. Search for 'viton zalando' or 'viton-hd'")
    print("  3. Add the dataset")

# Verify structure
!ls -la datasets/viton_hd/
!echo "Train folders:" && ls datasets/viton_hd/train/ | head -5
!echo "Test folders:" && ls datasets/viton_hd/test/ | head -5

# %% [code]
# ============== CELL 4: Prepare Data Structure ==============
# Check dataset structure
!find datasets/viton_hd -type d | head -20
!ls datasets/viton_hd/ | head -10

# %% [code]
# ============== CELL 5: Setup Python Path & Test Import Models ==============
import sys
import os

# IMPORTANT: Set Python path correctly for Kaggle
# This must be done BEFORE any project imports
PROJECT_ROOT = os.getcwd()  # Should be /kaggle/working/Virtual-try-on-clothes-v2
print(f"Project root: {PROJECT_ROOT}")

# Add to Python path if not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Verify the data module exists
print(f"data module exists: {os.path.exists(os.path.join(PROJECT_ROOT, 'data', '__init__.py'))}")
print(f"data/dataset.py exists: {os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'dataset.py'))}")

# Test imports
try:
    from models.cp_vton import CPVTON, GMM, TOM
    from models.hr_viton import HRVITON, ConditionGenerator, HRGenerator
    from data.dataset import VITONDataset, VITONHDDataset
    print("✓ All models imported successfully!")
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()

# %% [code]
# ============== CELL 6: Quick Model Test ==============
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Test CP-VTON GMM
# GMM expects: person (22 channels: pose+shape), cloth (3 channels RGB)
from models.cp_vton.gmm import GMM
gmm = GMM().to(device)

# Dummy inputs với đúng số channels
# person: 22 channels (18 pose heatmaps + 3 RGB + 1 shape mask)
# cloth: 3 channels RGB
dummy_person = torch.randn(1, 22, 256, 192).to(device)  # 22 channels, không phải 3!
dummy_cloth = torch.randn(1, 3, 256, 192).to(device)

with torch.no_grad():
    result = gmm(dummy_person, dummy_cloth)
print(f"✓ GMM output - warped_cloth shape: {result['warped_cloth'].shape}")
print(f"✓ GMM output - theta shape: {result['theta'].shape}")

# Test CP-VTON TOM
# TOM expects: concatenated input (25 channels = person 3 + warped_cloth 3 + pose 18 + shape 1)
from models.cp_vton.tom import TOM
tom = TOM().to(device)

# TOM input: concatenated tensor of 25 channels
dummy_person_rgb = torch.randn(1, 3, 256, 192).to(device)
dummy_pose = torch.randn(1, 18, 256, 192).to(device)  # 18 pose heatmaps
dummy_shape = torch.randn(1, 1, 256, 192).to(device)  # 1 body shape mask
warped_cloth = result['warped_cloth']

# Concatenate all inputs: person(3) + warped_cloth(3) + pose(18) + shape(1) = 25 channels
tom_input = torch.cat([dummy_person_rgb, warped_cloth, dummy_pose, dummy_shape], dim=1)

with torch.no_grad():
    rgb, mask = tom(tom_input)
print(f"✓ TOM output - rgb shape: {rgb.shape}, mask shape: {mask.shape}")

del gmm, tom
torch.cuda.empty_cache()
print("✓ Model tests passed!")

# %% [code]
# ============== CELL 7: Training Configuration ==============
import yaml

# Load config
with open('configs/cp_vton_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Adjust for Kaggle
config['training']['batch_size'] = 4  # Reduce for P100 16GB
config['training']['num_workers'] = 2
config['data']['data_root'] = './datasets/viton_hd'
config['training']['save_dir'] = '/kaggle/working/checkpoints'

print("Training config:")
for key, value in config['training'].items():
    print(f"  {key}: {value}")

# %% [code]
# ============== CELL 8: Train CP-VTON (Method 1 - Traditional) ==============
# CP-VTON sử dụng TPS warping, train 2 giai đoạn: GMM rồi TOM
# Thời gian: ~5-10 giờ mỗi stage trên P100

# Ensure we're in the right directory
import os
os.chdir('/kaggle/working/Virtual-try-on-clothes-v2')

# Import training modules (Python path already set in Cell 5)
from train.train_cp_vton import GMMTrainer, TOMTrainer
from models.cp_vton import GMM, TOM
from data.dataset import VITONDataset
from torch.utils.data import DataLoader
import torch

# === Stage 1: Train GMM (Geometric Matching Module) ===
print("="*50)
print("Training CP-VTON Stage 1: GMM")
print("="*50)

# Dataset - sử dụng resolution 256x192 cho CP-VTON
train_dataset = VITONDataset(
    data_root='./datasets/viton_hd',
    split='train',
    image_size=(256, 192)
)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

# Model
gmm = GMM().cuda()

# Train config
config = {
    'training': {
        'epochs': 100,
        'lr': 1e-4,
        'save_interval': 10
    },
    'checkpoint': {
        'save_dir': '/kaggle/working/checkpoints/cp_vton/gmm'
    }
}

# Create checkpoint directory
os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)

trainer = GMMTrainer(gmm, train_loader, config=config)
trainer.train(num_epochs=100)

# === Stage 2: Train TOM (Try-On Module) ===
print("="*50)
print("Training CP-VTON Stage 2: TOM")
print("="*50)

tom = TOM().cuda()
# Load trained GMM
gmm.load_state_dict(torch.load('/kaggle/working/checkpoints/cp_vton/gmm/best.pth'))
gmm.eval()

config['checkpoint']['save_dir'] = '/kaggle/working/checkpoints/cp_vton/tom'
os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)

trainer = TOMTrainer(tom, train_loader, config=config, gmm=gmm)
trainer.train(num_epochs=100)

print("✓ CP-VTON training complete!")

# %% [code]
# ============== CELL 9: Train HR-VITON (Method 2 - State-of-the-Art) ==============
# HR-VITON sử dụng appearance flow + SPADE generator, train end-to-end với GAN
# Yêu cầu: GPU với >= 16GB VRAM (P100 hoặc T4 x2)
# Thời gian: ~15-20 giờ trên P100

# Uncomment để train:
"""
from train.train_hr_viton import HRVITONTrainer
from models.hr_viton import HRVITON
from data.dataset import VITONHDDataset
from torch.utils.data import DataLoader
import yaml

print("="*50)
print("Training HR-VITON (High-Resolution)")
print("="*50)

# Load config
with open('configs/hr_viton_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Adjust for Kaggle
config['training']['batch_size'] = 2  # HR-VITON cần nhiều VRAM hơn
config['training']['num_workers'] = 2
config['data']['data_root'] = './datasets/viton_hd'
config['data']['image_size'] = [512, 384]  # Giảm resolution nếu thiếu VRAM
config['checkpoint']['save_dir'] = '/kaggle/working/checkpoints/hr_viton'

# Dataset - sử dụng resolution cao hơn (512x384 hoặc 1024x768)
train_dataset = VITONHDDataset(
    data_root='./datasets/viton_hd',
    split='train',
    image_size=(512, 384)  # Hoặc (1024, 768) nếu đủ VRAM
)
train_loader = DataLoader(
    train_dataset, 
    batch_size=config['training']['batch_size'], 
    shuffle=True, 
    num_workers=2,
    pin_memory=True
)

# Model
model = HRVITON(
    image_size=(512, 384),
    semantic_nc=13,
    ngf=64,
    ndf=64
).cuda()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# Trainer
trainer = HRVITONTrainer(
    model=model,
    train_loader=train_loader,
    device=torch.device('cuda'),
    config=config
)

# Train
trainer.train(num_epochs=config['training'].get('epochs', 100))

print("✓ HR-VITON training complete!")
"""

# %% [code]
# ============== CELL 10: Test HR-VITON Model ==============
# Kiểm tra HR-VITON có thể chạy không

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    from models.hr_viton import HRVITON, ConditionGenerator, HRGenerator
    
    # Test với resolution nhỏ để tiết kiệm VRAM
    model = HRVITON(image_size=(256, 192), semantic_nc=13).to(device)
    
    # Dummy inputs
    batch_size = 1
    person = torch.randn(batch_size, 3, 256, 192).to(device)
    cloth = torch.randn(batch_size, 3, 256, 192).to(device)
    cloth_mask = torch.randn(batch_size, 1, 256, 192).to(device)
    agnostic = torch.randn(batch_size, 3, 256, 192).to(device)
    parse = torch.randn(batch_size, 13, 256, 192).to(device)
    
    with torch.no_grad():
        output = model(person, cloth, cloth_mask, agnostic, parse)
    
    print(f"✓ HR-VITON test passed!")
    print(f"  - Output shape: {output['output'].shape}")
    print(f"  - Warped cloth shape: {output['warped_cloth'].shape}")
    print(f"  - Flow shape: {output['flow'].shape}")
    
    del model
    torch.cuda.empty_cache()
    
except Exception as e:
    print(f"✗ HR-VITON test failed: {e}")
    import traceback
    traceback.print_exc()

# %% [code]
# ============== CELL 11: Download Pre-trained Weights (Alternative) ==============
# If you have pre-trained weights, download them here
# Example:
# !gdown "your_weights_gdrive_link" -O checkpoints/cp_vton.pth

# %% [code]
# ============== CELL 12: Run Inference ==============
from inference.inference import VITONInference
import matplotlib.pyplot as plt
from PIL import Image
import glob

# Find sample images (VITON-HD structure)
person_images = sorted(glob.glob('datasets/viton_hd/test/image/*.jpg'))[:5]
cloth_images = sorted(glob.glob('datasets/viton_hd/test/cloth/*.jpg'))[:5]

print(f"Found {len(person_images)} person images")
print(f"Found {len(cloth_images)} cloth images")

if person_images and cloth_images:
    # Display samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, (person, cloth) in enumerate(zip(person_images[:5], cloth_images[:5])):
        axes[0, i].imshow(Image.open(person))
        axes[0, i].set_title(f'Person {i+1}')
        axes[0, i].axis('off')
        axes[1, i].imshow(Image.open(cloth))
        axes[1, i].set_title(f'Cloth {i+1}')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig('/kaggle/working/samples.png')
    plt.show()
else:
    print("Check dataset paths:")
    !ls datasets/viton_hd/test/ | head -5

# %% [code]
# ============== CELL 13: Save Outputs ==============
# Save checkpoints to Kaggle output
import shutil

output_dir = '/kaggle/working/outputs'
os.makedirs(output_dir, exist_ok=True)

# Copy any generated outputs
print(f"Outputs saved to {output_dir}")
print("Download from Kaggle Output tab when session ends.")

# %% [markdown]
# ## Notes:
# 
# 1. **Training Time**: Full training takes 10-20 hours per stage
# 2. **Save Checkpoints**: Kaggle sessions have 12-hour limit, save frequently
# 3. **Use Kaggle Datasets**: Upload VITON-HD as Kaggle Dataset for faster loading
# 4. **Commit Notebook**: Commit to save your work before session expires
# 
# ## Next Steps:
# 1. Upload trained checkpoints to Google Drive or Kaggle Dataset
# 2. Run inference on new images
# 3. Evaluate with SSIM, LPIPS, FID metrics
