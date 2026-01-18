"""
Data Preprocessing Pipeline for Virtual Try-On

This script preprocesses raw images to generate:
1. Person parsing maps (semantic segmentation)
2. Pose keypoints (OpenPose format)
3. Agnostic person representations
4. Cloth masks
5. Dense pose (optional)

Usage:
    python preprocess.py --data_dir ./data/viton-hd --output_dir ./data/viton-hd-processed
"""

import os
import argparse
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import mediapipe as mp
from typing import Dict, List, Tuple, Optional


# Parsing labels for ATR dataset format (used in VITON)
PARSING_LABELS = {
    0: 'background',
    1: 'hat',
    2: 'hair',
    3: 'sunglasses',
    4: 'upper_clothes',
    5: 'skirt',
    6: 'pants',
    7: 'dress',
    8: 'belt',
    9: 'left_shoe',
    10: 'right_shoe',
    11: 'face',
    12: 'left_leg',
    13: 'right_leg',
    14: 'left_arm',
    15: 'right_arm',
    16: 'bag',
    17: 'scarf',
}

# OpenPose keypoint indices
OPENPOSE_KEYPOINTS = {
    0: 'nose',
    1: 'neck',
    2: 'right_shoulder',
    3: 'right_elbow',
    4: 'right_wrist',
    5: 'left_shoulder',
    6: 'left_elbow',
    7: 'left_wrist',
    8: 'right_hip',
    9: 'right_knee',
    10: 'right_ankle',
    11: 'left_hip',
    12: 'left_knee',
    13: 'left_ankle',
    14: 'right_eye',
    15: 'left_eye',
    16: 'right_ear',
    17: 'left_ear',
}


class PoseEstimator:
    """Pose estimation using MediaPipe."""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def estimate(self, image: np.ndarray) -> Dict:
        """Estimate pose keypoints from image."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        h, w = image.shape[:2]
        keypoints = []
        
        if results.pose_landmarks:
            # Convert MediaPipe landmarks to OpenPose format
            landmarks = results.pose_landmarks.landmark
            
            # Mapping from MediaPipe to OpenPose indices
            mp_to_op = {
                0: 0,   # nose
                11: 5,  # left_shoulder
                12: 2,  # right_shoulder
                13: 6,  # left_elbow
                14: 3,  # right_elbow
                15: 7,  # left_wrist
                16: 4,  # right_wrist
                23: 11, # left_hip
                24: 8,  # right_hip
                25: 12, # left_knee
                26: 9,  # right_knee
                27: 13, # left_ankle
                28: 10, # right_ankle
            }
            
            # Initialize keypoints with zeros
            for i in range(18):
                keypoints.append([0, 0, 0])
                
            for mp_idx, op_idx in mp_to_op.items():
                lm = landmarks[mp_idx]
                keypoints[op_idx] = [
                    int(lm.x * w),
                    int(lm.y * h),
                    lm.visibility
                ]
                
            # Estimate neck as midpoint between shoulders
            if keypoints[2][2] > 0 and keypoints[5][2] > 0:
                keypoints[1] = [
                    (keypoints[2][0] + keypoints[5][0]) // 2,
                    (keypoints[2][1] + keypoints[5][1]) // 2,
                    (keypoints[2][2] + keypoints[5][2]) / 2
                ]
        else:
            # Return empty keypoints if no pose detected
            for i in range(18):
                keypoints.append([0, 0, 0])
                
        return {
            'keypoints': keypoints,
            'subset': [[i for i in range(18)] + [1.0, 18]]
        }
        
    def draw_pose(self, image: np.ndarray, pose_data: Dict) -> np.ndarray:
        """Draw pose keypoints on image."""
        img = image.copy()
        keypoints = pose_data['keypoints']
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.1:
                cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
                
        # Draw connections
        connections = [
            (1, 2), (2, 3), (3, 4),  # Right arm
            (1, 5), (5, 6), (6, 7),  # Left arm
            (1, 8), (8, 9), (9, 10),  # Right leg
            (1, 11), (11, 12), (12, 13),  # Left leg
            (0, 1), (0, 14), (0, 15),  # Head
        ]
        
        for i, j in connections:
            if keypoints[i][2] > 0.1 and keypoints[j][2] > 0.1:
                cv2.line(img, 
                         (int(keypoints[i][0]), int(keypoints[i][1])),
                         (int(keypoints[j][0]), int(keypoints[j][1])),
                         (0, 255, 255), 2)
                         
        return img
        
    def close(self):
        self.pose.close()


class HumanParser:
    """Simple human parsing using color-based segmentation."""
    
    # For production, use Self-Correction-Human-Parsing or similar
    # This is a placeholder implementation
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        # Load pretrained model if available
        
    def parse(self, image: np.ndarray, pose_keypoints: List) -> np.ndarray:
        """Generate parsing map from image and pose."""
        h, w = image.shape[:2]
        parsing = np.zeros((h, w), dtype=np.uint8)
        
        # Simple color-based segmentation (placeholder)
        # For real implementation, use deep learning model
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect skin color (face, arms)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Use pose to refine regions
        if pose_keypoints:
            keypoints = pose_keypoints
            
            # Face region (around nose)
            if keypoints[0][2] > 0.1:
                nose = (int(keypoints[0][0]), int(keypoints[0][1]))
                cv2.circle(parsing, nose, 40, 11, -1)  # Face label
                
            # Upper body region (between shoulders and hips)
            if (keypoints[2][2] > 0.1 and keypoints[5][2] > 0.1 and
                keypoints[8][2] > 0.1 and keypoints[11][2] > 0.1):
                pts = np.array([
                    [keypoints[2][0], keypoints[2][1]],
                    [keypoints[5][0], keypoints[5][1]],
                    [keypoints[11][0], keypoints[11][1]],
                    [keypoints[8][0], keypoints[8][1]],
                ], dtype=np.int32)
                cv2.fillPoly(parsing, [pts], 4)  # Upper clothes label
                
        return parsing


def generate_agnostic(
    person_image: np.ndarray,
    parsing_map: np.ndarray,
    pose_keypoints: List
) -> np.ndarray:
    """
    Generate agnostic person representation.
    
    Removes clothing from upper body while preserving:
    - Face
    - Hair
    - Lower body
    - Pose information
    """
    h, w = person_image.shape[:2]
    agnostic = person_image.copy()
    
    # Create mask for regions to remove (upper clothes, arms)
    remove_labels = [4, 14, 15]  # upper_clothes, left_arm, right_arm
    remove_mask = np.isin(parsing_map, remove_labels).astype(np.uint8) * 255
    
    # Dilate the mask to ensure complete removal
    kernel = np.ones((5, 5), np.uint8)
    remove_mask = cv2.dilate(remove_mask, kernel, iterations=2)
    
    # Fill removed regions with gray
    agnostic[remove_mask > 0] = [128, 128, 128]
    
    return agnostic


def generate_cloth_mask(cloth_image: np.ndarray) -> np.ndarray:
    """Generate binary mask for clothing item."""
    # Convert to grayscale
    gray = cv2.cvtColor(cloth_image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to separate cloth from white background
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Clean up mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def generate_pose_heatmap(
    image_shape: Tuple[int, int],
    keypoints: List,
    sigma: float = 6.0
) -> np.ndarray:
    """Generate pose heatmaps from keypoints."""
    h, w = image_shape
    n_keypoints = len(keypoints)
    heatmaps = np.zeros((n_keypoints, h, w), dtype=np.float32)
    
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.1 and 0 <= x < w and 0 <= y < h:
            # Generate Gaussian heatmap
            x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
            heatmaps[i] = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
            
    return heatmaps


def preprocess_single(
    image_path: str,
    cloth_path: str,
    output_dir: str,
    pose_estimator: PoseEstimator,
    human_parser: HumanParser,
    target_size: Tuple[int, int] = (1024, 768)
) -> Dict:
    """Preprocess a single image pair."""
    # Load images
    person_image = cv2.imread(image_path)
    cloth_image = cv2.imread(cloth_path)
    
    if person_image is None or cloth_image is None:
        return None
        
    # Resize images
    h, w = target_size
    person_image = cv2.resize(person_image, (w, h))
    cloth_image = cv2.resize(cloth_image, (w, h))
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 1. Estimate pose
    pose_data = pose_estimator.estimate(person_image)
    
    # 2. Generate parsing map
    parsing_map = human_parser.parse(person_image, pose_data['keypoints'])
    
    # 3. Generate agnostic representation
    agnostic = generate_agnostic(person_image, parsing_map, pose_data['keypoints'])
    
    # 4. Generate cloth mask
    cloth_mask = generate_cloth_mask(cloth_image)
    
    # 5. Generate pose heatmaps
    pose_heatmaps = generate_pose_heatmap((h, w), pose_data['keypoints'])
    
    # 6. Draw pose visualization
    pose_vis = pose_estimator.draw_pose(person_image, pose_data)
    
    # Save outputs
    outputs = {
        'person': os.path.join(output_dir, 'image', f'{base_name}.jpg'),
        'cloth': os.path.join(output_dir, 'cloth', f'{base_name}.jpg'),
        'cloth_mask': os.path.join(output_dir, 'cloth-mask', f'{base_name}.png'),
        'parsing': os.path.join(output_dir, 'image-parse', f'{base_name}.png'),
        'agnostic': os.path.join(output_dir, 'agnostic', f'{base_name}.jpg'),
        'pose_json': os.path.join(output_dir, 'openpose-json', f'{base_name}.json'),
        'pose_vis': os.path.join(output_dir, 'openpose-img', f'{base_name}.jpg'),
    }
    
    # Create directories
    for path in outputs.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
    # Save files
    cv2.imwrite(outputs['person'], person_image)
    cv2.imwrite(outputs['cloth'], cloth_image)
    cv2.imwrite(outputs['cloth_mask'], cloth_mask)
    cv2.imwrite(outputs['parsing'], parsing_map)
    cv2.imwrite(outputs['agnostic'], agnostic)
    cv2.imwrite(outputs['pose_vis'], pose_vis)
    
    with open(outputs['pose_json'], 'w') as f:
        json.dump({
            'people': [{
                'pose_keypoints_2d': [
                    coord for kp in pose_data['keypoints'] for coord in kp
                ]
            }]
        }, f)
        
    return outputs


def preprocess_dataset(
    data_dir: str,
    output_dir: str,
    split: str = 'train',
    target_size: Tuple[int, int] = (1024, 768)
):
    """Preprocess entire dataset."""
    print(f"Preprocessing {split} split...")
    
    # Initialize models
    pose_estimator = PoseEstimator()
    human_parser = HumanParser()
    
    # Get image paths
    image_dir = os.path.join(data_dir, split, 'image')
    cloth_dir = os.path.join(data_dir, split, 'cloth')
    
    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        return
        
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    
    output_split_dir = os.path.join(output_dir, split)
    
    success_count = 0
    fail_count = 0
    
    for img_file in tqdm(image_files, desc=f"Processing {split}"):
        image_path = os.path.join(image_dir, img_file)
        
        # Find corresponding cloth image
        cloth_file = img_file.replace('_0.jpg', '_1.jpg').replace('_0.png', '_1.png')
        cloth_path = os.path.join(cloth_dir, cloth_file)
        
        if not os.path.exists(cloth_path):
            # Try same filename
            cloth_path = os.path.join(cloth_dir, img_file)
            
        if not os.path.exists(cloth_path):
            fail_count += 1
            continue
            
        try:
            result = preprocess_single(
                image_path, cloth_path, output_split_dir,
                pose_estimator, human_parser, target_size
            )
            if result:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            fail_count += 1
            
    pose_estimator.close()
    
    print(f"\nPreprocessing complete!")
    print(f"Success: {success_count}, Failed: {fail_count}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess Virtual Try-On data")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Input data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "test", "all"],
        help="Which split to process"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Target height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Target width"
    )
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.data_dir
    target_size = (args.height, args.width)
    
    if args.split == "all":
        splits = ["train", "test"]
    else:
        splits = [args.split]
        
    for split in splits:
        preprocess_dataset(args.data_dir, output_dir, split, target_size)
        
    print("\nAll preprocessing complete!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
