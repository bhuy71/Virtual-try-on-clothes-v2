"""
Demo script for Virtual Try-On

This script demonstrates how to use both CP-VTON and HR-VITON models
for virtual clothing try-on.

Usage:
    # Quick demo with default settings
    python demo.py --person examples/person.jpg --cloth examples/cloth.jpg
    
    # Use specific method and checkpoint
    python demo.py --method hr_viton \
        --checkpoint checkpoints/hr_viton/best.pth \
        --person examples/person.jpg \
        --cloth examples/cloth.jpg \
        --output results/output.jpg
"""

import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import VITONInference
from utils.metrics import VITONMetrics, calculate_ssim, calculate_psnr


def create_comparison_grid(
    person: Image.Image,
    cloth: Image.Image,
    result: Image.Image,
    output_path: str = None
) -> Image.Image:
    """Create a side-by-side comparison image."""
    # Resize to same height
    target_height = 512
    
    def resize_to_height(img, height):
        ratio = height / img.height
        new_width = int(img.width * ratio)
        return img.resize((new_width, height), Image.LANCZOS)
    
    person_resized = resize_to_height(person, target_height)
    cloth_resized = resize_to_height(cloth, target_height)
    result_resized = resize_to_height(result, target_height)
    
    # Create combined image
    total_width = person_resized.width + cloth_resized.width + result_resized.width + 40
    combined = Image.new('RGB', (total_width, target_height + 60), color='white')
    
    # Paste images
    x_offset = 10
    combined.paste(person_resized, (x_offset, 10))
    x_offset += person_resized.width + 10
    combined.paste(cloth_resized, (x_offset, 10))
    x_offset += cloth_resized.width + 10
    combined.paste(result_resized, (x_offset, 10))
    
    # Add labels using PIL
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(combined)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
            
        y_text = target_height + 20
        draw.text((person_resized.width // 2, y_text), "Person", fill='black', anchor='mm', font=font)
        draw.text((person_resized.width + 10 + cloth_resized.width // 2, y_text), "Cloth", fill='black', anchor='mm', font=font)
        draw.text((person_resized.width + cloth_resized.width + 20 + result_resized.width // 2, y_text), "Result", fill='black', anchor='mm', font=font)
    except:
        pass  # Skip labels if font not available
    
    if output_path:
        combined.save(output_path)
        print(f"Comparison saved to: {output_path}")
        
    return combined


def demo_single_inference(args):
    """Demo with single person and cloth image."""
    print("\n" + "="*50)
    print("Virtual Try-On Demo")
    print("="*50)
    
    # Initialize model
    print(f"\nMethod: {args.method}")
    print(f"Device: {args.device}")
    
    model = VITONInference(
        method=args.method,
        checkpoint_path=args.checkpoint if hasattr(args, 'checkpoint') else None,
        config_path=args.config if hasattr(args, 'config') else None,
        device=args.device
    )
    
    # Load images
    print(f"\nPerson image: {args.person}")
    print(f"Cloth image: {args.cloth}")
    
    person_img = Image.open(args.person).convert('RGB')
    cloth_img = Image.open(args.cloth).convert('RGB')
    
    # Run inference
    print("\nRunning inference...")
    
    if args.show_all:
        results = model.try_on(args.person, args.cloth, return_all=True)
        result = results['output']
        
        # Save all outputs
        output_dir = os.path.dirname(args.output) or '.'
        os.makedirs(output_dir, exist_ok=True)
        
        for name, img in results.items():
            save_path = os.path.join(output_dir, f"{name}.jpg")
            img.save(save_path)
            print(f"  Saved {name} to: {save_path}")
    else:
        result = model.try_on(args.person, args.cloth)
    
    # Save result
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    result.save(args.output)
    print(f"\nResult saved to: {args.output}")
    
    # Create comparison
    if args.compare:
        compare_path = args.output.replace('.jpg', '_comparison.jpg').replace('.png', '_comparison.png')
        create_comparison_grid(person_img, cloth_img, result, compare_path)
    
    print("\nDone!")
    return result


def demo_batch_inference(args):
    """Demo with multiple images."""
    print("\n" + "="*50)
    print("Virtual Try-On Batch Demo")
    print("="*50)
    
    # Get image lists
    person_images = sorted([
        os.path.join(args.person_dir, f)
        for f in os.listdir(args.person_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    cloth_images = sorted([
        os.path.join(args.cloth_dir, f)
        for f in os.listdir(args.cloth_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    print(f"Found {len(person_images)} person images")
    print(f"Found {len(cloth_images)} cloth images")
    
    # Initialize model
    model = VITONInference(
        method=args.method,
        checkpoint_path=args.checkpoint if hasattr(args, 'checkpoint') else None,
        device=args.device
    )
    
    # Process
    results = model.batch_try_on(
        person_images=person_images,
        cloth_images=cloth_images,
        output_dir=args.output_dir,
        save_individual=True
    )
    
    print(f"\nProcessed {len(results)} images")
    print(f"Results saved to: {args.output_dir}")
    
    return results


def demo_evaluation(args):
    """Demo evaluation metrics."""
    print("\n" + "="*50)
    print("Virtual Try-On Evaluation Demo")
    print("="*50)
    
    # Initialize metrics
    metrics = VITONMetrics(device=args.device)
    
    # Get image lists
    generated_images = sorted([
        os.path.join(args.generated_dir, f)
        for f in os.listdir(args.generated_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    real_images = sorted([
        os.path.join(args.real_dir, f)
        for f in os.listdir(args.real_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    print(f"Generated images: {len(generated_images)}")
    print(f"Real images: {len(real_images)}")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    results = metrics.evaluate_from_paths(
        generated_images[:len(real_images)],
        real_images[:len(generated_images)],
        image_size=(256, 192)
    )
    
    print("\n" + "-"*30)
    print("Evaluation Results:")
    print("-"*30)
    print(f"  SSIM:  {results['ssim']:.4f}  (higher is better, max=1.0)")
    print(f"  LPIPS: {results['lpips']:.4f}  (lower is better)")
    print(f"  FID:   {results['fid']:.2f}  (lower is better)")
    print("-"*30)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Virtual Try-On Demo')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Single inference
    single_parser = subparsers.add_parser('single', help='Single image try-on')
    single_parser.add_argument('--method', type=str, default='hr_viton',
                               choices=['cp_vton', 'hr_viton'])
    single_parser.add_argument('--checkpoint', type=str, default=None)
    single_parser.add_argument('--config', type=str, default=None)
    single_parser.add_argument('--person', type=str, required=True)
    single_parser.add_argument('--cloth', type=str, required=True)
    single_parser.add_argument('--output', type=str, default='output.jpg')
    single_parser.add_argument('--device', type=str, default='auto')
    single_parser.add_argument('--compare', action='store_true')
    single_parser.add_argument('--show_all', action='store_true')
    
    # Batch inference
    batch_parser = subparsers.add_parser('batch', help='Batch image try-on')
    batch_parser.add_argument('--method', type=str, default='hr_viton')
    batch_parser.add_argument('--checkpoint', type=str, default=None)
    batch_parser.add_argument('--person_dir', type=str, required=True)
    batch_parser.add_argument('--cloth_dir', type=str, required=True)
    batch_parser.add_argument('--output_dir', type=str, default='results/')
    batch_parser.add_argument('--device', type=str, default='auto')
    
    # Evaluation
    eval_parser = subparsers.add_parser('eval', help='Evaluate results')
    eval_parser.add_argument('--generated_dir', type=str, required=True)
    eval_parser.add_argument('--real_dir', type=str, required=True)
    eval_parser.add_argument('--device', type=str, default='auto')
    
    # For backward compatibility, also allow direct arguments
    parser.add_argument('--method', type=str, default='hr_viton')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--person', type=str, default=None)
    parser.add_argument('--cloth', type=str, default=None)
    parser.add_argument('--output', type=str, default='output.jpg')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--show_all', action='store_true')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        demo_single_inference(args)
    elif args.command == 'batch':
        demo_batch_inference(args)
    elif args.command == 'eval':
        demo_evaluation(args)
    elif args.person and args.cloth:
        # Direct invocation without subcommand
        demo_single_inference(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
