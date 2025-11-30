import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import cv2

def load_image(path):
    """Loads an image and normalizes it to [0, 1]."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).float() / 255.0

def process_blender_data(base_dir, output_dir, split='train', half_res=False):
    """
    Converts Blender-style NeRF data (transforms.json) to the .pt format required by cNeRF.
    """
    base_path = Path(base_dir)
    json_path = base_path / f"transforms_{split}.json"
    
    if not json_path.exists():
        # Fallback to transforms.json if split-specific doesn't exist
        json_path = base_path / "transforms.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Could not find transforms_{split}.json or transforms.json in {base_dir}")

    print(f"Loading metadata from {json_path}...")
    with open(json_path, 'r') as f:
        meta = json.load(f)

    images = []
    poses = []
    
    camera_angle_x = meta.get('camera_angle_x', 0.0)
    
    frames = meta['frames']
    print(f"Processing {len(frames)} frames...")

    # Determine focal length from the first image (assuming all are same size)
    if frames:
        first_img_path = base_path / (frames[0]['file_path'] + ".png")
        if not first_img_path.exists():
             # Try without extension or different extension if needed, but standard is usually .png
             first_img_path = base_path / frames[0]['file_path']
        
        if first_img_path.exists():
            img = cv2.imread(str(first_img_path))
            H, W = img.shape[:2]
            focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
            if half_res:
                H = H // 2
                W = W // 2
                focal = focal / 2.0
        else:
             raise FileNotFoundError(f"Could not find first image to determine focal length: {first_img_path}")
    else:
        raise ValueError("No frames found in JSON.")

    for i, frame in enumerate(frames):
        fname = base_path / (frame['file_path'] + ".png")
        if not fname.exists():
             fname = base_path / frame['file_path']
        
        if not fname.exists():
            print(f"Warning: Skipping missing file {fname}")
            continue

        # Load Image
        img_tensor = load_image(fname)
        if half_res:
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.permute(2, 0, 1).unsqueeze(0), 
                size=(H, W), 
                mode='area'
            ).squeeze(0).permute(1, 2, 0)
        
        images.append(img_tensor)

        # Load Pose
        pose = np.array(frame['transform_matrix'], dtype=np.float32)
        poses.append(torch.from_numpy(pose))

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(frames)} images...")

    if not images:
        raise ValueError("No images were loaded.")

    # Stack into tensors
    images_tensor = torch.stack(images) # [N, H, W, 3]
    poses_tensor = torch.stack(poses)   # [N, 4, 4]
    focal_tensor = torch.tensor(focal, dtype=torch.float32)

    # Save to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving tensors to {output_path}...")
    torch.save(images_tensor, output_path / "images.pt")
    torch.save(poses_tensor, output_path / "poses.pt")
    torch.save(focal_tensor, output_path / "focal.pt")
    
    print("Conversion complete!")
    print(f"Images: {images_tensor.shape}")
    print(f"Poses: {poses_tensor.shape}")
    print(f"Focal: {focal_tensor.item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Blender/NeRF data to cNeRF format.")
    parser.add_argument("base_dir", type=str, help="Path to the directory containing transforms.json and images.")
    parser.add_argument("output_dir", type=str, help="Path where the .pt files will be saved.")
    parser.add_argument("--split", type=str, default="train", help="Data split to use (train, val, test).")
    parser.add_argument("--half_res", action="store_true", help="Downsample images by 2x.")
    
    args = parser.parse_args()
    
    process_blender_data(args.base_dir, args.output_dir, args.split, args.half_res)
