"""
Test script for structure-aware inpainting
Demonstrates line detection, vanishing point estimation, and geometric structure
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from structure_aware_inpaint import StructureAwareInpainter, enhance_lama_for_indoor_scenes


def visualize_structure_detection(img_path: str, mask_path: str = None):
    """
    Visualize the geometric structure detection on an image
    
    Args:
        img_path: Path to the image
        mask_path: Path to the mask (optional)
    """
    # Load image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create or load mask
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        # Create a dummy mask in the center
        h, w = img_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h//3:2*h//3, w//3:2*w//3] = 255
    
    # Detect structure
    inpainter = StructureAwareInpainter()
    structure = inpainter.detect_geometric_structure(img_rgb, mask)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Original image with mask
    ax = axes[0, 0]
    ax.imshow(img_rgb)
    mask_overlay = np.zeros_like(img_rgb)
    mask_overlay[mask > 127] = [255, 0, 0]
    ax.imshow(mask_overlay, alpha=0.3)
    ax.set_title('Original Image + Mask', fontsize=14)
    ax.axis('off')
    
    # 2. Detected lines
    ax = axes[0, 1]
    img_lines = img_rgb.copy()
    for line in structure.lines:
        rho, theta = line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    ax.imshow(img_lines)
    ax.set_title(f'Detected Lines ({len(structure.lines)} lines)', fontsize=14)
    ax.axis('off')
    
    # 3. Vanishing points
    ax = axes[1, 0]
    img_vp = img_rgb.copy()
    for vp in structure.vanishing_points:
        x, y = int(vp[0]), int(vp[1])
        # Draw only if within reasonable bounds
        h, w = img_vp.shape[:2]
        if -w < x < 2*w and -h < y < 2*h:
            cv2.circle(img_vp, (x, y), 15, (255, 0, 255), -1)
            cv2.putText(img_vp, 'VP', (x+20, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 0, 255), 2)
    
    ax.imshow(img_vp)
    ax.set_title(f'Vanishing Points ({len(structure.vanishing_points)} points)', fontsize=14)
    ax.axis('off')
    
    # 4. Dominant angles histogram
    ax = axes[1, 1]
    if len(structure.lines) > 0:
        angles_deg = [np.rad2deg(line[1]) for line in structure.lines]
        ax.hist(angles_deg, bins=36, color='blue', alpha=0.7, edgecolor='black')
        
        # Mark dominant angles
        for dominant_angle in structure.dominant_angles:
            ax.axvline(np.rad2deg(dominant_angle), color='red', 
                      linestyle='--', linewidth=2, label='Dominant')
        
        ax.set_xlabel('Angle (degrees)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Line Orientation Distribution', fontsize=14)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No lines detected', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Print structure info
    print("=" * 60)
    print("GEOMETRIC STRUCTURE ANALYSIS")
    print("=" * 60)
    print(f"Lines detected: {len(structure.lines)}")
    print(f"Vanishing points: {len(structure.vanishing_points)}")
    print(f"Dominant angles: {len(structure.dominant_angles)}")
    print()
    
    if len(structure.dominant_angles) > 0:
        print("Dominant Orientations:")
        for i, angle in enumerate(structure.dominant_angles):
            print(f"  {i+1}. {np.rad2deg(angle):.1f}°")
        print()
    
    if len(structure.vanishing_points) > 0:
        print("Vanishing Points:")
        for i, vp in enumerate(structure.vanishing_points):
            print(f"  {i+1}. ({vp[0]:.1f}, {vp[1]:.1f})")
        print()
    
    print("Detected Planes:")
    for plane in structure.planes:
        print(f"  - {plane['type']}: {plane['orientation']}")
    print("=" * 60)
    
    plt.show()
    
    return structure


def test_on_example_images():
    """Test on example images if available"""
    example_dir = Path("example")
    
    # Look for example images
    image_patterns = ["*.jpg", "*.png", "*.jpeg"]
    
    for pattern in image_patterns:
        for img_path in example_dir.rglob(pattern):
            print(f"\nProcessing: {img_path}")
            try:
                visualize_structure_detection(str(img_path))
                break  # Just test first image found
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue


def create_synthetic_test():
    """Create a synthetic room scene for testing"""
    # Create a simple room with door
    img = np.ones((600, 800, 3), dtype=np.uint8) * 200  # Gray wall
    
    # Add floor (darker)
    img[400:, :] = [150, 140, 130]
    
    # Add vertical lines (wall edges)
    cv2.line(img, (100, 0), (100, 400), (100, 100, 100), 3)
    cv2.line(img, (700, 0), (700, 400), (100, 100, 100), 3)
    
    # Add horizontal line (floor boundary)
    cv2.line(img, (0, 400), (800, 400), (80, 80, 80), 4)
    
    # Add door outline (to be removed)
    cv2.rectangle(img, (350, 150), (450, 400), (50, 50, 50), 2)
    
    # Create mask for door
    mask = np.zeros((600, 800), dtype=np.uint8)
    mask[150:400, 350:450] = 255
    
    # Save synthetic image
    cv2.imwrite('synthetic_room.png', img)
    cv2.imwrite('synthetic_mask.png', mask)
    
    print("Created synthetic room scene: synthetic_room.png")
    print("Testing structure detection...")
    
    # Test
    visualize_structure_detection('synthetic_room.png', 'synthetic_mask.png')


if __name__ == "__main__":
    print("Structure-Aware Inpainting Test")
    print("=" * 60)
    print()
    print("This script tests the geometric structure detection:")
    print("  - Line detection (Hough Transform)")
    print("  - Vanishing point estimation")
    print("  - Dominant angle detection")
    print("  - Plane classification")
    print()
    
    # Try example images first
    import os
    if os.path.exists("example"):
        print("Testing on example images...")
        test_on_example_images()
    else:
        print("No example directory found. Creating synthetic test...")
        create_synthetic_test()
