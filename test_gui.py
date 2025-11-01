"""
Quick Test Script for GUI Application

This script verifies that the GUI components work correctly.
Run this to ensure all dependencies are properly loaded.
"""

import sys
import torch
import numpy as np
from pathlib import Path

print("=" * 60)
print("Inpaint-Anything GUI - Dependency Check")
print("=" * 60)

# Check PyQt5
try:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QPixmap, QImage
    print("✓ PyQt5 imported successfully")
except ImportError as e:
    print(f"✗ PyQt5 import failed: {e}")
    sys.exit(1)

# Check SAM
try:
    from sam_segment import predict_masks_with_sam
    print("✓ SAM segment module loaded")
except ImportError as e:
    print(f"✗ SAM import failed: {e}")
    sys.exit(1)

# Check LaMa
try:
    from lama_inpaint import inpaint_img_with_lama
    print("✓ LaMa inpaint module loaded")
except ImportError as e:
    print(f"✗ LaMa import failed: {e}")
    sys.exit(1)

# Check Utils
try:
    from utils import load_img_to_array, save_array_to_img, dilate_mask
    print("✓ Utils module loaded")
except ImportError as e:
    print(f"✗ Utils import failed: {e}")
    sys.exit(1)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ PyTorch device: {device.upper()}")

# Check model files
sam_ckpt = Path("./weights/mobile_sam.pt")
lama_ckpt = Path("./pretrained_models/big-lama")
lama_config = Path("./lama/configs/prediction/default.yaml")

if sam_ckpt.exists():
    print(f"✓ SAM checkpoint found: {sam_ckpt}")
else:
    print(f"✗ SAM checkpoint missing: {sam_ckpt}")

if lama_ckpt.exists():
    print(f"✓ LaMa checkpoint found: {lama_ckpt}")
else:
    print(f"✗ LaMa checkpoint missing: {lama_ckpt}")

if lama_config.exists():
    print(f"✓ LaMa config found: {lama_config}")
else:
    print(f"✗ LaMa config missing: {lama_config}")

# Check example images
example_dir = Path("./example/remove-anything")
if example_dir.exists():
    example_images = list(example_dir.glob("*.jpg")) + list(example_dir.glob("*.png"))
    print(f"✓ Found {len(example_images)} example images")
else:
    print("✗ Example directory not found")

# Test numpy to QImage conversion (the fix we applied)
print("\n" + "=" * 60)
print("Testing NumPy to QImage conversion...")
print("=" * 60)

try:
    # Create a test array
    test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Convert to QImage using tobytes() (the fix)
    height, width, channel = test_array.shape
    bytes_per_line = 3 * width
    q_image = QImage(test_array.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
    
    if not q_image.isNull():
        print("✓ NumPy to QImage conversion working correctly!")
    else:
        print("✗ QImage is null - conversion failed")
except Exception as e:
    print(f"✗ Conversion test failed: {e}")

print("\n" + "=" * 60)
print("All checks completed!")
print("=" * 60)
print("\nYou can now run the GUI with:")
print("  python gui_app.py")
print("or double-click:")
print("  launch_gui.bat")
print("=" * 60)
