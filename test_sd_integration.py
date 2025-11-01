"""
Quick test to verify Stable Diffusion integration works
"""
import numpy as np
from sd_inpaint_integrated import inpaint_img_with_sd

print("🧪 Testing SD Integration...")
print("=" * 60)

# Create a simple test image and mask
print("\n1. Creating test image (256x256 RGB)...")
test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

print("2. Creating test mask (center square)...")
test_mask = np.zeros((256, 256), dtype=np.uint8)
test_mask[64:192, 64:192] = 255  # Center square

print(f"   Image shape: {test_img.shape}")
print(f"   Mask shape: {test_mask.shape}")
print(f"   Mask coverage: {(test_mask > 0).sum() / test_mask.size * 100:.1f}%")

try:
    print("\n3. Running SD inpainting...")
    print("   ⏱️ This will take 10-20 seconds (first time downloads ~5GB model)")
    print("   Note: Using 'cpu' for testing, will be slow!")
    
    result = inpaint_img_with_sd(
        test_img,
        test_mask,
        prompt="a clean white wall",
        device="cpu",  # Use CPU for testing (GPU will be faster)
        steps=10  # Reduced steps for faster testing
    )
    
    print(f"\n✅ SUCCESS! SD inpainting completed!")
    print(f"   Result shape: {result.shape}")
    print(f"   Result dtype: {result.dtype}")
    print(f"   Result range: [{result.min()}, {result.max()}]")
    
    print("\n" + "=" * 60)
    print("🎉 SD integration is working correctly!")
    print("\nYou can now:")
    print("1. Open the GUI: python gui_app.py")
    print("2. Load an indoor scene image")
    print("3. Segment a large object")
    print("4. Check 'Use Stable Diffusion' checkbox")
    print("5. Click 'Remove Object'")
    print("6. Enjoy MUCH better results!")
    
except ImportError as e:
    print(f"\n❌ IMPORT ERROR: {e}")
    print("\nMissing dependencies. Install with:")
    print("   pip install diffusers transformers accelerate xformers")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nCheck the error message above for details.")
    import traceback
    traceback.print_exc()
