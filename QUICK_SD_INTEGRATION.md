# Quick Integration: Add SD Toggle to GUI

## Step 1: Install Dependencies

```bash
# Activate your venv
cd D:\Inpainting\Inpaint-Anything
.\.venv\Scripts\activate

# Install Stable Diffusion libraries
pip install diffusers==0.24.0 transformers==4.36.0 accelerate==0.25.0

# HIGHLY RECOMMENDED for speed:
pip install xformers
```

## Step 2: Add SD Toggle to GUI

Add this code to `gui_app.py`:

### A. Add Import (Line ~17, after other imports):

```python
from sd_inpaint_integrated import inpaint_img_with_sd
```

### B. Add Checkbox in Settings (Around line ~500 in InpaintGUI.__init__):

```python
# After the existing checkboxes:
self.use_sd_checkbox = QCheckBox("🎨 Use Stable Diffusion (Better quality, slower)")
self.use_sd_checkbox.setChecked(False)  # Off by default
settings_layout.addWidget(self.use_sd_checkbox)

# Optional: Add prompt input
self.sd_prompt_input = QLineEdit()
self.sd_prompt_input.setPlaceholderText("Optional SD prompt (leave empty for auto)")
self.sd_prompt_input.setVisible(False)
settings_layout.addWidget(QLabel("SD Prompt:"))
settings_layout.addWidget(self.sd_prompt_input)

# Show/hide prompt input based on checkbox
self.use_sd_checkbox.stateChanged.connect(
    lambda: self.sd_prompt_input.setVisible(self.use_sd_checkbox.isChecked())
)
```

### C. Modify InpaintWorker.__init__ (Around line ~26):

```python
def __init__(self, img, mask, lama_config, lama_ckpt, device, 
             use_harmonization=True, use_sd=False, sd_prompt=None):
    super().__init__()
    self.img = img
    self.mask = mask
    self.lama_config = lama_config
    self.lama_ckpt = lama_ckpt
    self.device = device
    self.use_harmonization = use_harmonization
    self.use_sd = use_sd              # NEW
    self.sd_prompt = sd_prompt        # NEW
```

### D. Modify InpaintWorker.run() (Around line ~36):

```python
def run(self):
    try:
        # Step 1: Choose inpainting method
        if self.use_sd:
            # NEW: Use Stable Diffusion
            self.progress.emit("Running Stable Diffusion Inpainting...")
            img_inpainted = inpaint_img_with_sd(
                self.img,
                self.mask,
                prompt=self.sd_prompt if self.sd_prompt else None,
                device=self.device,
                steps=30
            )
        else:
            # Original: Use LaMa
            self.progress.emit("Running LaMa inpainting (original method)...")
            img_inpainted = inpaint_img_with_lama(
                self.img, 
                self.mask, 
                self.lama_config, 
                self.lama_ckpt, 
                mod=8,
                device=self.device
            )
        
        # Step 2: Post-processing (only if using LaMa and harmonization enabled)
        if not self.use_sd and self.use_harmonization:
            self.progress.emit("Analyzing scene structure...")
            img_inpainted = apply_intelligent_context_inpainting(
                self.img, img_inpainted, self.mask
            )
            # ... rest of harmonization code ...
        
        self.progress.emit("Inpainting complete!")
        self.finished.emit(img_inpainted)
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
        self.error.emit(error_msg)
```

### E. Modify the "Remove Object" button handler (Around line ~700):

```python
def on_remove_object(self):
    # ... existing validation code ...
    
    # Start worker with SD settings
    self.worker = InpaintWorker(
        self.img,
        combined_mask,
        self.lama_config,
        self.lama_ckpt,
        self.device,
        use_harmonization=self.intelligent_scene_checkbox.isChecked(),
        use_sd=self.use_sd_checkbox.isChecked(),           # NEW
        sd_prompt=self.sd_prompt_input.text()              # NEW
    )
    
    # ... rest of worker setup ...
```

## Step 3: Test It!

1. Run the GUI:
   ```bash
   python gui_app.py
   ```

2. Load your image (the room with furniture)

3. **Check the "Use Stable Diffusion" checkbox**

4. Segment and remove the object

5. **Wait 10-15 seconds** (first time will download model ~5GB)

6. See MUCH better results!

## Expected Behavior:

### With SD Disabled (LaMa):
- ⚡ Fast (1-2 seconds)
- ❌ Blurry floor/walls
- ❌ "Mirror effect" artifacts

### With SD Enabled:
- 🐌 Slower (10-15 seconds)
- ✅ Clean, coherent walls and floors
- ✅ Semantic understanding
- ✅ No artifacts

---

## Minimal Version (If You Want to Just Replace LaMa Completely):

If you want to **always use SD** and remove LaMa entirely:

### In `gui_app.py`, replace this:

```python
# OLD (Line ~40 in InpaintWorker.run()):
img_inpainted = inpaint_img_with_lama(
    self.img, self.mask, self.lama_config, self.lama_ckpt, 
    mod=8, device=self.device
)

# NEW:
from sd_inpaint_integrated import inpaint_img_with_sd
img_inpainted = inpaint_img_with_sd(
    self.img, self.mask, device=self.device, steps=30
)
```

That's it! Just 3 lines changed.

---

## Testing Script

Create `test_sd_inpaint.py` to test before integrating:

```python
import numpy as np
from PIL import Image
from sd_inpaint_integrated import inpaint_img_with_sd
from utils import load_img_to_array

# Load your test image
img = load_img_to_array("path/to/your/room/image.jpg")
mask = load_img_to_array("path/to/your/mask.png")

print("Testing Stable Diffusion Inpainting...")
print("This will take 10-15 seconds...")

# Test 1: Auto prompt
result_auto = inpaint_img_with_sd(
    img, mask,
    device="cuda",
    steps=30
)
Image.fromarray(result_auto).save("result_auto.jpg")
print("✓ Saved: result_auto.jpg")

# Test 2: Custom prompt
result_custom = inpaint_img_with_sd(
    img, mask,
    prompt="a clean white marble floor and wall, bright indoor lighting",
    device="cuda",
    steps=30
)
Image.fromarray(result_custom).save("result_custom.jpg")
print("✓ Saved: result_custom.jpg")

print("Done! Compare results with original LaMa output.")
```

Run it:
```bash
python test_sd_inpaint.py
```

---

## Summary

**To add SD as an option:**
1. Add import: `from sd_inpaint_integrated import inpaint_img_with_sd`
2. Add checkbox in GUI
3. Add `use_sd` parameter to InpaintWorker
4. Replace inpainting call with conditional

**To replace LaMa entirely:**
1. Add import
2. Replace the `inpaint_img_with_lama()` call with `inpaint_img_with_sd()`

**That's it!** The SD module handles everything else automatically (model loading, optimization, prompting, etc.).

Your indoor scene inpainting will be **dramatically better** with SD! 🚀
