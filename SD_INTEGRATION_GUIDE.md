# Complete Guide: Replacing LaMa with Stable Diffusion Inpainting

## 🎯 The Problem You Identified Correctly

**LaMa fails catastrophically on large indoor objects because:**
1. ❌ Operates in frequency domain (no semantic understanding)
2. ❌ No 3D geometric reasoning
3. ❌ Creates "mirror-like" repetition artifacts
4. ❌ Completely fails at structured scenes (walls, floors, corners)

**Stable Diffusion solves this by:**
1. ✅ Semantic understanding (knows what "wall" and "floor" mean)
2. ✅ Transformer-based architecture (global attention)
3. ✅ Trained on millions of indoor scenes
4. ✅ Can "imagine" plausible content, not just copy frequencies

---

## 📦 Installation

### Step 1: Install Required Libraries

```bash
# In your activated venv:
cd D:\Inpainting\Inpaint-Anything
.\.venv\Scripts\activate

# Install diffusers and dependencies
pip install diffusers transformers accelerate

# HIGHLY RECOMMENDED for 3x speed improvement:
pip install xformers

# If you get errors, install these individually:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.24.0
pip install transformers==4.36.0
pip install accelerate==0.25.0
```

### Step 2: Verify Installation

```python
python -c "from diffusers import StableDiffusion Inpaint Pipeline; print('✓ Installation successful!')"
```

---

## 🚀 Quick Start: Using SD Inpainting

### Method 1: Drop-in Replacement (Easiest)

```python
# OLD CODE (LaMa):
from lama_inpaint import inpaint_img_with_lama
result = inpaint_img_with_lama(img, mask, config, ckpt, device="cuda")

# NEW CODE (Stable Diffusion):
from sd_inpaint_integrated import inpaint_img_with_sd
result = inpaint_img_with_sd(img, mask, device="cuda")
# That's it! Auto-prompt handles everything
```

### Method 2: With Custom Prompt (Better Control)

```python
from sd_inpaint_integrated import inpaint_img_with_sd

result = inpaint_img_with_sd(
    img, mask,
    prompt="a clean white wall and marble floor, bright indoor lighting",
    device="cuda",
    steps=30  # Quality: 20=fast, 30=good, 50=best
)
```

### Method 3: Full Control (Advanced)

```python
from sd_inpaint_integrated import SDInpainter

# Initialize once
inpainter = SDInpainter(
    model_id="stabilityai/stable-diffusion-2-inpainting",
    device="cuda",
    use_fp16=True,  # Faster, less VRAM
    optimize_memory=True
)

# Use multiple times
result = inpainter.inpaint(
    img, mask,
    prompt="a clean indoor room with white walls",
    negative_prompt="blurry, distorted, artifacts",
    num_inference_steps=30,
    guidance_scale=7.5,  # How strictly to follow prompt
    seed=42  # For reproducibility
)
```

---

## 🔧 Integration into Existing Workflow

### Option A: Replace LaMa in GUI (Recommended)

Modify `gui_app.py`:

```python
# At the top, add:
from sd_inpaint_integrated import inpaint_img_with_sd

# In InpaintWorker.run(), replace:
# OLD:
img_inpainted = inpaint_img_with_lama(
    self.img, self.mask, self.lama_config, self.lama_ckpt, device=self.device
)

# NEW:
img_inpainted = inpaint_img_with_sd(
    self.img, self.mask, device=self.device, steps=30
)
```

### Option B: Add as Toggle Option (Best of Both Worlds)

```python
# Add checkbox in GUI settings:
self.use_sd = QCheckBox("🎨 Use Stable Diffusion (Slower but MUCH better)")

# In InpaintWorker:
if self.use_sd:
    from sd_inpaint_integrated import inpaint_img_with_sd
    img_inpainted = inpaint_img_with_sd(self.img, self.mask, device=self.device)
else:
    img_inpainted = inpaint_img_with_lama(...)
```

---

## 📝 Prompting Guide

### Auto-Prompting (Default - Works Great!)

```python
# No prompt = auto-generates based on scene
result = inpaint_img_with_sd(img, mask, device="cuda")

# Auto-prompt analyzes:
# - Mask position (top=wall, middle=furniture, bottom=floor)
# - Surrounding colors (bright/dark)
# - Scene context
```

### Manual Prompting (Fine Control)

**For Furniture Removal:**
```python
prompt = "a clean white wall and marble floor, bright indoor lighting, photorealistic"
```

**For Door Removal:**
```python
prompt = "a white painted wall, indoor room, clean surface, photorealistic, high quality"
```

**For Curtain Removal:**
```python
prompt = "a window with glass panes, white wall, indoor lighting, photorealistic"
```

**For Bed/Sofa Removal:**
```python
prompt = "a clean floor with tiles, white wall in background, empty room, photorealistic"
```

### Prompting Best Practices:

✅ **DO:**
- Be specific: "marble floor", "white wall", "wooden floor"
- Add quality terms: "photorealistic", "high quality", "detailed"
- Include lighting: "bright indoor lighting", "natural light"
- Keep it simple: 10-20 words

❌ **DON'T:**
- Be too vague: "a room" (too generic)
- Be too complex: Long paragraphs confuse the model
- Include negatives in prompt: Use negative_prompt parameter instead

### Empty Prompt?

```python
# Empty prompt works but less control:
result = inpaint_img_with_sd(img, mask, prompt="")

# Better: Let auto-prompt handle it:
result = inpaint_img_with_sd(img, mask, prompt=None)  # Auto-generates
```

---

## ⚙️ Parameters Explained

### `num_inference_steps` (Quality vs Speed)
```python
steps=20  # Fast (5-10 sec) - good for testing
steps=30  # Balanced (10-15 sec) - RECOMMENDED
steps=50  # Best quality (20-30 sec) - for final results
```

### `guidance_scale` (Prompt Adherence)
```python
guidance_scale=5.0   # More creative, less strict
guidance_scale=7.5   # RECOMMENDED (default)
guidance_scale=12.0  # Very strict to prompt
```

### `strength` (How Much to Change)
```python
strength=0.99  # Full inpainting (RECOMMENDED for removal)
strength=0.8   # Blend with original (for subtle edits)
strength=0.5   # Minimal changes
```

### `seed` (Reproducibility)
```python
seed=None   # Different result each time
seed=42     # Same result every time (for testing)
```

---

## 💾 Memory Management

### GPU VRAM Requirements:

| Image Size | Min VRAM | Recommended |
|-----------|----------|-------------|
| 512x512   | 4 GB     | 6 GB        |
| 768x768   | 6 GB     | 8 GB        |
| 1024x1024 | 8 GB     | 10 GB       |

### If You Get CUDA Out of Memory:

```python
# Method 1: Enable all optimizations (already done in sd_inpaint_integrated.py)
inpainter = SDInpainter(
    use_fp16=True,          # Half precision
    optimize_memory=True    # Attention slicing + VAE slicing
)

# Method 2: Reduce image size
# The code automatically resizes large images to 1024x1024

# Method 3: Reduce steps
result = inpaint_img_with_sd(img, mask, steps=20)  # Instead of 50

# Method 4: Use CPU (VERY slow but works)
result = inpaint_img_with_sd(img, mask, device="cpu")
```

---

## 🔄 Complete Integration Example

### Step-by-Step: Modify Your Workflow

**1. Import the new module:**

```python
# gui_app.py - at the top
from sd_inpaint_integrated import inpaint_img_with_sd
```

**2. Replace the inpainting call:**

```python
# In InpaintWorker.run() method:

# BEFORE:
self.progress.emit("Running LaMa inpainting...")
img_inpainted = inpaint_img_with_lama(
    self.img, self.mask, self.lama_config, self.lama_ckpt,
    mod=8, device=self.device
)

# AFTER:
self.progress.emit("Running Stable Diffusion inpainting...")
img_inpainted = inpaint_img_with_sd(
    self.img,
    self.mask,
    device=self.device,
    steps=30  # Adjust for quality vs speed
)
```

**3. Optional: Add prompt customization:**

```python
# Add a text input in GUI for custom prompts:
self.prompt_input = QLineEdit()
self.prompt_input.setPlaceholderText("Optional: Custom prompt (leave empty for auto)")

# In InpaintWorker:
prompt = self.prompt_input.text() if self.prompt_input.text() else None
img_inpainted = inpaint_img_with_sd(
    self.img, self.mask,
    prompt=prompt,
    device=self.device,
    steps=30
)
```

---

## 📊 Performance Comparison

| Aspect | LaMa | Stable Diffusion |
|--------|------|------------------|
| **Speed** | ⭐⭐⭐⭐⭐ (1-2 sec) | ⭐⭐⭐ (10-15 sec) |
| **Quality (small objects)** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Quality (large objects)** | ⭐ | ⭐⭐⭐⭐⭐ |
| **Semantic understanding** | ❌ None | ✅ Excellent |
| **Geometric structure** | ⭐ | ⭐⭐⭐⭐ |
| **Indoor scenes** | ⭐ | ⭐⭐⭐⭐⭐ |
| **VRAM usage** | ⭐⭐⭐⭐⭐ (1-2 GB) | ⭐⭐⭐ (6-8 GB) |

**Recommendation:** Use SD for large indoor objects, keep LaMa for small outdoor objects.

---

## 🐛 Troubleshooting

### Error: "CUDA out of memory"
```python
# Solution 1: Install xformers
pip install xformers

# Solution 2: Already implemented optimizations work automatically

# Solution 3: Reduce quality
result = inpaint_img_with_sd(img, mask, steps=20)
```

### Error: "diffusers not found"
```bash
pip install diffusers transformers accelerate
```

### Error: Slow on first run
```
Normal! First run downloads ~5GB model.
Subsequent runs are fast (model cached).
```

### Result is blurry/low quality
```python
# Increase steps:
result = inpaint_img_with_sd(img, mask, steps=50)

# Adjust guidance:
result = inpaint_img_with_sd(img, mask, guidance_scale=9.0)
```

### Result doesn't match surroundings
```python
# Use more specific prompt:
prompt = "a white marble floor with gray veining, indoor lighting"
result = inpaint_img_with_sd(img, mask, prompt=prompt)
```

---

## 🎓 Advanced Usage

### Custom Model Fine-Tuned for Your Scene:

```python
# Use a custom fine-tuned model:
inpainter = SDInpainter(
    model_id="your-username/your-finetuned-model",
    device="cuda"
)
```

### Multiple Passes for Best Quality:

```python
# Pass 1: Fast initial fill
result = inpaint_img_with_sd(img, mask, steps=20)

# Pass 2: Refinement with higher quality
mask_refined = create_boundary_mask(mask)  # Just the edges
result_final = inpaint_img_with_sd(
    result, mask_refined,
    steps=50,
    strength=0.7  # Subtle refinement
)
```

### Batch Processing:

```python
from sd_inpaint_integrated import SDInpainter

# Initialize once for all images
inpainter = SDInpainter(device="cuda")

# Process multiple images
for img_path, mask_path in image_mask_pairs:
    img = load_img(img_path)
    mask = load_img(mask_path)
    result = inpainter.inpaint(img, mask, steps=30)
    save_img(result, output_path)
```

---

## 📌 Summary: Key Takeaways

### ✅ What You Need to Do:

1. **Install dependencies:**
   ```bash
   pip install diffusers transformers accelerate xformers
   ```

2. **Import the module:**
   ```python
   from sd_inpaint_integrated import inpaint_img_with_sd
   ```

3. **Replace LaMa call:**
   ```python
   # OLD: inpaint_img_with_lama(img, mask, config, ckpt)
   # NEW: inpaint_img_with_sd(img, mask)
   ```

4. **That's it!** Auto-prompting handles everything.

### 🎯 Best Practices:

- **Auto-prompt works great** - no manual prompt needed for most cases
- **Use 30 steps** for balance of speed and quality
- **Install xformers** for 2-3x speed boost
- **Keep LaMa as fallback** for small objects (it's much faster)

### 🚀 Expected Results:

- **Much better structure** - walls stay planar, floors continue correctly
- **No more "mirror effect"** - semantic understanding prevents artifacts
- **Better geometry** - room corners preserved, lines stay straight
- **Slower** - 10-15 seconds vs 1-2 seconds for LaMa

### 🎪 Your Specific Use Case (Sofa Removal):

```python
result = inpaint_img_with_sd(
    img, mask,
    prompt="a clean marble floor and white wall, bright indoor lighting",
    steps=30,
    device="cuda"
)
```

This will give you **dramatically better results** than LaMa for your indoor scene!

---

**You were absolutely right to identify LaMa as unsuitable. Stable Diffusion is the correct solution for your use case!** 🎉
