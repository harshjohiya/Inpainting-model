# 🛠️ GUI Launch Troubleshooting Guide

## 🚨 Current Issue: PyTorch + Python 3.13 Compatibility

Your system is running **Python 3.13.5**, which has known compatibility issues with PyTorch/torchvision. This is causing the GUI to fail to launch.

### Error Message:
```
RuntimeError: resource deadlock would occur
```

This occurs when importing `segment_anything` because of torchvision incompatibility with Python 3.13.

## ✅ Solution: Downgrade to Python 3.11

Python 3.13 is very new and many deep learning libraries haven't caught up yet. **Python 3.11** is the recommended version for PyTorch/CUDA projects.

### Step 1: Install Python 3.11

**Download Python 3.11.x:**
https://www.python.org/downloads/release/python-3119/
- Choose "Windows installer (64-bit)"
- During installation, **CHECK "Add Python to PATH"**

### Step 2: Create New Virtual Environment with Python 3.11

```bash
# Navigate to project
cd D:\Inpainting\Inpaint-Anything

# Remove old virtual environment
Remove-Item -Recurse -Force .venv

# Create new venv with Python 3.11
py -3.11 -m venv .venv

# Activate it
.\.venv\Scripts\activate
```

### Step 3: Reinstall All Dependencies

```bash
# Make sure you're in the virtual environment
# You should see (.venv) in your terminal prompt

# Install PyTorch first (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install segment-anything
cd segment_anything
pip install -e .
cd ..

# Install other core dependencies
pip install opencv-python pillow numpy matplotlib PyQt5 PyYAML

# Install LaMa dependencies (if needed)
pip install omegaconf easydict scikit-image kornia webdataset torch-fidelity albumentations wldhx.yadisk-direct

# Install Stable Diffusion dependencies
pip install diffusers==0.24.0
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install xformers  # Optional but HIGHLY recommended for speed
```

### Step 4: Test Launch

```bash
# From D:\Inpainting\Inpaint-Anything
python gui_app.py
```

## 🎯 Quick Test: Verify PyTorch Works

After reinstalling with Python 3.11, test:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from segment_anything import SamPredictor; print('✅ SAM import successful')"
```

Expected output:
```
PyTorch: 2.x.x+cu118
CUDA: True
✅ SAM import successful
```

## 🔍 Alternative: Use Python 3.10 or 3.11

If you already have Python 3.10 or 3.11 installed:

```bash
# Check which Python versions you have
py -0

# Output example:
# -3.13-64 *
# -3.11-64
# -3.10-64

# Create venv with specific version
py -3.11 -m venv .venv
# or
py -3.10 -m venv .venv
```

## 📝 Why This Happens

**Python 3.13 Changes:**
- New GIL (Global Interpreter Lock) implementation
- Changes to C API
- Threading model changes

**PyTorch/torchvision:**
- Compiled C++ extensions need updates
- torchvision `_meta_registrations.py` deadlocks on 3.13
- Most DL libraries target Python 3.8-3.11

**Recommended Versions:**
- ✅ Python 3.10.x - Most stable for DL
- ✅ Python 3.11.x - Good compatibility, faster
- ⚠️ Python 3.12.x - Some compatibility issues
- ❌ Python 3.13.x - Too new, many incompatibilities

## 🚀 After Fixing Python Version

Once you're on Python 3.11:

1. Launch GUI: `python gui_app.py`
2. Look for "Use Stable Diffusion" checkbox ✨
3. Load image with large indoor object
4. Segment it
5. **Check "Use Stable Diffusion"**
6. Click "Remove Object"
7. Wait 10-20 seconds (first time downloads model)
8. See MUCH better results!

## 💡 Pro Tip: Virtual Environment Best Practices

Always use virtual environments for DL projects:

```bash
# Create
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Or (Windows CMD)
.\.venv\Scripts\activate.bat

# Deactivate
deactivate
```

## 📦 Full Dependency List (for Python 3.11)

Core:
- torch, torchvision, torchaudio (CUDA 11.8)
- opencv-python
- pillow
- numpy
- matplotlib
- PyQt5
- PyYAML

Inpainting:
- segment-anything (local install)
- omegaconf
- easydict
- scikit-image
- kornia

Stable Diffusion:
- diffusers==0.24.0
- transformers==4.36.0
- accelerate==0.25.0
- xformers (optional but recommended)

## ❓ Still Having Issues?

### Issue: "CUDA out of memory"
**Solution:** Install xformers or use LaMa instead of SD

### Issue: "No module named 'diffusers'"
**Solution:** `pip install diffusers transformers accelerate`

### Issue: SAM checkpoint not found
**Solution:** Download SAM checkpoint:
```bash
cd pretrained_models
# Download sam_vit_h_4b8939.pth from:
# https://github.com/facebookresearch/segment-anything#model-checkpoints
```

### Issue: GUI doesn't show SD checkbox
**Solution:** Make sure you saved `gui_app.py` with the latest changes and restarted

## ✅ Summary

**TLDR: Use Python 3.11, not 3.13**

```bash
# 1. Install Python 3.11
# 2. Create new venv
py -3.11 -m venv .venv
.\.venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
cd segment_anything; pip install -e .; cd ..
pip install opencv-python pillow numpy matplotlib PyQt5 PyYAML
pip install diffusers transformers accelerate xformers

# 4. Launch
python gui_app.py
```

Your SD integration is complete - you just need the right Python version! 🎉
