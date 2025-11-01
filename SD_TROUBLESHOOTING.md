# 🔧 Stable Diffusion Checkbox Troubleshooting

## ✅ Fixed Issue #1: Layout Overlap

**Problem:** Device label was overlapping with SD checkbox (both at row 3)
**Solution:** Moved device label to row 5

The GUI has been updated and should now display correctly.

## 🧪 Testing the SD Integration

### Quick Test (Without GUI):

Run this to verify SD works standalone:
```bash
python test_sd_integration.py
```

This will:
- Create a test image and mask
- Run SD inpainting
- Verify the integration is working
- Take 10-20 seconds (first time downloads ~5GB model)

### Full GUI Test:

1. **Restart the GUI:**
   ```bash
   .\.venv\Scripts\python.exe gui_app.py
   ```

2. **Look for the checkbox:**
   - Should see: "🎨 Use Stable Diffusion (Better for Indoor Scenes)"
   - Purple colored text
   - Below "🧠 Intelligent Scene Understanding"

3. **Test with an image:**
   ```
   Load Image → Segment Object → Check SD box → Remove Object
   ```

## 🐛 Common Issues & Solutions

### Issue 1: "Checkbox is there but nothing happens when I click Remove"

**Possible causes:**
- Missing dependencies
- Model download failed
- GPU out of memory
- Error in terminal output

**Check terminal output for errors:**
Look for messages like:
- "❌ diffusers library not installed!"
- "CUDA out of memory"
- "Model download failed"

**Solutions:**
```bash
# Install missing dependencies
pip install diffusers==0.24.0 transformers==4.36.0 accelerate==0.25.0

# If CUDA memory error, install xformers:
pip install xformers

# Or use CPU (slower but works):
# Edit line 48 in sd_inpaint_integrated.py:
# device = "cpu"  # Force CPU
```

### Issue 2: "First use takes forever"

**This is NORMAL!** First time:
- Downloads ~5GB Stable Diffusion model
- Takes 2-5 minutes depending on internet speed
- Shows progress in terminal
- Subsequent uses are fast (~10-20 seconds)

**Watch the terminal for:**
```
Downloading model...
Fetching 14 files: 100%|███████| 14/14
Loading model...
✓ Model loaded successfully!
```

### Issue 3: "Progress bar freezes at 'Running SD Inpainting'"

**Expected behavior:**
- GUI will appear frozen for 10-20 seconds
- This is normal - SD is processing
- Watch terminal for progress messages
- First use: Add 2-5 min for model download

### Issue 4: "Error: CUDA out of memory"

**Solutions:**
1. **Install xformers (reduces VRAM by 40%):**
   ```bash
   pip install xformers
   ```

2. **Use smaller image:**
   - Resize image to max 1024x1024 before loading

3. **Reduce SD steps:**
   Edit `gui_app.py` line 52:
   ```python
   steps=20  # Change from 30 to 20
   ```

4. **Force CPU mode (slow but works):**
   Edit `gui_app.py` line 25 in worker:
   ```python
   device="cpu"  # Force CPU
   ```

### Issue 5: "Results are blurry/bad even with SD"

**Check:**
- Is the checkbox actually CHECKED? (should be purple/highlighted)
- Look for "Running Stable Diffusion Inpainting" message in terminal
- If you see "Running LaMa inpainting" - SD is NOT being used

**Verify SD is active:**
- Terminal should show: "🎨 Running Stable Diffusion Inpainting..."
- NOT: "Running LaMa inpainting (original method)..."

### Issue 6: "Checkbox doesn't appear at all"

**Solutions:**
1. **Make sure you saved the file:**
   - Save `gui_app.py`
   - Restart GUI

2. **Check for syntax errors:**
   ```bash
   python -m py_compile gui_app.py
   ```

3. **Verify the code is correct:**
   ```bash
   grep -n "Use Stable Diffusion" gui_app.py
   ```
   Should show line ~392

## 📊 How to Verify SD is Working

### Terminal Messages (LaMa):
```
Running LaMa inpainting (original method)...
Analyzing scene structure...
Extending floor and wall patterns...
```

### Terminal Messages (SD - What you WANT to see):
```
🎨 Running Stable Diffusion Inpainting (semantic, structure-aware)...
⏱️ This takes 10-20 seconds (first time: model download ~5GB)...
Loading model...
✓ Model loaded successfully!
Running inference...
✓ Stable Diffusion inpainting complete!
```

## 🎯 Expected Behavior

### When SD Checkbox is UNCHECKED (Default):
- Uses LaMa (fast, 1-2 seconds)
- "Intelligent Scene Understanding" checkbox is enabled
- Good for small objects, outdoor scenes
- May have artifacts on large indoor objects

### When SD Checkbox is CHECKED:
- Uses Stable Diffusion (slower, 10-20 seconds)
- "Intelligent Scene Understanding" automatically DISABLED (grayed out)
- SD doesn't need post-processing enhancements
- MUCH better for large indoor objects
- Optional prompt field appears below checkbox

## 🧪 Debugging Commands

### Test if SD can import:
```bash
python -c "from sd_inpaint_integrated import inpaint_img_with_sd; print('✅ SD import OK')"
```

### Test if diffusers is installed:
```bash
pip show diffusers transformers accelerate
```

### Test minimal SD run:
```bash
python test_sd_integration.py
```

### Check Python version (should be 3.10 or 3.11):
```bash
python --version
```

### Check CUDA availability:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 💡 Pro Tips

1. **Watch the terminal while GUI is running**
   - Shows real-time progress
   - Shows errors immediately
   - Shows which backend is actually being used

2. **First SD use is slow**
   - Downloads model (~5GB)
   - Be patient, it's one-time only
   - Subsequent uses are 10-20 seconds

3. **Compare LaMa vs SD**
   - Process same image with SD unchecked (LaMa)
   - Then with SD checked
   - See the HUGE quality difference!

4. **Use custom prompts for fine control**
   - "a clean white wall and marble floor, bright lighting"
   - "empty room with wooden floor, indoor"
   - Or leave empty for auto-prompt (recommended)

## ✅ Final Checklist

Before reporting issues, verify:
- [ ] GUI restarted after saving changes
- [ ] Dependencies installed: `diffusers transformers accelerate`
- [ ] Checkbox is visible (purple text, row 3)
- [ ] Checkbox is actually CHECKED when removing object
- [ ] Terminal is open to see progress/errors
- [ ] First use: waited for model download (2-5 min)
- [ ] GPU has 6-8GB VRAM or using CPU mode
- [ ] Python 3.10 or 3.11 (not 3.13!)

## 🆘 Still Not Working?

**Share this information:**
1. Terminal output (copy/paste errors)
2. Python version: `python --version`
3. GPU info: `nvidia-smi` (if CUDA)
4. Package versions:
   ```bash
   pip show diffusers transformers torch
   ```
5. What happens when you:
   - Check the SD box
   - Click Remove Object
   - What messages appear in terminal?

---

**Most Common Issue:** First use takes 2-5 minutes for model download. Just be patient! 🎉
