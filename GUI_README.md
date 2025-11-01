# Inpaint-Anything GUI

A user-friendly graphical interface for the Inpaint-Anything project that allows you to remove objects from images with just a few clicks!

## 🚀 Quick Start

### Windows
Simply double-click `launch_gui.bat`

### Command Line
```bash
python gui_app.py
```

Or with the virtual environment:
```bash
D:/Inpainting/Inpaint-Anything/.venv/Scripts/python.exe gui_app.py
```

## 📖 How to Use

1. **Load Image**: Click "📁 Load Image" to select your photo
2. **Select Object**: Click directly on the object you want to remove in the image
3. **Segment Object**: Click "🎯 Segment Object" to run SAM segmentation
4. **Choose Mask**: Select the best mask from the dropdown (Mask 0, 1, or 2)
5. **Adjust Settings**: Optionally adjust the "Dilate Size" to expand/shrink the mask
6. **Remove Object**: Click "🗑️ Remove Object" to inpaint and remove the selected object
7. **Save Result**: Click "💾 Save Result" to save your edited image

## ✨ Features

- **Interactive Point Selection**: Click directly on objects to select them
- **Real-time Preview**: See segmentation masks and results instantly
- **Multiple Mask Options**: Choose from 3 different SAM-generated masks
- **Adjustable Dilation**: Fine-tune the mask size for better results
- **Background Processing**: UI remains responsive during processing
- **Easy Save**: Export results with one click

## 🎨 Interface Layout

### Left Panel (Controls)
- **Load Image Button**: Browse and load images
- **Instructions**: Step-by-step guide
- **Settings**: Mask selection and dilation controls
- **Action Buttons**: Segment, Remove, Save, and Reset
- **Status Display**: Real-time progress and messages

### Right Panel (Display)
- **Top Left**: Original image with selected point
- **Top Right**: Segmentation mask preview
- **Bottom**: Final inpainted result

## ⚙️ Settings

- **Select Mask**: Choose which of the 3 SAM masks to use
- **Dilate Size**: Expand the mask by 0-50 pixels (default: 15)
- **Device**: Automatically uses CUDA if available, otherwise CPU

## 🎯 Tips for Best Results

1. Click on the center of the object you want to remove
2. Try different masks if the first one doesn't capture the entire object
3. Increase dilate size if edges are visible after inpainting
4. For complex objects, you may need to click multiple times and re-segment

## 🔧 Technical Details

- **SAM Model**: MobileSAM (vit_t) for fast segmentation
- **Inpainting Model**: LaMa (big-lama) for high-quality inpainting
- **GUI Framework**: PyQt5
- **Processing**: Multi-threaded to keep UI responsive

## 🐛 Troubleshooting

**Issue**: GUI doesn't launch
- Make sure PyQt5 is installed: `pip install PyQt5`
- Check that the virtual environment is activated

**Issue**: Segmentation/Inpainting fails
- Ensure model checkpoints are in the correct locations:
  - SAM: `./weights/mobile_sam.pt`
  - LaMa: `./pretrained_models/big-lama/`

**Issue**: Out of memory
- Use CPU mode instead of CUDA for large images
- Reduce image size before loading

## 📝 Requirements

All requirements are already installed in your virtual environment:
- PyTorch
- PyQt5
- segment-anything
- LaMa dependencies

## 🎉 Examples

The GUI works with any of the example images in `./example/remove-anything/`:
- Dog
- Cat
- Person
- Bridge
- Boat
- Baseball

Try them out to see the power of Inpaint-Anything!
