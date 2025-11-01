import sys
import torch
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QSpinBox, QMessageBox, QProgressBar,
                             QComboBox, QGroupBox, QGridLayout, QCheckBox)
from PyQt5.QtCore import Qt, QPoint, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont

from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask
from intelligent_harmonizer import intelligent_harmonize
from advanced_inpainting import ContextAwareInpainter, refine_sam_mask
from context_intelligence import apply_intelligent_context_inpainting
from structure_aware_inpaint import enhance_lama_for_indoor_scenes
from aggressive_indoor_inpaint import super_aggressive_indoor_inpaint
from sd_inpaint_integrated import inpaint_img_with_sd


class InpaintWorker(QThread):
    """Worker thread for running inpainting in background"""
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, img, mask, lama_config, lama_ckpt, device, use_harmonization=True, use_sd=False, sd_prompt=None):
        super().__init__()
        self.img = img
        self.mask = mask
        self.lama_config = lama_config
        self.lama_ckpt = lama_ckpt
        self.device = device
        self.use_harmonization = use_harmonization
        self.use_sd = use_sd
        self.sd_prompt = sd_prompt
        
    def run(self):
        try:
            # Step 1: Choose inpainting backend
            if self.use_sd:
                # NEW: Use Stable Diffusion Inpainting
                self.progress.emit("🎨 Running Stable Diffusion Inpainting (semantic, structure-aware)...")
                self.progress.emit("⏱️ This takes 10-20 seconds (first time: model download ~5GB)...")
                
                img_inpainted = inpaint_img_with_sd(
                    self.img,
                    self.mask,
                    prompt=self.sd_prompt if self.sd_prompt else None,
                    device=self.device,
                    steps=30  # Good balance of quality and speed
                )
                
                self.progress.emit("✓ Stable Diffusion inpainting complete!")
            else:
                # Original: LaMa inpainting
                self.progress.emit("Running LaMa inpainting (original method)...")
                img_inpainted = inpaint_img_with_lama(
                    self.img, 
                    self.mask, 
                    self.lama_config, 
                    self.lama_ckpt, 
                    mod=8,  # ALWAYS 8!
                    device=self.device
                )
            
            # Step 2: Post-processing enhancements (only for LaMa)
            if not self.use_sd and self.use_harmonization:
                self.progress.emit("Analyzing scene structure (floor, walls, edges)...")
                # Use intelligent context system that understands the scene
                img_inpainted = apply_intelligent_context_inpainting(
                    self.img,
                    img_inpainted,
                    self.mask
                )
                
                self.progress.emit("Extending floor and wall patterns intelligently...")
                # Additional refinement with the advanced inpainter
                inpainter = ContextAwareInpainter()
                img_inpainted = inpainter.enhance_lama_result(
                    self.img,
                    img_inpainted,
                    self.mask,
                    use_advanced=True
                )
                
                # Step 3: NEW - Structure-aware enhancement for indoor scenes
                self.progress.emit("Detecting lines, vanishing points, and 3D structure...")
                img_inpainted = enhance_lama_for_indoor_scenes(
                    self.img,
                    img_inpainted,
                    self.mask,
                    use_structure_detection=True
                )
                
                # Step 4: SUPER AGGRESSIVE - Direct texture copying (bypasses LaMa's failures)
                self.progress.emit("Applying AGGRESSIVE floor/wall reconstruction...")
                img_inpainted = super_aggressive_indoor_inpaint(
                    self.img,
                    img_inpainted,
                    self.mask
                )
            
            self.progress.emit("Inpainting complete!")
            self.finished.emit(img_inpainted)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class SegmentWorker(QThread):
    """Worker thread for running segmentation in background"""
    finished = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)
    
    def __init__(self, img, point_coords, point_labels, sam_model_type, sam_ckpt, device, refine_masks=True):
        super().__init__()
        self.img = img
        self.point_coords = point_coords  # Now can be multiple points
        self.point_labels = point_labels
        self.sam_model_type = sam_model_type
        self.sam_ckpt = sam_ckpt
        self.device = device
        self.refine_masks = refine_masks
        
    def run(self):
        try:
            self.progress.emit(f"Running SAM segmentation with {len(self.point_coords)} point(s)...")
            masks, scores, logits = predict_masks_with_sam(
                self.img,
                self.point_coords,  # Pass all points
                self.point_labels,
                model_type=self.sam_model_type,
                ckpt_p=self.sam_ckpt,
                device=self.device,
            )
            
            # Refine masks for better accuracy
            if self.refine_masks:
                self.progress.emit("Refining masks with GrabCut...")
                refined_masks = []
                for i, mask in enumerate(masks):
                    refined = refine_sam_mask(self.img, (mask * 255).astype(np.uint8), use_grabcut=True)
                    refined_masks.append(refined.astype(bool))
                masks = np.array(refined_masks)
            
            self.progress.emit("Segmentation complete!")
            self.finished.emit(masks, scores, logits)
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class ImageLabel(QLabel):
    """Custom QLabel that allows clicking to select points"""
    clicked = pyqtSignal(int, int)
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.setScaledContents(False)
        self.points = []
        self.original_pixmap = None
        self.image_rect = None  # Store actual image rectangle
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.pixmap() and not self.pixmap().isNull():
            # Get click position in label coordinates
            click_x = event.pos().x()
            click_y = event.pos().y()
            
            # Get current pixmap (scaled)
            current_pixmap = self.pixmap()
            label_width = self.width()
            label_height = self.height()
            pixmap_width = current_pixmap.width()
            pixmap_height = current_pixmap.height()
            
            # Calculate image position (centered)
            x_offset = (label_width - pixmap_width) // 2
            y_offset = (label_height - pixmap_height) // 2
            
            # Check if click is within image
            if (x_offset <= click_x <= x_offset + pixmap_width and 
                y_offset <= click_y <= y_offset + pixmap_height):
                
                # Convert to pixmap coordinates
                pixmap_x = click_x - x_offset
                pixmap_y = click_y - y_offset
                
                # Scale to original image coordinates
                if self.original_pixmap:
                    orig_width = self.original_pixmap.width()
                    orig_height = self.original_pixmap.height()
                    
                    scale_x = orig_width / pixmap_width
                    scale_y = orig_height / pixmap_height
                    
                    # Convert to original image coordinates
                    img_x = int(pixmap_x * scale_x)
                    img_y = int(pixmap_y * scale_y)
                    
                    # Clamp to image bounds
                    img_x = max(0, min(img_x, orig_width - 1))
                    img_y = max(0, min(img_y, orig_height - 1))
                    
                    self.clicked.emit(img_x, img_y)
                
    def setImageWithPoints(self, pixmap, points):
        """Set image and draw points on it"""
        self.original_pixmap = pixmap
        self.points = points
        self.updateDisplay()
        
    def updateDisplay(self):
        """Update the display with current points"""
        if self.original_pixmap is None:
            return
            
        pixmap = self.original_pixmap.copy()
        if self.points:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            for idx, point in enumerate(self.points):
                x, y = point
                # Draw larger, more visible circles
                # Outer circle (white border)
                painter.setPen(QPen(QColor(255, 255, 255), 4))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPoint(x, y), 12, 12)
                
                # Inner circle (red fill)
                painter.setPen(QPen(QColor(255, 0, 0), 3))
                painter.setBrush(QColor(255, 0, 0, 180))
                painter.drawEllipse(QPoint(x, y), 10, 10)
                
                # Draw point number
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.setFont(QFont('Arial', 14, QFont.Bold))
                painter.drawText(QPoint(x - 6, y + 6), str(idx + 1))
                
            painter.end()
        
        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)


class InpaintGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inpaint Anything - GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # State variables
        self.current_image = None
        self.current_image_array = None
        self.clicked_points = []  # List of (x, y) tuples for multiple points
        self.masks = None
        self.selected_mask_idx = 0
        self.mask_checkboxes = []  # Store checkbox references
        self.inpainted_result = None
        
        # Model settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_model_type = "vit_t"
        self.sam_ckpt = "./weights/mobile_sam.pt"
        self.lama_config = "./lama/configs/prediction/default.yaml"
        self.lama_ckpt = "./pretrained_models/big-lama"
        
        self.initUI()
        
    def initUI(self):
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(350)
        
        # Title
        title = QLabel("Inpaint Anything")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50;")
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)
        
        # Load Image Button
        self.load_btn = QPushButton("📁 Load Image")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.load_btn.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_btn)
        
        # Instructions
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout()
        instructions = QLabel(
            "MULTI-OBJECT SELECTION:\n\n"
            "1. Load an image\n"
            "2. Click MULTIPLE times on objects\n"
            "   (bed, furniture, etc.)\n"
            "3. Click 'Segment' to generate masks\n"
            "4. Try all 3 masks\n"
            "5. Click 'Clear Points' to start over\n"
            "6. Enable Harmonization ✓\n"
            "7. Click 'Remove Object'\n\n"
            "TIP: Click on each piece of\n"
            "furniture separately for best results!"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("padding: 10px; font-size: 11px;")
        instructions_layout.addWidget(instructions)
        instructions_group.setLayout(instructions_layout)
        left_layout.addWidget(instructions_group)
        
        # Settings Group
        settings_group = QGroupBox("Settings")
        settings_layout = QGridLayout()
        
        # Dilate kernel size
        settings_layout.addWidget(QLabel("Dilate Size:"), 0, 0)
        self.dilate_spin = QSpinBox()
        self.dilate_spin.setRange(0, 50)
        self.dilate_spin.setValue(15)  # Original repo default
        self.dilate_spin.setSuffix(" px")
        self.dilate_spin.setToolTip("Expand mask to avoid edge artifacts (original default: 15)")
        settings_layout.addWidget(self.dilate_spin, 0, 1)
        
        # Mask refinement
        self.refine_mask_check = QCheckBox("✨ GrabCut Mask Refinement")
        self.refine_mask_check.setChecked(True)
        self.refine_mask_check.setToolTip(
            "Refine SAM masks for better accuracy:\n"
            "• Uses GrabCut algorithm\n"
            "• Better object boundary detection\n"
            "• Removes small artifacts\n"
            "• Smooths mask edges\n\n"
            "Recommended for precise object selection!"
        )
        self.refine_mask_check.setStyleSheet(
            "QCheckBox { font-weight: bold; color: #e74c3c; padding: 5px; }"
        )
        settings_layout.addWidget(self.refine_mask_check, 1, 0, 1, 2)
        
        # Advanced inpainting
        self.harmonize_check = QCheckBox("🧠 Intelligent Scene Understanding")
        self.harmonize_check.setChecked(True)
        self.harmonize_check.setToolTip(
            "AI-powered scene intelligence:\n"
            "• Identifies floors, walls, edges\n"
            "• Extends floor patterns naturally (marble, tiles)\n"
            "• Continues wall textures seamlessly\n"
            "• Preserves structural boundaries\n"
            "• Patch-based texture synthesis\n"
            "• Lighting gradient adjustment\n"
            "• Multi-scale detail enhancement\n\n"
            "Thinks: 'How would this look without the object?'"
        )
        self.harmonize_check.setStyleSheet(
            "QCheckBox { font-weight: bold; color: #3498db; padding: 5px; }"
        )
        settings_layout.addWidget(self.harmonize_check, 2, 0, 1, 2)
        
        # NEW: Stable Diffusion option
        self.use_sd_check = QCheckBox("🎨 Use Stable Diffusion (Better for Indoor Scenes)")
        self.use_sd_check.setChecked(False)  # Off by default
        self.use_sd_check.setToolTip(
            "Stable Diffusion Inpainting (Semantic & Structure-Aware):\n\n"
            "BEST FOR:\n"
            "✓ Large objects (furniture, doors, curtains)\n"
            "✓ Indoor scenes (walls, floors)\n"
            "✓ Semantic understanding (knows what a 'wall' is)\n"
            "✓ Geometric structure preservation\n"
            "✓ No 'mirror effect' or smudging\n\n"
            "TRADE-OFFS:\n"
            "⏱️ Slower (10-20 seconds vs 1-2 seconds)\n"
            "💾 More VRAM (6-8 GB vs 1-2 GB)\n"
            "🎨 MUCH better quality for structured scenes\n\n"
            "NOTE: First use downloads model (~5GB, one-time)"
        )
        self.use_sd_check.setStyleSheet(
            "QCheckBox { font-weight: bold; color: #9b59b6; padding: 5px; }"
        )
        settings_layout.addWidget(self.use_sd_check, 3, 0, 1, 2)
        
        # SD Prompt input (optional)
        from PyQt5.QtWidgets import QLineEdit
        self.sd_prompt_label = QLabel("SD Prompt (optional):")
        self.sd_prompt_input = QLineEdit()
        self.sd_prompt_input.setPlaceholderText("Leave empty for auto-prompt (recommended)")
        self.sd_prompt_input.setToolTip(
            "Custom prompt for Stable Diffusion:\n\n"
            "Examples:\n"
            "• 'a clean white wall and marble floor'\n"
            "• 'empty room with wooden floor'\n"
            "• 'white painted wall, indoor lighting'\n\n"
            "Leave EMPTY for automatic prompt generation\n"
            "(analyzes scene and generates appropriate prompt)"
        )
        self.sd_prompt_input.setVisible(False)  # Hidden by default
        self.sd_prompt_label.setVisible(False)
        
        # Show/hide prompt input based on SD checkbox
        def toggle_sd_prompt():
            visible = self.use_sd_check.isChecked()
            self.sd_prompt_input.setVisible(visible)
            self.sd_prompt_label.setVisible(visible)
            # Disable harmonization when using SD (SD doesn't need it)
            if visible:
                self.harmonize_check.setChecked(False)
                self.harmonize_check.setEnabled(False)
            else:
                self.harmonize_check.setEnabled(True)
        
        self.use_sd_check.stateChanged.connect(toggle_sd_prompt)
        
        settings_layout.addWidget(self.sd_prompt_label, 4, 0)
        settings_layout.addWidget(self.sd_prompt_input, 4, 1)
        
        # Device info
        device_label = QLabel(f"Device: {self.device.upper()}")
        device_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 10px;")
        device_label.setWordWrap(True)
        settings_layout.addWidget(device_label, 5, 0, 1, 2)
        
        settings_group.setLayout(settings_layout)
        left_layout.addWidget(settings_group)
        
        # Mask Selection Group (for multi-object removal)
        mask_selection_group = QGroupBox("🎭 Mask Selection")
        mask_selection_layout = QVBoxLayout()
        
        # Preview dropdown
        preview_layout = QHBoxLayout()
        preview_layout.addWidget(QLabel("Preview Mask:"))
        self.mask_combo = QComboBox()
        self.mask_combo.addItems(["No masks yet"])
        self.mask_combo.currentIndexChanged.connect(self.on_mask_changed)
        self.mask_combo.setEnabled(False)
        preview_layout.addWidget(self.mask_combo, 1)
        mask_selection_layout.addLayout(preview_layout)
        
        # Checkboxes container (will be populated dynamically)
        self.mask_checkboxes_widget = QWidget()
        self.mask_checkboxes_layout = QVBoxLayout()
        self.mask_checkboxes_layout.setContentsMargins(0, 5, 0, 5)
        self.mask_checkboxes_widget.setLayout(self.mask_checkboxes_layout)
        mask_selection_layout.addWidget(self.mask_checkboxes_widget)
        
        # Select/Deselect All buttons
        select_buttons_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("✓ All")
        self.select_all_btn.setEnabled(False)
        self.select_all_btn.clicked.connect(self.select_all_masks)
        self.deselect_all_btn = QPushButton("✗ None")
        self.deselect_all_btn.setEnabled(False)
        self.deselect_all_btn.clicked.connect(self.deselect_all_masks)
        select_buttons_layout.addWidget(self.select_all_btn)
        select_buttons_layout.addWidget(self.deselect_all_btn)
        mask_selection_layout.addLayout(select_buttons_layout)
        
        # Helper label
        self.mask_selection_label = QLabel("Segment objects first to see masks")
        self.mask_selection_label.setStyleSheet("color: #7f8c8d; font-style: italic; padding: 5px;")
        self.mask_selection_label.setWordWrap(True)
        mask_selection_layout.addWidget(self.mask_selection_label)
        
        mask_selection_group.setLayout(mask_selection_layout)
        left_layout.addWidget(mask_selection_group)
        
        # Action Buttons
        self.segment_btn = QPushButton("🎯 Segment Object")
        self.segment_btn.setEnabled(False)
        self.segment_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.segment_btn.clicked.connect(self.segment_object)
        left_layout.addWidget(self.segment_btn)
        
        # Clear Points Button
        self.clear_points_btn = QPushButton("🔄 Clear Points")
        self.clear_points_btn.setEnabled(False)
        self.clear_points_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                border: none;
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.clear_points_btn.clicked.connect(self.clear_points)
        left_layout.addWidget(self.clear_points_btn)
        
        self.remove_btn = QPushButton("🗑️ Remove Object")
        self.remove_btn.setEnabled(False)
        self.remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.remove_btn.clicked.connect(self.remove_object)
        left_layout.addWidget(self.remove_btn)
        
        # Save Result Button
        self.save_btn = QPushButton("💾 Save Result")
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
        """)
        self.save_btn.clicked.connect(self.save_result)
        left_layout.addWidget(self.save_btn)
        
        # Reset Button
        reset_btn = QPushButton("🔄 Reset")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 10px;
                font-size: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
        """)
        reset_btn.clicked.connect(self.reset)
        left_layout.addWidget(reset_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximum(0)  # Indeterminate
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready. Load an image to start.")
        self.status_label.setStyleSheet("padding: 10px; background-color: #ecf0f1; border-radius: 5px;")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)
        
        left_layout.addStretch()
        
        # Right panel - Image displays
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Top row - Original and Mask
        top_row = QHBoxLayout()
        
        # Original image with point
        original_group = QGroupBox("Original Image (Click to select object)")
        original_layout = QVBoxLayout()
        self.original_label = ImageLabel()
        self.original_label.clicked.connect(self.on_image_clicked)
        original_layout.addWidget(self.original_label)
        original_group.setLayout(original_layout)
        top_row.addWidget(original_group)
        
        # Mask preview
        mask_group = QGroupBox("Segmentation Mask")
        mask_layout = QVBoxLayout()
        self.mask_label = QLabel()
        self.mask_label.setMinimumSize(400, 400)
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.mask_label.setScaledContents(False)
        mask_layout.addWidget(self.mask_label)
        mask_group.setLayout(mask_layout)
        top_row.addWidget(mask_group)
        
        right_layout.addLayout(top_row)
        
        # Bottom - Result
        result_group = QGroupBox("Inpainted Result")
        result_layout = QVBoxLayout()
        self.result_label = QLabel()
        self.result_label.setMinimumSize(800, 400)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.result_label.setScaledContents(False)
        result_layout.addWidget(self.result_label)
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
    def load_image(self):
        """Load image from file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if file_name:
            try:
                self.current_image_array = load_img_to_array(file_name)
                self.current_image = file_name
                
                # Convert to QPixmap
                height, width, channel = self.current_image_array.shape
                bytes_per_line = 3 * width
                q_image = QImage(
                    self.current_image_array.tobytes(), 
                    width, height, bytes_per_line, 
                    QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(q_image)
                
                # Display image
                self.original_label.setImageWithPoints(pixmap, [])
                
                # Reset state
                self.clicked_points = []  # Reset to empty list
                self.masks = None
                self.inpainted_result = None
                self.mask_label.clear()
                self.result_label.clear()
                self.segment_btn.setEnabled(False)
                self.clear_points_btn.setEnabled(False)  # Disable clear button
                self.remove_btn.setEnabled(False)
                self.save_btn.setEnabled(False)
                self.mask_combo.setEnabled(False)
                
                self.status_label.setText(f"✅ Image loaded: {Path(file_name).name}\nClick on the object you want to remove.")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")
                
    def on_image_clicked(self, x, y):
        """Handle click on image - supports multiple points"""
        if self.current_image_array is None:
            return
        
        # Add point to list
        self.clicked_points.append([x, y])
        
        # Update display with all points
        height, width, channel = self.current_image_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            self.current_image_array.tobytes(), 
            width, height, bytes_per_line, 
            QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        self.original_label.setImageWithPoints(pixmap, self.clicked_points)
        
        # Enable buttons
        self.segment_btn.setEnabled(True)
        self.clear_points_btn.setEnabled(True)
        
        num_points = len(self.clicked_points)
        self.status_label.setText(
            f"📍 {num_points} point(s) selected. Last: ({x}, {y})\n"
            f"Click more points or click 'Segment Object'."
        )
    
    def clear_points(self):
        """Clear all selected points"""
        self.clicked_points = []
        
        if self.current_image_array is not None:
            # Redisplay image without points
            height, width, channel = self.current_image_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(
                self.current_image_array.tobytes(), 
                width, height, bytes_per_line, 
                QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(q_image)
            self.original_label.setPixmap(pixmap.scaled(
                self.original_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        
        # Reset masks
        self.masks = None
        self.mask_label.clear()
        self.mask_label.setText("Segmentation mask will appear here")
        
        # Disable buttons
        self.segment_btn.setEnabled(False)
        self.clear_points_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
        self.mask_combo.setEnabled(False)
        
        self.status_label.setText("Points cleared. Click on image to select objects.")
        
    def segment_object(self):
        """Run SAM segmentation with multiple points"""
        if not self.clicked_points:
            return
            
        self.progress_bar.setVisible(True)
        self.segment_btn.setEnabled(False)
        self.clear_points_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        
        # Create point labels (all foreground points = 1)
        point_labels = [1] * len(self.clicked_points)
        
        # Get refinement setting
        refine_masks = self.refine_mask_check.isChecked()
        
        # Start worker thread
        self.segment_worker = SegmentWorker(
            self.current_image_array,
            self.clicked_points,  # Pass all points
            point_labels,
            self.sam_model_type,
            self.sam_ckpt,
            self.device,
            refine_masks=refine_masks
        )
        self.segment_worker.finished.connect(self.on_segmentation_complete)
        self.segment_worker.error.connect(self.on_worker_error)
        self.segment_worker.progress.connect(self.update_status)
        self.segment_worker.start()
        
    def on_segmentation_complete(self, masks, scores, logits):
        """Handle segmentation completion"""
        self.masks = (masks.astype(np.uint8) * 255).copy()
        
        # Apply dilation
        dilate_size = self.dilate_spin.value()
        if dilate_size > 0:
            self.masks = [dilate_mask(mask, dilate_size) for mask in self.masks]
        else:
            self.masks = list(self.masks)
        
        # Populate mask combo box
        self.mask_combo.clear()
        self.mask_combo.addItems([f"Mask {i+1}" for i in range(len(self.masks))])
        self.mask_combo.setEnabled(True)
        
        # Populate checkboxes
        self.populate_mask_checkboxes()
        
        # Show first mask
        self.selected_mask_idx = 0
        self.display_mask(0)
        
        self.select_all_btn.setEnabled(True)
        self.deselect_all_btn.setEnabled(True)
        self.remove_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.segment_btn.setEnabled(True)
        self.clear_points_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        
        self.status_label.setText(
            f"✅ Segmentation complete! {len(self.masks)} masks generated.\n"
            "Check masks you want to remove, then click 'Remove Object'."
        )
        
    def on_mask_changed(self, index):
        """Handle mask selection change"""
        if self.masks is not None:
            self.selected_mask_idx = index
            self.display_mask(index)
            
    def display_mask(self, index):
        """Display selected mask"""
        if self.masks is None or index >= len(self.masks):
            return
            
        mask = self.masks[index]
        height, width = mask.shape
        
        # Convert to RGB for display
        mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        mask_rgb[mask > 0] = [255, 100, 100]  # Red tint
        
        # Overlay on original
        overlay = self.current_image_array.copy()
        alpha = 0.5
        overlay[mask > 0] = (
            overlay[mask > 0] * (1 - alpha) + 
            mask_rgb[mask > 0] * alpha
        ).astype(np.uint8)
        
        # Convert to QPixmap
        bytes_per_line = 3 * width
        q_image = QImage(overlay.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit
        scaled_pixmap = pixmap.scaled(
            self.mask_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.mask_label.setPixmap(scaled_pixmap)
    
    def populate_mask_checkboxes(self):
        """Create checkboxes for each mask"""
        # Clear existing checkboxes
        while self.mask_checkboxes_layout.count():
            item = self.mask_checkboxes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.mask_checkboxes = []
        
        # Create checkbox for each mask
        for i in range(len(self.masks)):
            checkbox = QCheckBox(f"Include Mask {i+1}")
            checkbox.setChecked(False)  # Start unchecked so user can select specific ones
            checkbox.stateChanged.connect(self.on_checkbox_changed)
            self.mask_checkboxes.append(checkbox)
            self.mask_checkboxes_layout.addWidget(checkbox)
        
        # Update label
        self.mask_selection_label.setText(
            f"{len(self.masks)} masks detected. Check which ones to remove together."
        )
    
    def on_checkbox_changed(self):
        """Handle checkbox state change"""
        selected_count = sum(1 for cb in self.mask_checkboxes if cb.isChecked())
        if selected_count > 0:
            self.mask_selection_label.setText(
                f"{selected_count} mask(s) selected for removal"
            )
            self.mask_selection_label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px;")
        else:
            self.mask_selection_label.setText(
                f"{len(self.masks)} masks detected. Check which ones to remove together."
            )
            self.mask_selection_label.setStyleSheet("color: #7f8c8d; font-style: italic; padding: 5px;")
    
    def select_all_masks(self):
        """Select all mask checkboxes"""
        for checkbox in self.mask_checkboxes:
            checkbox.setChecked(True)
    
    def deselect_all_masks(self):
        """Deselect all mask checkboxes"""
        for checkbox in self.mask_checkboxes:
            checkbox.setChecked(False)
        
    def remove_object(self):
        """Run inpainting to remove object"""
        if self.masks is None:
            return
        
        # Combine selected masks
        selected_indices = [i for i, cb in enumerate(self.mask_checkboxes) if cb.isChecked()]
        
        if len(selected_indices) == 0:
            QMessageBox.warning(
                self,
                "No Masks Selected",
                "Please check at least one mask to remove."
            )
            return
        
        self.progress_bar.setVisible(True)
        self.remove_btn.setEnabled(False)
        self.segment_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        
        # Combine selected masks using OR operation
        combined_mask = np.zeros_like(self.masks[0])
        for idx in selected_indices:
            combined_mask = np.maximum(combined_mask, self.masks[idx])
        
        mask = combined_mask
        
        # Get harmonization setting
        use_harmonization = self.harmonize_check.isChecked()
        
        # Start worker thread
        self.inpaint_worker = InpaintWorker(
            self.current_image_array,
            mask,
            self.lama_config,
            self.lama_ckpt,
            self.device,
            use_harmonization=use_harmonization,
            use_sd=self.use_sd_check.isChecked(),
            sd_prompt=self.sd_prompt_input.text() if self.sd_prompt_input.text() else None
        )
        self.inpaint_worker.finished.connect(self.on_inpainting_complete)
        self.inpaint_worker.error.connect(self.on_worker_error)
        self.inpaint_worker.progress.connect(self.update_status)
        self.inpaint_worker.start()
        
    def on_inpainting_complete(self, result):
        """Handle inpainting completion"""
        self.inpainted_result = result
        
        # Display result
        height, width, channel = result.shape
        bytes_per_line = 3 * width
        q_image = QImage(result.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit
        scaled_pixmap = pixmap.scaled(
            self.result_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.result_label.setPixmap(scaled_pixmap)
        
        self.save_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.remove_btn.setEnabled(True)
        self.segment_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        
        self.status_label.setText("✅ Object removed successfully!\nClick 'Save Result' to export the image.")
        
    def on_worker_error(self, error_msg):
        """Handle worker thread errors"""
        self.progress_bar.setVisible(False)
        self.segment_btn.setEnabled(True)
        self.remove_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Error", f"Operation failed:\n{error_msg}")
        self.status_label.setText(f"❌ Error: {error_msg}")
        
    def update_status(self, message):
        """Update status label"""
        self.status_label.setText(message)
        
    def save_result(self):
        """Save inpainted result"""
        if self.inpainted_result is None:
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Result", "", 
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        
        if file_name:
            try:
                save_array_to_img(self.inpainted_result, file_name)
                QMessageBox.information(self, "Success", f"Result saved to:\n{file_name}")
                self.status_label.setText(f"✅ Result saved: {Path(file_name).name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
                
    def reset(self):
        """Reset the application state"""
        self.current_image = None
        self.current_image_array = None
        self.clicked_points = []  # Reset to empty list
        self.masks = None
        self.inpainted_result = None
        
        self.original_label.clear()
        self.original_label.original_pixmap = None
        self.original_label.points = []
        self.mask_label.clear()
        self.result_label.clear()
        
        self.segment_btn.setEnabled(False)
        self.clear_points_btn.setEnabled(False)  # Disable clear button
        self.remove_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.mask_combo.setEnabled(False)
        
        self.status_label.setText("Ready. Load an image to start.")


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    window = InpaintGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
