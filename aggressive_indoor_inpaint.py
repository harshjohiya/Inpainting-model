"""
AGGRESSIVE Indoor Scene Inpainting - Addresses LaMa's Fundamental Failure
Uses exemplar-based inpainting for floors and structure-aware for walls
"""

import numpy as np
import cv2
from typing import Tuple


def aggressive_floor_reconstruction(
    original_img: np.ndarray,
    lama_result: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    AGGRESSIVE floor reconstruction using exemplar-based inpainting
    This directly addresses the floor smudging problem you're seeing
    
    Args:
        original_img: Original RGB image
        lama_result: LaMa result (which is probably terrible for floor)
        mask: Binary mask
        
    Returns:
        Image with properly reconstructed floor
    """
    result = lama_result.copy()
    h, w = original_img.shape[:2]
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Detect floor region (bottom 40-50% typically)
    floor_y_start = int(h * 0.55)
    
    # For each row in the floor region
    for y in range(floor_y_start, h):
        # Find masked pixels in this row
        masked_pixels = np.where(mask_binary[y, :] > 0)[0]
        
        if len(masked_pixels) == 0:
            continue
        
        # Find unmaksed pixels in this row
        unmasked_pixels = np.where(mask_binary[y, :] == 0)[0]
        
        if len(unmasked_pixels) < 10:
            # Not enough reference, try nearby rows
            for offset in [1, 2, 3, 5, 10, 20]:
                if y - offset >= 0:
                    unmasked_pixels = np.where(mask_binary[y - offset, :] == 0)[0]
                    if len(unmasked_pixels) >= 10:
                        # Copy from this row
                        for x in masked_pixels:
                            # Find closest unmasked pixel
                            closest_idx = np.argmin(np.abs(unmasked_pixels - x))
                            sample_x = unmasked_pixels[closest_idx]
                            result[y, x] = original_img[y - offset, sample_x]
                        break
        else:
            # Copy from same row
            for x in masked_pixels:
                # Find closest unmasked pixel
                closest_idx = np.argmin(np.abs(unmasked_pixels - x))
                sample_x = unmasked_pixels[closest_idx]
                
                # Use original texture (100% replacement for floor!)
                result[y, x] = original_img[y, sample_x]
    
    # Blend at boundaries to avoid seams
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)
    mask_eroded = cv2.erode(mask_binary, kernel, iterations=1)
    blend_zone = mask_dilated - mask_eroded
    blend_zone_3ch = cv2.cvtColor(blend_zone, cv2.COLOR_GRAY2RGB).astype(float) / 255.0
    
    # Smooth blend at boundaries
    result_blurred = cv2.GaussianBlur(result, (15, 15), 0)
    result = (result * (1 - blend_zone_3ch * 0.5) + 
             result_blurred * blend_zone_3ch * 0.5).astype(np.uint8)
    
    return result


def aggressive_wall_reconstruction(
    original_img: np.ndarray,
    lama_result: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    AGGRESSIVE wall reconstruction - extend wall texture horizontally
    
    Args:
        original_img: Original RGB image
        lama_result: LaMa result
        mask: Binary mask
        
    Returns:
        Image with better wall reconstruction
    """
    result = lama_result.copy()
    h, w = original_img.shape[:2]
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Wall region (top 60%)
    wall_y_end = int(h * 0.6)
    
    # For each column in wall region
    for x in range(w):
        for y in range(wall_y_end):
            if mask_binary[y, x] > 0:
                # Find nearest unmasked pixel in same row (left or right)
                sample_x = None
                
                # Try left
                for offset in range(1, min(x + 1, 150)):
                    if mask_binary[y, x - offset] == 0:
                        sample_x = x - offset
                        break
                
                # Try right if left failed
                if sample_x is None:
                    for offset in range(1, min(w - x, 150)):
                        if mask_binary[y, x + offset] == 0:
                            sample_x = x + offset
                            break
                
                if sample_x is not None:
                    # Mix: 70% original texture, 30% LaMa
                    result[y, x] = (0.7 * original_img[y, sample_x] + 
                                   0.3 * result[y, x]).astype(np.uint8)
    
    return result


def super_aggressive_indoor_inpaint(
    original_img: np.ndarray,
    lama_result: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    SUPER AGGRESSIVE indoor scene inpainting
    Completely bypasses LaMa's terrible frequency-domain approach
    Uses direct texture copying from nearby regions
    
    This is what you ACTUALLY need for indoor scenes!
    
    Args:
        original_img: Original RGB image
        lama_result: LaMa result (we'll mostly ignore this)
        mask: Binary mask
        
    Returns:
        Much better result for indoor scenes
    """
    # Step 1: Aggressive floor reconstruction (ignore LaMa for floor!)
    result = aggressive_floor_reconstruction(original_img, lama_result, mask)
    
    # Step 2: Aggressive wall reconstruction
    result = aggressive_wall_reconstruction(original_img, result, mask)
    
    # Step 3: Use OpenCV's inpainting for any remaining issues
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Find small remaining issues
    remaining_mask = mask_binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    remaining_mask = cv2.erode(remaining_mask, kernel, iterations=2)
    
    if np.sum(remaining_mask) > 0:
        # Use traditional inpainting for touch-up
        result = cv2.inpaint(result, remaining_mask, 3, cv2.INPAINT_TELEA)
    
    # Step 4: Final smoothing at boundaries
    mask_boundary = cv2.Canny(mask_binary, 50, 150)
    mask_boundary_dilated = cv2.dilate(mask_boundary, kernel, iterations=3)
    
    smoothed = cv2.GaussianBlur(result, (7, 7), 0)
    boundary_3ch = cv2.cvtColor(mask_boundary_dilated, cv2.COLOR_GRAY2RGB).astype(float) / 255.0
    
    result = (result * (1 - boundary_3ch * 0.3) + 
             smoothed * boundary_3ch * 0.3).astype(np.uint8)
    
    return result
