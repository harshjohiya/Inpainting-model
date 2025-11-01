"""
Enhanced LaMa Inpainting with Post-Processing

This module provides improved inpainting with better quality settings.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def enhance_inpainted_result(original_img, inpainted_img, mask, blend_width=10):
    """
    Enhance inpainting result with better blending at edges
    
    Args:
        original_img: Original image (H, W, 3)
        inpainted_img: Inpainted result (H, W, 3)
        mask: Binary mask (H, W) where 255 = inpainted region
        blend_width: Width of blending zone in pixels
        
    Returns:
        Enhanced inpainted image
    """
    # Convert mask to binary
    mask_binary = (mask > 127).astype(np.uint8)
    
    # Create distance transform for smooth blending
    dist_transform = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
    
    # Normalize distance transform to [0, 1] for blending weights
    if dist_transform.max() > 0:
        blend_mask = np.clip(dist_transform / blend_width, 0, 1)
    else:
        blend_mask = mask_binary.astype(float)
    
    # Expand to 3 channels
    blend_mask = blend_mask[:, :, np.newaxis]
    
    # Blend the images
    result = (inpainted_img * blend_mask + 
              original_img * (1 - blend_mask)).astype(np.uint8)
    
    return result


def sharpen_result(img, amount=0.3):
    """
    Apply subtle sharpening to improve detail
    
    Args:
        img: Input image (H, W, 3)
        amount: Sharpening amount (0.0 to 1.0)
        
    Returns:
        Sharpened image
    """
    # Create sharpening kernel
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    # Apply sharpening
    sharpened = cv2.filter2D(img, -1, kernel)
    
    # Blend with original
    result = cv2.addWeighted(img, 1 - amount, sharpened, amount, 0)
    
    return result.astype(np.uint8)


def match_color_statistics(source, target, mask):
    """
    Match color statistics of inpainted region to surrounding area
    
    Args:
        source: Inpainted image
        target: Original image (for reference)
        mask: Binary mask of inpainted region
        
    Returns:
        Color-corrected image
    """
    result = source.copy()
    mask_binary = (mask > 127)
    
    # Dilate mask to get surrounding region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    surrounding = (dilated > 0) & ~mask_binary
    
    if not surrounding.any():
        return result
    
    # Match each channel
    for c in range(3):
        source_channel = source[:, :, c]
        target_channel = target[:, :, c]
        
        # Get statistics from surrounding area
        target_mean = target_channel[surrounding].mean()
        target_std = target_channel[surrounding].std()
        
        # Get statistics from inpainted area
        source_mean = source_channel[mask_binary].mean()
        source_std = source_channel[mask_binary].std()
        
        # Match statistics
        if source_std > 0:
            normalized = (source_channel[mask_binary] - source_mean) / source_std
            matched = normalized * target_std + target_mean
            result[mask_binary, c] = np.clip(matched, 0, 255)
    
    return result.astype(np.uint8)


def denoise_inpainted_region(img, mask, strength=3):
    """
    Apply selective denoising to inpainted region
    
    Args:
        img: Input image
        mask: Binary mask of inpainted region
        strength: Denoising strength
        
    Returns:
        Denoised image
    """
    result = img.copy()
    mask_binary = (mask > 127)
    
    # Apply bilateral filter to preserve edges while denoising
    denoised = cv2.bilateralFilter(img, 9, strength * 25, strength * 25)
    
    # Apply only to inpainted region
    result[mask_binary] = denoised[mask_binary]
    
    return result
