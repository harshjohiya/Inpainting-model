"""
Advanced Inpainting Enhancement with Better Texture and Color Matching
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def poisson_blend(source, target, mask):
    """
    Poisson blending for seamless compositing
    
    Args:
        source: Source image (inpainted region)
        target: Target image (original)
        mask: Binary mask
        
    Returns:
        Blended image
    """
    try:
        # Find center of mask for Poisson blending
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return source
            
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
        
        # Create mask for seamlessClone
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Poisson blending
        result = cv2.seamlessClone(source, target, mask_uint8, center, cv2.NORMAL_CLONE)
        return result
    except:
        # Fallback to regular blending if Poisson fails
        return source


def advanced_color_matching(inpainted, original, mask, kernel_size=50):
    """
    Advanced color and luminance matching using local statistics
    
    Args:
        inpainted: Inpainted image
        original: Original image
        mask: Binary mask (0 or 1)
        kernel_size: Size of local region for statistics
        
    Returns:
        Color-matched image
    """
    result = inpainted.copy().astype(np.float32)
    mask_bool = mask > 0.5
    
    # Erode mask to get border region for sampling
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    border_region = (mask.astype(np.uint8) - eroded) > 0
    
    # If border region is too small, use dilated mask instead
    if border_region.sum() < 100:
        dilated = cv2.dilate((1 - mask).astype(np.uint8), kernel, iterations=1)
        border_region = dilated > 0
    
    # Convert to LAB color space for better color matching
    inpainted_lab = cv2.cvtColor(inpainted.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    original_lab = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Match each channel (L, A, B)
    for ch in range(3):
        # Get statistics from border region
        border_pixels = original_lab[:, :, ch][border_region]
        if len(border_pixels) > 0:
            target_mean = np.mean(border_pixels)
            target_std = np.std(border_pixels)
            
            # Get statistics from inpainted region
            inpaint_pixels = inpainted_lab[:, :, ch][mask_bool]
            if len(inpaint_pixels) > 0:
                source_mean = np.mean(inpaint_pixels)
                source_std = np.std(inpaint_pixels) + 1e-6
                
                # Apply color transfer
                inpainted_lab[:, :, ch][mask_bool] = (
                    (inpainted_lab[:, :, ch][mask_bool] - source_mean) * 
                    (target_std / source_std) + target_mean
                )
    
    # Convert back to RGB
    inpainted_lab = np.clip(inpainted_lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(inpainted_lab, cv2.COLOR_LAB2RGB)
    
    return result


def texture_preserving_smooth(img, mask, strength=0.5):
    """
    Apply edge-preserving smoothing to reduce artifacts while preserving texture
    
    Args:
        img: Input image
        mask: Binary mask of inpainted region
        strength: Smoothing strength (0-1)
        
    Returns:
        Smoothed image
    """
    result = img.copy()
    mask_bool = mask > 0.5
    
    # Apply bilateral filter for edge-preserving smoothing
    smoothed = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Blend original and smoothed based on strength
    result[mask_bool] = (
        img[mask_bool] * (1 - strength) + 
        smoothed[mask_bool] * strength
    ).astype(np.uint8)
    
    return result


def multi_scale_blend(original, inpainted, mask, levels=3, blend_width=30):
    """
    Multi-scale blending using Laplacian pyramid for seamless transitions
    
    Args:
        original: Original image
        inpainted: Inpainted image
        mask: Binary mask
        levels: Number of pyramid levels
        blend_width: Width of blending zone
        
    Returns:
        Blended image
    """
    # Create extended blend mask with smooth falloff
    kernel = np.ones((blend_width, blend_width), np.uint8)
    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    
    # Distance transform for smooth blending
    dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)
    blend_mask = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    
    # Apply stronger Gaussian blur for smoother blending
    blend_mask = cv2.GaussianBlur(blend_mask, (blend_width*2+1, blend_width*2+1), 0)
    
    # Build Gaussian pyramids
    gauss_pyr_orig = [original.astype(np.float32)]
    gauss_pyr_inp = [inpainted.astype(np.float32)]
    gauss_pyr_mask = [np.stack([blend_mask]*3, axis=-1)]
    
    for i in range(levels):
        gauss_pyr_orig.append(cv2.pyrDown(gauss_pyr_orig[i]))
        gauss_pyr_inp.append(cv2.pyrDown(gauss_pyr_inp[i]))
        gauss_pyr_mask.append(cv2.pyrDown(gauss_pyr_mask[i]))
    
    # Build Laplacian pyramids
    lap_pyr_orig = [gauss_pyr_orig[levels]]
    lap_pyr_inp = [gauss_pyr_inp[levels]]
    
    for i in range(levels, 0, -1):
        size = (gauss_pyr_orig[i-1].shape[1], gauss_pyr_orig[i-1].shape[0])
        lap_pyr_orig.append(gauss_pyr_orig[i-1] - cv2.pyrUp(gauss_pyr_orig[i], dstsize=size))
        lap_pyr_inp.append(gauss_pyr_inp[i-1] - cv2.pyrUp(gauss_pyr_inp[i], dstsize=size))
    
    # Blend Laplacian pyramids
    blended_pyr = []
    for i in range(levels + 1):
        mask_level = gauss_pyr_mask[levels - i]
        blended = lap_pyr_inp[i] * mask_level + lap_pyr_orig[i] * (1 - mask_level)
        blended_pyr.append(blended)
    
    # Reconstruct image from pyramid
    result = blended_pyr[0]
    for i in range(1, levels + 1):
        size = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + blended_pyr[i]
    
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_inpainted_result(original, inpainted, mask, mode='advanced'):
    """
    Main enhancement function with multiple processing steps
    
    Args:
        original: Original image
        inpainted: Inpainted image
        mask: Binary mask (0 or 1)
        mode: 'basic', 'advanced', or 'maximum'
        
    Returns:
        Enhanced result
    """
    result = inpainted.copy()
    
    # Convert mask to binary if needed
    if mask.max() > 1:
        mask = (mask / 255.0).astype(np.float32)
    
    if mode == 'basic':
        # Just color matching
        result = advanced_color_matching(result, original, mask, kernel_size=40)
        
    elif mode == 'advanced':
        # Color matching + smoothing + multi-scale blend
        result = advanced_color_matching(result, original, mask, kernel_size=50)
        result = texture_preserving_smooth(result, mask, strength=0.3)
        result = multi_scale_blend(original, result, mask, levels=3, blend_width=25)
        
    elif mode == 'maximum':
        # All enhancements including Poisson blending
        result = advanced_color_matching(result, original, mask, kernel_size=60)
        result = texture_preserving_smooth(result, mask, strength=0.4)
        result = poisson_blend(result, original, mask)
        result = multi_scale_blend(original, result, mask, levels=4, blend_width=30)
    
    return result
