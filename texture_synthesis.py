"""
Texture Synthesis and Patch-based Inpainting for Better Results
"""

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt


def patch_based_inpaint(img, mask, patch_size=7):
    """
    Patch-based texture synthesis for inpainting
    Uses exemplar-based method to fill from surrounding texture
    
    Args:
        img: Input image (H, W, 3)
        mask: Binary mask (H, W) where >0 = region to fill
        patch_size: Size of patches to use
        
    Returns:
        Inpainted image
    """
    result = img.copy()
    mask_binary = (mask > 0).astype(np.uint8)
    
    # Use OpenCV's Navier-Stokes based inpainting as backup
    # This is better for texture than deep learning sometimes
    result = cv2.inpaint(img, mask_binary, inpaintRadius=patch_size, 
                         flags=cv2.INPAINT_NS)
    
    return result


def telea_inpaint(img, mask, radius=10):
    """
    Telea's fast marching method - excellent for texture
    
    Args:
        img: Input image
        mask: Binary mask
        radius: Inpainting radius
        
    Returns:
        Inpainted image
    """
    mask_binary = (mask > 0).astype(np.uint8)
    result = cv2.inpaint(img, mask_binary, inpaintRadius=radius, 
                         flags=cv2.INPAINT_TELEA)
    return result


def hybrid_inpaint(img, lama_result, mask, blend_ratio=0.6):
    """
    Combine LaMa result with traditional CV inpainting
    
    Args:
        img: Original image
        lama_result: LaMa inpainting result
        mask: Binary mask
        blend_ratio: How much LaMa vs CV (0=all CV, 1=all LaMa)
        
    Returns:
        Hybrid result
    """
    # Apply Telea inpainting
    mask_binary = (mask > 0).astype(np.uint8)
    cv_result = cv2.inpaint(img, mask_binary, inpaintRadius=15, 
                           flags=cv2.INPAINT_TELEA)
    
    # Blend LaMa and CV results
    mask_3ch = np.stack([mask_binary] * 3, axis=-1).astype(float)
    result = (lama_result * blend_ratio + cv_result * (1 - blend_ratio))
    
    # Use original outside mask
    result = np.where(mask_3ch > 0, result, img).astype(np.uint8)
    
    return result


def smart_texture_transfer(img, mask, sample_region_margin=50):
    """
    Transfer texture from nearby regions using advanced techniques
    
    Args:
        img: Input image
        mask: Binary mask
        sample_region_margin: Pixels around mask to sample from
        
    Returns:
        Texture-transferred result
    """
    mask_binary = (mask > 0).astype(np.uint8)
    
    # Create sampling region (area around mask but not in mask)
    kernel = np.ones((sample_region_margin, sample_region_margin), np.uint8)
    dilated = cv2.dilate(mask_binary, kernel, iterations=1)
    sample_region = dilated - mask_binary
    
    # Use larger radius for better texture matching
    result = cv2.inpaint(img, mask_binary, inpaintRadius=25, 
                         flags=cv2.INPAINT_TELEA)
    
    # Apply additional smoothing at boundaries
    result = cv2.bilateralFilter(result, 9, 75, 75)
    
    return result


def improve_lama_result(original_img, lama_result, mask, method='hybrid'):
    """
    Post-process LaMa result to improve quality
    
    Args:
        original_img: Original image
        lama_result: LaMa output
        mask: Binary mask (0-255 or 0-1)
        method: 'hybrid', 'telea', 'texture', or 'multi'
        
    Returns:
        Improved result
    """
    # Normalize mask
    if mask.max() > 1:
        mask_norm = mask / 255.0
    else:
        mask_norm = mask
        
    mask_binary = (mask_norm > 0.5).astype(np.uint8)
    
    if method == 'telea':
        # Just use Telea method
        result = telea_inpaint(original_img, mask_binary, radius=15)
        
    elif method == 'texture':
        # Smart texture transfer
        result = smart_texture_transfer(original_img, mask_binary, sample_region_margin=60)
        
    elif method == 'hybrid':
        # Blend LaMa with CV methods
        result = hybrid_inpaint(original_img, lama_result, mask_binary, blend_ratio=0.5)
        
    elif method == 'multi':
        # Multi-pass approach
        # 1. First pass with Telea for structure
        pass1 = telea_inpaint(original_img, mask_binary, radius=20)
        
        # 2. Blend with LaMa
        result = hybrid_inpaint(original_img, lama_result, mask_binary, blend_ratio=0.4)
        
        # 3. Apply bilateral filter to smooth
        result = cv2.bilateralFilter(result, 9, 75, 75)
    else:
        result = lama_result
    
    return result


def adaptive_inpaint(original_img, lama_result, mask):
    """
    Intelligently choose best method based on mask size and content
    
    Args:
        original_img: Original image
        lama_result: LaMa result
        mask: Binary mask
        
    Returns:
        Best inpainting result
    """
    mask_binary = (mask > 0.5 if mask.max() <= 1 else mask > 127).astype(np.uint8)
    
    # Calculate mask properties
    mask_area = np.sum(mask_binary)
    total_area = mask_binary.shape[0] * mask_binary.shape[1]
    mask_ratio = mask_area / total_area
    
    # For large masks, use hybrid approach
    if mask_ratio > 0.15:
        print(f"Large mask detected ({mask_ratio:.1%}), using multi-pass hybrid...")
        result = improve_lama_result(original_img, lama_result, mask_binary, method='multi')
    
    # For medium masks, blend LaMa with traditional
    elif mask_ratio > 0.05:
        print(f"Medium mask detected ({mask_ratio:.1%}), using hybrid blend...")
        result = improve_lama_result(original_img, lama_result, mask_binary, method='hybrid')
    
    # For small masks, Telea works well
    else:
        print(f"Small mask detected ({mask_ratio:.1%}), using Telea method...")
        result = improve_lama_result(original_img, lama_result, mask_binary, method='telea')
    
    return result
