"""
Advanced Inpainting with Context-Aware Processing
Inspired by state-of-the-art results (like Gemini Inpainting)

Key Improvements:
1. Better mask generation with SAM parameter tuning
2. Context-aware texture synthesis
3. Multi-scale processing
4. Edge-preserving blending
5. Perspective-aware inpainting
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import scipy.ndimage as ndimage
from scipy.interpolate import griddata


class AdvancedMaskRefiner:
    """
    Refines SAM masks to be more accurate and object-aware
    """
    
    @staticmethod
    def refine_mask_with_grabcut(img: np.ndarray, initial_mask: np.ndarray, iterations: int = 5) -> np.ndarray:
        """
        Use GrabCut to refine mask based on color statistics
        
        Args:
            img: RGB image
            initial_mask: Initial binary mask from SAM
            iterations: Number of GrabCut iterations
            
        Returns:
            Refined binary mask
        """
        if img is None or initial_mask is None:
            return initial_mask
            
        # Prepare mask for GrabCut
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[initial_mask > 0] = cv2.GC_PR_FGD  # Probable foreground
        mask[initial_mask == 0] = cv2.GC_PR_BGD  # Probable background
        
        # Shrink initial mask slightly to get definite foreground
        kernel = np.ones((15, 15), np.uint8)
        sure_fg = cv2.erode(initial_mask.astype(np.uint8), kernel, iterations=1)
        mask[sure_fg > 0] = cv2.GC_FGD  # Definite foreground
        
        # Expand initial mask to get definite background
        sure_bg = cv2.dilate(initial_mask.astype(np.uint8), kernel, iterations=3)
        mask[sure_bg == 0] = cv2.GC_BGD  # Definite background
        
        # Initialize GrabCut models
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        
        try:
            # Run GrabCut
            cv2.grabCut(img, mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
            
            # Create refined mask
            refined_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            
            return refined_mask
        except Exception as e:
            print(f"GrabCut refinement failed: {e}")
            return initial_mask
    
    @staticmethod
    def remove_small_components(mask: np.ndarray, min_size: int = 500) -> np.ndarray:
        """
        Remove small disconnected components from mask
        
        Args:
            mask: Binary mask
            min_size: Minimum component size to keep
            
        Returns:
            Cleaned mask
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        # Create new mask keeping only large components
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # Skip background (0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                cleaned_mask[labels == i] = 255
        
        return cleaned_mask
    
    @staticmethod
    def smooth_mask_boundary(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Smooth mask boundaries while preserving shape
        
        Args:
            mask: Binary mask
            kernel_size: Size of smoothing kernel
            
        Returns:
            Smoothed mask
        """
        # Apply morphological closing to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Apply Gaussian blur then threshold
        blurred = cv2.GaussianBlur(closed.astype(np.float32), (kernel_size, kernel_size), 0)
        smoothed = (blurred > 127).astype(np.uint8) * 255
        
        return smoothed


class ContextAwareInpainter:
    """
    Advanced inpainting with context awareness and texture synthesis
    """
    
    def __init__(self):
        self.mask_refiner = AdvancedMaskRefiner()
    
    def enhance_lama_result(
        self,
        original_img: np.ndarray,
        lama_result: np.ndarray,
        mask: np.ndarray,
        use_advanced: bool = True
    ) -> np.ndarray:
        """
        Enhance LaMa inpainting result with context-aware processing
        
        Args:
            original_img: Original RGB image
            lama_result: LaMa inpainted result
            mask: Binary mask of inpainted region
            use_advanced: Whether to use advanced enhancements
            
        Returns:
            Enhanced inpainted image
        """
        if not use_advanced:
            return lama_result
        
        result = lama_result.copy()
        mask_binary = (mask > 127).astype(np.float32) / 255.0
        
        # Step 1: Perspective-aware color correction
        result = self._perspective_color_correction(original_img, result, mask_binary)
        
        # Step 2: Multi-scale texture transfer
        result = self._multi_scale_texture_transfer(original_img, result, mask_binary)
        
        # Step 3: Edge-preserving detail enhancement
        result = self._enhance_details(result, mask_binary)
        
        # Step 4: Advanced boundary blending
        result = self._advanced_boundary_blend(original_img, result, mask_binary)
        
        return result
    
    def _perspective_color_correction(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Correct colors considering perspective and lighting gradients
        """
        result = inpainted.copy().astype(np.float32)
        
        # Extract border region for reference
        kernel = np.ones((30, 30), np.uint8)
        dilated = cv2.dilate((mask > 0.5).astype(np.uint8), kernel)
        border = (dilated > 0) & (mask < 0.5)
        
        if border.sum() < 100:
            return inpainted
        
        # Convert to LAB for better color handling
        original_lab = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        result_lab = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # For each channel, perform spatial color correction
        for c in range(3):
            # Get reference colors from border
            border_colors = original_lab[:, :, c][border]
            
            if len(border_colors) > 0:
                # Create spatial color map using interpolation
                y_coords, x_coords = np.where(border)
                points = np.column_stack([x_coords, y_coords])
                values = border_colors
                
                # Create grid for interpolation
                mask_region = mask > 0.5
                if mask_region.sum() > 0:
                    y_mask, x_mask = np.where(mask_region)
                    grid_points = np.column_stack([x_mask, y_mask])
                    
                    # Interpolate reference colors into masked region
                    try:
                        interpolated = griddata(
                            points, values, grid_points,
                            method='linear', fill_value=np.mean(values)
                        )
                        
                        # Blend interpolated colors with inpainted colors
                        alpha = 0.4  # Adjust blend strength
                        result_lab[y_mask, x_mask, c] = (
                            alpha * interpolated +
                            (1 - alpha) * result_lab[y_mask, x_mask, c]
                        )
                    except:
                        pass
        
        # Convert back to RGB
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        
        return result
    
    def _multi_scale_texture_transfer(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Transfer texture at multiple scales for more realistic results
        """
        result = inpainted.copy().astype(np.float32)
        mask_binary = mask > 0.5
        
        # Build Gaussian pyramids
        scales = 3
        original_pyramid = [original.astype(np.float32)]
        inpainted_pyramid = [inpainted.astype(np.float32)]
        
        for i in range(scales - 1):
            original_pyramid.append(cv2.pyrDown(original_pyramid[-1]))
            inpainted_pyramid.append(cv2.pyrDown(inpainted_pyramid[-1]))
        
        # Process each scale
        for scale_idx in range(scales):
            scale_factor = 2 ** scale_idx
            
            # Get textures at this scale
            orig_level = original_pyramid[scale_idx]
            inpaint_level = inpainted_pyramid[scale_idx]
            
            # Compute high-frequency components (details)
            orig_details = cv2.Laplacian(orig_level, cv2.CV_32F)
            inpaint_details = cv2.Laplacian(inpaint_level, cv2.CV_32F)
            
            # Extract texture strength from surroundings
            mask_scaled = cv2.resize(mask.astype(np.float32), 
                                     (orig_level.shape[1], orig_level.shape[0]))
            
            # Create border region at this scale
            kernel_size = max(5, 15 // scale_factor)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated = cv2.dilate((mask_scaled > 0.5).astype(np.uint8), kernel)
            border = (dilated > 0) & (mask_scaled < 0.5)
            
            if border.sum() > 50:
                # Measure texture strength in border
                border_detail_strength = np.std(orig_details[border])
                inpaint_detail_strength = np.std(inpaint_details[mask_scaled > 0.5]) + 1e-6
                
                # Adjust detail strength to match surroundings
                detail_scale = border_detail_strength / inpaint_detail_strength
                detail_scale = np.clip(detail_scale, 0.5, 2.0)
                
                # Apply scaled details back to result
                mask_region_scaled = mask_scaled > 0.5
                if mask_region_scaled.sum() > 0:
                    adjusted_level = inpaint_level.copy()
                    adjusted_level[mask_region_scaled] += (
                        inpaint_details[mask_region_scaled] * (detail_scale - 1) * 0.3
                    )
                    
                    # Upsample and blend into result
                    for _ in range(scale_idx):
                        adjusted_level = cv2.pyrUp(adjusted_level)
                    
                    # Resize to match result
                    adjusted_level = cv2.resize(adjusted_level, 
                                               (result.shape[1], result.shape[0]))
                    
                    # Blend with decreasing weight for higher scales
                    weight = 0.3 / (scale_idx + 1)
                    result[mask_binary] = (
                        (1 - weight) * result[mask_binary] +
                        weight * adjusted_level[mask_binary]
                    )
        
        return np.clip(result, 0, 255)
    
    def _enhance_details(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Enhance fine details in inpainted region to reduce blur
        """
        result = img.copy().astype(np.float32)
        mask_binary = mask > 0.5
        
        # Unsharp masking for detail enhancement
        blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
        sharpened = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)
        
        # Apply only in masked region
        result[mask_binary] = sharpened[mask_binary]
        
        # Edge enhancement using bilateral filter
        # Preserves edges while smoothing flat areas
        result = cv2.bilateralFilter(result.astype(np.uint8), 5, 50, 50).astype(np.float32)
        
        return np.clip(result, 0, 255)
    
    def _advanced_boundary_blend(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Advanced multi-band blending for seamless boundaries
        """
        try:
            # Create smooth alpha mask with wider feathering
            feather_size = 40
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # Create distance transform for smooth falloff
            dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
            dist_transform = np.clip(dist_transform / feather_size, 0, 1)
            
            # Apply Gaussian blur for extra smoothness
            alpha = cv2.GaussianBlur(dist_transform.astype(np.float32), (41, 41), 0)
            alpha = np.expand_dims(alpha, axis=2)
            
            # Perform Laplacian pyramid blending for multi-scale seamless blend
            result = self._laplacian_blend(original, inpainted, alpha)
            
            return result
        except Exception as e:
            print(f"Laplacian blending failed: {e}, using simple alpha blending")
            # Fallback to simple alpha blending
            feather_size = 40
            mask_uint8 = (mask * 255).astype(np.uint8)
            dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
            dist_transform = np.clip(dist_transform / feather_size, 0, 1)
            alpha = cv2.GaussianBlur(dist_transform.astype(np.float32), (41, 41), 0)
            alpha = np.expand_dims(alpha, axis=2)
            alpha = np.repeat(alpha, 3, axis=2)  # Make 3-channel
            
            result = original.astype(np.float32) * (1 - alpha) + inpainted.astype(np.float32) * alpha
            return np.clip(result, 0, 255).astype(np.uint8)
    
    def _laplacian_blend(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        mask: np.ndarray,
        levels: int = 4
    ) -> np.ndarray:
        """
        Multi-band blending using Laplacian pyramids
        """
        # Ensure mask has 3 channels
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        if mask.shape[2] == 1:
            mask = np.repeat(mask, 3, axis=2)
        
        # Build Gaussian pyramids
        gauss_pyr_img1 = [img1.astype(np.float32)]
        gauss_pyr_img2 = [img2.astype(np.float32)]
        gauss_pyr_mask = [mask.astype(np.float32)]
        
        for i in range(levels):
            gauss_pyr_img1.append(cv2.pyrDown(gauss_pyr_img1[-1]))
            gauss_pyr_img2.append(cv2.pyrDown(gauss_pyr_img2[-1]))
            gauss_pyr_mask.append(cv2.pyrDown(gauss_pyr_mask[-1]))
        
        # Build Laplacian pyramids
        lapl_pyr_img1 = [gauss_pyr_img1[levels]]
        lapl_pyr_img2 = [gauss_pyr_img2[levels]]
        
        for i in range(levels, 0, -1):
            size = (gauss_pyr_img1[i-1].shape[1], gauss_pyr_img1[i-1].shape[0])
            lap1 = gauss_pyr_img1[i-1] - cv2.pyrUp(gauss_pyr_img1[i], dstsize=size)
            lap2 = gauss_pyr_img2[i-1] - cv2.pyrUp(gauss_pyr_img2[i], dstsize=size)
            lapl_pyr_img1.append(lap1)
            lapl_pyr_img2.append(lap2)
        
        # Blend Laplacian pyramids
        blended_pyr = []
        for l1, l2, m in zip(lapl_pyr_img1, lapl_pyr_img2, gauss_pyr_mask[::-1]):
            # Ensure mask matches image dimensions
            if m.shape[:2] != l1.shape[:2]:
                m = cv2.resize(m, (l1.shape[1], l1.shape[0]))
            if len(m.shape) == 2:
                m = np.expand_dims(m, axis=2)
            if m.shape[2] == 1:
                m = np.repeat(m, 3, axis=2)
            
            blended = l1 * (1 - m) + l2 * m
            blended_pyr.append(blended)
        
        # Reconstruct from blended pyramid
        result = blended_pyr[0]
        for i in range(1, len(blended_pyr)):
            size = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
            result = cv2.pyrUp(result, dstsize=size) + blended_pyr[i]
        
        return np.clip(result, 0, 255).astype(np.uint8)


def refine_sam_mask(img: np.ndarray, sam_mask: np.ndarray, use_grabcut: bool = True) -> np.ndarray:
    """
    Main function to refine SAM mask for better object selection
    
    Args:
        img: Original RGB image
        sam_mask: Binary mask from SAM
        use_grabcut: Whether to use GrabCut refinement
        
    Returns:
        Refined binary mask
    """
    refiner = AdvancedMaskRefiner()
    
    # Remove small components
    mask = refiner.remove_small_components(sam_mask, min_size=500)
    
    # Smooth boundaries
    mask = refiner.smooth_mask_boundary(mask, kernel_size=5)
    
    # Apply GrabCut refinement if requested
    if use_grabcut and img is not None:
        mask = refiner.refine_mask_with_grabcut(img, mask, iterations=5)
    
    return mask
