"""
Intelligent Texture & Color Harmonization
Neural-guided post-processing that respects LaMa's structure while fixing color/texture
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F


class IntelligentTextureHarmonizer:
    """
    Smart texture and color harmonization using statistical matching
    and edge-aware blending
    """
    
    def __init__(self):
        self.debug = False
    
    def extract_reference_statistics(self, img, mask, border_width=40):
        """
        Extract color and texture statistics from the border region
        
        Args:
            img: RGB image (H, W, 3)
            mask: Binary mask (H, W) where 1 = inpainted region
            border_width: Width of border to sample from
            
        Returns:
            Dictionary with reference statistics
        """
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # Create border region (around mask, not in mask)
        kernel = np.ones((border_width, border_width), np.uint8)
        dilated = cv2.dilate(mask_binary, kernel, iterations=1)
        eroded = cv2.erode(mask_binary, kernel, iterations=1)
        border_region = (dilated.astype(bool)) & (~mask_binary.astype(bool))
        
        # Fallback if border is too small
        if border_region.sum() < 100:
            # Use area just outside mask
            kernel_large = np.ones((60, 60), np.uint8)
            dilated_large = cv2.dilate(mask_binary, kernel_large, iterations=1)
            border_region = (dilated_large.astype(bool)) & (~mask_binary.astype(bool))
        
        if border_region.sum() < 50:
            # Last resort - use entire non-masked region
            border_region = ~mask_binary.astype(bool)
        
        # Extract statistics
        border_pixels = img[border_region]
        
        stats = {
            'mean': np.mean(border_pixels, axis=0),
            'std': np.std(border_pixels, axis=0) + 1e-6,
            'median': np.median(border_pixels, axis=0),
            'min': np.min(border_pixels, axis=0),
            'max': np.max(border_pixels, axis=0),
            'border_region': border_region
        }
        
        return stats
    
    def match_histogram(self, source, reference_stats, mask):
        """
        Match histogram of source to reference in masked region
        
        Args:
            source: Image to adjust (H, W, 3)
            reference_stats: Statistics from border
            mask: Binary mask
            
        Returns:
            Histogram-matched image
        """
        result = source.copy().astype(np.float32)
        mask_binary = (mask > 0.5)
        
        # For each channel
        for c in range(3):
            if mask_binary.sum() > 0:
                # Get source pixels in mask
                source_pixels = source[:, :, c][mask_binary]
                
                if len(source_pixels) > 0:
                    # Calculate CDF of source
                    source_hist, bins = np.histogram(source_pixels.flatten(), 256, [0, 256])
                    source_cdf = source_hist.cumsum()
                    source_cdf = source_cdf / source_cdf[-1]  # Normalize
                    
                    # Create target distribution (approximate Gaussian around reference mean)
                    target_mean = reference_stats['mean'][c]
                    target_std = reference_stats['std'][c]
                    
                    # Create lookup table
                    lookup_table = np.zeros(256, dtype=np.uint8)
                    for i in range(256):
                        # Map source CDF value to target value
                        target_val = int(np.clip(
                            target_mean + (i - 128) * (target_std / 40),
                            0, 255
                        ))
                        lookup_table[i] = target_val
                    
                    # Apply lookup table
                    result[:, :, c][mask_binary] = lookup_table[
                        source[:, :, c][mask_binary].astype(np.uint8)
                    ]
        
        return result.astype(np.uint8)
    
    def transfer_color_statistics(self, inpainted, original, mask, reference_stats):
        """
        Transfer color statistics from reference to inpainted region
        
        Args:
            inpainted: LaMa output (H, W, 3)
            original: Original image (H, W, 3)
            mask: Binary mask (H, W)
            reference_stats: Border statistics
            
        Returns:
            Color-matched image
        """
        result = inpainted.copy().astype(np.float32)
        mask_binary = (mask > 0.5)
        
        # Work in LAB space for perceptually accurate color matching
        inpainted_lab = cv2.cvtColor(inpainted.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        original_lab = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Get reference LAB statistics from border
        border_pixels_lab = original_lab[reference_stats['border_region']]
        ref_mean_lab = np.mean(border_pixels_lab, axis=0)
        ref_std_lab = np.std(border_pixels_lab, axis=0) + 1e-6
        
        # Transfer statistics in LAB space
        if mask_binary.sum() > 0:
            for c in range(3):
                inpainted_pixels = inpainted_lab[:, :, c][mask_binary]
                if len(inpainted_pixels) > 0:
                    src_mean = np.mean(inpainted_pixels)
                    src_std = np.std(inpainted_pixels) + 1e-6
                    
                    # Transfer: (x - src_mean) / src_std * ref_std + ref_mean
                    inpainted_lab[:, :, c][mask_binary] = (
                        (inpainted_lab[:, :, c][mask_binary] - src_mean) * 
                        (ref_std_lab[c] / src_std) + ref_mean_lab[c]
                    )
        
        # Clip LAB values to valid range
        inpainted_lab[:, :, 0] = np.clip(inpainted_lab[:, :, 0], 0, 100)  # L: 0-100
        inpainted_lab[:, :, 1] = np.clip(inpainted_lab[:, :, 1], -127, 127)  # A: -127 to 127
        inpainted_lab[:, :, 2] = np.clip(inpainted_lab[:, :, 2], -127, 127)  # B: -127 to 127
        
        # Convert back to RGB
        result = cv2.cvtColor(inpainted_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return result
    
    def guided_filter_smooth(self, img, mask, radius=15, eps=1e-6):
        """
        Edge-preserving smoothing using guided filter
        
        Args:
            img: Input image (H, W, 3)
            mask: Binary mask
            radius: Filter radius
            eps: Regularization
            
        Returns:
            Smoothed image
        """
        result = img.copy()
        mask_binary = (mask > 0.5)
        
        # Apply guided filter only in masked region
        for c in range(3):
            filtered = cv2.ximgproc.guidedFilter(
                guide=img.astype(np.float32),
                src=img[:, :, c].astype(np.float32),
                radius=radius,
                eps=eps
            )
            result[:, :, c][mask_binary] = filtered[mask_binary]
        
        return result.astype(np.uint8)
    
    def smart_feather_blend(self, inpainted, original, mask, feather_size=25):
        """
        Smart feathering with edge detection
        
        Args:
            inpainted: Inpainted image
            original: Original image
            mask: Binary mask
            feather_size: Feather width
            
        Returns:
            Blended image
        """
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # Create distance-based feather
        dist = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 5)
        feather = np.clip(dist / feather_size, 0, 1).astype(np.float32)
        
        # Smooth the feather mask
        feather = cv2.GaussianBlur(feather, (feather_size*2+1, feather_size*2+1), 0)
        
        # Apply power curve for smoother transition
        feather = np.power(feather, 0.6)
        
        # Expand to 3 channels
        feather_3ch = np.stack([feather] * 3, axis=-1)
        
        # Blend
        result = (inpainted * feather_3ch + original * (1 - feather_3ch)).astype(np.uint8)
        
        return result
    
    def harmonize(self, original, inpainted, mask):
        """
        Main harmonization pipeline
        
        Args:
            original: Original image (H, W, 3) uint8
            inpainted: LaMa inpainted result (H, W, 3) uint8
            mask: Binary mask (H, W) 0-1 or 0-255
            
        Returns:
            Harmonized image
        """
        # Normalize mask
        if mask.max() > 1:
            mask = mask / 255.0
        
        print("\n🎨 Intelligent Texture Harmonization:")
        print("   📊 Analyzing border region...")
        
        # Step 1: Extract reference statistics
        ref_stats = self.extract_reference_statistics(original, mask, border_width=50)
        print(f"   ✓ Reference color: RGB({ref_stats['mean'][0]:.0f}, "
              f"{ref_stats['mean'][1]:.0f}, {ref_stats['mean'][2]:.0f})")
        
        # Step 2: Transfer color statistics in LAB space
        print("   🎨 Transferring color statistics...")
        result = self.transfer_color_statistics(inpainted, original, mask, ref_stats)
        
        # Step 3: Histogram matching for fine-tuning
        print("   📈 Matching histogram distribution...")
        result = self.match_histogram(result, ref_stats, mask)
        
        # Step 4: Edge-aware smoothing (only if guidedFilter available)
        try:
            print("   ✨ Edge-preserving smoothing...")
            result = self.guided_filter_smooth(result, mask, radius=12, eps=1e-4)
        except:
            # Fallback to bilateral filter
            print("   ✨ Bilateral smoothing...")
            mask_binary = (mask > 0.5)
            smoothed = cv2.bilateralFilter(result, d=9, sigmaColor=75, sigmaSpace=75)
            result[mask_binary] = smoothed[mask_binary]
        
        # Step 5: Smart feathering at boundaries
        print("   🪶 Feathering boundaries...")
        result = self.smart_feather_blend(result, original, mask, feather_size=20)
        
        print("   ✅ Harmonization complete!\n")
        
        return result


def intelligent_harmonize(original, lama_output, mask):
    """
    Convenience function for intelligent harmonization
    
    Args:
        original: Original image (H, W, 3)
        lama_output: LaMa inpainting result (H, W, 3)
        mask: Binary mask (H, W)
        
    Returns:
        Harmonized result matching surrounding texture/color
    """
    harmonizer = IntelligentTextureHarmonizer()
    return harmonizer.harmonize(original, lama_output, mask)
