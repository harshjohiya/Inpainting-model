"""
Context Intelligence System
Understands the scene structure (floor, walls, edges) and extends patterns naturally
"""

import numpy as np
import cv2
from typing import Tuple, Dict, Optional
from scipy import ndimage


class SceneAnalyzer:
    """
    Analyzes the scene to identify floors, walls, edges, and patterns
    """
    
    def __init__(self):
        self.debug = False
    
    def analyze_scene_structure(self, img: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Analyze the image to understand scene structure
        
        Returns dict with:
        - floor_region: Binary mask of floor area
        - wall_region: Binary mask of wall area
        - edge_region: Binary mask of floor-wall boundary
        - dominant_directions: Detected line orientations
        """
        height, width = img.shape[:2]
        mask_binary = (mask > 127).astype(np.uint8)
        
        # Get context region around mask
        kernel = np.ones((50, 50), np.uint8)
        context_region = cv2.dilate(mask_binary, kernel) - mask_binary
        
        # Detect edges in context region
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        context_edges = edges * context_region
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(context_edges, 1, np.pi/180, threshold=50, 
                                minLineLength=30, maxLineGap=10)
        
        # Classify regions based on position (simple heuristic: bottom = floor, top = wall)
        floor_threshold = int(height * 0.4)  # Bottom 60% likely floor
        floor_region = np.zeros((height, width), dtype=np.uint8)
        floor_region[floor_threshold:, :] = 1
        
        wall_region = np.zeros((height, width), dtype=np.uint8)
        wall_region[:floor_threshold, :] = 1
        
        # Detect floor-wall boundary
        edge_region = self._detect_floor_wall_edge(img, mask_binary)
        
        return {
            'floor_region': floor_region,
            'wall_region': wall_region,
            'edge_region': edge_region,
            'lines': lines,
            'context_region': context_region
        }
    
    def _detect_floor_wall_edge(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Detect the boundary between floor and wall
        """
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Detect horizontal edges (floor-wall boundary is usually horizontal)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Horizontal edges have strong vertical gradient
        horizontal_edges = np.abs(sobely) > np.abs(sobelx) * 2
        horizontal_edges = horizontal_edges.astype(np.uint8) * 255
        
        # Clean up
        kernel = np.ones((3, 20), np.uint8)
        horizontal_edges = cv2.morphologyEx(horizontal_edges, cv2.MORPH_CLOSE, kernel)
        
        return horizontal_edges


class PatternExtender:
    """
    Intelligently extends patterns from surrounding areas
    """
    
    def __init__(self):
        self.debug = False
    
    def extend_floor_pattern(
        self, 
        img: np.ndarray, 
        mask: np.ndarray,
        inpainted: np.ndarray
    ) -> np.ndarray:
        """
        Extend floor pattern (like marble) into the inpainted region
        """
        mask_binary = (mask > 127).astype(bool)
        result = inpainted.copy()
        
        # Identify floor region in mask
        height = img.shape[0]
        floor_start = int(height * 0.4)
        floor_mask = mask_binary.copy()
        floor_mask[:floor_start, :] = False  # Only consider bottom part
        
        if not floor_mask.any():
            return inpainted
        
        # Extract floor texture from surrounding area
        kernel = np.ones((40, 40), np.uint8)
        floor_context = cv2.dilate(floor_mask.astype(np.uint8), kernel) - floor_mask.astype(np.uint8)
        floor_context = floor_context.astype(bool)
        
        if not floor_context.any():
            return inpainted
        
        # Get floor texture samples
        floor_pixels = img[floor_context]
        
        # Analyze texture characteristics
        mean_color = np.mean(floor_pixels, axis=0)
        std_color = np.std(floor_pixels, axis=0)
        
        # Create texture using patch-based synthesis
        result = self._patch_based_texture_synthesis(
            img, result, mask, floor_mask, floor_context
        )
        
        return result
    
    def _patch_based_texture_synthesis(
        self,
        source_img: np.ndarray,
        target_img: np.ndarray,
        mask: np.ndarray,
        fill_region: np.ndarray,
        sample_region: np.ndarray,
        patch_size: int = 15
    ) -> np.ndarray:
        """
        Fill masked region using patch-based texture synthesis
        """
        mask_binary = (mask > 127).astype(bool)
        result = target_img.copy().astype(np.float32)
        
        # Get list of pixels to fill (prioritize from edges inward)
        dist = cv2.distanceTransform(mask_binary.astype(np.uint8), cv2.DIST_L2, 5)
        fill_order = np.argsort(dist[mask_binary])[::-1]  # Fill from edges first
        
        fill_coords = np.argwhere(mask_binary)
        
        # Sample patches from source region
        sample_coords = np.argwhere(sample_region)
        
        if len(sample_coords) < patch_size * patch_size:
            return target_img
        
        half_patch = patch_size // 2
        
        # Fill pixels in order
        for idx in fill_order[:min(len(fill_order), 500)]:  # Limit for performance
            y, x = fill_coords[idx]
            
            # Skip if outside valid range
            if (y < half_patch or y >= result.shape[0] - half_patch or
                x < half_patch or x >= result.shape[1] - half_patch):
                continue
            
            # Find best matching patch from sample region
            best_patch, best_score = self._find_best_patch(
                source_img, result, (y, x), sample_coords, 
                patch_size, mask_binary
            )
            
            if best_patch is not None:
                # Copy center pixel from best patch
                result[y, x] = best_patch[half_patch, half_patch]
        
        return result.astype(np.uint8)
    
    def _find_best_patch(
        self,
        source_img: np.ndarray,
        current_result: np.ndarray,
        target_pos: Tuple[int, int],
        sample_coords: np.ndarray,
        patch_size: int,
        mask: np.ndarray,
        n_samples: int = 50
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Find best matching patch from sample region
        """
        y, x = target_pos
        half_patch = patch_size // 2
        
        # Get target patch (with known and unknown regions)
        y1, y2 = y - half_patch, y + half_patch + 1
        x1, x2 = x - half_patch, x + half_patch + 1
        
        if y1 < 0 or y2 > current_result.shape[0] or x1 < 0 or x2 > current_result.shape[1]:
            return None, float('inf')
        
        target_patch = current_result[y1:y2, x1:x2]
        target_mask = ~mask[y1:y2, x1:x2]  # Known pixels
        
        if not target_mask.any():
            return None, float('inf')
        
        # Sample random patches from source
        best_patch = None
        best_score = float('inf')
        
        sample_indices = np.random.choice(len(sample_coords), 
                                         min(n_samples, len(sample_coords)), 
                                         replace=False)
        
        for idx in sample_indices:
            sy, sx = sample_coords[idx]
            
            sy1, sy2 = sy - half_patch, sy + half_patch + 1
            sx1, sx2 = sx - half_patch, sx + half_patch + 1
            
            if (sy1 < 0 or sy2 > source_img.shape[0] or 
                sx1 < 0 or sx2 > source_img.shape[1]):
                continue
            
            source_patch = source_img[sy1:sy2, sx1:sx2]
            
            # Compare only on known pixels
            if source_patch.shape == target_patch.shape:
                diff = np.sum((source_patch[target_mask] - target_patch[target_mask]) ** 2)
                
                if diff < best_score:
                    best_score = diff
                    best_patch = source_patch
        
        return best_patch, best_score
    
    def extend_wall_pattern(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        inpainted: np.ndarray
    ) -> np.ndarray:
        """
        Extend wall texture into inpainted region
        """
        mask_binary = (mask > 127).astype(bool)
        result = inpainted.copy()
        
        # Identify wall region in mask (upper part)
        height = img.shape[0]
        wall_end = int(height * 0.6)
        wall_mask = mask_binary.copy()
        wall_mask[wall_end:, :] = False  # Only consider upper part
        
        if not wall_mask.any():
            return inpainted
        
        # Extract wall context
        kernel = np.ones((40, 40), np.uint8)
        wall_context = cv2.dilate(wall_mask.astype(np.uint8), kernel) - wall_mask.astype(np.uint8)
        wall_context = wall_context.astype(bool)
        
        if not wall_context.any():
            return inpainted
        
        # Use patch-based synthesis for wall texture
        result = self._patch_based_texture_synthesis(
            img, result, mask, wall_mask, wall_context
        )
        
        return result


class ContextIntelligentInpainter:
    """
    Main intelligent inpainting system that understands scene context
    """
    
    def __init__(self):
        self.scene_analyzer = SceneAnalyzer()
        self.pattern_extender = PatternExtender()
    
    def intelligent_inpaint(
        self,
        original_img: np.ndarray,
        lama_result: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply intelligent context-aware inpainting
        
        Args:
            original_img: Original image
            lama_result: LaMa inpainting result
            mask: Binary mask of inpainted region
            
        Returns:
            Enhanced result with intelligent pattern extension
        """
        # Step 1: Analyze scene structure
        scene_info = self.scene_analyzer.analyze_scene_structure(original_img, mask)
        
        # Step 2: Start with LaMa result
        result = lama_result.copy()
        
        # Step 3: Intelligently extend floor patterns
        result = self.pattern_extender.extend_floor_pattern(
            original_img, mask, result
        )
        
        # Step 4: Intelligently extend wall patterns
        result = self.pattern_extender.extend_wall_pattern(
            original_img, mask, result
        )
        
        # Step 5: Enhance structural edges (floor-wall boundary)
        result = self._enhance_structural_edges(
            original_img, result, mask, scene_info
        )
        
        # Step 6: Final color/lighting adjustment
        result = self._adjust_lighting_gradient(
            original_img, result, mask
        )
        
        return result
    
    def _enhance_structural_edges(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        mask: np.ndarray,
        scene_info: Dict
    ) -> np.ndarray:
        """
        Enhance structural edges like floor-wall boundaries
        """
        result = inpainted.copy()
        mask_binary = (mask > 127).astype(bool)
        
        # Find horizontal edges in the context
        edge_region = scene_info['edge_region']
        
        # Detect strong horizontal lines near mask
        kernel = np.ones((30, 30), np.uint8)
        context = cv2.dilate(mask_binary.astype(np.uint8), kernel)
        context_edges = edge_region * context
        
        # Find dominant horizontal line (floor-wall boundary)
        if context_edges.any():
            # Get y-coordinates of edge pixels
            edge_y_coords = np.where(context_edges > 0)[0]
            if len(edge_y_coords) > 0:
                # Most common y-coordinate is likely the floor-wall boundary
                boundary_y = int(np.median(edge_y_coords))
                
                # Ensure continuity of this line in inpainted region
                x_range = np.where(mask_binary[boundary_y, :])[0]
                if len(x_range) > 0:
                    # Sample colors from left and right of the boundary line
                    if boundary_y < mask_binary.shape[0] - 5:
                        for x in x_range:
                            if x > 0 and x < result.shape[1] - 1:
                                # Interpolate boundary pixel colors
                                left_color = result[boundary_y, max(0, x-20)]
                                right_color = result[boundary_y, min(result.shape[1]-1, x+20)]
                                result[boundary_y, x] = (left_color + right_color) // 2
        
        return result
    
    def _adjust_lighting_gradient(
        self,
        original: np.ndarray,
        inpainted: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Adjust lighting to match the natural gradient in the room
        """
        mask_binary = (mask > 127).astype(bool)
        result = inpainted.copy().astype(np.float32)
        
        # Analyze brightness gradient in surrounding area
        gray_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        
        # Create context region
        kernel = np.ones((50, 50), np.uint8)
        context_region = cv2.dilate(mask_binary.astype(np.uint8), kernel) - mask_binary.astype(np.uint8)
        context_region = context_region.astype(bool)
        
        if not context_region.any():
            return inpainted
        
        # Get context coordinates and brightness
        context_coords = np.argwhere(context_region)
        context_brightness = gray_orig[context_region]
        
        # Fit a plane to the brightness (lighting gradient)
        if len(context_coords) > 100:
            from scipy.interpolate import griddata
            
            # Get mask coordinates
            mask_coords = np.argwhere(mask_binary)
            
            # Interpolate expected brightness
            try:
                expected_brightness = griddata(
                    context_coords, context_brightness, mask_coords,
                    method='linear', fill_value=np.mean(context_brightness)
                )
                
                # Adjust inpainted region brightness
                gray_result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2GRAY)
                current_brightness = gray_result[mask_binary]
                
                # Calculate adjustment factor
                brightness_ratio = expected_brightness / (current_brightness + 1e-6)
                brightness_ratio = np.clip(brightness_ratio, 0.8, 1.2)
                
                # Apply adjustment per pixel
                for idx, (y, x) in enumerate(mask_coords):
                    result[y, x] = result[y, x] * brightness_ratio[idx]
                
            except:
                pass  # Fallback: no adjustment
        
        return np.clip(result, 0, 255).astype(np.uint8)


def apply_intelligent_context_inpainting(
    original_img: np.ndarray,
    lama_result: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Main function to apply intelligent context-aware inpainting
    """
    inpainter = ContextIntelligentInpainter()
    result = inpainter.intelligent_inpaint(original_img, lama_result, mask)
    return result
