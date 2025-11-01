"""
Structure-Aware Inpainting Enhancement for Large Indoor Objects
Addresses LaMa's fundamental architectural limitations by adding:
1. 3D geometric structure detection
2. Perspective-aware line extension
3. Plane-based reconstruction
4. Edge-preserving blending
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class GeometricStructure:
    """Represents detected geometric structures in the scene"""
    lines: List[np.ndarray]  # Detected lines (rho, theta)
    vanishing_points: List[np.ndarray]  # Perspective vanishing points
    dominant_angles: List[float]  # Dominant line orientations
    planes: List[dict]  # Detected planes (wall, floor, ceiling)


class StructureAwareInpainter:
    """
    Enhance LaMa results with geometric and structural understanding
    Compensates for FFC's lack of 3D/semantic reasoning
    """
    
    def __init__(self):
        self.min_line_length = 50
        self.line_gap = 10
        
    def detect_geometric_structure(
        self, 
        img: np.ndarray, 
        mask: np.ndarray
    ) -> GeometricStructure:
        """
        Detect lines, vanishing points, and dominant orientations
        
        Args:
            img: Original RGB image
            mask: Binary mask (255 for region to inpaint)
            
        Returns:
            GeometricStructure with detected features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Create inverse mask (area to analyze for structure)
        inverse_mask = cv2.bitwise_not((mask > 127).astype(np.uint8) * 255)
        
        # Apply mask to focus on visible regions
        masked_gray = cv2.bitwise_and(gray, gray, mask=inverse_mask)
        
        # Edge detection with Canny
        edges = cv2.Canny(masked_gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is None:
            lines = []
        else:
            lines = lines[:, 0, :]  # Shape: (N, 2) - (rho, theta)
        
        # Find vanishing points
        vanishing_points = self._find_vanishing_points(lines)
        
        # Find dominant angles (typically 0°, 90° for walls/floors)
        dominant_angles = self._find_dominant_angles(lines)
        
        # Detect planes (simplified - based on image regions)
        planes = self._detect_planes(img, mask, lines)
        
        return GeometricStructure(
            lines=lines,
            vanishing_points=vanishing_points,
            dominant_angles=dominant_angles,
            planes=planes
        )
    
    def _find_vanishing_points(
        self, 
        lines: List[np.ndarray], 
        threshold: float = 10.0
    ) -> List[np.ndarray]:
        """
        Find vanishing points by finding intersections of parallel line groups
        
        Args:
            lines: List of lines in (rho, theta) format
            threshold: Angle threshold for considering lines parallel (degrees)
            
        Returns:
            List of vanishing points (x, y)
        """
        if len(lines) < 2:
            return []
        
        vanishing_points = []
        
        # Group lines by angle
        angle_threshold_rad = np.deg2rad(threshold)
        line_groups = []
        
        for line in lines:
            rho, theta = line
            
            # Find which group this line belongs to
            found_group = False
            for group in line_groups:
                if abs(theta - group[0][1]) < angle_threshold_rad:
                    group.append(line)
                    found_group = True
                    break
            
            if not found_group:
                line_groups.append([line])
        
        # For each pair of line groups, find intersection (vanishing point)
        for i in range(len(line_groups)):
            for j in range(i + 1, len(line_groups)):
                group1 = line_groups[i]
                group2 = line_groups[j]
                
                # Use representative lines from each group
                if len(group1) > 0 and len(group2) > 0:
                    vp = self._line_intersection(group1[0], group2[0])
                    if vp is not None:
                        vanishing_points.append(vp)
        
        return vanishing_points
    
    def _line_intersection(
        self, 
        line1: np.ndarray, 
        line2: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Find intersection point of two lines in (rho, theta) format
        
        Args:
            line1: First line (rho1, theta1)
            line2: Second line (rho2, theta2)
            
        Returns:
            Intersection point (x, y) or None if parallel
        """
        rho1, theta1 = line1
        rho2, theta2 = line2
        
        # Convert to Cartesian form: a*x + b*y = c
        a1 = np.cos(theta1)
        b1 = np.sin(theta1)
        c1 = rho1
        
        a2 = np.cos(theta2)
        b2 = np.sin(theta2)
        c2 = rho2
        
        # Solve system of equations
        det = a1 * b2 - a2 * b1
        
        if abs(det) < 1e-6:  # Parallel lines
            return None
        
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        
        return np.array([x, y])
    
    def _find_dominant_angles(
        self, 
        lines: List[np.ndarray], 
        bins: int = 36
    ) -> List[float]:
        """
        Find dominant line orientations (typically 0° and 90° for indoor scenes)
        
        Args:
            lines: List of lines in (rho, theta) format
            bins: Number of angle bins
            
        Returns:
            List of dominant angles in radians
        """
        if len(lines) == 0:
            return []
        
        # Extract angles
        angles = [line[1] for line in lines]
        
        # Create histogram
        hist, bin_edges = np.histogram(angles, bins=bins, range=(0, np.pi))
        
        # Find peaks
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > np.mean(hist):
                angle = (bin_edges[i] + bin_edges[i + 1]) / 2
                peaks.append(angle)
        
        return peaks
    
    def _detect_planes(
        self, 
        img: np.ndarray, 
        mask: np.ndarray, 
        lines: List[np.ndarray]
    ) -> List[dict]:
        """
        Detect major planes (walls, floor, ceiling) based on image regions
        
        Args:
            img: Original RGB image
            mask: Binary mask
            lines: Detected lines
            
        Returns:
            List of plane dictionaries with type and boundaries
        """
        h, w = img.shape[:2]
        planes = []
        
        # Simple heuristic: floor is bottom third, walls are middle
        # In a real implementation, this would use depth estimation
        
        # Floor plane (bottom region with horizontal lines)
        floor_region = (int(h * 0.66), h)
        planes.append({
            'type': 'floor',
            'region': floor_region,
            'orientation': 'horizontal'
        })
        
        # Wall plane (middle region with vertical lines)
        wall_region = (int(h * 0.33), int(h * 0.66))
        planes.append({
            'type': 'wall',
            'region': wall_region,
            'orientation': 'vertical'
        })
        
        # Ceiling plane (top region)
        ceiling_region = (0, int(h * 0.33))
        planes.append({
            'type': 'ceiling',
            'region': ceiling_region,
            'orientation': 'horizontal'
        })
        
        return planes
    
    def extend_lines_through_mask(
        self, 
        img: np.ndarray, 
        mask: np.ndarray, 
        structure: GeometricStructure
    ) -> np.ndarray:
        """
        Extend detected lines through the masked region
        Helps preserve geometric structure that LaMa destroys
        
        Args:
            img: Original RGB image
            mask: Binary mask
            structure: Detected geometric structure
            
        Returns:
            Image with extended lines (for guidance)
        """
        # Create a guide image
        guide = np.zeros_like(img)
        
        # Draw detected lines extended through mask
        for line in structure.lines:
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Extend line across entire image
            x1 = int(x0 + 3000 * (-b))
            y1 = int(y0 + 3000 * (a))
            x2 = int(x0 - 3000 * (-b))
            y2 = int(y0 - 3000 * (a))
            
            cv2.line(guide, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        return guide
    
    def enhance_with_structure(
        self,
        original_img: np.ndarray,
        lama_result: np.ndarray,
        mask: np.ndarray,
        structure: GeometricStructure,
        blend_strength: float = 0.3
    ) -> np.ndarray:
        """
        Enhance LaMa result using geometric structure
        
        Args:
            original_img: Original RGB image
            lama_result: LaMa inpainting result
            mask: Binary mask
            structure: Detected geometric structure
            blend_strength: How much to enforce structure (0-1)
            
        Returns:
            Structure-enhanced result
        """
        result = lama_result.copy()
        mask_binary = (mask > 127).astype(np.uint8)
        
        # 1. Extend edges through masked region
        result = self._extend_edges(original_img, result, mask_binary, structure)
        
        # 2. Enforce plane consistency (walls should be planar)
        result = self._enforce_plane_consistency(result, mask_binary, structure)
        
        # 3. Perspective-aware texture transfer
        result = self._perspective_texture_transfer(original_img, result, mask_binary, structure)
        
        return result
    
    def _extend_edges(
        self,
        original: np.ndarray,
        result: np.ndarray,
        mask: np.ndarray,
        structure: GeometricStructure
    ) -> np.ndarray:
        """Extend strong edges through the masked region - AGGRESSIVE VERSION"""
        # Detect edges in original image (outside mask)
        gray_orig = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        inverse_mask = cv2.bitwise_not(mask)
        edges_orig = cv2.Canny(gray_orig, 30, 100)  # Lower threshold for more edges
        edges_orig = cv2.bitwise_and(edges_orig, edges_orig, mask=inverse_mask)
        
        # Dilate edges AGGRESSIVELY into mask region
        kernel_size = 25  # Larger kernel
        for angle in structure.dominant_angles:
            # Create directional kernel
            angle_deg = np.rad2deg(angle)
            if abs(angle_deg) < 45 or abs(angle_deg - 180) < 45:
                # Horizontal
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 5))
            else:
                # Vertical
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, kernel_size))
            
            extended_edges = cv2.dilate(edges_orig, kernel, iterations=5)  # More iterations
        
        # Use extended edges to guide sharpening in result
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        edge_mask = cv2.cvtColor(extended_edges, cv2.COLOR_GRAY2RGB) / 255.0
        
        # AGGRESSIVE sharpening along edges
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        sharpened = cv2.addWeighted(result, 2.0, blurred, -1.0, 0)  # Much stronger
        
        # Blend: use sharpened version HEAVILY along edges
        result = (result * (1 - edge_mask * 0.8) + sharpened * edge_mask * 0.8).astype(np.uint8)
        
        return result
    
    def _enforce_plane_consistency(
        self,
        result: np.ndarray,
        mask: np.ndarray,
        structure: GeometricStructure
    ) -> np.ndarray:
        """
        Enforce planar consistency for walls/floors - AGGRESSIVE VERSION
        LaMa doesn't understand planes - we add this constraint STRONGLY
        """
        # For each detected plane, ensure smooth gradients
        for plane in structure.planes:
            if plane['type'] in ['wall', 'floor']:
                # Extract region
                y_start, y_end = plane['region']
                plane_mask = np.zeros_like(mask)
                plane_mask[y_start:y_end, :] = mask[y_start:y_end, :]
                
                if np.sum(plane_mask) > 0:
                    # AGGRESSIVE bilateral filter for smoothness
                    smoothed = cv2.bilateralFilter(result, 15, 100, 100)
                    
                    # VERY AGGRESSIVE Blend in plane region (70% smoothed!)
                    plane_mask_3ch = cv2.cvtColor(plane_mask, cv2.COLOR_GRAY2RGB) / 255.0
                    result = (result * (1 - plane_mask_3ch * 0.7) + 
                             smoothed * plane_mask_3ch * 0.7).astype(np.uint8)
        
        return result
    
    def _perspective_texture_transfer(
        self,
        original: np.ndarray,
        result: np.ndarray,
        mask: np.ndarray,
        structure: GeometricStructure
    ) -> np.ndarray:
        """
        Transfer texture with perspective awareness
        Sample from nearby regions following vanishing point directions
        """
        if len(structure.vanishing_points) == 0:
            return result
        
        # Use first vanishing point
        vp = structure.vanishing_points[0]
        
        # For pixels in mask, sample from direction toward vanishing point
        mask_coords = np.argwhere(mask > 0)
        
        for coord in mask_coords[::5]:  # Sample every 5 pixels for speed
            y, x = coord
            
            # Direction from pixel to vanishing point
            dx = vp[0] - x
            dy = vp[1] - y
            norm = np.sqrt(dx**2 + dy**2)
            
            if norm > 0:
                dx /= norm
                dy /= norm
                
                # Sample from outside mask in this direction
                sample_dist = 20
                sample_x = int(x + dx * sample_dist)
                sample_y = int(y + dy * sample_dist)
                
                # Check bounds
                h, w = mask.shape
                if 0 <= sample_x < w and 0 <= sample_y < h:
                    if mask[sample_y, sample_x] == 0:
                        # Transfer texture
                        result[y, x] = (0.7 * result[y, x] + 
                                       0.3 * original[sample_y, sample_x])
        
        return result.astype(np.uint8)


def enhance_lama_for_indoor_scenes(
    original_img: np.ndarray,
    lama_result: np.ndarray,
    mask: np.ndarray,
    use_structure_detection: bool = True
) -> np.ndarray:
    """
    Main function to enhance LaMa results for indoor scenes
    Addresses FFC limitations by adding geometric understanding
    
    Args:
        original_img: Original RGB image
        lama_result: LaMa inpainting result
        mask: Binary mask (255 = inpaint region)
        use_structure_detection: Whether to use full structure detection
        
    Returns:
        Enhanced inpainting result
    """
    if not use_structure_detection:
        return lama_result
    
    # AGGRESSIVE floor/wall reconstruction
    result = lama_result.copy()
    mask_binary = (mask > 127).astype(np.uint8)
    
    # 1. Detect floor region (bottom part with horizontal lines)
    h, w = original_img.shape[:2]
    floor_y_threshold = int(h * 0.6)  # Bottom 40% likely floor
    
    # Check if mask intersects floor region
    floor_mask = np.zeros_like(mask_binary)
    floor_mask[floor_y_threshold:, :] = mask_binary[floor_y_threshold:, :]
    
    if np.sum(floor_mask) > 100:  # Significant floor area to inpaint
        # AGGRESSIVE: Copy floor texture from nearby unmaksed floor
        for y in range(floor_y_threshold, h):
            for x in range(w):
                if floor_mask[y, x] > 0:
                    # Sample from left or right at same height
                    sample_x = None
                    # Try left
                    for offset in range(5, min(x, 100), 5):
                        if mask_binary[y, x - offset] == 0:
                            sample_x = x - offset
                            break
                    # Try right if left failed
                    if sample_x is None:
                        for offset in range(5, min(w - x, 100), 5):
                            if x + offset < w and mask_binary[y, x + offset] == 0:
                                sample_x = x + offset
                                break
                    
                    if sample_x is not None:
                        # Aggressive replacement - 80% original texture
                        result[y, x] = (0.2 * result[y, x] + 0.8 * original_img[y, sample_x]).astype(np.uint8)
    
    # 2. Now do structure-aware enhancement for walls
    inpainter = StructureAwareInpainter()
    structure = inpainter.detect_geometric_structure(original_img, mask)
    result = inpainter.enhance_with_structure(
        original_img,
        result,
        mask,
        structure,
        blend_strength=0.6  # More aggressive blending
    )
    
    return result
