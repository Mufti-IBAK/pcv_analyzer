"""
Enhanced PCV Analyzer - Python Version
Multi-Stage Hybrid Analysis Algorithm with Robust Detection
Dr. Mufti & Team - Professional Medical Image Analysis
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List, Optional, NamedTuple
import json
from dataclasses import dataclass, asdict
from scipy import ndimage
from skimage import filters, measure, segmentation
from skimage.morphology import closing, opening, disk
import argparse


@dataclass
class BoundaryResult:
    """Result of boundary detection"""
    y_position: int
    confidence: float
    method_used: str


@dataclass
class AnalysisResult:
    """Complete PCV analysis result"""
    pcv: float
    hemoglobin: float
    total_height: int
    packed_height: int
    boundaries: Dict[str, BoundaryResult]
    tube_rect: Tuple[int, int, int, int]  # x, y, w, h
    detection_method: str
    confidence_overall: float
    processing_time_ms: int
    warnings: List[str]
    success: bool
    error_message: Optional[str] = None


class EnhancedPCVAnalyzer:
    """Enhanced PCV Analyzer with Multi-Stage Hybrid Detection"""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.debug_images = {}
        
    def analyze_image(self, image_path: str) -> AnalysisResult:
        """Main analysis pipeline"""
        import time
        start_time = time.time()
        
        try:
            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                return AnalysisResult(
                    pcv=0, hemoglobin=0, total_height=0, packed_height=0,
                    boundaries={}, tube_rect=(0,0,0,0), detection_method="none",
                    confidence_overall=0, processing_time_ms=0, warnings=[],
                    success=False, error_message="Failed to load image"
                )
            
            # Convert BGR to RGB for processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.debug_images['original'] = image_rgb
            
            warnings = []
            
            # Pre-processing: Enhanced image preparation
            processed_image = self._preprocess_image(image_rgb)
            
            # Stage 1: Enhanced Global Localization (Multiple Methods)
            tube_result = self._find_tube_enhanced(processed_image, image_rgb)
            if not tube_result['success']:
                return AnalysisResult(
                    pcv=0, hemoglobin=0, total_height=0, packed_height=0,
                    boundaries={}, tube_rect=(0,0,0,0), detection_method="failed",
                    confidence_overall=0, processing_time_ms=int((time.time() - start_time) * 1000),
                    warnings=warnings, success=False, 
                    error_message=f"Tube detection failed. Tried: {', '.join(tube_result['methods_attempted'])}"
                )
            
            tube_rect = tube_result['rect']
            
            # Ensure vertical orientation with blue plasticine at bottom
            oriented_image, oriented_rect = self._ensure_vertical_orientation_advanced(image_rgb, tube_rect)
            
            tube_roi = oriented_image[oriented_rect[1]:oriented_rect[1]+oriented_rect[3], 
                                    oriented_rect[0]:oriented_rect[0]+oriented_rect[2]]
            self.debug_images['tube_roi'] = tube_roi
            self.debug_images['oriented_image'] = oriented_image
            
            # Update tube_rect to oriented coordinates
            tube_rect = oriented_rect
            
            # Stage 2: Advanced Vertical Intensity Profiling
            profiles = self._create_enhanced_profiles(tube_roi)
            
            # Stage 3: Intelligent Boundary Detection
            boundaries = self._detect_boundaries_enhanced(profiles, tube_roi.shape[0])
            
            # Validate boundaries
            if not self._validate_boundaries(boundaries):
                warnings.append("Boundary relationships appear unusual")
            
            # Calculate PCV using medical-grade precision
            pcv_result = self._calculate_pcv_enhanced(boundaries, warnings)
            
            # Create final result
            processing_time = int((time.time() - start_time) * 1000)
            
            result = AnalysisResult(
                pcv=pcv_result['pcv'],
                hemoglobin=pcv_result['hemoglobin'], 
                total_height=pcv_result['total_height'],
                packed_height=pcv_result['packed_height'],
                boundaries=boundaries,
                tube_rect=tube_rect,
                detection_method=tube_result['method'],
                confidence_overall=pcv_result['confidence'],
                processing_time_ms=processing_time,
                warnings=warnings,
                success=True
            )
            
            # Generate annotated image
            self._create_annotated_image(image_rgb, result)
            
            return result
            
        except Exception as e:
            return AnalysisResult(
                pcv=0, hemoglobin=0, total_height=0, packed_height=0,
                boundaries={}, tube_rect=(0,0,0,0), detection_method="error",
                confidence_overall=0, processing_time_ms=int((time.time() - start_time) * 1000),
                warnings=[], success=False, error_message=str(e)
            )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing with multiple techniques"""
        # Noise reduction with edge preservation
        denoised = cv2.bilateralFilter(image, 15, 80, 80)
        
        # Enhance contrast using CLAHE on each channel
        enhanced = np.zeros_like(denoised)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        
        for i in range(3):
            enhanced[:,:,i] = clahe.apply(denoised[:,:,i])
        
        # Histogram equalization for better contrast
        for i in range(3):
            enhanced[:,:,i] = cv2.equalizeHist(enhanced[:,:,i])
        
        self.debug_images['preprocessed'] = enhanced
        return enhanced
    
    def _find_tube_enhanced(self, processed: np.ndarray, original: np.ndarray) -> Dict:
        """Enhanced tube detection with multiple robust methods"""
        methods_attempted = []
        
        # Method 1: Advanced Edge Detection with Morphological Operations
        methods_attempted.append("Advanced Edge Detection")
        tube_rect = self._detect_tube_by_advanced_edges(processed)
        if tube_rect is not None:
            return {'success': True, 'rect': tube_rect, 'method': 'advanced_edges', 'methods_attempted': methods_attempted}
        
        # Method 2: Enhanced Color-based Detection
        methods_attempted.append("Enhanced Color Detection")  
        tube_rect = self._detect_tube_by_enhanced_color(original)
        if tube_rect is not None:
            return {'success': True, 'rect': tube_rect, 'method': 'enhanced_color', 'methods_attempted': methods_attempted}
        
        # Method 3: Template Matching with Multiple Templates
        methods_attempted.append("Template Matching")
        tube_rect = self._detect_tube_by_template_matching(processed)
        if tube_rect is not None:
            return {'success': True, 'rect': tube_rect, 'method': 'template_matching', 'methods_attempted': methods_attempted}
        
        # Method 4: Contour Analysis with Shape Filtering
        methods_attempted.append("Contour Shape Analysis")
        tube_rect = self._detect_tube_by_shape_analysis(processed)
        if tube_rect is not None:
            return {'success': True, 'rect': tube_rect, 'method': 'shape_analysis', 'methods_attempted': methods_attempted}
        
        # Method 5: Machine Learning-based Detection (Simplified)
        methods_attempted.append("ML-based Detection")
        tube_rect = self._detect_tube_by_ml_approach(processed)
        if tube_rect is not None:
            return {'success': True, 'rect': tube_rect, 'method': 'ml_detection', 'methods_attempted': methods_attempted}
        
        # Method 6: Last resort - Intelligent Center Rectangle
        methods_attempted.append("Intelligent Center Fallback")
        tube_rect = self._detect_tube_intelligent_fallback(processed)
        if tube_rect is not None:
            return {'success': True, 'rect': tube_rect, 'method': 'intelligent_fallback', 'methods_attempted': methods_attempted}
        
        return {'success': False, 'rect': None, 'method': 'none', 'methods_attempted': methods_attempted}
    
    def _ensure_vertical_orientation_advanced(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Advanced vertical orientation with plasticine detection"""
        if image is None or image.size == 0:
            raise ValueError("Empty image provided")
            
        x, y, w, h = roi
        
        # Validate ROI bounds
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            return image, (x, y, max(1, w), max(1, h))
        
        # Extract tube ROI with bounds checking
        tube_roi = image[y:y+h, x:x+w]
        
        if tube_roi.size == 0:
            return image, roi
            
        # Check if tube needs rotation (horizontal to vertical)
        if w > h:  # Horizontal tube
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_height, new_width = rotated_image.shape[:2]
            new_x = y
            new_y = new_width - (x + w)
            new_w = h
            new_h = w
            new_roi = (new_x, new_y, new_w, new_h)
            return self._ensure_vertical_orientation_advanced(rotated_image, new_roi)
        
        # Advanced plasticine detection for orientation
        plasticine_bottom_score = self._detect_plasticine_position(tube_roi, 'bottom')
        plasticine_top_score = self._detect_plasticine_position(tube_roi, 'top')
        
        # If plasticine detected at top more than bottom, flip tube
        if plasticine_top_score > plasticine_bottom_score + 0.2:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            new_height, new_width = rotated_image.shape[:2]
            new_x = new_width - (x + w)
            new_y = new_height - (y + h)
            new_roi = (new_x, new_y, w, h)
            return rotated_image, new_roi
        
        return image, roi
    
    def _detect_plasticine_position(self, tube_roi: np.ndarray, position: str) -> float:
        """Detect plasticine (blue/green) at specified position"""
        if tube_roi.size == 0:
            return 0.0
            
        h = tube_roi.shape[0]
        if position == 'bottom':
            section = tube_roi[int(h*0.8):, :] if h > 10 else tube_roi
        else:  # top
            section = tube_roi[:int(h*0.2), :] if h > 10 else tube_roi
        
        if section.size == 0:
            return 0.0
        
        try:
            # Convert to HSV for better color detection
            hsv_section = cv2.cvtColor(section, cv2.COLOR_RGB2HSV)
            
            # Blue plasticine detection (hue 100-140)
            blue_mask = cv2.inRange(hsv_section, np.array([100, 50, 50]), np.array([140, 255, 255]))
            blue_score = np.sum(blue_mask) / (section.shape[0] * section.shape[1] * 255)
            
            # Green plasticine detection (hue 40-80)
            green_mask = cv2.inRange(hsv_section, np.array([40, 50, 50]), np.array([80, 255, 255]))
            green_score = np.sum(green_mask) / (section.shape[0] * section.shape[1] * 255)
            
            return max(blue_score, green_score)
            
        except Exception as e:
            print(f"Plasticine detection error: {e}")
            return 0.0
    
    def _detect_tube_by_advanced_edges(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Advanced edge detection with morphological operations"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Multiple edge detection approaches
        # 1. Canny with adaptive thresholds
        sigma = 0.33
        median = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        edges_canny = cv2.Canny(gray, lower, upper)
        
        # 2. Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges_sobel = np.sqrt(sobelx**2 + sobely**2)
        edges_sobel = np.uint8(edges_sobel / edges_sobel.max() * 255)
        
        # 3. Laplacian edge detection
        edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edges_laplacian = np.uint8(np.absolute(edges_laplacian))
        
        # Combine edge maps
        edges_combined = cv2.bitwise_or(edges_canny, cv2.bitwise_or(
            cv2.threshold(edges_sobel, 50, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(edges_laplacian, 50, 255, cv2.THRESH_BINARY)[1]
        ))
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))  # Vertical kernel for tubes
        edges_morphed = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.debug_images['advanced_edges'] = edges_combined
        
        return self._filter_tube_contours(contours, image.shape)
    
    def _detect_tube_by_enhanced_color(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Enhanced color-based detection with multiple color spaces"""
        # Convert to multiple color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        masks = []
        
        # 1. Blue sealant detection (multiple blue ranges)
        blue_ranges = [
            ([100, 50, 50], [130, 255, 255]),   # Standard blue
            ([90, 60, 60], [120, 255, 255]),    # Darker blue  
            ([110, 40, 40], [140, 255, 255]),   # Lighter blue
        ]
        
        for lower, upper in blue_ranges:
            mask_blue = cv2.inRange(hsv, np.array(lower), np.array(upper))
            masks.append(mask_blue)
        
        # 2. Red blood cell detection
        red_ranges = [
            ([0, 120, 120], [10, 255, 255]),    # Red range 1
            ([170, 120, 120], [180, 255, 255]), # Red range 2
        ]
        
        for lower, upper in red_ranges:
            mask_red = cv2.inRange(hsv, np.array(lower), np.array(upper))
            masks.append(mask_red)
        
        # 3. Glass tube detection (looking for edges/transparency)
        # Detect glass-like regions using color variance
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask_glass = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
        masks.append(mask_glass)
        
        # Combine all masks
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.debug_images['enhanced_color'] = combined_mask
        
        # Find tube-like contours and extend upward for full tube
        tube_rect = self._filter_tube_contours(contours, image.shape)
        if tube_rect:
            # Extend upward assuming sealant is at bottom
            x, y, w, h = tube_rect
            extended_y = max(0, y - h * 8)  # Extend up 8x sealant height
            extended_h = min(image.shape[0] - extended_y, y + h - extended_y + h * 2)
            return (x, extended_y, w, extended_h)
        
        return None
    
    def _detect_tube_by_template_matching(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Template matching approach for tube detection"""
        # Create synthetic tube templates of different sizes
        templates = []
        
        for height in [100, 150, 200, 250]:
            for width in [20, 30, 40, 50]:
                # Create a simple tube template (vertical rectangle with edges)
                template = np.zeros((height, width), dtype=np.uint8)
                cv2.rectangle(template, (2, 2), (width-3, height-3), 128, 1)  # Tube walls
                cv2.rectangle(template, (0, height-10), (width, height), 255, -1)  # Bottom sealant
                templates.append(template)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        best_match = None
        best_score = 0
        
        for template in templates:
            # Template matching
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_score and max_val > 0.3:  # Threshold for good match
                best_score = max_val
                best_match = (max_loc[0], max_loc[1], template.shape[1], template.shape[0])
        
        return best_match
    
    def _detect_tube_by_shape_analysis(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Advanced shape analysis for tube detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply multiple filters
        # 1. Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 3. Morphological operations to enhance tube-like shapes
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        enhanced = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Find contours
        contours, _ = cv2.findContours(enhanced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Advanced shape filtering
        tube_candidates = []
        for contour in contours:
            # Get contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area < 500:  # Too small
                continue
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w
            
            # Rectangle area vs contour area (solidity)
            rect_area = w * h
            solidity = area / rect_area
            
            # Extent (contour area vs bounding box area)
            extent = area / rect_area
            
            # Score based on tube-like properties
            score = 0
            if aspect_ratio > 2:  # Tall and thin
                score += min(aspect_ratio * 10, 50)
            if solidity > 0.3:  # Reasonably solid
                score += solidity * 30
            if extent > 0.3:  # Good extent
                score += extent * 20
            if area > 1000:  # Reasonable size
                score += min(np.log(area) * 5, 20)
            
            tube_candidates.append((score, (x, y, w, h)))
        
        # Return best candidate
        if tube_candidates:
            tube_candidates.sort(reverse=True)
            return tube_candidates[0][1]
        
        return None
    
    def _detect_tube_by_ml_approach(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Simplified ML-based detection using feature analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Extract features using local binary patterns
        from skimage.feature import local_binary_pattern
        
        # Calculate LBP
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Find regions with tube-like texture patterns
        # Tubes typically have consistent vertical patterns
        
        # Apply thresholding on LBP
        lbp_thresh = np.zeros_like(lbp)
        lbp_thresh[lbp < 10] = 255  # Areas with low variation (tube-like)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
        lbp_morphed = cv2.morphologyEx(lbp_thresh.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(lbp_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return self._filter_tube_contours(contours, image.shape)
    
    def _detect_tube_intelligent_fallback(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Intelligent fallback that analyzes image structure"""
        height, width = image.shape[:2]
        
        # Analyze vertical intensity profiles across the image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate variance for each column (tubes have consistent vertical patterns)
        column_variances = []
        for x in range(0, width, 5):  # Sample every 5 pixels
            column = gray[:, x:x+5].mean(axis=1)
            variance = np.var(column)
            column_variances.append((variance, x))
        
        # Find the region with consistent vertical pattern (low variance)
        column_variances.sort()  # Low variance first
        
        # Take the middle region with lowest variance as tube center
        best_x = column_variances[len(column_variances)//4][1]  # Not the absolute lowest (might be background)
        
        # Estimate tube width based on image size
        estimated_width = max(20, min(width // 8, 60))
        tube_x = max(0, best_x - estimated_width // 2)
        tube_width = min(width - tube_x, estimated_width)
        
        # Estimate tube height (assume it takes up most of the image)
        tube_height = int(height * 0.8)
        tube_y = int(height * 0.1)
        
        return (tube_x, tube_y, tube_width, tube_height)
    
    def _filter_tube_contours(self, contours: List, image_shape: Tuple) -> Optional[Tuple[int, int, int, int]]:
        """Filter contours to find the most tube-like one"""
        height, width = image_shape[:2]
        
        best_score = 0
        best_rect = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:  # Too small
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate tube-like properties
            aspect_ratio = h / w
            height_ratio = h / height
            area_ratio = area / (w * h)
            
            # Score calculation
            score = 0
            
            # Prefer tall, thin shapes
            if aspect_ratio > 1.5:
                score += min(aspect_ratio * 5, 25)
            
            # Should occupy reasonable height
            if height_ratio > 0.3:
                score += height_ratio * 20
            
            # Should be reasonably filled
            if area_ratio > 0.3:
                score += area_ratio * 15
            
            # Size bonus
            if area > 1000:
                score += min(np.log(area), 10)
            
            # Position bonus (tubes often in center-ish area)
            center_x = x + w // 2
            distance_from_center = abs(center_x - width // 2) / (width // 2)
            if distance_from_center < 0.5:  # Within center 50%
                score += (1 - distance_from_center) * 10
            
            if score > best_score:
                best_score = score
                best_rect = (x, y, w, h)
        
        return best_rect if best_score > 15 else None
    
    def _create_enhanced_profiles(self, tube_roi: np.ndarray) -> Dict:
        """Create enhanced intensity profiles with multiple channels"""
        height = tube_roi.shape[0]
        
        # Convert to multiple color spaces for analysis
        hsv = cv2.cvtColor(tube_roi, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(tube_roi, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(tube_roi, cv2.COLOR_RGB2GRAY)
        
        profiles = {
            'hue': [],
            'saturation': [],
            'value': [],
            'red': [],
            'green': [],
            'blue': [],
            'lightness': [],
            'a_channel': [],
            'b_channel': [],
            'gray': []
        }
        
        # Sample from center 60% to avoid edge artifacts
        start_x = int(tube_roi.shape[1] * 0.2)
        end_x = int(tube_roi.shape[1] * 0.8)
        
        for y in range(height):
            # RGB profiles
            profiles['red'].append(np.mean(tube_roi[y, start_x:end_x, 0]))
            profiles['green'].append(np.mean(tube_roi[y, start_x:end_x, 1]))
            profiles['blue'].append(np.mean(tube_roi[y, start_x:end_x, 2]))
            
            # HSV profiles
            profiles['hue'].append(np.mean(hsv[y, start_x:end_x, 0]))
            profiles['saturation'].append(np.mean(hsv[y, start_x:end_x, 1]))
            profiles['value'].append(np.mean(hsv[y, start_x:end_x, 2]))
            
            # LAB profiles
            profiles['lightness'].append(np.mean(lab[y, start_x:end_x, 0]))
            profiles['a_channel'].append(np.mean(lab[y, start_x:end_x, 1]))
            profiles['b_channel'].append(np.mean(lab[y, start_x:end_x, 2]))
            
            # Grayscale profile
            profiles['gray'].append(np.mean(gray[y, start_x:end_x]))
        
        # Apply smoothing to all profiles
        for key in profiles:
            profiles[key] = self._smooth_profile(profiles[key])
        
        # Calculate gradients for boundary detection
        profiles['gradients'] = {}
        for key in ['hue', 'saturation', 'value', 'red', 'lightness', 'a_channel', 'b_channel']:
            profiles['gradients'][key] = self._calculate_gradient(profiles[key])
        
        return profiles
    
    def _smooth_profile(self, profile: List[float], window_size: int = 5) -> List[float]:
        """Apply smoothing to profile data"""
        if len(profile) < window_size:
            return profile
        
        smoothed = []
        for i in range(len(profile)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(profile), i + window_size // 2 + 1)
            smoothed.append(np.mean(profile[start_idx:end_idx]))
        
        return smoothed
    
    def _calculate_gradient(self, profile: List[float]) -> List[float]:
        """Calculate gradient of profile"""
        gradient = [0]  # First element
        
        for i in range(1, len(profile) - 1):
            gradient.append((profile[i + 1] - profile[i - 1]) / 2)
        
        gradient.append(0)  # Last element
        return gradient
    
    def _detect_boundaries_enhanced(self, profiles: Dict, roi_height: int) -> Dict[str, BoundaryResult]:
        """Enhanced boundary detection using multiple signals"""
        boundaries = {}
        
        # Boundary A: Top of Plasma (scan from top)
        plasma_top = self._find_plasma_top_enhanced(profiles, roi_height)
        boundaries['plasma_top'] = plasma_top
        
        # Boundary C: Bottom of RBCs (scan from bottom) 
        rbc_bottom = self._find_rbc_bottom_enhanced(profiles, roi_height)
        boundaries['rbc_bottom'] = rbc_bottom
        
        # Boundary B: Top of RBCs (scan in middle region)
        if plasma_top.y_position != -1 and rbc_bottom.y_position != -1:
            rbc_top = self._find_rbc_top_enhanced(profiles, plasma_top.y_position, rbc_bottom.y_position)
        else:
            rbc_top = BoundaryResult(-1, 0.0, "failed")
        
        boundaries['rbc_top'] = rbc_top
        
        return boundaries
    
    def _find_plasma_top_enhanced(self, profiles: Dict, roi_height: int) -> BoundaryResult:
        """Enhanced plasma top detection using multiple signals"""
        search_end = min(roi_height - 1, roi_height // 2)
        
        candidates = []
        
        # Method 1: Value gradient (brightness increase)
        for y in range(5, search_end):
            value_grad = profiles['gradients']['value'][y]
            if value_grad > 1.0:  # Significant brightness increase
                candidates.append((y, value_grad * 0.4, 'value_gradient'))
        
        # Method 2: Lightness change (L*a*b*)
        for y in range(5, search_end):
            lightness_grad = profiles['gradients']['lightness'][y]
            if lightness_grad > 2.0:
                candidates.append((y, lightness_grad * 0.3, 'lightness_change'))
        
        # Method 3: Color variance (transition from air to liquid)
        for y in range(5, search_end):
            # Look for increase in color variance
            if y > 10:
                var_before = np.var([profiles['red'][y-5:y], profiles['green'][y-5:y], profiles['blue'][y-5:y]])
                var_after = np.var([profiles['red'][y:y+5], profiles['green'][y:y+5], profiles['blue'][y:y+5]])
                if var_after > var_before * 1.5:
                    candidates.append((y, (var_after - var_before) * 0.01, 'color_variance'))
        
        # Select best candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0]
            return BoundaryResult(best[0], min(1.0, best[1]), best[2])
        
        return BoundaryResult(-1, 0.0, "failed")
    
    def _find_rbc_bottom_enhanced(self, profiles: Dict, roi_height: int) -> BoundaryResult:
        """Enhanced RBC bottom detection using hue and saturation analysis"""
        search_start = roi_height // 2
        
        candidates = []
        
        # Method 1: Hue shift (RBC to clay/sealant)
        for y in range(roi_height - 5, search_start, -1):
            hue_grad = abs(profiles['gradients']['hue'][y])
            if hue_grad > 5.0:  # Significant hue change
                candidates.append((y, hue_grad * 0.1, 'hue_shift'))
        
        # Method 2: Saturation drop (vibrant RBC to dull sealant)
        for y in range(roi_height - 5, search_start, -1):
            sat_grad = -profiles['gradients']['saturation'][y]  # Negative = drop
            if sat_grad > 10.0:
                candidates.append((y, sat_grad * 0.05, 'saturation_drop'))
        
        # Method 3: Enhanced Blue Plasticine Detection
        for y in range(roi_height - 5, search_start, -1):
            # Multi-criteria blue plasticine detection
            blue_increase = profiles['blue'][y] - profiles['blue'][y-5] if y >= 5 else 0
            hue_value = profiles['hue'][y]
            saturation_value = profiles['saturation'][y]
            
            # Blue plasticine characteristics:
            # - High blue channel value
            # - Hue in blue range (100-140 in OpenCV HSV)
            # - Moderate to high saturation
            blue_score = 0
            if blue_increase > 15:  # Blue channel increase
                blue_score += blue_increase * 0.03
            if 100 <= hue_value <= 140:  # Blue hue range
                blue_score += 20
            if saturation_value > 80:  # High saturation indicates vibrant blue
                blue_score += saturation_value * 0.1
            
            if blue_score > 15:
                candidates.append((y, blue_score * 0.05, 'blue_plasticine_detection'))
        
        # Method 4: A-channel change (green-red axis in LAB)
        for y in range(roi_height - 5, search_start, -1):
            a_grad = abs(profiles['gradients']['a_channel'][y])
            if a_grad > 3.0:
                candidates.append((y, a_grad * 0.2, 'a_channel_change'))
        
        # Method 5: Blue Plasticine Texture Analysis
        for y in range(roi_height - 10, search_start, -1):
            if y >= 10:
                # Analyze texture consistency (blue plasticine is usually uniform)
                blue_variance = np.var(profiles['blue'][y-5:y+5])
                hue_variance = np.var(profiles['hue'][y-5:y+5])
                
                # Low variance indicates uniform blue plasticine
                if blue_variance < 100 and hue_variance < 5 and profiles['blue'][y] > 100:
                    uniformity_score = (200 - blue_variance) * 0.01 + (10 - hue_variance) * 2
                    candidates.append((y, uniformity_score, 'blue_plasticine_texture'))
        
        # Select best candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0]
            return BoundaryResult(best[0], min(1.0, best[1]), best[2])
        
        return BoundaryResult(-1, 0.0, "failed")
    
    def _find_rbc_top_enhanced(self, profiles: Dict, plasma_top_y: int, rbc_bottom_y: int) -> BoundaryResult:
        """Enhanced RBC top detection within defined search zone"""
        search_start = plasma_top_y + 10
        search_end = rbc_bottom_y - 10
        
        if search_start >= search_end:
            return BoundaryResult(-1, 0.0, "search_zone_invalid")
        
        candidates = []
        
        # Method 1: Value drop (bright plasma to dark RBCs)
        for y in range(search_start, search_end):
            value_drop = -profiles['gradients']['value'][y]  # Negative = drop
            if value_drop > 2.0:
                candidates.append((y, value_drop * 0.3, 'value_drop'))
        
        # Method 2: Red increase (plasma to red blood cells)
        for y in range(search_start, search_end):
            red_increase = profiles['gradients']['red'][y]
            if red_increase > 5.0:
                candidates.append((y, red_increase * 0.15, 'red_increase'))
        
        # Method 3: Saturation increase (pale plasma to vibrant RBCs)
        for y in range(search_start, search_end):
            sat_increase = profiles['gradients']['saturation'][y]
            if sat_increase > 8.0:
                candidates.append((y, sat_increase * 0.08, 'saturation_increase'))
        
        # Method 4: Combined multi-signal approach
        for y in range(search_start, search_end):
            # Combined score using multiple channels
            value_signal = -profiles['gradients']['value'][y] if profiles['gradients']['value'][y] < 0 else 0
            red_signal = profiles['gradients']['red'][y] if profiles['gradients']['red'][y] > 0 else 0
            sat_signal = profiles['gradients']['saturation'][y] if profiles['gradients']['saturation'][y] > 0 else 0
            
            combined_score = (value_signal * 0.4 + red_signal * 0.3 + sat_signal * 0.3) * 0.1
            if combined_score > 0.5:
                candidates.append((y, combined_score, 'combined_signal'))
        
        # Select best candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0]
            return BoundaryResult(best[0], min(1.0, best[1]), best[2])
        
        return BoundaryResult(-1, 0.0, "failed")
    
    def _validate_boundaries(self, boundaries: Dict[str, BoundaryResult]) -> bool:
        """Enhanced boundary validation for vertical tube orientation"""
        plasma_top = boundaries.get('plasma_top', BoundaryResult(-1, 0, ""))
        rbc_top = boundaries.get('rbc_top', BoundaryResult(-1, 0, ""))
        rbc_bottom = boundaries.get('rbc_bottom', BoundaryResult(-1, 0, ""))
        
        # Check if all boundaries were detected
        if plasma_top.y_position == -1 or rbc_top.y_position == -1 or rbc_bottom.y_position == -1:
            return False
        
        # Vertical orientation validation: A (plasma top) < B (RBC top) < C (RBC bottom)
        if not (plasma_top.y_position < rbc_top.y_position < rbc_bottom.y_position):
            return False
        
        # Enhanced validation checks for medical accuracy
        # 1. Minimum distances between boundaries (avoid too close boundaries)
        min_plasma_rbc_distance = 10  # At least 1mm between plasma and RBC
        min_rbc_height = 20  # RBC layer should be at least 2mm thick
        
        plasma_to_rbc_distance = rbc_top.y_position - plasma_top.y_position
        rbc_height = rbc_bottom.y_position - rbc_top.y_position
        
        if plasma_to_rbc_distance < min_plasma_rbc_distance:
            return False
        
        if rbc_height < min_rbc_height:
            return False
        
        # 2. Total height should be reasonable (not too small or too large)
        total_height = rbc_bottom.y_position - plasma_top.y_position
        if total_height < 50 or total_height > 500:  # 5mm to 50mm range
            return False
        
        # 3. PCV should be in medically reasonable range
        pcv = (rbc_height / total_height) * 100
        if pcv < 10 or pcv > 70:  # Medical PCV range
            return False
        
        # 4. Confidence scores should be reasonable
        avg_confidence = (plasma_top.confidence + rbc_top.confidence + rbc_bottom.confidence) / 3
        if avg_confidence < 0.3:  # At least 30% confidence
            return False
        
        return True
    
    def _calculate_pcv_enhanced(self, boundaries: Dict[str, BoundaryResult], warnings: List[str]) -> Dict:
        """Calculate PCV with enhanced precision and validation"""
        plasma_top = boundaries['plasma_top']
        rbc_top = boundaries['rbc_top']
        rbc_bottom = boundaries['rbc_bottom']
        
        if plasma_top.y_position == -1 or rbc_top.y_position == -1 or rbc_bottom.y_position == -1:
            return {
                'pcv': 0.0, 'hemoglobin': 0.0, 'total_height': 0, 'packed_height': 0, 'confidence': 0.0
            }
        
        # Calculate heights
        total_height = rbc_bottom.y_position - plasma_top.y_position
        packed_height = rbc_bottom.y_position - rbc_top.y_position
        
        if total_height <= 0:
            warnings.append("Invalid total height measurement")
            return {
                'pcv': 0.0, 'hemoglobin': 0.0, 'total_height': 0, 'packed_height': 0, 'confidence': 0.0
            }
        
        # Calculate PCV with high precision
        pcv = (packed_height / total_height) * 100.0
        
        # Calculate hemoglobin using PCV/3 rule
        hemoglobin = pcv / 3.0
        
        # Calculate overall confidence
        confidence = (plasma_top.confidence + rbc_top.confidence + rbc_bottom.confidence) / 3.0
        
        # Validate PCV range
        if pcv < 10 or pcv > 70:
            warnings.append(f"PCV value ({pcv:.1f}%) outside expected range (10-70%)")
        
        if hemoglobin < 3 or hemoglobin > 25:
            warnings.append(f"Estimated Hb ({hemoglobin:.1f} g/dL) outside expected range (3-25 g/dL)")
        
        return {
            'pcv': round(pcv, 1),
            'hemoglobin': round(hemoglobin, 1), 
            'total_height': total_height,
            'packed_height': packed_height,
            'confidence': confidence
        }
    
    def _create_annotated_image(self, original_image: np.ndarray, result: AnalysisResult) -> None:
        """Create annotated image with detected boundaries"""
        if not result.success:
            return
        
        # Create copy for annotation
        annotated = original_image.copy()
        
        # Extract tube region
        x, y, w, h = result.tube_rect
        
        # Draw tube rectangle
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
        # Draw boundary lines
        boundaries = result.boundaries
        colors = {
            'plasma_top': (255, 223, 0),    # Gold
            'rbc_top': (0, 255, 0),         # Green  
            'rbc_bottom': (255, 0, 0),      # Red
        }
        
        for boundary_name, boundary in boundaries.items():
            if boundary.y_position != -1:
                color = colors.get(boundary_name, (255, 255, 255))
                line_y = y + boundary.y_position
                
                # Draw line across tube
                cv2.line(annotated, (x, line_y), (x + w, line_y), color, 3)
                
                # Draw confidence indicator
                confidence_text = f"{boundary.confidence:.2f}"
                cv2.putText(annotated, confidence_text, (x + w + 5, line_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw method used
                method_text = boundary.method_used
                cv2.putText(annotated, method_text, (x - 100, line_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add PCV result text
        cv2.putText(annotated, f"PCV: {result.pcv}%", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"Hb: {result.hemoglobin} g/dL", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"Method: {result.detection_method}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Confidence: {result.confidence_overall:.2f}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        self.debug_images['annotated'] = annotated
    
    def save_debug_images(self, output_dir: str) -> None:
        """Save all debug images for analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for name, image in self.debug_images.items():
            if image is not None:
                # Convert RGB to BGR for OpenCV saving
                if len(image.shape) == 3:
                    save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    save_image = image
                
                cv2.imwrite(str(output_path / f"{name}.png"), save_image)
    
    def display_results(self, result: AnalysisResult) -> None:
        """Display analysis results"""
        print("="*60)
        print("ENHANCED PCV ANALYZER RESULTS")
        print("="*60)
        
        if result.success:
            print(f"✅ Analysis Status: SUCCESS")
            print(f"📊 PCV: {result.pcv}%")
            print(f"🩸 Estimated Hemoglobin: {result.hemoglobin} g/dL")
            print(f"📏 Total Height: {result.total_height} pixels")
            print(f"📏 Packed Height: {result.packed_height} pixels")
            print(f"🔍 Detection Method: {result.detection_method}")
            print(f"📈 Overall Confidence: {result.confidence_overall:.2f}")
            print(f"⏱️  Processing Time: {result.processing_time_ms} ms")
            print(f"📍 Tube Location: x={result.tube_rect[0]}, y={result.tube_rect[1]}, w={result.tube_rect[2]}, h={result.tube_rect[3]}")
            
            print("\n🎯 BOUNDARY DETECTION DETAILS:")
            for name, boundary in result.boundaries.items():
                if boundary.y_position != -1:
                    print(f"  {name}: Y={boundary.y_position}px, Confidence={boundary.confidence:.2f}, Method={boundary.method_used}")
                else:
                    print(f"  {name}: FAILED")
            
            if result.warnings:
                print(f"\n⚠️  WARNINGS:")
                for warning in result.warnings:
                    print(f"  - {warning}")
        else:
            print(f"❌ Analysis Status: FAILED")
            print(f"❌ Error: {result.error_message}")
        
        print("="*60)


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Enhanced PCV Analyzer')
    parser.add_argument('image_path', help='Path to the PCV tube image')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--output', default='debug_output', help='Output directory for debug images')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EnhancedPCVAnalyzer(debug_mode=args.debug)
    
    # Analyze image
    result = analyzer.analyze_image(args.image_path)
    
    # Display results
    analyzer.display_results(result)
    
    # Save debug images if requested
    if args.debug:
        analyzer.save_debug_images(args.output)
        print(f"\n🖼️  Debug images saved to: {args.output}")
    
    # Save results to JSON
    output_file = Path(args.output) / "analysis_result.json"
    with open(output_file, 'w') as f:
        json.dump(asdict(result), f, indent=2)
    
    print(f"📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
