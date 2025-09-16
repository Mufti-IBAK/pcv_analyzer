"""
Enhanced PCV Analyzer - Python Version
Multi-Stage Hybrid Analysis Algorithm with Robust Detection
Dr. Mufti & Team - Professional Medical Image Analysis
"""

import os
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
        self.training_data = []
        self.training_file = 'manual_corrections.json'
        self._load_training_data()
        
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
            
            # Validate oriented rectangle bounds
            x, y, w, h = oriented_rect
            img_h, img_w = oriented_image.shape[:2]
            
            if x < 0 or y < 0 or x + w > img_w or y + h > img_h or w <= 0 or h <= 0:
                return AnalysisResult(
                    pcv=0, hemoglobin=0, total_height=0, packed_height=0,
                    boundaries={}, tube_rect=(0,0,0,0), detection_method="invalid_roi",
                    confidence_overall=0, processing_time_ms=int((time.time() - start_time) * 1000),
                    warnings=warnings, success=False, 
                    error_message=f"Invalid tube ROI bounds: {oriented_rect} for image size {img_w}x{img_h}"
                )
            
            # Extract tube ROI safely
            tube_roi = oriented_image[y:y+h, x:x+w]
            
            # Validate tube ROI
            if tube_roi.size == 0:
                return AnalysisResult(
                    pcv=0, hemoglobin=0, total_height=0, packed_height=0,
                    boundaries={}, tube_rect=(0,0,0,0), detection_method="empty_roi",
                    confidence_overall=0, processing_time_ms=int((time.time() - start_time) * 1000),
                    warnings=warnings, success=False, 
                    error_message="Empty tube ROI extracted"
                )
            
            if len(tube_roi.shape) != 3 or tube_roi.shape[2] != 3:
                return AnalysisResult(
                    pcv=0, hemoglobin=0, total_height=0, packed_height=0,
                    boundaries={}, tube_rect=(0,0,0,0), detection_method="invalid_roi_shape",
                    confidence_overall=0, processing_time_ms=int((time.time() - start_time) * 1000),
                    warnings=warnings, success=False, 
                    error_message=f"Invalid tube ROI shape: {tube_roi.shape}. Expected (H,W,3)"
                )
            
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
    
    def save_manual_correction(self, correction_data: dict):
        """Save manual correction data for training"""
        import json
        import datetime
        
        correction_data['timestamp'] = datetime.datetime.now().isoformat()
        self.training_data.append(correction_data)
        
        try:
            with open(self.training_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save training data: {e}")
    
    def _load_training_data(self):
        """Load existing training data"""
        import json
        
        try:
            if os.path.exists(self.training_file):
                with open(self.training_file, 'r') as f:
                    self.training_data = json.load(f)
        except Exception as e:
            print(f"Failed to load training data: {e}")
            self.training_data = []
    
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
            
            # Fix coordinate transformation for 90-degree counterclockwise rotation
            new_x = max(0, y)
            new_y = max(0, new_width - (x + w))
            new_w = min(h, new_width - new_x)
            new_h = min(w, new_height - new_y)
            
            # Ensure valid bounds
            if new_x + new_w > new_width:
                new_w = new_width - new_x
            if new_y + new_h > new_height:
                new_h = new_height - new_y
                
            new_roi = (new_x, new_y, new_w, new_h)
            return self._ensure_vertical_orientation_advanced(rotated_image, new_roi)
        
        # Advanced plasticine detection for orientation
        plasticine_bottom_score = self._detect_plasticine_position(tube_roi, 'bottom')
        plasticine_top_score = self._detect_plasticine_position(tube_roi, 'top')
        
        # If plasticine detected at top more than bottom, flip tube
        if plasticine_top_score > plasticine_bottom_score + 0.2:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            new_height, new_width = rotated_image.shape[:2]
            
            # Fix coordinate transformation for 180-degree rotation
            new_x = max(0, min(new_width - w, new_width - (x + w)))
            new_y = max(0, min(new_height - h, new_height - (y + h)))
            
            # Ensure valid bounds
            new_w = min(w, new_width - new_x)
            new_h = min(h, new_height - new_y)
            
            new_roi = (new_x, new_y, new_w, new_h)
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
        """Enhanced color-based detection with robust background removal"""
        # Step 1: Advanced background removal for any background color
        tube_mask = self._remove_background_advanced(image)
        
        # Step 2: Detect tube-specific colors within the mask
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        tube_content_masks = []
        
        # Enhanced plasticine detection (blue/green)
        plasticine_ranges = [
            ([100, 50, 50], [140, 255, 255]),   # Blue plasticine
            ([40, 50, 50], [80, 255, 255]),     # Green plasticine
        ]
        
        for lower, upper in plasticine_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            tube_content_masks.append(mask)
        
        # Enhanced RBC detection (red shades)
        rbc_ranges = [
            ([0, 80, 80], [15, 255, 255]),      # Bright red
            ([165, 80, 80], [180, 255, 255]),   # Deep red
            ([0, 40, 60], [15, 255, 200]),      # Dark red
        ]
        
        for lower, upper in rbc_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            tube_content_masks.append(mask)
        
        # Plasma/buffy coat detection (pale/yellow)
        plasma_ranges = [
            ([15, 20, 100], [35, 150, 255]),    # Yellow plasma
            ([0, 0, 150], [180, 50, 255]),      # Pale/white buffy coat
        ]
        
        for lower, upper in plasma_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            tube_content_masks.append(mask)
        
        # Combine tube content masks
        tube_content = np.zeros_like(tube_mask)
        for mask in tube_content_masks:
            tube_content = cv2.bitwise_or(tube_content, mask)
        
        # Combine with background-removed mask
        final_mask = cv2.bitwise_and(tube_content, tube_mask)
        
        # Morphological operations to clean up
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)
        
        self.debug_images['enhanced_color'] = final_mask
        
        # Find contours and detect tube
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return self._filter_tube_contours(contours, image.shape)
    
    def _remove_background_advanced(self, image: np.ndarray) -> np.ndarray:
        """Advanced background removal that works with any background color"""
        height, width = image.shape[:2]
        
        # Method 1: Edge-based background removal
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Method 2: Color clustering to identify background
        # Sample border pixels to estimate background color
        border_samples = []
        
        # Top and bottom borders
        border_samples.extend(image[0:5, :].reshape(-1, 3))
        border_samples.extend(image[-5:, :].reshape(-1, 3))
        
        # Left and right borders  
        border_samples.extend(image[:, 0:5].reshape(-1, 3))
        border_samples.extend(image[:, -5:].reshape(-1, 3))
        
        border_samples = np.array(border_samples)
        
        # Estimate background color (median of border samples)
        if len(border_samples) > 0:
            bg_color = np.median(border_samples, axis=0)
            
            # Create mask for background color (with tolerance)
            tolerance = 40
            bg_mask = np.zeros((height, width), dtype=np.uint8)
            
            for i in range(3):  # RGB channels
                channel_diff = np.abs(image[:,:,i].astype(float) - bg_color[i])
                bg_mask = np.logical_or(bg_mask, channel_diff > tolerance)
            
            # Invert to get foreground mask
            fg_mask = bg_mask.astype(np.uint8) * 255
        else:
            # Fallback: use edge detection only
            fg_mask = edges
        
        # Method 3: Combine with adaptive thresholding
        adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
        
        # Combine all methods
        combined_mask = cv2.bitwise_or(fg_mask, edges)
        combined_mask = cv2.bitwise_or(combined_mask, adaptive_mask)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        self.debug_images['background_removal'] = combined_mask
        
        return combined_mask
    
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
        # Validate input
        if tube_roi is None or tube_roi.size == 0:
            raise ValueError("Empty tube ROI provided to profile creation")
        
        if len(tube_roi.shape) != 3 or tube_roi.shape[2] != 3:
            raise ValueError(f"Invalid tube ROI shape: {tube_roi.shape}. Expected (H, W, 3)")
        
        height, width = tube_roi.shape[:2]
        
        if height < 10 or width < 5:
            raise ValueError(f"Tube ROI too small: {width}x{height}. Minimum size: 5x10")
        
        try:
            # Convert to multiple color spaces for analysis
            hsv = cv2.cvtColor(tube_roi, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(tube_roi, cv2.COLOR_RGB2LAB)
            gray = cv2.cvtColor(tube_roi, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            raise ValueError(f"Color space conversion failed: {e}")
        
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
        """Advanced sequential color detection following medical hematocrit methodology"""
        boundaries = {}
        
        # Step 1: Detect plasticine bottom (blue/green) - starting point
        plasticine_bottom = self._detect_plasticine_bottom(profiles, roi_height)
        
        # Step 2: Detect RBC bottom (transition from RBC red to plasticine)
        rbc_bottom = self._detect_rbc_bottom_transition(profiles, roi_height, plasticine_bottom)
        boundaries['rbc_bottom'] = rbc_bottom
        
        # Step 3: Detect RBC top (transition from red to white/buffy coat)
        rbc_top = self._detect_rbc_top_transition(profiles, roi_height, rbc_bottom)
        boundaries['rbc_top'] = rbc_top
        
        # Step 4: Detect buffy coat boundaries (white to pale yellow)
        buffy_boundaries = self._detect_buffy_coat_layer(profiles, roi_height, rbc_top)
        boundaries['buffy_bottom'] = buffy_boundaries['bottom']
        boundaries['buffy_top'] = buffy_boundaries['top']
        
        # Step 5: Detect plasma top (transition from plasma yellow to air/glass)
        plasma_top = self._detect_plasma_top_transition(profiles, roi_height, buffy_boundaries['top'])
        boundaries['plasma_top'] = plasma_top
        
        return boundaries
    
    def _detect_plasticine_bottom(self, profiles: Dict, roi_height: int) -> BoundaryResult:
        """Step 1: Detect plasticine bottom (blue/green clay)"""
        # Search from bottom up to find the plasticine layer
        search_start = roi_height - 5
        search_end = max(int(roi_height * 0.7), search_start - 50)  # Don't go too far up
        
        candidates = []
        
        for y in range(search_start, search_end, -1):
            # Multi-criteria plasticine detection
            hue = profiles['hue'][y]
            saturation = profiles['saturation'][y]
            blue_value = profiles['blue'][y]
            
            # Blue plasticine characteristics (hue 100-140)
            blue_score = 0
            if 100 <= hue <= 140 and saturation > 80:
                blue_score = saturation * 0.1 + (blue_value - 128) * 0.05
            
            # Green plasticine characteristics (hue 40-80)
            green_score = 0
            if 40 <= hue <= 80 and saturation > 70:
                green_score = saturation * 0.08 + profiles['green'][y] * 0.03
            
            total_score = max(blue_score, green_score)
            if total_score > 5:
                candidates.append((y, total_score, 'plasticine_detection'))
        
        # Return the topmost (lowest y) plasticine detection
        if candidates:
            candidates.sort(key=lambda x: x[0])  # Sort by y position
            best = candidates[0]
            return BoundaryResult(best[0], min(1.0, best[1] / 20), best[2])
        
        # Fallback: use bottom portion as plasticine
        fallback_y = roi_height - 10
        return BoundaryResult(fallback_y, 0.3, 'fallback_bottom')
    
    def _detect_rbc_bottom_transition(self, profiles: Dict, roi_height: int, plasticine_bottom: BoundaryResult) -> BoundaryResult:
        """Step 2: Detect transition from RBC (red) to plasticine"""
        if plasticine_bottom.y_position == -1:
            return BoundaryResult(-1, 0.0, "no_plasticine_reference")
        
        # Search upward from plasticine bottom
        search_start = plasticine_bottom.y_position - 5
        search_end = max(int(roi_height * 0.4), search_start - 100)
        
        candidates = []
        
        for y in range(search_start, search_end, -1):
            # Look for red blood cell characteristics
            red_value = profiles['red'][y]
            hue = profiles['hue'][y]
            saturation = profiles['saturation'][y]
            
            # RBC characteristics: high red, hue 0-20 or 160-180
            rbc_score = 0
            if (0 <= hue <= 20 or 160 <= hue <= 180) and red_value > 120:
                rbc_score = red_value * 0.1 + saturation * 0.05
            
            # Also check for dark red (lower value but high red channel)
            if red_value > 100 and profiles['value'][y] < 100:
                rbc_score = max(rbc_score, red_value * 0.08)
            
            if rbc_score > 8:
                candidates.append((y, rbc_score, 'rbc_detection'))
        
        # Find the bottommost RBC detection (highest y in search range)
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best = candidates[0]
            return BoundaryResult(best[0], min(1.0, best[1] / 25), best[2])
        
        # Fallback
        fallback_y = plasticine_bottom.y_position - 20
        return BoundaryResult(max(0, fallback_y), 0.4, 'fallback_rbc_bottom')
    
    def _detect_rbc_top_transition(self, profiles: Dict, roi_height: int, rbc_bottom: BoundaryResult) -> BoundaryResult:
        """Step 3: Detect transition from RBC (red) to buffy coat (white)"""
        if rbc_bottom.y_position == -1:
            return BoundaryResult(-1, 0.0, "no_rbc_bottom_reference")
        
        # Search upward from RBC bottom
        search_start = rbc_bottom.y_position - 5
        search_end = max(int(roi_height * 0.2), search_start - 150)
        
        candidates = []
        
        for y in range(search_start, search_end, -1):
            # Look for transition from red to white/pale
            red_drop = profiles['red'][y-5] - profiles['red'][y] if y >= 5 else 0
            value_increase = profiles['value'][y] - profiles['value'][y+5] if y < len(profiles['value'])-5 else 0
            saturation_drop = profiles['saturation'][y-5] - profiles['saturation'][y] if y >= 5 else 0
            
            # Score for RBC to buffy coat transition
            transition_score = 0
            if red_drop > 10:  # Red decreasing
                transition_score += red_drop * 0.1
            if value_increase > 10:  # Brightness increasing
                transition_score += value_increase * 0.1
            if saturation_drop > 10:  # Less saturated (more white)
                transition_score += saturation_drop * 0.1
            
            if transition_score > 2:
                candidates.append((y, transition_score, 'rbc_to_buffy_transition'))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0]
            return BoundaryResult(best[0], min(1.0, best[1] / 5), best[2])
        
        # Fallback
        fallback_y = rbc_bottom.y_position - 50
        return BoundaryResult(max(0, fallback_y), 0.3, 'fallback_rbc_top')
    
    def _detect_buffy_coat_layer(self, profiles: Dict, roi_height: int, rbc_top: BoundaryResult) -> Dict[str, BoundaryResult]:
        """Step 4: Detect thin buffy coat layer (white to pale yellow)"""
        if rbc_top.y_position == -1:
            return {
                'bottom': BoundaryResult(-1, 0.0, "no_rbc_top_reference"),
                'top': BoundaryResult(-1, 0.0, "no_rbc_top_reference")
            }
        
        # Buffy coat is typically very thin (2-5 pixels)
        buffy_bottom = rbc_top.y_position  # Start of buffy coat is end of RBC
        
        # Search upward for buffy coat characteristics
        search_start = rbc_top.y_position - 2
        search_end = max(int(roi_height * 0.1), search_start - 20)
        
        buffy_candidates = []
        
        for y in range(search_start, search_end, -1):
            # Buffy coat characteristics: whitish with slight yellow tint
            value = profiles['value'][y]  # Brightness
            saturation = profiles['saturation'][y]  # Should be low (white-ish)
            hue = profiles['hue'][y]
            b_channel = profiles['b_channel'][y]  # Yellow component in LAB
            
            # White/pale characteristics
            buffy_score = 0
            if value > 150 and saturation < 50:  # Bright but not saturated
                buffy_score += (value - 150) * 0.1 + (50 - saturation) * 0.1
            
            # Slight yellow tint (hue 20-60 or positive b_channel)
            if (20 <= hue <= 60 and saturation < 30) or b_channel > 130:
                buffy_score += 2
            
            if buffy_score > 1:
                buffy_candidates.append((y, buffy_score, 'buffy_coat'))
        
        # Find buffy coat top
        if buffy_candidates:
            buffy_candidates.sort(key=lambda x: x[0])  # Topmost
            buffy_top_y = buffy_candidates[0][0]
            buffy_top_confidence = min(1.0, buffy_candidates[0][1] / 5)
        else:
            # Very thin buffy coat fallback
            buffy_top_y = max(0, rbc_top.y_position - 3)
            buffy_top_confidence = 0.2
        
        return {
            'bottom': BoundaryResult(buffy_bottom, rbc_top.confidence, 'buffy_bottom_from_rbc_top'),
            'top': BoundaryResult(buffy_top_y, buffy_top_confidence, 'buffy_coat_detection')
        }
    
    def _detect_plasma_top_transition(self, profiles: Dict, roi_height: int, buffy_top: BoundaryResult) -> BoundaryResult:
        """Step 5: Detect transition from plasma (yellow) to air/glass (colorless)"""
        if buffy_top.y_position == -1:
            return BoundaryResult(-1, 0.0, "no_buffy_top_reference")
        
        # Search upward from buffy coat top
        search_start = buffy_top.y_position - 2
        search_end = max(0, search_start - 100)
        
        candidates = []
        
        for y in range(search_start, search_end, -1):
            # Look for transition from plasma (yellowish) to air/glass (colorless)
            value_drop = profiles['value'][y-5] - profiles['value'][y] if y >= 5 else 0
            saturation_drop = profiles['saturation'][y-5] - profiles['saturation'][y] if y >= 5 else 0
            b_channel_drop = profiles['b_channel'][y-5] - profiles['b_channel'][y] if y >= 5 else 0
            
            # Score for plasma to air transition
            transition_score = 0
            if value_drop > 15:  # Brightness drops (liquid to air)
                transition_score += value_drop * 0.1
            if saturation_drop > 5:  # Color disappears
                transition_score += saturation_drop * 0.2
            if b_channel_drop > 5:  # Yellow disappears
                transition_score += b_channel_drop * 0.1
            
            # Also check for very low values (air/glass)
            if profiles['value'][y] < 80 and profiles['saturation'][y] < 20:
                transition_score += 2
            
            if transition_score > 1.5:
                candidates.append((y, transition_score, 'plasma_to_air_transition'))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0]
            return BoundaryResult(best[0], min(1.0, best[1] / 5), best[2])
        
        # Fallback
        fallback_y = max(0, buffy_top.y_position - 30)
        return BoundaryResult(fallback_y, 0.3, 'fallback_plasma_top')
    
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
        """Enhanced boundary validation for sequential layer detection"""
        plasma_top = boundaries.get('plasma_top', BoundaryResult(-1, 0, ""))
        buffy_top = boundaries.get('buffy_top', BoundaryResult(-1, 0, ""))
        buffy_bottom = boundaries.get('buffy_bottom', BoundaryResult(-1, 0, ""))  # Same as RBC top
        rbc_top = boundaries.get('rbc_top', BoundaryResult(-1, 0, ""))
        rbc_bottom = boundaries.get('rbc_bottom', BoundaryResult(-1, 0, ""))
        
        # Check if critical boundaries were detected
        if plasma_top.y_position == -1 or rbc_top.y_position == -1 or rbc_bottom.y_position == -1:
            return False
        
        # Sequential validation: plasma_top < buffy_top < rbc_top < rbc_bottom
        # Note: buffy layers might be very thin or undetected, so be flexible
        if not (plasma_top.y_position < rbc_top.y_position < rbc_bottom.y_position):
            return False
        
        # Enhanced validation checks for medical accuracy
        # 1. Minimum distances between boundaries
        min_plasma_rbc_distance = 5   # Reduced for buffy coat consideration
        min_rbc_height = 15  # Minimum RBC layer thickness
        
        # Calculate layer sizes
        plasma_height = rbc_top.y_position - plasma_top.y_position
        rbc_height = rbc_bottom.y_position - rbc_top.y_position
        total_height = rbc_bottom.y_position - plasma_top.y_position
        
        if plasma_height < min_plasma_rbc_distance:
            return False
        
        if rbc_height < min_rbc_height:
            return False
        
        # 2. Total height should be reasonable
        if total_height < 40 or total_height > 600:  # 4mm to 60mm range
            return False
        
        # 3. PCV should be in medically reasonable range
        pcv = (rbc_height / total_height) * 100
        if pcv < 8 or pcv > 75:  # Slightly expanded medical PCV range
            return False
        
        # 4. Confidence scores should be reasonable for critical boundaries
        critical_confidence = (plasma_top.confidence + rbc_top.confidence + rbc_bottom.confidence) / 3
        if critical_confidence < 0.25:  # Reduced threshold
            return False
        
        return True
    
    def _calculate_pcv_enhanced(self, boundaries: Dict[str, BoundaryResult], warnings: List[str]) -> Dict:
        """Comprehensive measurement system with all layer sizes"""
        plasma_top = boundaries.get('plasma_top', BoundaryResult(-1, 0, ""))
        buffy_top = boundaries.get('buffy_top', BoundaryResult(-1, 0, ""))
        buffy_bottom = boundaries.get('buffy_bottom', BoundaryResult(-1, 0, ""))
        rbc_top = boundaries.get('rbc_top', BoundaryResult(-1, 0, ""))
        rbc_bottom = boundaries.get('rbc_bottom', BoundaryResult(-1, 0, ""))
        
        # Check critical boundaries
        if plasma_top.y_position == -1 or rbc_top.y_position == -1 or rbc_bottom.y_position == -1:
            return {
                'pcv': 0.0, 'hemoglobin': 0.0, 'total_height': 0, 'packed_height': 0, 
                'rbc_size': 0, 'buffy_size': 0, 'plasma_size': 0, 'total_size': 0, 'confidence': 0.0
            }
        
        # Calculate comprehensive layer measurements
        # Total size: from plasma top to RBC bottom (as specified)
        total_size = rbc_bottom.y_position - plasma_top.y_position
        
        # RBC size: height of red blood cell layer
        rbc_size = rbc_bottom.y_position - rbc_top.y_position
        
        # Plasma size: from plasma top to buffy coat bottom (or RBC top)
        if buffy_bottom.y_position != -1:
            plasma_size = buffy_bottom.y_position - plasma_top.y_position
        else:
            plasma_size = rbc_top.y_position - plasma_top.y_position
        
        # Buffy coat size: very thin layer between RBC and plasma
        if buffy_top.y_position != -1 and buffy_bottom.y_position != -1:
            buffy_size = buffy_bottom.y_position - buffy_top.y_position
        else:
            # Estimate minimal buffy coat if not detected
            buffy_size = 2  # Minimal buffy coat assumption
        
        # Validation
        if total_size <= 0:
            warnings.append("Invalid total height measurement")
            return {
                'pcv': 0.0, 'hemoglobin': 0.0, 'total_height': 0, 'packed_height': 0,
                'rbc_size': 0, 'buffy_size': 0, 'plasma_size': 0, 'total_size': 0, 'confidence': 0.0
            }
        
        # Calculate PCV: RBC size divided by Total size × 100
        pcv = (rbc_size / total_size) * 100.0
        
        # Calculate hemoglobin using PCV/3 rule
        hemoglobin = pcv / 3.0
        
        # Calculate overall confidence (prioritize critical boundaries)
        confidence = (plasma_top.confidence * 0.3 + rbc_top.confidence * 0.4 + rbc_bottom.confidence * 0.3)
        
        # Validate PCV range
        if pcv < 8 or pcv > 75:
            warnings.append(f"PCV value ({pcv:.1f}%) outside expected range (8-75%)")
        
        if hemoglobin < 2.5 or hemoglobin > 25:
            warnings.append(f"Estimated Hb ({hemoglobin:.1f} g/dL) outside expected range (2.5-25 g/dL)")
        
        # Check layer proportions
        if plasma_size / total_size < 0.2:  # Plasma should be at least 20%
            warnings.append("Plasma layer appears unusually small")
        
        if rbc_size / total_size > 0.8:  # RBC shouldn't exceed 80%
            warnings.append("RBC layer appears unusually large")
        
        return {
            'pcv': round(pcv, 1),
            'hemoglobin': round(hemoglobin, 1), 
            'total_height': total_size,  # For backward compatibility
            'packed_height': rbc_size,  # For backward compatibility
            'rbc_size': rbc_size,
            'buffy_size': buffy_size,
            'plasma_size': plasma_size,
            'total_size': total_size,
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
