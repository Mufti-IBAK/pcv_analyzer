"""
Flask Web Application for Enhanced PCV Analyzer
Professional Medical Interface with Manual Correction
Dr. Mufti & Team
"""

import os
import json
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
from pcv_analyzer import EnhancedPCVAnalyzer
from pathlib import Path
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'pcv-analyzer-secret-key-2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global analyzer instance
analyzer = EnhancedPCVAnalyzer(debug_mode=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_array, max_size=800):
    """Convert numpy image array to base64 string with size optimization"""
    # Resize for display if too large
    display_img = image_array.copy()
    h, w = display_img.shape[:2]
    
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    if len(display_img.shape) == 3:
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = display_img
    
    # Encode to PNG
    _, buffer = cv2.imencode('.png', image_bgr)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Analyze image
        result = analyzer.analyze_image(file_path)
        
        # Convert result to JSON-serializable format
        result_dict = {
            'success': result.success,
            'pcv': result.pcv,
            'hemoglobin': result.hemoglobin,
            'total_height': result.total_height,
            'packed_height': result.packed_height,
            'detection_method': result.detection_method,
            'confidence_overall': result.confidence_overall,
            'processing_time_ms': result.processing_time_ms,
            'warnings': result.warnings,
            'error_message': result.error_message,
            'tube_rect': result.tube_rect,
            'boundaries': {}
        }
        
        # Convert boundaries
        for name, boundary in result.boundaries.items():
            result_dict['boundaries'][name] = {
                'y_position': boundary.y_position,
                'confidence': boundary.confidence,
                'method_used': boundary.method_used
            }
        
        # Add debug images as base64
        debug_images = {}
        for name, image in analyzer.debug_images.items():
            if image is not None:
                debug_images[name] = image_to_base64(image)
        
        result_dict['debug_images'] = debug_images
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify(result_dict)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/manual_correction', methods=['POST'])
def manual_correction():
    """Handle manual boundary correction and save training data"""
    try:
        data = request.json
        
        # Extract boundary positions
        plasma_top_y = data.get('plasma_top_y', 0)
        rbc_top_y = data.get('rbc_top_y', 0) 
        rbc_bottom_y = data.get('rbc_bottom_y', 0)
        
        # Calculate new PCV
        total_height = rbc_bottom_y - plasma_top_y
        packed_height = rbc_bottom_y - rbc_top_y
        
        if total_height <= 0:
            return jsonify({'error': 'Invalid boundary positions'}), 400
        
        pcv = (packed_height / total_height) * 100.0
        hemoglobin = pcv / 3.0
        
        # Save training data if requested
        if data.get('save_training', False):
            analyzer.save_manual_correction({
                'plasma_top_y': plasma_top_y,
                'rbc_top_y': rbc_top_y,
                'rbc_bottom_y': rbc_bottom_y,
                'pcv': pcv,
                'hemoglobin': hemoglobin,
                'total_height': total_height,
                'packed_height': packed_height
            })
        
        # Validate ranges
        warnings = []
        if pcv < 10 or pcv > 70:
            warnings.append(f"PCV value ({pcv:.1f}%) outside expected range (10-70%)")
        if hemoglobin < 3 or hemoglobin > 25:
            warnings.append(f"Estimated Hb ({hemoglobin:.1f} g/dL) outside expected range (3-25 g/dL)")
        
        return jsonify({
            'pcv': round(pcv, 1),
            'hemoglobin': round(hemoglobin, 1),
            'total_height': total_height,
            'packed_height': packed_height,
            'warnings': warnings
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'version': '1.0.0'})

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Enhanced PCV Analyzer - Python Version")
    print("🔬 Professional Medical Image Analysis")
    print("💻 Starting Flask development server...")
    print("📍 Access at: http://localhost:5000")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
