"""
Simple test Flask app to debug the upload error
"""

import os
from flask import Flask, request, jsonify, render_template
import uuid
from werkzeug.utils import secure_filename
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("🔥 Upload endpoint called!")
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"✅ File received: {file.filename}")
        print(f"✅ Content type: {file.content_type}")
        print(f"✅ Content length: {request.content_length}")
        
        # Save the uploaded file
        if file:
            # Generate unique filename
            unique_id = str(uuid.uuid4())
            filename = f"{unique_id}_{secure_filename(file.filename)}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            print(f"💾 Saving file to: {file_path}")
            file.save(file_path)
            print(f"✅ File saved successfully!")
            
            # Simple successful response
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully!',
                'filename': filename,
                'pcv': 45.0,  # Dummy value for testing
                'hemoglobin': 15.0,  # Dummy value
                'confidence': 0.95,
                'boundaries': {
                    'plasma_top': {'y_position': 20, 'confidence': 0.9},
                    'rbc_top': {'y_position': 60, 'confidence': 0.8}, 
                    'rbc_bottom': {'y_position': 120, 'confidence': 0.85}
                },
                'tube_location': {'x': 100, 'y': 50, 'width': 40, 'height': 200},
                'warnings': [],
                'debug_images': {
                    'original': 'data:image/png;base64,test123',
                    'tube_roi': 'data:image/png;base64,test456'
                }
            })
    
    except Exception as e:
        print(f"🚨 ERROR in upload: {str(e)}")
        print(f"🚨 Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting TEST Flask server...")
    print("📍 Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
