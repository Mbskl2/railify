import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from classical_image_recognition.pipline import run_preprocessing_pipeline, run_main_pipeline

INPUT_DIR = 'backend/files/input_pdfs'
OUTPUT_PNG_DIR = 'backend/files/output_pngs'
OUTPUT_SVG_DIR = 'backend/files/output_svgs'

# initialize flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# sample api endpoint
@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_name = file.filename
    file_media_type = file.mimetype

    file_path = os.path.join(INPUT_DIR, file_name)
    global file_path_stupid_global_variable
    file_path_stupid_global_variable = file_path
    print(file_path_stupid_global_variable)
    file.save(file_path)

    if file_media_type != 'application/pdf':
        return jsonify({'error': f'Invalid file type: {file_media_type}'}), 400

    print(f'File name: {file_name}')
    print(f'File media type: {file_media_type}')

    # Read the processed files
    png_file_path = run_preprocessing_pipeline(file_path)

    if not os.path.exists(png_file_path):
        return jsonify({'error': 'Processed files not found'}), 404

    with open(png_file_path, 'rb') as png_file:
        png_data = base64.b64encode(png_file.read()).decode('utf-8')

    return jsonify({
        'message': 'File processed successfully',
        'png_data': png_data
    }), 201

@app.route('/api/process', methods=['POST'])
def process():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    x = int(data.get('left'))
    y = int(data.get('top'))
    width = int(data.get('width'))
    height = int(data.get('height'))

    if x is None or y is None or width is None or height is None:
        return jsonify({'error': 'Missing required parameters'}), 400

    print(f'x: {x}, y: {y}, width: {width}, height: {height}')

    print(file_path_stupid_global_variable)
    # Read the processed files
    png_file_path, svg_file_path = run_main_pipeline(file_path_stupid_global_variable, x, y, width, height)

    if not os.path.exists(png_file_path) or not os.path.exists(svg_file_path):
        return jsonify({'error': 'Processed files not found'}), 404

    with open(png_file_path, 'rb') as png_file:
        png_data = base64.b64encode(png_file.read()).decode('utf-8')

    with open(svg_file_path, 'rb') as svg_file:
        svg_data = base64.b64encode(svg_file.read()).decode('utf-8')

    return jsonify({
        'message': 'File processed successfully',
        'png_data': png_data,
        'svg_data': svg_data
    }), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
