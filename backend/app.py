import os
from flask import Flask, request, jsonify
from flask_cors import CORS

FILES_DIR = 'files'

# initialize flask application
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# sample api endpoint
@app.route('/api/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # file_content = file.read()
    file_name = file.filename
    file_media_type = file.mimetype

    file_path = os.path.join(FILES_DIR, file_name)
    file.save(file_path)

    if file_media_type != 'application/pdf':
        return jsonify({'error': f'Invalid file type: {file_media_type}'}), 400

    print(f'File name: {file_name}')
    print(f'File media type: {file_media_type}')
    
    return jsonify({'message': 'File processed successfully'}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
