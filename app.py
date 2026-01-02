
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import shutil
from omr_engine import OMREngine

app = Flask(__name__)
UPLOAD_DIR = "web_uploads"
OUTPUT_DIR = "static/web_outputs"

# Initialize directories
for d in [UPLOAD_DIR, os.path.join(UPLOAD_DIR, "test"), os.path.join(UPLOAD_DIR, "answer"), OUTPUT_DIR]:
    if not os.path.exists(d): os.makedirs(d)

engine = None # Lazy load model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file_type = request.form.get('type') # 'test' or 'answer'
    files = request.files.getlist('files')
    
    target_dir = os.path.join(UPLOAD_DIR, file_type)
    # Clear old files if answer key
    if file_type == 'answer':
        for f in os.listdir(target_dir): os.remove(os.path.join(target_dir, f))
    
    saved_files = []
    for f in files:
        path = os.path.join(target_dir, f.filename)
        f.save(path)
        saved_files.append(f.filename)
        
    return jsonify({"status": "success", "files": saved_files})

@app.route('/process', methods=['POST'])
def process():
    global engine
    if engine is None:
        engine = OMREngine() # Load CNN
        
    test_dir = os.path.join(UPLOAD_DIR, "test")
    answer_files = os.listdir(os.path.join(UPLOAD_DIR, "answer"))
    if not answer_files:
        return jsonify({"status": "error", "message": "No Answer Key uploaded"}), 400
        
    answer_path = os.path.join(UPLOAD_DIR, "answer", answer_files[0])
    
    # Process
    results = engine.process_all(test_dir, answer_path, OUTPUT_DIR)
    
    return jsonify({"status": "success", "results": results})

@app.route('/clear', methods=['POST'])
def clear():
    for d in [UPLOAD_DIR, OUTPUT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
            os.makedirs(d)
    os.makedirs(os.path.join(UPLOAD_DIR, "test"))
    os.makedirs(os.path.join(UPLOAD_DIR, "answer"))
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
