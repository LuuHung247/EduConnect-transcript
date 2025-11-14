from flask import Flask, render_template, request, jsonify, send_file
from faster_whisper import WhisperModel
import torch
import os
from pathlib import Path
import time
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'mp3', 'wav', 'avi', 'mov', 'webm', 'm4a', 'flac', 'ogg'}

# Tạo thư mục nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Khởi tạo model toàn cục
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model(model_size="small"):
    """
    Load Faster-Whisper model
    Model sizes: tiny, base, small, medium, large-v2, large-v3
    Compute types: 
        - GPU: float16 (fastest), float32
        - CPU: int8 (fastest), int8_float16, float32
    """
    global MODEL
    if MODEL is None:
        print(f"🔄 Đang tải Faster-Whisper model '{model_size}' trên {DEVICE}...")
        print(f"   Compute type: {COMPUTE_TYPE}")
        
        # Tham số cho WhisperModel
        model_kwargs = {
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "num_workers": 1,
        }
        
        # Chỉ thêm cpu_threads nếu dùng CPU
        if DEVICE == "cpu":
            model_kwargs["cpu_threads"] = 4
        
        MODEL = WhisperModel(model_size, **model_kwargs)
        print("✅ Model đã sẵn sàng!")
    return MODEL

def transcribe_video(video_path, language=None, use_fast=True):
    """
    Transcribe video với Faster-Whisper
    Nhanh hơn 2-4x so với Whisper gốc!
    """
    model = load_model("small")
    
    # Cấu hình cho tốc độ hoặc độ chính xác
    if use_fast:
        # Chế độ nhanh
        beam_size = 1
        best_of = 1
        vad_filter = True  # Voice Activity Detection - tăng tốc đáng kể
        vad_parameters = {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": float('inf'),
            "min_silence_duration_ms": 100,
            "window_size_samples": 512,
        }
    else:
        # Chế độ chính xác
        beam_size = 5
        best_of = 5
        vad_filter = False
        vad_parameters = None
    
    print(f"⚡ Bắt đầu transcribe: {os.path.basename(video_path)}")
    print(f"   Mode: {'Fast' if use_fast else 'Accurate'}")
    print(f"   VAD: {'Enabled' if vad_filter else 'Disabled'}")
    
    # Thực hiện transcribe
    segments_generator, info = model.transcribe(
        video_path,
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        # Các tối ưu bổ sung
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )
    
    # Thu thập kết quả
    segments = []
    full_text = []
    
    for segment in segments_generator:
        segments.append({
            'start': round(segment.start, 2),
            'end': round(segment.end, 2),
            'text': segment.text.strip()
        })
        full_text.append(segment.text.strip())
    
    result = {
        'text': ' '.join(full_text),
        'language': info.language,
        'language_probability': round(info.language_probability, 4),
        'duration': round(info.duration, 2),
        'segments': segments
    }
    
    print(f"✅ Hoàn tất! Phát hiện {len(segments)} đoạn")
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'Không có file nào được upload'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'Không có file nào được chọn'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Định dạng file không được hỗ trợ'}), 400
    
    # Lưu file
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    return jsonify({
        'success': True,
        'filename': unique_filename,
        'message': 'Upload thành công!'
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json
    filename = data.get('filename')
    language = data.get('language', None)
    use_fast = data.get('fast', True)
    
    if not filename:
        return jsonify({'error': 'Thiếu tên file'}), 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File không tồn tại'}), 404
    
    try:
        start_time = time.time()
        
        # Thực hiện transcribe với Faster-Whisper
        result = transcribe_video(filepath, language=language, use_fast=use_fast)
        
        elapsed_time = time.time() - start_time
        
        # Lưu kết quả
        output_filename = Path(filename).stem + "_transcript.txt"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        
        # Tính speedup (real-time factor)
        video_duration = result.get('duration', 0)
        speedup = video_duration / elapsed_time if elapsed_time > 0 else 0
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'language': result['language'],
            'language_probability': result.get('language_probability', 0),
            'segments': result['segments'],
            'processing_time': round(elapsed_time, 2),
            'video_duration': video_duration,
            'speedup': round(speedup, 2),
            'output_file': output_filename
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download(filename):
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File không tồn tại'}), 404

@app.route('/video/<filename>')
def serve_video(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return jsonify({'error': 'Video không tồn tại'}), 404

@app.route('/stats')
def stats():
    """API để kiểm tra thông tin hệ thống"""
    return jsonify({
        'device': DEVICE,
        'compute_type': COMPUTE_TYPE,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'model_loaded': MODEL is not None
    })

if __name__ == '__main__':
    # Banner
    print("=" * 60)
    print("🚀 FASTER-WHISPER TRANSCRIPTION SERVER")
    print("=" * 60)
    
    # Kiểm tra GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA: {torch.version.cuda}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   Compute Type: {COMPUTE_TYPE}")
    else:
        print("⚠️  Không phát hiện GPU, sử dụng CPU")
        print(f"   Compute Type: {COMPUTE_TYPE}")
    
    print()
    
    # Load model khi khởi động (pre-warming)
    print("🔥 Pre-loading model...")
    load_model("small")
    
    print()
    print("=" * 60)
    print(f"🌐 Server: http://localhost:5001")
    print(f"📊 Stats API: http://localhost:5001/stats")
    print("=" * 60)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)