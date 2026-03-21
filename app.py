import os
import io
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
from faster_whisper import WhisperModel
import tempfile
import base64
import time

app = Flask(__name__)
CORS(app)

# 音頻限制設定（這些要在 Flask config 之前定義）
MAX_AUDIO_SIZE = int(os.environ.get("MAX_AUDIO_SIZE_MB", 100)) * 1024 * 1024  # 預設 100MB
MAX_RECORDING_TIME = int(os.environ.get("MAX_RECORDING_SECONDS", 3600))  # 預設 1 小時
MAX_TRANSCRIPTION_TIME = int(os.environ.get("MAX_TRANSCRIPTION_TIME", 300))  # 預設 5 分鐘

# Flask 上傳設定
app.config['MAX_CONTENT_LENGTH'] = MAX_AUDIO_SIZE
app.config['REQUEST_TIMEOUT'] = MAX_TRANSCRIPTION_TIME

MODEL_SIZE = os.environ.get("MODEL_SIZE", "small")
MODEL_PATH = os.environ.get("MODEL_PATH", None)

print(f"Loading Whisper model: {MODEL_SIZE}...")
if MODEL_PATH:
    model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")
else:
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Model loaded successfully!")

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎙️ Whisper 語音轉文字</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            width: 100%;
            max-width: 600px;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 1.8rem;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9rem;
        }
        
        /* Tabs */
        .tabs {
            display: flex;
            margin-bottom: 25px;
            border-bottom: 2px solid #eee;
        }
        .tab {
            flex: 1;
            padding: 12px;
            text-align: center;
            cursor: pointer;
            font-weight: 600;
            color: #999;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }
        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        
        /* Upload Tab */
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 12px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }
        .upload-area:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .upload-area.dragover {
            border-color: #667eea;
            background: #f0f3ff;
        }
        .upload-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        .upload-text {
            color: #666;
            font-size: 1rem;
        }
        .upload-hint {
            color: #999;
            font-size: 0.85rem;
            margin-top: 8px;
        }
        .upload-limit {
            color: #48bb78;
            font-size: 0.85rem;
            margin-top: 8px;
            font-weight: 600;
        }
        
        /* Record Tab */
        .record-area {
            text-align: center;
            padding: 30px 20px;
        }
        .record-btn {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 48px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        .record-btn:hover {
            transform: scale(1.05);
        }
        .record-btn.recording {
            background: linear-gradient(135deg, #e66767 0%, #c0392b 100%);
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        .record-status {
            margin-top: 20px;
            color: #666;
            font-size: 1rem;
        }
        .record-timer {
            font-size: 2rem;
            font-weight: 700;
            color: #333;
            margin-top: 10px;
        }
        .limit-info {
            font-size: 0.8rem;
            color: #999;
            margin-top: 10px;
        }
        
        /* Common */
        #fileInput {
            display: none;
        }
        .file-info {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 12px 15px;
            margin-bottom: 20px;
            display: none;
        }
        .file-name {
            font-weight: 600;
            color: #333;
        }
        .file-size {
            color: #666;
            font-size: 0.85rem;
            margin-top: 5px;
        }
        .btn {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 15px;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        .btn-primary:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .language-select {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
            margin-bottom: 20px;
            background: white;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading-text {
            color: #666;
        }
        .error {
            background: #fee;
            color: #c00;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }
        .result-area {
            margin-top: 25px;
            display: none;
        }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .result-title {
            font-weight: 600;
            color: #333;
        }
        .copy-btn {
            background: #f0f0f0;
            border: none;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: all 0.3s;
        }
        .copy-btn:hover {
            background: #e0e0e0;
        }
        .result-text {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            line-height: 1.8;
            color: #333;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .audio-preview {
            margin-top: 15px;
            display: none;
        }
        .audio-preview audio {
            width: 100%;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Whisper 語音轉文字</h1>
        <p class="subtitle">基於 Faster Whisper - 支援錄音和檔案上傳</p>
        
        <!-- Tabs -->
        <div class="tabs">
            <div class="tab active" onclick="switchTab('upload')">📁 上傳檔案</div>
            <div class="tab" onclick="switchTab('record')">🎤 錄音</div>
        </div>
        
        <!-- Upload Tab -->
        <div class="tab-content active" id="uploadTab">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📁</div>
                <div class="upload-text">點擊或拖曳音檔到这里</div>
                <div class="upload-hint">支援 MP3, WAV, M4A, OGG, FLAC, WebM</div>
                <div class="upload-limit">📦 最大支援 {{ max_size_mb }}MB 音檔</div>
            </div>
            <input type="file" id="fileInput" accept="audio/*">
            
            <div class="file-info" id="fileInfo">
                <div class="file-name" id="fileName"></div>
                <div class="file-size" id="fileSize"></div>
            </div>
            
            <select class="language-select" id="language">
                <option value="">自動偵測語言</option>
                <option value="zh">中文 (Chinese)</option>
                <option value="en">English (英文)</option>
                <option value="ja">日本語 (日文)</option>
                <option value="ko">한국어 (韓文)</option>
                <option value="fr">Français (法文)</option>
                <option value="de">Deutsch (德文)</option>
                <option value="es">Español (西班牙文)</option>
                <option value="id">Bahasa Indonesia (印尼文)</option>
                <option value="vi">Tiếng Việt (越南文)</option>
                <option value="th">ภาษาไทย (泰文)</option>
            </select>
            
            <button class="btn btn-primary" id="transcribeBtn" disabled>開始轉換</button>
        </div>
        
        <!-- Record Tab -->
        <div class="tab-content" id="recordTab">
            <div class="record-area">
                <button class="record-btn" id="recordBtn" onclick="toggleRecording()">🎤</button>
                <div class="record-status" id="recordStatus">點擊開始錄音</div>
                <div class="record-timer" id="recordTimer">00:00:00</div>
                <div class="limit-info">🎙️ 最長支援 {{ max_hours }} 小時連續錄音</div>
            </div>
            
            <select class="language-select" id="languageRecord">
                <option value="">自動偵測語言</option>
                <option value="zh">中文 (Chinese)</option>
                <option value="en">English (英文)</option>
                <option value="ja">日本語 (日文)</option>
                <option value="ko">한국어 (韓文)</option>
                <option value="id">Bahasa Indonesia (印尼文)</option>
                <option value="vi">Tiếng Việt (越南文)</option>
                <option value="th">ภาษาไทย (泰文)</option>
            </select>
            
            <button class="btn btn-primary" id="transcribeRecordBtn" onclick="transcribeRecording()" disabled>轉換錄音</button>
            
            <div class="audio-preview" id="audioPreview">
                <audio id="audioPlayer" controls></audio>
            </div>
        </div>
        
        <!-- Loading & Result -->
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div class="loading-text" id="loadingText">正在轉換中，請稍候...</div>
            <div class="limit-info" style="margin-top:10px;">較大的檔案可能需要幾分鐘處理</div>
        </div>
        
        <div class="error" id="error"></div>
        
        <div class="result-area" id="resultArea">
            <div class="result-header">
                <span class="result-title">📝 轉換結果</span>
                <button class="copy-btn" onclick="copyResult()">複製文字</button>
            </div>
            <div class="result-text" id="resultText"></div>
        </div>
    </div>
    
    <script>
        // Tab switching
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab:nth-child(${tab === 'upload' ? 1 : 2})`).classList.add('active');
            document.getElementById(tab + 'Tab').classList.add('active');
        }
        
        // Upload functionality
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const transcribeBtn = document.getElementById('transcribeBtn');
        
        let selectedFile = null;
        
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            if (!file.type.startsWith('audio/')) {
                showError('請選擇音訊檔案');
                return;
            }
            selectedFile = file;
            fileName.textContent = file.name;
            fileSize.textContent = formatSize(file.size);
            fileInfo.style.display = 'block';
            transcribeBtn.disabled = false;
            error.style.display = 'none';
            resultArea.style.display = 'none';
        }
        
        function formatSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }
        
        transcribeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            transcribeBtn.disabled = true;
            showLoading();
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            const lang = document.getElementById('language').value;
            if (lang) formData.append('language', lang);
            
            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || '轉換失敗');
                showResult(data.text || '沒有偵測到文字');
            } catch (err) {
                showError(err.message);
            } finally {
                transcribeBtn.disabled = false;
            }
        });
        
        // Recording functionality
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let recordedBlob = null;
        let timerInterval = null;
        let seconds = 0;
        
        const recordBtn = document.getElementById('recordBtn');
        const recordStatus = document.getElementById('recordStatus');
        const recordTimer = document.getElementById('recordTimer');
        const transcribeRecordBtn = document.getElementById('transcribeRecordBtn');
        const audioPreview = document.getElementById('audioPreview');
        const audioPlayer = document.getElementById('audioPlayer');
        
        async function toggleRecording() {
            if (!isRecording) {
                // Start recording
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = (e) => {
                        audioChunks.push(e.data);
                    };
                    
                    mediaRecorder.onstop = () => {
                        recordedBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const audioUrl = URL.createObjectURL(recordedBlob);
                        audioPlayer.src = audioUrl;
                        audioPreview.style.display = 'block';
                        transcribeRecordBtn.disabled = false;
                        stream.getTracks().forEach(track => track.stop());
                    };
                    
                    mediaRecorder.start();
                    isRecording = true;
                    recordBtn.classList.add('recording');
                    recordBtn.textContent = '⏹️';
                    recordStatus.textContent = '錄音中...點擊停止';
                    seconds = 0;
                    timerInterval = setInterval(() => {
                        seconds++;
                        const hrs = Math.floor(seconds / 3600).toString().padStart(2, '0');
                        const mins = Math.floor((seconds % 3600) / 60).toString().padStart(2, '0');
                        const secs = (seconds % 60).toString().padStart(2, '0');
                        recordTimer.textContent = `${hrs}:${mins}:${secs}`;
                    }, 1000);
                } catch (err) {
                    showError('無法訪問麥克風: ' + err.message);
                }
            } else {
                // Stop recording
                mediaRecorder.stop();
                isRecording = false;
                recordBtn.classList.remove('recording');
                recordBtn.textContent = '🎤';
                recordStatus.textContent = '錄音完成';
                clearInterval(timerInterval);
            }
        }
        
        async function transcribeRecording() {
            if (!recordedBlob) return;
            
            transcribeRecordBtn.disabled = true;
            showLoading();
            
            const formData = new FormData();
            formData.append('file', recordedBlob, 'recording.webm');
            const lang = document.getElementById('languageRecord').value;
            if (lang) formData.append('language', lang);
            
            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || '轉換失敗');
                showResult(data.text || '沒有偵測到文字');
            } catch (err) {
                showError(err.message);
            } finally {
                transcribeRecordBtn.disabled = false;
            }
        }
        
        // Common
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const resultArea = document.getElementById('resultArea');
        const resultText = document.getElementById('resultText');
        
        function showLoading() {
            loading.style.display = 'block';
            error.style.display = 'none';
            resultArea.style.display = 'none';
        }
        
        function showError(msg) {
            loading.style.display = 'none';
            error.textContent = msg;
            error.style.display = 'block';
        }
        
        function showResult(text) {
            loading.style.display = 'none';
            error.style.display = 'none';
            resultText.textContent = text;
            resultArea.style.display = 'block';
        }
        
        function copyResult() {
            navigator.clipboard.writeText(resultText.textContent).then(() => {
                const btn = document.querySelector('.copy-btn');
                btn.textContent = '已複製!';
                setTimeout(() => btn.textContent = '複製文字', 2000);
            });
        }
    </script>
</body>
</html>
'''

@app.route("/")
def index():
    max_size_mb = int(os.environ.get("MAX_AUDIO_SIZE_MB", 100))
    max_hours = int(os.environ.get("MAX_RECORDING_SECONDS", 3600)) // 3600
    return render_template_string(HTML_TEMPLATE, max_size_mb=max_size_mb, max_hours=max_hours)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_SIZE})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    # 檢查檔案大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_AUDIO_SIZE:
        max_mb = MAX_AUDIO_SIZE / (1024 * 1024)
        return jsonify({
            "error": f"檔案太大，請上傳小於 {max_mb:.0f}MB 的音檔"
        }), 400
    
    language = request.form.get("language", None)
    
    # 保留原始副檔名
    ext = os.path.splitext(file.filename)[1] if file.filename else ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        print(f"Processing audio file: {file_size / (1024*1024):.2f} MB")
        start_time = time.time()
        
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        full_text = " ".join([seg.text for seg in segments])
        
        elapsed = time.time() - start_time
        print(f"Transcription completed in {elapsed:.2f} seconds")
        
        result = {
            "text": full_text.strip(),
            "language": info.language if hasattr(info, 'language') else language or "auto",
            "duration": info.duration if hasattr(info, 'duration') else None,
            "file_size_mb": round(file_size / (1024 * 1024), 2)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.route("/models", methods=["GET"])
def list_models():
    return jsonify({
        "current_model": MODEL_SIZE,
        "available_models": ["tiny", "base", "small", "medium", "large-v3"],
        "limits": {
            "max_file_size_mb": MAX_AUDIO_SIZE // (1024 * 1024),
            "max_recording_seconds": MAX_RECORDING_TIME,
            "max_transcription_seconds": MAX_TRANSCRIPTION_TIME
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
