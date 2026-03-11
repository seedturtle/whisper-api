import os
import io
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
from faster_whisper import WhisperModel
import tempfile

app = Flask(__name__)
CORS(app)

# Model size: tiny, base, small, medium, large-v3
MODEL_SIZE = os.environ.get("MODEL_SIZE", "base")
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
            transition: all 0.3s ease;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ Whisper 語音轉文字</h1>
        <p class="subtitle">基於 Faster Whisper - 本地運行，無需 API Key</p>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📁</div>
            <div class="upload-text">點擊或拖曳音檔到这里</div>
            <div class="upload-hint">支援 MP3, WAV, M4A, OGG, FLAC, WebM</div>
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
        </select>
        
        <button class="btn btn-primary" id="transcribeBtn" disabled>開始轉換</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div class="loading-text">正在轉換中，請稍候...</div>
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
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const transcribeBtn = document.getElementById('transcribeBtn');
        const language = document.getElementById('language');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const resultArea = document.getElementById('resultArea');
        const resultText = document.getElementById('resultText');
        
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
            loading.style.display = 'block';
            error.style.display = 'none';
            resultArea.style.display = 'none';
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            if (language.value) {
                formData.append('language', language.value);
            }
            
            try {
                const response = await fetch('/transcribe', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || '轉換失敗');
                }
                
                resultText.textContent = data.text || '沒有偵測到文字';
                resultArea.style.display = 'block';
            } catch (err) {
                showError(err.message);
            } finally {
                loading.style.display = 'none';
                transcribeBtn.disabled = false;
            }
        });
        
        function showError(msg) {
            error.textContent = msg;
            error.style.display = 'block';
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
    return render_template_string(HTML_TEMPLATE)

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
    
    language = request.form.get("language", None)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True
        )
        
        full_text = " ".join([seg.text for seg in segments])
        
        result = {
            "text": full_text.strip(),
            "language": info.language if hasattr(info, 'language') else language or "auto",
            "duration": info.duration if hasattr(info, 'duration') else None
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
        "available_models": ["tiny", "base", "small", "medium", "large-v3"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
