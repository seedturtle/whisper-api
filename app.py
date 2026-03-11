import os
import io
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from faster_whisper import WhisperModel
import tempfile

app = Flask(__name__)
CORS(app)

# Model size: tiny, base, small, medium, large-v3
MODEL_SIZE = os.environ.get("MODEL_SIZE", "base")
MODEL_PATH = os.environ.get("MODEL_PATH", None)  # Optional: local model path

print(f"Loading Whisper model: {MODEL_SIZE}...")
if MODEL_PATH:
    model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8")
else:
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Model loaded successfully!")

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
    
    # Get language parameter (optional)
    language = request.form.get("language", None)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Run transcription
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            beam_size=5,
            vad_filter=True
        )
        
        # Collect all segments
        full_text = " ".join([seg.text for seg in segments])
        
        # Build response
        result = {
            "text": full_text.strip(),
            "language": info.language if hasattr(info, 'language') else language or "auto",
            "language_probability": info.language_probability if hasattr(info, 'language_probability') else None,
            "duration": info.duration if hasattr(info, 'duration') else None
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.route("/models", methods=["GET"])
def list_models():
    """List available model sizes"""
    return jsonify({
        "current_model": MODEL_SIZE,
        "available_models": ["tiny", "base", "small", "medium", "large-v3"]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
